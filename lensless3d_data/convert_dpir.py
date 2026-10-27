#!/usr/bin/env python3
"""Run lensless HQS-DPIR reconstruction for one measurement .npy file.

The reconstruction follows the active HQS-DPIR method in the supplied reference
script: balanced Wiener initialization, FFT data-fidelity updates on the full
measurement canvas, reflect-padded DRUNet denoising on the scene ROI, and
chroma-only cleanup to reduce color speckle.
"""

import argparse
import os
import sys
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# -----------------------------------------------------------------------------
# Arguments and constants
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Reconstruct a lensless measurement using HQS-DPIR."
)
parser.add_argument(
    "--input", type=str, required=True,
    help="Path to the input measurement .npy file (full 1280x1408 canvas).",
)
parser.add_argument(
    "--psf", type=str, default="psf.npy",
    help="Path to the PSF .npy file (default: psf.npy).",
)
parser.add_argument(
    "--scene_dir", type=str, required=True,
    help="Scene output directory; the result is saved in dpir_clean_sharp/.",
)
args = parser.parse_args()

DPIR_PATH = "/4tb_hdd/Sumedha/CP_Project/DPIR"
if not os.path.isdir(DPIR_PATH):
    raise FileNotFoundError(
        f"DPIR repository was not found at {DPIR_PATH!r}. "
        "Update DPIR_PATH in convert_dpir.py to point to your DPIR checkout."
    )
sys.path.append(DPIR_PATH)
from models.network_unet import UNetRes as DRUNet  # noqa: E402

PSF_CENTER_HW = (808, 965)
CROP_SHAPE_HW = (1280, 1408)
SCENE_HW = (360, 640)

# Scene ROI within the full PSF/measurement canvas, matching the reference code.
CANVAS_H, CANVAS_W = CROP_SHAPE_HW
SCENE_H, SCENE_W = SCENE_HW
ROI_TOP = (CANVAS_H - SCENE_H) // 2  # 460
ROI_LEFT = (CANVAS_W - SCENE_W) // 2  # 384
ROI_BOTTOM = ROI_TOP + SCENE_H
ROI_RIGHT = ROI_LEFT + SCENE_W

# HQS-DPIR settings copied from the active method in the supplied reference.
ITER_NUM = 20
WIENER_REG = 3e-4
PAD_SIZE = 32


def safe_load_npy(path):
    """Load numeric .npy data, including a scalar object containing a dict."""
    data = np.load(path, allow_pickle=True)
    if data.dtype == object and data.ndim == 0:
        item = data.item()
        if isinstance(item, dict):
            if not item:
                raise ValueError(f"The dictionary in {path!r} is empty.")
            return np.asarray(next(iter(item.values())), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def align_measurement(raw, path):
    """Convert supported measurement layouts to a (1, 3, H, W) tensor."""
    if raw.shape == (CANVAS_H, CANVAS_W, 3):          # H, W, C
        chw = np.transpose(raw, (2, 0, 1))
    elif raw.shape == (3, CANVAS_H, CANVAS_W):        # C, H, W
        chw = raw
    elif raw.shape == (CANVAS_W, CANVAS_H, 3):        # W, H, C
        chw = np.transpose(raw, (2, 1, 0))
    elif raw.shape == (3, CANVAS_W, CANVAS_H):        # C, W, H
        chw = np.transpose(raw, (0, 2, 1))
    else:
        raise ValueError(
            f"Unexpected measurement shape {raw.shape} in {path!r}. Expected "
            f"({CANVAS_H}, {CANVAS_W}, 3), (3, {CANVAS_H}, {CANVAS_W}), "
            f"({CANVAS_W}, {CANVAS_H}, 3), or (3, {CANVAS_W}, {CANVAS_H})."
        )
    return torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)


def prepare_psf(psf_full):
    """Crop and normalize the PSF, then center its origin for FFT convolution."""
    if psf_full.ndim == 3:
        if psf_full.shape[-1] in (1, 3):
            psf_full = psf_full[:, :, 0]
        elif psf_full.shape[0] in (1, 3):
            psf_full = psf_full[0]
        else:
            raise ValueError(
                f"Cannot identify the channel dimension in PSF shape {psf_full.shape}."
            )
    if psf_full.ndim != 2:
        raise ValueError(f"PSF must be 2D after channel selection; got {psf_full.shape}.")

    cy, cx = PSF_CENTER_HW
    crop_h, crop_w = CROP_SHAPE_HW
    y0, x0 = cy - crop_h // 2, cx - crop_w // 2
    y1, x1 = y0 + crop_h, x0 + crop_w
    if y0 < 0 or x0 < 0 or y1 > psf_full.shape[0] or x1 > psf_full.shape[1]:
        raise ValueError(
            f"PSF shape {psf_full.shape} is too small for the requested crop "
            f"[{y0}:{y1}, {x0}:{x1}]."
        )

    psf_crop = np.asarray(psf_full[y0:y1, x0:x1], dtype=np.float32)
    psf_sum = float(np.sum(psf_crop))
    if not np.isfinite(psf_sum) or abs(psf_sum) < 1e-12:
        raise ValueError("Cropped PSF has a zero or invalid sum; cannot normalize it.")
    psf_crop /= psf_sum
    # Equivalent to the ifftshift operation in the supplied reference code.
    return np.fft.ifftshift(psf_crop).astype(np.float32, copy=False)


def suppress_chroma_noise(rgb_tensor, kernel_size=5):
    """Smooth Cb/Cr only, retaining the original luminance detail."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")

    r = rgb_tensor[:, 0:1]
    g = rgb_tensor[:, 1:2]
    b = rgb_tensor[:, 2:3]

    # YCbCr luminance and chrominance.
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b

    pad = kernel_size // 2
    cb_clean = F.avg_pool2d(
        F.pad(cb, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1
    )
    cr_clean = F.avg_pool2d(
        F.pad(cr, (pad, pad, pad, pad), mode="reflect"), kernel_size, stride=1
    )

    r_out = y + 1.402 * cr_clean
    g_out = y - 0.344136 * cb_clean - 0.714136 * cr_clean
    b_out = y + 1.772 * cb_clean
    return torch.clamp(torch.cat((r_out, g_out, b_out), dim=1), 0.0, 1.0)


def save_tensor_image(tensor_bchw, output_path):
    """Save the first BCHW image as an 8-bit RGB PNG."""
    image = tensor_bchw[0].detach().float().clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    image_u8 = np.round(image * 255.0).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(image_u8, mode="RGB").save(output_path)
    print(f"Saved DPIR reconstruction: {output_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. Load the pretrained DRUNet color denoiser
    # -------------------------------------------------------------------------
    model_zoo_path = os.path.join(DPIR_PATH, "model_zoo")
    os.makedirs(model_zoo_path, exist_ok=True)
    model_path = os.path.join(model_zoo_path, "drunet_color.pth")
    if not os.path.exists(model_path):
        url = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth"
        print(f"Downloading pretrained weights from {url}")
        urllib.request.urlretrieve(url, model_path)

    model = DRUNet(
        in_nc=4,
        out_nc=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------------------
    # 2. Load the full measurement and prepare the cropped PSF
    # -------------------------------------------------------------------------
    print(f"Loading measurement from: {args.input}")
    raw = safe_load_npy(args.input)
    Y = align_measurement(raw, args.input).to(device=device, dtype=torch.float32)

    print(f"Loading PSF from: {args.psf}")
    psf_full = safe_load_npy(args.psf)
    psf_shifted = prepare_psf(psf_full)
    if tuple(Y.shape[-2:]) != CROP_SHAPE_HW:
        raise ValueError(
            f"Measurement canvas {tuple(Y.shape[-2:])} does not match PSF canvas "
            f"{CROP_SHAPE_HW}."
        )

    # PSF is shared across RGB channels; broadcasting handles the channel axis.
    K = torch.from_numpy(psf_shifted).to(device).unsqueeze(0).unsqueeze(0)
    FK = torch.fft.rfft2(K)
    FK_conj = torch.conj(FK)
    FK2 = torch.abs(FK) ** 2
    FY = torch.fft.rfft2(Y)

    print(f"Measurement tensor: {tuple(Y.shape)} | PSF tensor: {tuple(K.shape)}")
    print(f"Scene ROI: rows {ROI_TOP}:{ROI_BOTTOM}, cols {ROI_LEFT}:{ROI_RIGHT}")

    # -------------------------------------------------------------------------
    # 3. HQS-DPIR schedule from the active method in the reference code
    # -------------------------------------------------------------------------
    sigmas = torch.logspace(
        np.log10(35.0 / 255.0), np.log10(4.0 / 255.0), ITER_NUM, device=device
    )
    rhos = torch.logspace(
        np.log10(2.5e-4), np.log10(3.0e-3), ITER_NUM, device=device
    )

    # Balanced Wiener initialization: use it only to initialize the scene ROI.
    Z_wiener = torch.fft.irfft2(
        (FK_conj * FY) / (FK2 + WIENER_REG), s=Y.shape[-2:]
    )
    x_wiener = torch.clamp(
        Z_wiener[:, :, ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT], 0.0, 1.0
    )

    # Keep the full canvas zero outside the target scene support.
    X_canvas = torch.zeros_like(Y)
    X_canvas[:, :, ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = x_wiener

    # -------------------------------------------------------------------------
    # 4. Iterative HQS data-fidelity update + reflect-padded DRUNet prior
    # -------------------------------------------------------------------------
    with torch.no_grad():
        for i in range(ITER_NUM):
            rho = rhos[i]
            sigma = sigmas[i]

            # Exact FFT data-fidelity update over the entire measurement canvas.
            FX = torch.fft.rfft2(X_canvas)
            Z = torch.fft.irfft2(
                (FK_conj * FY + rho * FX) / (FK2 + rho),
                s=Y.shape[-2:],
            )

            # Only denoise the 360x640 scene ROI.
            x_crop = torch.clamp(
                Z[:, :, ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT], 0.0, 1.0
            )

            # Early-iteration chroma cleanup reduces rainbow color spikes.
            if i < 10:
                x_crop = suppress_chroma_noise(x_crop, kernel_size=5)

            # Reflect padding avoids artificial dark borders in the denoiser.
            x_padded = F.pad(
                x_crop,
                (PAD_SIZE, PAD_SIZE, PAD_SIZE, PAD_SIZE),
                mode="reflect",
            )
            noise_map = sigma.view(1, 1, 1, 1).expand(
                x_padded.shape[0], 1, x_padded.shape[2], x_padded.shape[3]
            )
            denoiser_input = torch.cat((x_padded, noise_map), dim=1)
            x_denoised_padded = model(denoiser_input)
            x_crop_denoised = torch.clamp(
                x_denoised_padded[
                    :, :, PAD_SIZE:-PAD_SIZE, PAD_SIZE:-PAD_SIZE
                ],
                0.0,
                1.0,
            )

            # Reimpose zero support outside the 360x640 scene at every iteration.
            X_canvas = torch.zeros_like(Z)
            X_canvas[:, :, ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT] = x_crop_denoised

            if (i + 1) % 4 == 0 or i == 0:
                print(
                    f"Iter {i + 1:02d}/{ITER_NUM} | "
                    f"rho: {rho.item():.2e} | sigma: {sigma.item() * 255.0:.1f}/255"
                )

        # Final light chroma cleanup on the reconstructed ROI.
        x_final_tensor = suppress_chroma_noise(
            X_canvas[:, :, ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT],
            kernel_size=3,
        )

    # -------------------------------------------------------------------------
    # 5. Save the final reconstruction
    # -------------------------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(args.input))[0] + ".png"
    output_path = os.path.join(args.scene_dir, "dpir_clean_sharp", base_name)
    save_tensor_image(x_final_tensor, output_path)


if __name__ == "__main__":
    main()
