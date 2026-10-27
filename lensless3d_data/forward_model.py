"""Lensless camera forward model used to generate the measurements.

    y = clamp( H * pad(x) + n , 0, 1 )

x      : scene image, (B, 360, 640, 3) float in [0, 1]
pad    : x is placed at the centre of a black 1280x1408 canvas
H *    : circular 2D convolution with the cropped, sum-normalised PSF (same PSF
         for all three colour channels)
n      : white Gaussian noise, per-image SNR of about 25 dB w.r.t. H * pad(x)

A simple baseline inversion is the Wiener filter:
    w = crop( ifft( conj(H) / (|H|^2 + K) * fft(y) ) ),  K = 1e-4,  clamped to [0, 1]

Measurements are stored as fp16 arrays of shape (3, 1280, 1408), channels first.
"""

import numpy as np
import torch

PSF_CENTER_HW = (808, 965)      # optical centre in psf.npy coordinates
CROP_SHAPE_HW = (1280, 1408)    # sensor area the measurements cover
SCENE_HW = (360, 640)
WIENER_K = 1e-4


def build_operator(psf, device=None):
    """FFT of the cropped, sum-normalised PSF, rolled so its centre sits at index (0, 0)."""
    psf = torch.as_tensor(np.asarray(psf, dtype=np.float32))
    cy, cx = PSF_CENTER_HW
    ch, cw = CROP_SHAPE_HW
    y0, x0 = cy - ch // 2, cx - cw // 2
    crop = psf[y0:y0 + ch, x0:x0 + cw]
    crop = crop / crop.sum()
    crop = torch.roll(crop, shifts=(-(cy - y0), -(cx - x0)), dims=(0, 1))
    return torch.fft.fft2(crop.to(device) if device is not None else crop)


def _pad(img):
    B, h, w, C = img.shape
    ch, cw = CROP_SHAPE_HW
    top, left = (ch - h) // 2, (cw - w) // 2
    canvas = img.new_zeros((B, ch, cw, C))
    canvas[:, top:top + h, left:left + w] = img
    return canvas, (top, left)


def forward(img, H):
    """Noise-free measurement H * pad(img). img: (B, 360, 640, 3) -> (B, 1280, 1408, 3)."""
    canvas, _ = _pad(img)
    return torch.real(torch.fft.ifft2(torch.fft.fft2(canvas, dim=(1, 2)) * H[None, ..., None], dim=(1, 2)))


def wiener(meas, H, K=WIENER_K):
    """Wiener reconstruction of a measurement (B, 1280, 1408, 3) -> (B, 360, 640, 3)."""
    Hb = H[None, ..., None]
    rec = torch.real(torch.fft.ifft2(torch.conj(Hb) / (Hb.abs() ** 2 + K) * torch.fft.fft2(meas, dim=(1, 2)),
                                     dim=(1, 2)))
    ch, cw = CROP_SHAPE_HW
    h, w = SCENE_HW
    top, left = (ch - h) // 2, (cw - w) // 2
    return rec[:, top:top + h, left:left + w].clamp(0, 1)


def load_measurement(path):
    """measurements/NNNN.npy -> (1280, 1408, 3) float32 tensor."""
    return torch.from_numpy(np.load(path).astype(np.float32)).permute(1, 2, 0)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PIL import Image

    # Write the Wiener reconstruction of every train measurement of a scene as PNGs.
    #   python forward_model.py scenes/scene_01 scenes/scene_01/images
    scene = Path(sys.argv[1] if len(sys.argv) > 1 else "scenes/scene_08")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else scene / "images")
    out.mkdir(parents=True, exist_ok=True)
    H = build_operator(np.load("/4tb_hdd/Sumedha/CP_Project/lensless3d_data/psf.npy"))
    paths = sorted((scene / "measurements").glob("*.npy"))
    for m in paths:
        rec = wiener(load_measurement(m)[None], H)[0].numpy()
        Image.fromarray((rec * 255 + 0.5).astype(np.uint8)).save(out / f"{m.stem}.png")
    print(f"wrote {len(paths)} Wiener images to {out}")
