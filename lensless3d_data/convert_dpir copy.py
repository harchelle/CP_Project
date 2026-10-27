import argparse
import os
import sys
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

DPIR_PATH = "/4tb_hdd/Sumedha/CP_Project/DPIR" 
sys.path.append(DPIR_PATH)

from models.network_unet import UNetRes as DRUNet

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="Run DRUNet reconstructions with custom paths.")
parser.add_argument('--input', type=str, required=True, help="Path to the input .npy image file")
parser.add_argument('--psf', type=str, default='psf.npy', help="Path to the PSF .npy file")
parser.add_argument('--scene_dir', type=str, required=True, help="Path to the target scene directory (e.g., scenes/scene_01)")
args = parser.parse_args()

# --- Optical Constants from Forward Model ---
PSF_CENTER_HW = (808, 965)
CROP_SHAPE_HW = (1280, 1408)
SCENE_HW = (360, 640)
WIENER_K = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Load Pretrained DRUNet ---
model_zoo_path = os.path.join(DPIR_PATH, 'model_zoo')
os.makedirs(model_zoo_path, exist_ok=True)
model_path = os.path.join(model_zoo_path, 'drunet_color.pth')
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth", 
        model_path
    )

model = DRUNet(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', 
               downsample_mode="strideconv", upsample_mode="convtranspose")
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval().to(device)
for p in model.parameters():
    p.requires_grad = False

def run_drunet(x_bhwc, sigma, scale=1.0):
    x_bchw = x_bhwc.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    H_orig, W_orig = x_bchw.shape[2], x_bchw.shape[3]
    
    if scale != 1.0:
        h_s = int(round(H_orig * scale / 8) * 8)
        w_s = int(round(W_orig * scale / 8) * 8)
        x_in = F.interpolate(x_bchw, size=(h_s, w_s), mode='area')
    else:
        x_in = x_bchw

    noise_map = torch.full((x_in.shape[0], 1, x_in.shape[2], x_in.shape[3]), 
                           sigma, device=device)
    with torch.no_grad():
        out = model(torch.cat([x_in, noise_map], dim=1)).clamp(0.0, 1.0)
        
    if scale != 1.0:
        out = F.interpolate(out, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        
    return out.permute(0, 2, 3, 1)

# --- 2. Forward Operator & Wiener Helper ---
def build_operator(psf, device=None):
    psf = torch.as_tensor(np.asarray(psf, dtype=np.float32))
    cy, cx = PSF_CENTER_HW
    ch, cw = CROP_SHAPE_HW
    y0, x0 = cy - ch // 2, cx - cw // 2
    crop = psf[y0:y0 + ch, x0:x0 + cw]
    crop = crop / crop.sum()
    crop = torch.roll(crop, shifts=(-(cy - y0), -(cx - x0)), dims=(0, 1))
    return torch.fft.fft2(crop.to(device) if device is not None else crop)

def crop_roi(full_tensor):
    ch, cw = CROP_SHAPE_HW
    h, w = SCENE_HW
    top, left = (ch - h) // 2, (cw - w) // 2
    return full_tensor[:, top:top + h, left:left + w].clamp(0.0, 1.0)

# Load data using command-line arguments
print(f"Loading input data from: {args.input}")
raw = np.load(args.input).astype(np.float32)
if raw.shape[0] == 3:
    raw = np.transpose(raw, (1, 2, 0))

print(f"Loading PSF from: {args.psf}")
psf_data = np.load(args.psf)
H = build_operator(psf_data, device=device)
Hb = H[None, ..., None]
Hb_conj = torch.conj(Hb)
Hb_abs2 = Hb.abs() ** 2

meas_tensor = torch.from_numpy(raw).unsqueeze(0).to(device)
Y_fft = torch.fft.fft2(meas_tensor, dim=(1, 2))

# Baseline Wiener (K = 1e-4)
wiener_full_base = torch.real(torch.fft.ifft2(Hb_conj / (Hb_abs2 + WIENER_K) * Y_fft, dim=(1, 2)))
y_wiener = crop_roi(wiener_full_base)

# =========================================================================
# METHOD 1: Coarse-to-Fine Multi-Scale DRUNet on Wiener
# =========================================================================
coarse_clean = run_drunet(y_wiener, sigma=75.0 / 255.0, scale=0.5)
fine_blend = 0.75 * coarse_clean + 0.25 * y_wiener
recon_multiscale = run_drunet(fine_blend, sigma=35.0 / 255.0, scale=1.0)

# =========================================================================
# METHOD 2: True HQS-DPIR Annealed From High K
# =========================================================================
iter_num = 8
alphas = np.logspace(np.log10(1.5e-3), np.log10(1.5e-4), iter_num).astype(np.float32)
sigmas = np.logspace(np.log10(70.0 / 255.0), np.log10(20.0 / 255.0), iter_num).astype(np.float32)

ch, cw = CROP_SHAPE_HW
h, w = SCENE_HW
top, left = (ch - h) // 2, (cw - w) // 2

z_full = torch.real(torch.fft.ifft2(Hb_conj / (Hb_abs2 + alphas[0]) * Y_fft, dim=(1, 2))).clamp(0.0, 1.0)
z_full[:, top:top + h, left:left + w] = run_drunet(crop_roi(z_full), sigma=sigmas[0], scale=0.5)

for i in range(iter_num):
    alpha_k = alphas[i]
    Z_fft = torch.fft.fft2(z_full, dim=(1, 2))
    X_fft = (Hb_conj * Y_fft + alpha_k * Z_fft) / (Hb_abs2 + alpha_k)
    x_full = torch.real(torch.fft.ifft2(X_fft, dim=(1, 2)))
    
    s_factor = 0.5 if i < 2 else 1.0
    z_roi = run_drunet(crop_roi(x_full), sigma=sigmas[i], scale=s_factor)
    
    z_full = x_full.clamp(0.0, 1.0)
    z_full[:, top:top + h, left:left + w] = z_roi

recon_hqs_annealed = z_roi

# =========================================================================
# METHOD 3: Iterative Proximal Relaxation Directly on Wiener
# =========================================================================
z_prox = run_drunet(y_wiener, sigma=70.0 / 255.0, scale=0.5)
sigmas_prox = np.logspace(np.log10(50.0 / 255.0), np.log10(15.0 / 255.0), 6).astype(np.float32)

for i, sig in enumerate(sigmas_prox):
    x_prox = 0.15 * y_wiener + 0.85 * z_prox
    z_prox = run_drunet(x_prox, sigma=sig, scale=1.0)

recon_prox = z_prox

# Extract base file name (e.g., "0001") to name the output PNG file
base_name = os.path.splitext(os.path.basename(args.input))[0] + ".png"

def save_tensor_image(tensor_bhwc, method_folder):
    target_dir = os.path.join(args.scene_dir, method_folder)
    os.makedirs(target_dir, exist_ok=True)
    img_np = (tensor_bhwc[0].detach().cpu().numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(img_np)
    out_path = os.path.join(target_dir, base_name)
    img.save(out_path)
    print(f"Saved: {out_path}")

# Save outputs into respective scene subfolders
save_tensor_image(recon_multiscale, "mscale_dru_wien")
save_tensor_image(recon_hqs_annealed, "true_anneal")
save_tensor_image(recon_prox, "proximal_hqs")