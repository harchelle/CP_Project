#!/usr/bin/env python3
"""Generate synthetic lensless measurements directly from DIV2K images.

This version intentionally RESIZES THE ENTIRE SOURCE IMAGE to 640x360
(non-uniformly, i.e. it squishes the original aspect ratio), rather than cropping.
It then generates the lensless measurement with the project's specified PSF
forward model and 25 dB AWGN.

It saves ONLY measurement .npy files plus a CSV manifest. It does NOT save
resized ground-truth PNGs or Wiener-deconvolved targets.

Default split:
  - 720 train images and 80 validation images from DIV2K_train_HR (seed 42)
  - 100 test images from DIV2K_valid_HR

Default roots follow the custom_rgb_v3 project layout. Output defaults to a NEW
folder so it does not overwrite or mix with an existing Stage-1 dataset.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

try:
    from scipy import fft as sp_fft
except ImportError:
    sp_fft = np.fft

VALID_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SOURCE_H, SOURCE_W = 360, 640
MEAS_H, MEAS_W = 1280, 1408
PAD_TOP, PAD_LEFT = 460, 384
PSF_CENTER_Y, PSF_CENTER_X = 808, 965

DEFAULT_RAW_ROOT = Path("/4tb_hdd/Sumedha/PhoCoLens/data/custom_rgb_v2/div2k/raw")
DEFAULT_OUTPUT_ROOT = Path("/4tb_hdd/Sumedha/PhoCoLens/data/custom_rgb_v3_measurements_only")
DEFAULT_PSF = Path("/4tb_hdd/Sumedha/CP_Project/lensless3d_data/psf.npy")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT,
                   help="Contains DIV2K_train_HR/ and DIV2K_valid_HR/.")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                   help="Output root for {split}/measurements and manifest.")
    p.add_argument("--psf", type=Path, default=DEFAULT_PSF,
                   help="Full measured PSF .npy, expected shape 1518x2012.")
    p.add_argument("--snr-db", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=2026,
                   help="Base seed for per-image AWGN; same IDs get reproducible noise.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the 720/80 train/validation split from DIV2K_train_HR.")
    p.add_argument("--train-count", type=int, default=720)
    p.add_argument("--val-count", type=int, default=80)
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                   default=["train", "val", "test"])
    p.add_argument("--sample-id", default=None,
                   help="Optional: process one source filename stem, e.g. 0001.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing measurement files.")
    return p.parse_args()


def image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {folder}")
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VALID_EXTS)


def per_sample_seed(base_seed: int, split: str, sample_id: str) -> int:
    split_offset = {"train": 0, "val": 1_000_000, "test": 2_000_000}[split]
    try:
        number = int(sample_id)
    except ValueError:
        digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
        number = int.from_bytes(digest[:4], "little")
    seq = np.random.SeedSequence([int(base_seed), split_offset, number])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def make_splits(raw_root: Path, split_seed: int, train_count: int, val_count: int):
    train_src = image_files(raw_root / "DIV2K_train_HR")
    test_src = image_files(raw_root / "DIV2K_valid_HR")
    if len(train_src) != 800:
        raise ValueError(f"Expected 800 DIV2K_train_HR images; found {len(train_src)}")
    if len(test_src) != 100:
        raise ValueError(f"Expected 100 DIV2K_valid_HR images; found {len(test_src)}")
    if train_count <= 0 or val_count <= 0 or train_count + val_count != 800:
        raise ValueError("train-count + val-count must equal 800 (default: 720 + 80).")
    shuffled = train_src.copy()
    random.Random(split_seed).shuffle(shuffled)
    train = sorted(shuffled[:train_count])
    val = sorted(shuffled[train_count:train_count + val_count])
    return {"train": train, "val": val, "test": test_src}


def make_normalized_psf_transfer(psf_path: Path):
    full = np.load(psf_path).astype(np.float32)
    if full.shape != (1518, 2012):
        raise ValueError(f"Expected PSF shape (1518,2012), got {full.shape}")
    if not np.isfinite(full).all():
        raise ValueError("PSF contains NaN/Inf")
    y0 = PSF_CENTER_Y - MEAS_H // 2
    x0 = PSF_CENTER_X - MEAS_W // 2
    crop = full[y0:y0 + MEAS_H, x0:x0 + MEAS_W]
    if crop.shape != (MEAS_H, MEAS_W):
        raise ValueError(f"Unexpected PSF crop shape: {crop.shape}")
    total = float(crop.astype(np.float64).sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid PSF crop sum: {total}")
    normalized = (crop.astype(np.float64) / total).astype(np.float32)
    # Centre the PSF origin at [0,0] for circular convolution.
    transfer = sp_fft.fft2(np.fft.ifftshift(normalized)).astype(np.complex64, copy=False)
    return transfer, total


def load_and_squish(src_path: Path) -> np.ndarray:
    """Read source RGB and resize the ENTIRE image to HxW=(360,640), distorting AR."""
    bgr = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read DIV2K image: {src_path}")
    # Intentional non-uniform resize: no crop, padding, or aspect-ratio preservation.
    resized_bgr = cv2.resize(bgr, (SOURCE_W, SOURCE_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if rgb.shape != (SOURCE_H, SOURCE_W, 3):
        raise RuntimeError(f"Unexpected resized image shape {rgb.shape} for {src_path}")
    if not np.isfinite(rgb).all() or not np.any(rgb):
        raise ValueError(f"Resized DIV2K image is invalid or all black: {src_path}")
    return rgb


def generate_measurement(source_rgb: np.ndarray, transfer: np.ndarray,
                         snr_db: float, seed: int):
    padded = np.zeros((MEAS_H, MEAS_W, 3), dtype=np.float32)
    padded[PAD_TOP:PAD_TOP + SOURCE_H, PAD_LEFT:PAD_LEFT + SOURCE_W, :] = source_rgb
    clean = sp_fft.ifft2(
        sp_fft.fft2(padded, axes=(0, 1)) * transfer[:, :, None], axes=(0, 1)
    ).real.astype(np.float32)
    clean_power = float(np.mean(clean.astype(np.float64) ** 2))
    if not np.isfinite(clean_power) or clean_power <= 0:
        raise ValueError(f"Invalid clean measurement power: {clean_power}")
    sigma = float(np.sqrt(clean_power / (10.0 ** (snr_db / 10.0))))
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=clean.shape).astype(np.float32)
    unclipped = clean + noise
    measurement = np.clip(unclipped, 0.0, 1.0).astype(np.float32)
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    achieved_snr = float(10.0 * np.log10(clean_power / noise_power))
    clipped_fraction = float(np.mean((unclipped < 0.0) | (unclipped > 1.0)))
    return measurement, sigma, achieved_snr, clipped_fraction


def main() -> int:
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    psf_path = args.psf.expanduser().resolve()
    if not psf_path.is_file():
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    if not raw_root.is_dir():
        raise FileNotFoundError(f"DIV2K raw root does not exist: {raw_root}")
    if not np.isfinite(args.snr_db):
        raise ValueError("snr-db must be finite")

    splits = make_splits(raw_root, args.split_seed, args.train_count, args.val_count)
    transfer, psf_crop_sum = make_normalized_psf_transfer(psf_path)
    print("Raw DIV2K root:", raw_root)
    print("Output root:", output_root)
    print("PSF:", psf_path)
    print("PSF crop sum before normalization:", psf_crop_sum)
    print("SNR (dB):", args.snr_db)
    print("Resize: entire source image squished directly to 640x360 (no crop)")
    print("Outputs: measurement .npy only; no GTS PNGs and no Wiener targets")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "synthetic_measurements_only.csv"
    fieldnames = ["split", "sample_id", "source_file", "measurement_file", "seed",
                  "source_width", "source_height", "resized_width", "resized_height",
                  "snr_requested_db", "snr_achieved_db_before_clipping", "noise_sigma",
                  "clipped_fraction", "psf_crop_sum", "status"]
    manifest_rows = []
    found_requested_sample = args.sample_id is None

    for split in args.splits:
        files = splits[split]
        if args.sample_id is not None:
            files = [p for p in files if p.stem == args.sample_id]
            if not files:
                continue
            found_requested_sample = True
        out_dir = output_root / split / "measurements"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split}] processing {len(files)} images", flush=True)
        for idx, src in enumerate(files, start=1):
            sample_id = src.stem
            out_path = out_dir / f"{sample_id}.npy"
            seed = per_sample_seed(args.seed, split, sample_id)
            if out_path.exists() and not args.overwrite:
                print(f"  skip existing: {out_path}", flush=True)
                src_bgr_existing = cv2.imread(str(src), cv2.IMREAD_COLOR)
                if src_bgr_existing is None:
                    raise OSError(f"Could not read DIV2K image: {src}")
                source_h_existing, source_w_existing = src_bgr_existing.shape[:2]
                manifest_rows.append({
                    "split": split, "sample_id": sample_id, "source_file": str(src),
                    "measurement_file": str(out_path), "seed": seed,
                    "source_width": source_w_existing, "source_height": source_h_existing,
                    "resized_width": SOURCE_W, "resized_height": SOURCE_H,
                    "snr_requested_db": args.snr_db,
                    "snr_achieved_db_before_clipping": "", "noise_sigma": "",
                    "clipped_fraction": "", "psf_crop_sum": psf_crop_sum,
                    "status": "skipped_existing",
                })
                continue

            source_rgb = load_and_squish(src)
            # Read original source dimensions for the manifest.
            src_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            source_h, source_w = src_bgr.shape[:2]
            measurement, sigma, achieved_snr, clipped_fraction = generate_measurement(
                source_rgb, transfer, args.snr_db, seed
            )
            # Save float32 CHW [0,1], matching the custom_rgb_v3 measurement loader.
            tmp_path = out_path.with_name(out_path.name + ".tmp")
            with tmp_path.open("wb") as f:
                np.save(f, measurement.transpose(2, 0, 1).astype(np.float32))
            tmp_path.replace(out_path)

            manifest_rows.append({
                "split": split, "sample_id": sample_id, "source_file": str(src),
                "measurement_file": str(out_path), "seed": seed,
                "source_width": source_w, "source_height": source_h,
                "resized_width": SOURCE_W, "resized_height": SOURCE_H,
                "snr_requested_db": args.snr_db,
                "snr_achieved_db_before_clipping": achieved_snr,
                "noise_sigma": sigma, "clipped_fraction": clipped_fraction,
                "psf_crop_sum": psf_crop_sum, "status": "generated",
            })
            if idx == 1 or idx % 25 == 0 or idx == len(files):
                print(f"  {idx:4d}/{len(files)} | {sample_id} | sigma={sigma:.6g} "
                      f"SNR={achieved_snr:.3f} dB | clip={clipped_fraction:.4f}", flush=True)

    if args.sample_id is not None and not found_requested_sample:
        raise FileNotFoundError(
            f"sample-id {args.sample_id!r} was not found in the selected split(s): {args.splits}"
        )

    # Keep a manifest of generated/skipped files; no GTS or Wiener targets are created.
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\nGeneration complete.")
    for split in args.splits:
        n = len(list((output_root / split / "measurements").glob("*.npy")))
        print(f"{split}: {n} measurement files under {output_root / split / 'measurements'}")
    print("Manifest:", manifest_path)
    print("Measurements are float32 CHW [0,1], shape (3,1280,1408).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
