
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import fftconvolve


# ---------------- CONFIGURATION ----------------

INPUT_ROOT = Path("/path/to/dataset")
OUTPUT_ROOT = Path("/path/to/output")
PSF_PATH = Path("/path/to/psf.npy")

# Wiener regularization parameter.
# Increase it for stronger noise suppression.
BALANCE = 0.01

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


# ---------------- WIENER DECONVOLUTION ----------------

def wiener_deconvolution(image, psf, balance=0.01):
    """
    Wiener deconvolution in the Fourier domain.

    image: 2D floating-point image
    psf:   2D PSF array
    balance: non-negative regularization parameter
    """

    image = np.asarray(image, dtype=np.float64)
    psf = np.asarray(psf, dtype=np.float64)

    if image.ndim != 2 or psf.ndim != 2:
        raise ValueError("Image and PSF must both be 2D arrays.")

    if not np.all(np.isfinite(psf)) or psf.sum() <= 0:
        raise ValueError("PSF must be finite and have a positive sum.")

    if balance < 0:
        raise ValueError("BALANCE must be non-negative.")

    # Normalize PSF so its total energy is 1.
    psf = psf / psf.sum()

    # Pad PSF to image dimensions, then center its origin.
    padded_psf = np.zeros_like(image)
    kh, kw = psf.shape

    if kh > image.shape[0] or kw > image.shape[1]:
        raise ValueError("PSF must not be larger than the image.")

    padded_psf[:kh, :kw] = psf
    padded_psf = np.roll(
        padded_psf,
        shift=(-(kh // 2), -(kw // 2)),
        axis=(0, 1),
    )

    # Fourier-domain Wiener filter.
    image_fft = np.fft.fft2(image)
    psf_fft = np.fft.fft2(padded_psf)

    denominator = np.abs(psf_fft) ** 2 + balance
    restored_fft = (
        np.conj(psf_fft) * image_fft / denominator
    )

    restored = np.fft.ifft2(restored_fft).real

    return restored


# ---------------- IMAGE PROCESSING ----------------

def restore_image(image_array, psf, balance):
    """Restore grayscale or RGB/RGBA images."""

    original_dtype = image_array.dtype
    image = image_array.astype(np.float64)

    # Process each color channel independently.
    if image.ndim == 2:
        restored = wiener_deconvolution(image, psf, balance)

    elif image.ndim == 3:
        channels = [
            wiener_deconvolution(image[..., c], psf, balance)
            for c in range(image.shape[2])
        ]
        restored = np.stack(channels, axis=-1)

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Clip to valid range for integer image formats.
    if np.issubdtype(original_dtype, np.integer):
        limits = np.iinfo(original_dtype)
        restored = np.clip(restored, limits.min, limits.max)

    return restored.astype(original_dtype)


def main():
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Input directory not found: {INPUT_ROOT}")

    if not PSF_PATH.is_file():
        raise FileNotFoundError(f"PSF file not found: {PSF_PATH}")

    psf = np.load(PSF_PATH).astype(np.float64)

    scene_folders = sorted(
        p for p in INPUT_ROOT.iterdir()
        if p.is_dir()
    )

    total = 0

    for scene_folder in scene_folders:
        images_folder = scene_folder / "images"

        if not images_folder.is_dir():
            print(f"Skipping {scene_folder.name}: no images folder")
            continue

        # Preserve scene name in output.
        output_folder = (
            OUTPUT_ROOT / scene_folder.name / "images"
        )
        output_folder.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in images_folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    image_array = np.array(img)

                restored = restore_image(
                    image_array, psf, BALANCE
                )

                output_path = output_folder / image_path.name
                Image.fromarray(restored).save(output_path)

                print(f"Saved: {output_path}")
                total += 1

            except Exception as exc:
                print(f"Error processing {image_path}: {exc}")

    print(f"\nDone. Successfully processed {total} images.")


if __name__ == "__main__":
    main()
