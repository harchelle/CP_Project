
"""Pack rendered test views into submission.csv safely."""

import csv
import os
import sys
import tempfile
import cv2

EXPECTED_SCENES = [
    f"scene_{i:02d}" for i in range(1, 9)
]

EXPECTED_VIEWS = [
    f"{i:04d}" for i in range(0, 64, 8)
]

H, W = 360, 640

RUN_DIR = "/4tb_hdd/Sumedha/CP_Project/runs/20261015_102627"


def collect_scene_files(root):
    """Validate all scene folders and view filenames before writing."""
    scene_files = {}

    for scene in EXPECTED_SCENES:
        scene_path = os.path.join(
            root, scene, "test_cameras", "ours_30000", "renders"
        )

        if not os.path.isdir(scene_path):
            raise ValueError(f"Missing renders folder: {scene_path}")

        image_files = sorted(
            f for f in os.listdir(scene_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        views = [os.path.splitext(f)[0] for f in image_files]

        missing = sorted(set(EXPECTED_VIEWS) - set(views))
        extra = sorted(set(views) - set(EXPECTED_VIEWS))
        duplicates = sorted({
            v for v in views if views.count(v) > 1
        })

        if missing or extra or duplicates:
            raise ValueError(
                f"Invalid views in {scene_path}\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}\n"
                f"Duplicates: {duplicates}"
            )

        scene_files[scene] = image_files

    return scene_files


def write_submission(root, output):
    # Validate all scene directories before starting the CSV write.
    scene_files = collect_scene_files(root)

    output_path = os.path.abspath(output)
    output_dir = os.path.dirname(output_path)

    if not os.path.isdir(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")

    # Write to a temporary file in the same directory.
    fd, temp_path = tempfile.mkstemp(
        prefix="submission_",
        suffix=".tmp",
        dir=output_dir
    )

    n_images = 0
    n_rows = 0

    try:
        with os.fdopen(
            fd, "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["id", "image"])

            for scene in EXPECTED_SCENES:
                scene_path = os.path.join(
                    root, scene, "test_cameras",
                    "ours_30000", "renders"
                )

                for fname in scene_files[scene]:
                    view = os.path.splitext(fname)[0]
                    img_path = os.path.join(scene_path, fname)

                    img = cv2.imread(img_path)

                    if img is None:
                        raise ValueError(f"Cannot read image: {img_path}")

                    if img.shape != (H, W, 3):
                        raise ValueError(
                            f"{img_path} has shape {img.shape}; "
                            f"expected {(H, W, 3)}"
                        )

                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # Stream rows instead of storing all images in memory.
                    for y in range(H):
                        pixels = " ".join(
                            map(str, rgb[y].reshape(-1))
                        )
                        row_id = f"{scene}_{view}_r{y:03d}"

                        writer.writerow([row_id, pixels])
                        n_rows += 1

                    n_images += 1
                    print(
                        f"Processed {n_images}/64 images: "
                        f"{scene}/{view} "
                        f"({n_rows} rows written)"
                    )

        # Replace the final file only after a successful complete write.
        os.replace(temp_path, output_path)

        print("\nCSV generated successfully!")
        print(f"Output: {output_path}")
        print(f"Images processed: {n_images}")
        print(f"Data rows: {n_rows}")
        print(f"Total CSV records including header: {n_rows + 1}")

    except BaseException:
        # Preserve the previous final CSV if anything goes wrong.
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


if __name__ == "__main__":
    output = (
        sys.argv[1] if len(sys.argv) > 1
        else "submission.csv"
    )

    if not os.path.isdir(RUN_DIR):
        sys.exit(f"Run directory not found: {RUN_DIR}")

    write_submission(RUN_DIR, output)
