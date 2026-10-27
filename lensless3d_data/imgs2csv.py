"""Pack a folder of rendered test views into submission.csv.
"""

import os
import sys
import cv2
import pandas as pd


EXPECTED_SCENES = [
    "scene_01",
    "scene_02",
    "scene_03",
    "scene_04",
    "scene_05",
    "scene_06",
    "scene_07",
    "scene_08"
]

EXPECTED_VIEWS = [f"{i:04d}" for i in range(0, 64, 8)]  # 0000, 0008, ..., 0056

H, W = 360, 640

# Point this to your run folder (e.g., where scene_01, scene_02 reside)
RUN_DIR = "/4tb_hdd/Sumedha/CP_Project/runs/latest"


def image_to_rows(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [" ".join(map(str, r.flatten())) for r in img]


def images_to_df(root):
    rows = []
    n_images = 0

    for scene in EXPECTED_SCENES:
        # Updated path to match your nested structure:
        # runs/20261015_102627/scene_XX/test_cameras/ours_30000/renders
        scene_path = os.path.join(root, scene, "test_cameras", "ours_30000", "renders")
        
        if not os.path.isdir(scene_path):
            raise ValueError(f"Missing renders folder for scene: {scene_path}")

        image_files = sorted(
            f for f in os.listdir(scene_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        views = [os.path.splitext(f)[0] for f in image_files]
        missing = sorted(set(EXPECTED_VIEWS) - set(views))
        extra = sorted(set(views) - set(EXPECTED_VIEWS))
        dupes = sorted({v for v in views if views.count(v) > 1})
        
        if missing:
            raise ValueError(f"Missing views in {scene_path}: {missing}\nExpected: {EXPECTED_VIEWS}")
        if extra:
            raise ValueError(f"Unexpected views in {scene_path}: {extra}\nExpected: {EXPECTED_VIEWS}")
        if dupes:
            raise ValueError(f"Duplicate views in {scene_path}: {dupes}")

        for fname in image_files:
            img_path = os.path.join(scene_path, fname)
            view = os.path.splitext(fname)[0]
            img = cv2.imread(img_path)

            if img is None:
                raise ValueError(f"Failed to read: {img_path}")
            if img.shape != (H, W, 3):
                raise ValueError(f"{img_path} is {img.shape[1]}x{img.shape[0]}, expected {W}x{H}")

            for y, pixels in enumerate(image_to_rows(img)):
                rows.append({"id": f"{scene}_{view}_r{y:03d}", "image": pixels})
            n_images += 1

    print(f"Processed images: {n_images}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"

    if not os.path.isdir(RUN_DIR):
        sys.exit(f"Run directory not found: {RUN_DIR}")

    df = images_to_df(RUN_DIR)
    df.to_csv(output, index=False)

    print(f"{output} ready ({len(df)} rows)")