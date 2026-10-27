#!/usr/bin/env bash
# Train physics-informed 3DGS independently on every scene.
# Outputs: runs/<timestamp>/<scene>/
# Usage:
#   ./train_all_physics.sh
#   ./train_all_physics.sh --iterations 21500
#   ./train_all_physics.sh --iterations 21500 --random_init_points 50000
#
# Extra arguments are forwarded to train_physics.py. For single-value options,
# arguments supplied by the caller come after the defaults and therefore win.

set -euo pipefail

ROOT=/4tb_hdd/Sumedha/CP_Project
SCENES="$ROOT/lensless3d_data/scenes"
GS="$ROOT/gaussian-splatting"
PY=/4tb_hdd/envs/gaussian_splatting/bin/python
TRAIN_SCRIPT="$GS/train.py"

# These are defaults appropriate for the described 52-train / 8-test split.
# --eval is enabled by default; --disable_viewer makes batch runs non-interactive.
DEFAULT_ARGS=(
    --eval
    --disable_viewer
    --lambda_dssim 0.2
    --images dpir
)

# Validate fixed paths before creating a run directory.
if [[ ! -x "$PY" ]]; then
    echo "ERROR: Python executable not found or not executable: $PY" >&2
    exit 1
fi
if [[ ! -d "$SCENES" ]]; then
    echo "ERROR: Scene dataset directory not found: $SCENES" >&2
    exit 1
fi
if [[ ! -d "$GS" ]]; then
    echo "ERROR: Gaussian Splatting repository not found: $GS" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT" >&2
    echo "Copy train_physics.py into the Gaussian Splatting repository or update TRAIN_SCRIPT." >&2
    exit 1
fi

shopt -s nullglob
scene_dirs=("$SCENES"/scene_*/)
if (( ${#scene_dirs[@]} == 0 )); then
    echo "ERROR: No scene_* directories found under $SCENES" >&2
    exit 1
fi

# Preflight each scene's expected directory layout.
for scene in "${scene_dirs[@]}"; do
    for required in "$scene/sparse/0" "$scene/measurements" "$scene/dpir"; do
        if [[ ! -d "$required" ]]; then
            echo "ERROR: Required dataset directory not found: $required" >&2
            exit 1
        fi
    done
done

RUN="$ROOT/runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"
echo "Run folder: $RUN"
ln -sfn "$RUN" "$ROOT/runs/latest"

for scene in "${scene_dirs[@]}"; do
    name=$(basename "$scene")
    echo
    echo "=== Training $name ==="
    echo "Dataset: $scene"
    echo "Output:  $RUN/$name"

    mkdir -p "$RUN/$name"
    (
        cd "$GS"
        "$PY" "$TRAIN_SCRIPT" \
            -s "$scene" \
            -m "$RUN/$name" \
            "${DEFAULT_ARGS[@]}" \
            "$@"
    ) 2>&1 | tee "$RUN/$name.train.log"
done

echo
echo "Done. Models and logs are in: $RUN"
