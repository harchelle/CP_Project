#!/usr/bin/env bash
# Render every scene's trained model from its training cameras and its separate test_cameras/.
# Usage: ./test_all.sh [run_folder]   (default: runs/latest)
# Renders go to <run_folder>/<scene>/{train,test_cameras}/ours_<iter>/renders/
set -euo pipefail

ROOT=/4tb_hdd/Sumedha/CP_Project
SCENES=$ROOT/lensless3d_data/scenes
GS=$ROOT/gaussian-splatting
PY=/4tb_hdd/envs/gaussian_splatting/bin/python

RUN=$(readlink -f "${1:-$ROOT/runs/latest}")
echo "Using run folder: $RUN"

for scene in "$SCENES"/scene_*/; do
    name=$(basename "$scene")
    if [ ! -d "$RUN/$name" ]; then echo "Skipping $name (no trained model)"; continue; fi
    echo "=== Rendering $name ==="
    (cd "$GS" && $PY render_test.py -m "$RUN/$name" \
        --test_cameras "$scene/test_cameras") 2>&1 | tee "$RUN/$name.test.log"
done
echo "Done. Renders in $RUN/<scene>/{train,test_cameras}/ours_*/renders"
