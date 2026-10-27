#!/bin/bash

# Loop through scenes 1 to 8
for i in {1..8}; do
    # Format scene name with leading zero (scene_01, scene_02, etc.)
    scene_name=$(printf "scene_%02d" $i)
    measurements_dir="scenes/${scene_name}/measurements"
    
    # Check if the measurements directory exists
    if [ -d "$measurements_dir" ]; then
        echo "Processing $scene_name..."
        
        # Loop through all available .npy files in measurements
        for npy_file in "$measurements_dir"/*.npy; do
            if [ -f "$npy_file" ]; then
                python convert_dpir.py \
                    --input "$npy_file" \
                    --psf "psf.npy" \
                    --scene_dir "scenes/${scene_name}"
            fi
        done
    else
        echo "Skipping $scene_name (measurements folder not found)"
    fi
done

echo "All batch reconstructions completed!"