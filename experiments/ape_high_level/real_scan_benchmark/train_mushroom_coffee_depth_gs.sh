#!/usr/bin/env bash
set -u

# Run on a temporary CUDA compute host after installing upstream DN-Splatter.
# The iPhone archive and all authored code remain reproducible from this repo.
# This is a feasibility trial, not the 20-room benchmark or a body qualification.
cd /workspace
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
rm -f /workspace/dn_train.exit
ns-train dn-splatter \
  --output-dir /workspace/output \
  --experiment-name ape_coffee_depth_faro \
  --method-name dn-splatter \
  --max-num-iterations 7000 \
  --steps-per-save 7000 \
  --vis tensorboard \
  --pipeline.model.use-depth-loss True \
  --pipeline.model.depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True \
  --pipeline.model.use-normal-tv-loss True \
  --pipeline.model.normal-supervision depth \
  mushroom \
  --data /workspace/room_datasets/coffee_room \
  --mode iphone \
  --eval-mode with \
  --normals-from depth \
  --load-pcd-normals False \
  --iphone-ply-name faro_aligned_pointcloud.ply \
  --num-init-points 1000000
status=$?
printf '%s\n' "$status" > /workspace/dn_train.exit
exit "$status"
