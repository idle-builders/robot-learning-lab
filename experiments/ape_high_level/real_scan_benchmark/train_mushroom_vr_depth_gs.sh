#!/usr/bin/env bash
set -u

# Run on temporary CUDA compute after preparing the MD5-verified VR room
# iPhone capture, Faro-aligned 1M-point seed and two extra low held-out frames.
cd /workspace
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
rm -f /workspace/vr_train.exit
ns-train dn-splatter \
  --output-dir /workspace/output \
  --experiment-name ape_vr_depth_faro \
  --method-name dn-splatter \
  --max-num-iterations 7000 \
  --steps-per-save 1000 \
  --vis tensorboard \
  --pipeline.model.use-depth-loss True \
  --pipeline.model.depth-lambda 0.2 \
  --pipeline.model.use-normal-loss True \
  --pipeline.model.use-normal-tv-loss True \
  --pipeline.model.normal-supervision depth \
  mushroom \
  --data /workspace/room_datasets/vr_room \
  --mode iphone \
  --eval-mode with \
  --normals-from depth \
  --load-pcd-normals False \
  --iphone-ply-name faro_aligned_pointcloud.ply \
  --num-init-points 1000000
status=$?
printf '%s\n' "$status" > /workspace/vr_train.exit
exit "$status"
