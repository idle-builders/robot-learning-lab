#!/usr/bin/env bash
set -u

while [ ! -f /workspace/vr_train.exit ]; do
  sleep 15
done
if [ "$(cat /workspace/vr_train.exit)" != "0" ]; then
  printf 'VR training failed with exit %s\n' "$(cat /workspace/vr_train.exit)" > /workspace/vr_export.log
  printf '1\n' > /workspace/vr_pipeline.exit
  exit 1
fi

run_dir=$(find /workspace/output/ape_vr_depth_faro/dn-splatter -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
ns-export gaussian-splat \
  --load-config "$run_dir/config.yml" \
  --output-dir /workspace/export_vr \
  > /workspace/vr_export.log 2>&1
export_status=$?
printf '%s\n' "$export_status" > /workspace/vr_export.exit
if [ "$export_status" != "0" ]; then
  printf '%s\n' "$export_status" > /workspace/vr_pipeline.exit
  exit "$export_status"
fi

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
python /workspace/render_mushroom_native_heldout.py \
  --config "$run_dir/config.yml" \
  --output /workspace/vr_native_heldout \
  --frame frame_00045 --frame frame_00076 \
  > /workspace/vr_native_heldout.log 2>&1
heldout_status=$?
printf '%s\n' "$heldout_status" > /workspace/vr_native_heldout.exit
if [ "$heldout_status" != "0" ]; then
  printf '%s\n' "$heldout_status" > /workspace/vr_pipeline.exit
  exit "$heldout_status"
fi

python /workspace/render_mushroom_native_route.py \
  --config "$run_dir/config.yml" \
  --route /workspace/vr_proxy_route.json \
  --output /workspace/vr_native_route \
  > /workspace/vr_native_route.log 2>&1
route_status=$?
printf '%s\n' "$route_status" > /workspace/vr_pipeline.exit
exit "$route_status"
