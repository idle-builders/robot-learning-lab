#!/usr/bin/env bash
# One full-task arm with a bounded observed-free route fallback.
set -euo pipefail

scene=${1:?scene required}
arm=${2:?none, walk, route, or final required}
case "$arm" in none|walk|route|final) ;; *) exit 2 ;; esac
repo=/root/robot-learning-lab
cache=/root/replica-sdk-src
export LD_LIBRARY_PATH="$cache/build/ReplicaSDK:$cache/3rdparty/Pangolin/build/src:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
cd "$cache/ReplicaSDK"
exec /root/ape-venv/bin/python \
  "$repo/experiments/ape_high_level/real_scan_benchmark/run_replica_owner_task.py" \
  --renderer "$cache/build/ReplicaSDK/ReplicaRenderer" \
  --source-scene "/root/replica-source/$scene" \
  --converted-mesh "/root/replica-source/$scene/mesh_yup.ply" \
  --reference "$repo/assets/real_indoor_scenes/replica/ptex_routes/$scene" \
  --scanned-person "$repo/assets/real_indoor_scenes/scansceneverse/human_previews/sitting_person_isolated.glb" \
  --output "$repo/assets/real_indoor_scenes/replica_owner_integrated_20261015/$scene/$arm" \
  --device cuda --range-gap-stage "$arm" --route-fallback
