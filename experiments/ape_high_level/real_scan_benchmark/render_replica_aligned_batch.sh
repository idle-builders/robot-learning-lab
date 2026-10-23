#!/usr/bin/env bash
# Rebuild source-mesh-aligned 0.42 m RGB-D routes, then add official HDR PTex RGB.
set -u

work=${1:-/workspace/replica}
habitat="$work/mamba/envs/habitat/bin/python"
renderer=/workspace/ape-replica-sdk-src/build/ReplicaSDK/ReplicaRenderer
converted="$work/converted"
routes="$work/aligned_routes"
ptex="$work/aligned_ptex_routes"
mkdir -p "$converted" "$routes" "$ptex"
status="$ptex/render_status.tsv"
printf 'scene\tmesh_exit\troute_exit\tptex_exit\tpackage_exit\n' > "$status"

scenes=(apartment_0 apartment_1 apartment_2
        frl_apartment_0 frl_apartment_1 frl_apartment_2
        frl_apartment_3 frl_apartment_4 frl_apartment_5
        hotel_0 office_0 office_1 office_2 office_3 office_4
        room_0 room_1 room_2)

for scene in "${scenes[@]}"; do
  mesh="$converted/$scene.ply"
  route="$routes/$scene"
  hdr="$ptex/$scene"
  mesh_exit=0
  route_exit=0
  ptex_exit=0
  package_exit=0

  if [[ ! -s "$mesh" ]]; then
    python /workspace/transform_replica_mesh_to_habitat.py \
      "$work/data/$scene/mesh.ply" "$mesh" || mesh_exit=$?
  fi
  if [[ "$mesh_exit" == 0 && (
        ! -s "$route/trace.json" || ! -s "$route/depth_0000.png" ||
        ! -s "$route/ape_height_route.mp4" ) ]]; then
    rm -rf "$route"
    mkdir -p "$route"
    "$habitat" /workspace/render_replica_ape_route.py \
      --dataset-root "$converted" --scene "$mesh" --scene-label "$scene" \
      --recompute-navmesh --save-depth --output "$route" \
      > "$route/render.log" 2>&1 || route_exit=$?
    if [[ "$route_exit" == 0 ]]; then
      "$habitat" /workspace/package_replica_ape_route.py "$route" \
        > "$route/package.log" 2>&1 || package_exit=$?
    fi
  fi
  if [[ "$mesh_exit" == 0 && "$route_exit" == 0 && "$package_exit" == 0 \
        && ( ! -s "$hdr/trace.json" || ! -s "$hdr/depth_0000.png" ||
             ! -s "$hdr/ape_height_route.mp4" ) ]]; then
    rm -rf "$hdr"
    mkdir -p "$hdr"
    "$habitat" /workspace/write_replica_sdk_route.py \
      "$route/trace.json" "$hdr/route.txt"
    "$habitat" - "$route/trace.json" "$hdr/trace.json" <<'PY'
import json, sys
from pathlib import Path
trace = json.loads(Path(sys.argv[1]).read_text())
for frame in trace["frames"]:
    frame["image"] = frame["image"].replace(".png", ".jpg")
trace["appearance_renderer"] = "Official Replica SDK HDR PTex with mirror/glass surfaces"
trace["geometry_frame"] = "Source PLY explicitly rotated to Habitat Y-up; navmesh recomputed from that mesh"
Path(sys.argv[2]).write_text(json.dumps(trace, indent=2) + "\n")
PY
    cp "$route"/depth_*.png "$hdr/"
    glass=()
    if [[ -s "$work/data/$scene/glass.sur" ]]; then
      glass=("$work/data/$scene/glass.sur")
    fi
    "$renderer" "$work/data/$scene/mesh.ply" "$work/data/$scene/textures" \
      "$hdr/route.txt" "$hdr" "${glass[@]}" \
      > "$hdr/render.log" 2>&1 || ptex_exit=$?
    if [[ "$ptex_exit" == 0 ]]; then
      "$habitat" /workspace/package_replica_ape_route.py "$hdr" \
        > "$hdr/package.log" 2>&1 || package_exit=$?
    fi
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$scene" "$mesh_exit" "$route_exit" \
    "$ptex_exit" "$package_exit" >> "$status"
  echo "$scene mesh=$mesh_exit route=$route_exit ptex=$ptex_exit package=$package_exit"
done

if awk 'NR>1 && ($2 != 0 || $3 != 0 || $4 != 0 || $5 != 0) {bad=1} END {exit !bad}' "$status"; then
  exit 1
fi
