#!/usr/bin/env bash
# Stream official split Replica v1 into ephemeral compute, retaining only named rooms.
set -euo pipefail

destination=${1:?usage: fetch_replica_selected_stream.sh DESTINATION SCENE...}
shift
if [[ $# -eq 0 ]]; then
  echo 'at least one Replica scene is required' >&2
  exit 2
fi
mkdir -p "$destination"
base=https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0
for suffix in {a..q}; do
  printf 'FETCH_PART_%s\n' "$suffix" >&2
  curl -fLsS --retry 2 "$base/replica_v1_0.tar.gz.parta${suffix}"
done | gzip -dc | tar --no-same-owner -xf - -C "$destination" -- "$@"
for scene in "$@"; do
  test -s "$destination/$scene/mesh.ply"
  test -s "$destination/$scene/preseg.json"
done
echo REPLICA_SELECTED_SCENES_READY
