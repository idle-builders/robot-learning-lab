#!/usr/bin/env bash
# Download and unpack the official Replica v1 real-scan archive on large temporary compute storage.
set -euo pipefail

work=${1:-/workspace/replica}
mkdir -p "$work/data"

if [[ ! -s "$work/download.exit" ]]; then
  APE_REPLICA_STORAGE_READY=1 bash /workspace/fetch_replica_parts.sh "$work" 6
  printf '0\n' > "$work/download.exit"
fi

# Release each compressed part after the extractor has consumed it. Otherwise
# keeping both the 34 GB archive and all expanded scenes can hit the pod quota.
for suffix in {a..q}; do
  part="$work/replica_v1_0.tar.gz.parta${suffix}"
  cat "$part"
  rm "$part"
done | pigz -dc -p 8 | tar --no-same-owner -xf - -C "$work/data"
curl -fLs https://dl.fbaipublicfiles.com/habitat/Replica/additional_habitat_configs.zip \
  -o "$work/additional_habitat_configs.zip"
python -m zipfile -e "$work/additional_habitat_configs.zip" "$work/data"
test -f "$work/data/replica.scene_dataset_config.json"
printf '0\n' > "$work/extract.exit"
echo REPLICA_OFFICIAL_ARCHIVE_READY
