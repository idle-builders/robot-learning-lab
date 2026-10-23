#!/usr/bin/env bash
# Fetch the official real-scan Replica archive in resumable parts on compute.
set -euo pipefail

# The official archive is about 34 GB compressed; Runpod's shared df can show
# cluster-wide free space even when this pod's own 20 GB quota is exhausted.
: "${APE_REPLICA_STORAGE_READY:?verify a sufficiently large pod volume and set APE_REPLICA_STORAGE_READY=1}"

destination=${1:?usage: fetch_replica_parts.sh DESTINATION [PARALLELISM]}
parallelism=${2:-6}
mkdir -p "$destination"
cd "$destination"

base=https://github.com/facebookresearch/Replica-Dataset/releases/download/v1.0
for suffix in {a..q}; do
  printf '%s\n' "$suffix"
done | xargs -P "$parallelism" -I '{}' bash -c '
  suffix=$1
  base=$2
  file="replica_v1_0.tar.gz.parta${suffix}"
  wget --continue --tries=5 --waitretry=5 --quiet "$base/$file" -O "$file"
  test -s "$file"
' _ '{}' "$base"

echo "FETCHED_REPLICA_PARTS_A_TO_Q"
