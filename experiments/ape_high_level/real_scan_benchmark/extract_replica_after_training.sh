#!/usr/bin/env bash
# Keep archive extraction from competing with the MuSHRoom GPU fit for CPU/I/O.
set -euo pipefail

work=${1:?usage: extract_replica_after_training.sh WORK_DIR VR_TRAIN_EXIT}
train_exit=${2:?usage: extract_replica_after_training.sh WORK_DIR VR_TRAIN_EXIT}
while [[ ! -f "$train_exit" || ! -f "$work/download.exit" ]]; do
  sleep 30
done
[[ $(<"$train_exit") == 0 ]]
[[ $(<"$work/download.exit") == 0 ]]

mkdir -p "$work/data"
cat "$work"/replica_v1_0.tar.gz.part{aa..aq} | gzip -dc | tar -xf - -C "$work/data"
test -f "$work/data/replica.scene_dataset_config.json"
echo REPLICA_EXTRACTED
