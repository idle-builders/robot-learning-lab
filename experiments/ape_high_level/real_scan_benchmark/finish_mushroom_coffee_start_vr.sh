#!/usr/bin/env bash
set -u

# Temporary Runpod compute orchestration; source captures and results are kept
# in this repository's local asset store. Never treat a failed fit as evidence.
while [ ! -f /workspace/dn_train.exit ]; do
  sleep 15
done

if [ "$(cat /workspace/dn_train.exit)" = "0" ]; then
  run_dir=$(find /workspace/output/ape_coffee_depth_faro/dn-splatter -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
  ns-export gaussian-splat \
    --load-config "$run_dir/config.yml" \
    --output-dir /workspace/export_coffee \
    > /workspace/export_coffee.log 2>&1
  printf '%s\n' "$?" > /workspace/export_coffee.exit
else
  printf 'coffee training failed with exit %s\n' "$(cat /workspace/dn_train.exit)" > /workspace/export_coffee.log
  printf '1\n' > /workspace/export_coffee.exit
fi

# The second room has separately aligned Faro geometry and a frozen low-view
# test split. Its result remains independent of the coffee-room appearance.
nohup bash /workspace/train_mushroom_vr_depth_gs.sh \
  > /workspace/vr_train.log 2>&1 < /dev/null &
printf '%s\n' "$!" > /workspace/vr_train.pid
