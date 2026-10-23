#!/usr/bin/env bash
set -euo pipefail

root=/workspace/robot-learning-lab/compact-morphology-policy-20260927
action_scale=${1:-0.3}
checkpoint=${2:-$root/runs/CompactJoystick-20260927-114155-ape_short_torso/checkpoints}
steps=${3:-327680}
stage=${4:-forward}
cd "$root"
export COMPACT_SHAPE=ape_short_torso
export COMPACT_WALK_ACTION_SCALE="$action_scale"
export COMPACT_WALK_STAGE="$stage"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax-compact

exec timeout 2h /root/compact-ppo-venv/bin/python -u playground_walk.py \
  --env_name=CompactForwardWalk \
  --impl=jax \
  --load_checkpoint_path="$checkpoint" \
  --num_timesteps="$steps" \
  --num_envs=512 \
  --num_eval_envs=32 \
  --num_evals=2 \
  --episode_length=300 \
  --num_videos=1 \
  --seed=7 \
  --suffix="ape-$stage-scale$action_scale" \
  --logdir="$root/runs"
