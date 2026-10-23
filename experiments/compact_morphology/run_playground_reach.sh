#!/usr/bin/env bash
set -euo pipefail

shape=${1:?shape required}
locomotion_checkpoint=${2:?locomotion checkpoint directory required}
steps=${3:-4000000}
root=/workspace/robot-learning-lab/compact-morphology-policy-20260927

cd "$root"
export COMPACT_SHAPE="$shape"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax-compact

exec timeout 3h /root/compact-ppo-venv/bin/python -u playground_reach.py \
  --env_name=CompactFloorReach \
  --impl=jax \
  --load_checkpoint_path="$locomotion_checkpoint" \
  --num_timesteps="$steps" \
  --num_envs=512 \
  --num_eval_envs=32 \
  --num_evals=5 \
  --episode_length=300 \
  --num_videos=1 \
  --seed=7 \
  --suffix="$shape-floor-reach" \
  --logdir="$root/runs"
