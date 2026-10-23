#!/usr/bin/env bash
set -euo pipefail

shape=ape_short_torso
stable_checkpoint=${1:?stable-reach checkpoint directory required}
steps=${2:-3000000}
root=/workspace/robot-learning-lab/compact-morphology-policy-20260927
python_bin=${COMPACT_PYTHON:-/root/compact-ppo-venv/bin/python}

cd "$root"
export COMPACT_SHAPE="$shape"
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_triton_gemm_any=True"
export JAX_COMPILATION_CACHE_DIR=/root/.cache/jax-compact

exec timeout 3h "$python_bin" -u playground_contact_curriculum.py \
  --env_name=CompactContactCurriculum \
  --impl=jax \
  --load_checkpoint_path="$stable_checkpoint" \
  --num_timesteps="$steps" \
  --num_envs=512 \
  --num_eval_envs=32 \
  --num_evals=5 \
  --episode_length=300 \
  --num_videos=1 \
  --seed=7 \
  --suffix="$shape-contact-curriculum" \
  --logdir="$root/runs"
