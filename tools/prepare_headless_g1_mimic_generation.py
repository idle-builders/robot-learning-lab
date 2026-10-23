"""Adapt the pinned Mimic generator to Arena's stateful success term."""

import sys
from pathlib import Path


source = Path(sys.argv[1]).read_text()
old = "    env = gym.make(env_name, cfg=env_cfg).unwrapped\n"
new = old + '''\
    # Mimic removes the termination manager and then invokes cfg.func directly.
    # Arena's cfg.func is a ManagerTermBase class, so give the Mimic environment
    # the two Arena scene helpers it needs and instantiate that exact native term.
    from isaaclab_arena.environments.arena_world import ArenaWorld
    from isaaclab_arena.tasks.predicates.object_settling import ObjectInitialRestPoseRecorder

    env.arena_world = ArenaWorld(env.scene)
    env.progress_tracker = None
    env.object_initial_rest_pose_recorder = ObjectInitialRestPoseRecorder(env.num_envs, env.device)
    success_term.func = success_term.func(success_term, env)
    env.progress_tracker = env._progress_tracker
    original_reset_idx = env._reset_idx

    def reset_idx_with_arena_success(env_ids):
        success_term.func.reset(env_ids)
        return original_reset_idx(env_ids)

    env._reset_idx = reset_idx_with_arena_success
'''
if source.count(old) != 1:
    raise SystemExit(f"Expected one upstream insertion point, found {source.count(old)}")
Path(sys.argv[2]).write_text(source.replace(old, new))
