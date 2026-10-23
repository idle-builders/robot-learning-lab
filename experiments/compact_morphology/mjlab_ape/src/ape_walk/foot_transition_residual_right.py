"""Right-foot counterpart of the validated reference-centered ape tracker."""

from copy import deepcopy
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .foot_transition_residual import env_cfg as left_env_cfg, rl_cfg as left_rl_cfg


TASK_ID = "Mjlab-Tracking-Ape-Foot-Transition-Residual-Right"
MOTION = Path(__file__).resolve().parents[5] / "experiments/compact_morphology/motions/ape_foot_transition_right_mirror_v2.npz"


def env_cfg(play=False):
    cfg = left_env_cfg(play=play)
    cfg.commands["motion"].motion_file = str(MOTION)
    return cfg


rl_cfg = deepcopy(left_rl_cfg)
rl_cfg.experiment_name = "ape_foot_transition_residual_right"
rl_cfg.run_name = "mirrored_reference_v1"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
