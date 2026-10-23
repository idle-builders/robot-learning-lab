"""Single forward step on the contact-aware ape using MJLab reference tracking."""

from copy import deepcopy
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .foot_transition_residual import env_cfg as foot_env_cfg, rl_cfg as foot_rl_cfg


ROOT = Path(__file__).resolve().parents[5]


def env_cfg(side: str, play: bool = False):
    cfg = foot_env_cfg(play=play)
    cfg.commands["motion"].motion_file = str(
        ROOT / f"experiments/compact_morphology/motions/ape_single_forward_step_{side}_v1.npz"
    )
    cfg.episode_length_s = 13.2
    return cfg


for _side in ("left", "right"):
    _rl_cfg = deepcopy(foot_rl_cfg)
    _rl_cfg.experiment_name = f"ape_single_step_{_side}"
    _rl_cfg.run_name = "forward_reference_v1"
    register_mjlab_task(
        task_id=f"Mjlab-Tracking-Ape-Single-Step-{_side.title()}",
        env_cfg=env_cfg(_side),
        play_env_cfg=env_cfg(_side, play=True),
        rl_cfg=_rl_cfg,
        runner_cls=MotionTrackingOnPolicyRunner,
    )
