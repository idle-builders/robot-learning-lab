"""Targeted hold-phase curriculum using MJLab's stock velocity-push event."""

from copy import deepcopy

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .foot_transition_residual_push import env_cfg as push_env_cfg, rl_cfg as push_rl_cfg


TASK_ID = "Mjlab-Tracking-Ape-Foot-Transition-Residual-Hold-Push"


def env_cfg(play=False):
    cfg = push_env_cfg(play=play)
    if not play:
        # The event clock is episode-relative; start every episode at motion
        # frame zero so the 6.9-7.4 s push lands during the actual foot hold.
        cfg.commands["motion"].sampling_mode = "start"
        event = cfg.events["push_robot"]
        event.interval_range_s = (6.9, 7.4)
        event.params["velocity_range"] = {"y": (-0.08, 0.08)}
    return cfg


rl_cfg = deepcopy(push_rl_cfg)
rl_cfg.experiment_name = "ape_foot_transition_residual_hold_push"
rl_cfg.run_name = "upstream_hold_velocity_push_v1"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
