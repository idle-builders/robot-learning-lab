"""MJLab reference tracking with its upstream random velocity-push event."""

from copy import deepcopy

from mjlab.envs.mdp import push_by_setting_velocity
from mjlab.managers import EventTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .foot_transition_residual import env_cfg as residual_env_cfg, rl_cfg as residual_rl_cfg


TASK_ID = "Mjlab-Tracking-Ape-Foot-Transition-Residual-Push"


def env_cfg(play=False):
    cfg = residual_env_cfg(play=play)
    if not play:
        # MJLab G1 tracking uses this same event at up to 0.5 m/s. Here the
        # range is reduced for a 35 cm ape and applied once or twice per skill.
        cfg.events["push_robot"] = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={"velocity_range": {
                "x": (-0.04, 0.04), "y": (-0.08, 0.08),
                "yaw": (-0.1, 0.1),
            }},
        )
    return cfg


rl_cfg = deepcopy(residual_rl_cfg)
rl_cfg.experiment_name = "ape_foot_transition_residual_push"
rl_cfg.run_name = "upstream_scaled_velocity_push_v1"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=rl_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
