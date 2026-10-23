"""Teach the native MJLab ape to track one physically valid pickup motion.

Adapted from mjlab.tasks.tracking (BeyondMimic): the motion is a teacher,
while the policy must still actuate the body and maintain jaw-cube contact.
"""

from copy import deepcopy
from pathlib import Path

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from .grasp import held_env_cfg, held_runner_cfg


TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Jaw-Motion-Right"
MOTION = Path(__file__).resolve().parents[3] / "motions" / "ape_jaw_held_lift_demo.npz"


def tracking_env_cfg(play: bool = False):
    cfg = held_env_cfg(play=play)
    cfg.episode_length_s = 3.0
    cfg.commands["motion"] = MotionCommandCfg(
        entity_name="robot",
        motion_file=str(MOTION),
        anchor_body_name="pelvis",
        body_names=("pelvis", "right_arm", "right_forearm", "right_finger"),
        resampling_time_range=(1.0e9, 1.0e9),
        sampling_mode="start",
        pose_range={},
        velocity_range={},
        joint_position_range=(0.0, 0.0),
        debug_vis=False,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["motion"] = ObservationTermCfg(
            func=tracking_mdp.generated_commands,
            params={"command_name": "motion"},
        )
    # Train source-style motion tracking alone. The prior mixed trial paid an
    # immediate contact-lift bonus, causing a fast yank before the teacher's
    # slow pickup; keep contact and height as independent evaluation gates.
    cfg.rewards["lift"].weight = 0.0
    cfg.rewards["lift_precise"].weight = 0.0
    cfg.rewards["lifted"].weight = 0.0
    # The elbow body can match while the jaw is still centimeters off the toy.
    # Put the same total position weight on the distal finger body instead.
    # A 1 cm kernel makes a 9 mm error costly for a 2 cm cube; the prior 3 cm
    # kernel paid almost full reward at that miss distance.
    cfg.rewards["motion_forearm_pos"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_position_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.03,
                "body_names": ("right_forearm",)},
    )
    cfg.rewards["motion_finger_pos"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "std": 0.01,
                "body_names": ("right_finger",)},
    )
    # MJLab's reference tracking task rewards both pose components. Position
    # alone lets a rotated jaw hit the target point while sweeping the toy
    # out of its grip. Its 0.4 rad kernel is scaled to this 2 cm toy.
    cfg.rewards["motion_finger_ori"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.1,
                "body_names": ("right_finger",)},
    )
    # The v6 finger travels about 0.18-0.24 m/s on average while its teacher
    # moves about 0.028 m/s, then sweeps the cube away. Restore MJLab's
    # reference linear-velocity tracking term. A 0.1 m/s kernel gave almost
    # zero reward during early exploration; broaden to 0.3 m/s so the learned
    # motions still receive a gradient, changing no other term.
    cfg.rewards["motion_finger_lin_vel"] = RewardTermCfg(
        func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3,
                "body_names": ("right_finger",)},
    )
    return cfg


runner_cfg = deepcopy(held_runner_cfg)
runner_cfg.experiment_name = "ape_jaw_motion_toy"
runner_cfg.run_name = "ape_right_jaw_track_pickup"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=tracking_env_cfg(),
    play_env_cfg=tracking_env_cfg(play=True),
    rl_cfg=runner_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
