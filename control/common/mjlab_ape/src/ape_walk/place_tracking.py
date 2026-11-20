"""Native MJLab motion-imitation lesson for setting down a held floor toy."""

from copy import deepcopy
from pathlib import Path

from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from .grasp import COLLISION_MODEL, held_env_cfg, held_runner_cfg


TASK_ID = "Mjlab-Place-Floor-Toy-Ape-Jaw-Motion-Right"
BC_TASK_ID = "Mjlab-Place-Floor-Toy-Ape-Jaw-Collision-Right"
MOTION = Path(__file__).resolve().parents[3] / "motions" / "ape_jaw_place_withdraw_demo.npz"


def place_env_cfg(play: bool = False):
    cfg = held_env_cfg(play=play)
    cfg.episode_length_s = 3.0
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r"^right_shoulder_pitch$"] = 0.103
    pose[r"^right_elbow_pitch$"] = -1.754
    pose[r"^right_gripper$"] = 0.35
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.5
    cmd = cfg.commands["lift_height"]
    cmd.object_pose_range.x = (0.088, 0.088)
    cmd.object_pose_range.y = (-0.0845, -0.0845)
    cmd.object_pose_range.z = (0.066, 0.066)
    cmd.target_position_range.x = (0.08, 0.08)
    cmd.target_position_range.y = (-0.0845, -0.0845)
    cmd.target_position_range.z = (0.01, 0.01)
    cfg.commands["motion"] = MotionCommandCfg(
        entity_name="robot", motion_file=str(MOTION), anchor_body_name="pelvis",
        body_names=("pelvis", "right_arm", "right_forearm", "right_finger"),
        resampling_time_range=(1.0e9, 1.0e9), sampling_mode="start",
        pose_range={}, velocity_range={}, joint_position_range=(0.0, 0.0),
        debug_vis=False,
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["motion"] = ObservationTermCfg(
            func=tracking_mdp.generated_commands,
            params={"command_name": "motion"},
        )
    for name in ("lift", "lift_precise", "lifted"):
        cfg.rewards[name].weight = 0.0
    cfg.rewards["motion_finger_pos"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_position_error_exp,
        weight=3.0, params={"command_name": "motion", "std": 0.03,
                            "body_names": ("right_finger",)},
    )
    # With a 0.25 rad orientation kernel, the first policy closed to 0.50 rad
    # against the teacher's 0.35 rad hold and squeezed the toy out. The
    # gripper's 0.15 rad error now loses most of the orientation reward.
    cfg.rewards["motion_finger_ori"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0, params={"command_name": "motion", "std": 0.1,
                            "body_names": ("right_finger",)},
    )
    return cfg


def place_bc_env_cfg(play: bool = False):
    """Physics-only v5 held-to-floor lesson for demonstration/BC replay."""
    cfg = held_env_cfg(play=play, model=COLLISION_MODEL)
    cfg.episode_length_s = 3.0
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r"^right_shoulder_pitch$"] = 0.103
    pose[r"^right_elbow_pitch$"] = -1.754
    pose[r"^right_gripper$"] = 0.35
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.5
    cmd = cfg.commands["lift_height"]
    cmd.object_pose_range.x = (0.087, 0.089)
    cmd.object_pose_range.y = (-0.0855, -0.0835)
    cmd.object_pose_range.z = (0.066, 0.066)
    cmd.target_position_range.x = (0.079, 0.081)
    cmd.target_position_range.y = (-0.0855, -0.0835)
    cmd.target_position_range.z = (0.01, 0.01)
    return cfg


runner_cfg = deepcopy(held_runner_cfg)
runner_cfg.experiment_name = "ape_jaw_place_toy"
runner_cfg.run_name = "ape_right_jaw_track_place"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=place_env_cfg(),
    play_env_cfg=place_env_cfg(play=True),
    rl_cfg=runner_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

bc_runner_cfg = deepcopy(held_runner_cfg)
bc_runner_cfg.run_name = "ape_right_jaw_collision_place_bc"
register_mjlab_task(
    task_id=BC_TASK_ID,
    env_cfg=place_bc_env_cfg(),
    play_env_cfg=place_bc_env_cfg(play=True),
    rl_cfg=bc_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)
