"""Two short MJLab tracking lessons for a held-to-floor placement."""

from copy import deepcopy
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

from .place_tracking import place_env_cfg, runner_cfg


ROOT = Path(__file__).resolve().parents[3] / "motions"
LOWER_TASK_ID = "Mjlab-Lower-Held-Toy-Ape-Jaw-Motion-Right"
RELEASE_TASK_ID = "Mjlab-Release-Floor-Toy-Ape-Jaw-Motion-Right"


def lower_env_cfg(play: bool = False):
    cfg = place_env_cfg(play=play)
    cfg.episode_length_s = 1.5
    cfg.commands["motion"].motion_file = str(ROOT / "ape_jaw_lower_held_demo.npz")
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.0
    cfg.rewards["motion_finger_pos"].weight = 4.0
    cfg.rewards["motion_finger_pos"].params["std"] = 0.01
    cfg.rewards["motion_finger_ori"].weight = 0.0
    return cfg


def release_env_cfg(play: bool = False):
    cfg = place_env_cfg(play=play)
    cfg.episode_length_s = 1.5
    cfg.commands["motion"].motion_file = str(ROOT / "ape_jaw_open_withdraw_demo.npz")
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    pose[r"^right_shoulder_pitch$"] = -0.042
    pose[r"^right_elbow_pitch$"] = -1.0013
    pose[r"^right_gripper$"] = 0.3391
    cmd = cfg.commands["lift_height"]
    cmd.object_pose_range.x = (0.0886, 0.0886)
    cmd.object_pose_range.y = (-0.0852, -0.0852)
    cmd.object_pose_range.z = (0.0108, 0.0108)
    cfg.rewards["motion_finger_pos"].params["std"] = 0.01
    return cfg


lower_runner_cfg = deepcopy(runner_cfg)
lower_runner_cfg.experiment_name = "ape_jaw_lower_toy"
lower_runner_cfg.run_name = "ape_right_jaw_lower_held"
register_mjlab_task(
    task_id=LOWER_TASK_ID,
    env_cfg=lower_env_cfg(),
    play_env_cfg=lower_env_cfg(play=True),
    rl_cfg=lower_runner_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)

release_runner_cfg = deepcopy(runner_cfg)
release_runner_cfg.experiment_name = "ape_jaw_release_toy"
release_runner_cfg.run_name = "ape_right_jaw_open_withdraw"
register_mjlab_task(
    task_id=RELEASE_TASK_ID,
    env_cfg=release_env_cfg(),
    play_env_cfg=release_env_cfg(play=True),
    rl_cfg=release_runner_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
