"""Ape floor-toy lesson adapted from MJLab's native YAM lift-cube task.

This first stage starts in the already demonstrated squat pose and uses the
right hand. It keeps MJLab's object command, observations, PPO and staged
reach/lift reward. The 2 cm toy is the YAM cube scaled by 1/2 in length and
1/8 in mass (same density). Its spawn range is inside the ape's measured
right-hand kinematic reach from a 0.117 m squat, not from standing.
"""

from copy import deepcopy
from functools import partial

from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.config.yam.env_cfgs import get_cube_spec
from mjlab.tasks.manipulation.config.yam.rl_cfg import yam_lift_cube_ppo_runner_cfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .task import SIDES, LEG_JOINTS, robot_cfg, env_cfg


TASK_ID = "Mjlab-Reach-Floor-Toy-Ape-Elbows-Right"


def reach_env_cfg(play: bool = False):
    cfg = make_lift_cube_env_cfg()
    ape = robot_cfg(elbowed=True)
    ape.init_state.pos = (0.0, 0.0, 0.117)
    ape.init_state.joint_pos[r".*hip_pitch.*"] = -0.95
    ape.init_state.joint_pos[r".*knee.*"] = 1.9
    ape.init_state.joint_pos[r".*ankle_pitch.*"] = -0.95
    cfg.scene.entities = {
        "robot": ape,
        "cube": EntityCfg(spec_fn=partial(get_cube_spec, cube_size=0.01, mass=0.00625)),
    }
    gait = env_cfg(elbowed=True)
    cfg.scene.sensors = gait.scene.sensors[:2]
    cfg.scene.terrain.terrain_type = "plane"
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 8.0
    cfg.viewer.body_name = "pelvis"
    cfg.viewer.distance = 0.7

    # Keep the ape's rated XML servos. The zero-action squat survives 8 seconds;
    # isolate hand reach before training a combined balance/reach policy.
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.actuator_names = (r".*",)
    action.scale = {
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in LEG_JOINTS},
        **{f"^left_{joint}$": 0.0 for joint in (
            "shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"
        )},
        **{f"^right_{joint}$": 0.5 for joint in (
            "shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"
        )},
        "^head_yaw$": 0.0,
        "^head_pitch$": 0.0,
    }

    command = cfg.commands["lift_height"]
    command.resampling_time_range = (8.0, 8.0)
    command.object_pose_range.x = (0.06, 0.09)
    command.object_pose_range.y = (-0.09, -0.07)
    command.object_pose_range.z = (0.01, 0.01)
    command.target_position_range.x = (0.06, 0.09)
    command.target_position_range.y = (-0.09, -0.07)
    command.target_position_range.z = (0.07, 0.10)

    hand = ("right_hand_center",)
    cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = hand
    cfg.observations["critic"].terms["ee_to_cube"].params["asset_cfg"].site_names = hand
    cfg.rewards["lift"].params["asset_cfg"].site_names = hand
    # The ape's 8 cm reach is about 1/4 of YAM's; scale the reference reward
    # widths with that geometry rather than introduce an unrelated shape.
    cfg.rewards["lift"].params["reaching_std"] = 0.05
    cfg.rewards["lift"].params["bringing_std"] = 0.08
    cfg.rewards["lift_precise"].params["std"] = 0.015
    cfg.rewards["upright"] = deepcopy(gait.rewards["upright"])
    cfg.terminations["fell_over"] = deepcopy(gait.terminations["fell_over"])
    # Floor contact is part of this lesson: the hand must get under a 2 cm
    # object, so the native YAM forearm/ground tripwire and gait hand tripwire
    # would end the episode precisely at the target pose.
    cfg.terminations.pop("ee_ground_collision")
    cfg.commands["lift_height"].success_threshold = 0.015
    for name in ("fingertip_friction_slide", "fingertip_friction_spin",
                 "fingertip_friction_roll"):
        cfg.events[name].params["asset_cfg"].geom_names = r"^(right_hand|right_finger_tip)$"
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum = {}
    return cfg


runner_cfg = yam_lift_cube_ppo_runner_cfg()
runner_cfg.experiment_name = "ape_floor_toy"
runner_cfg.run_name = "ape_right_reach_from_squat"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=reach_env_cfg(),
    play_env_cfg=reach_env_cfg(play=True),
    rl_cfg=runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)
