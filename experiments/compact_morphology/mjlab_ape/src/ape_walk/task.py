"""Native mjlab flat-velocity task for the 35 cm ape.

The environment, PPO runner, and gait terms are upstream mjlab/Microduck. This
file only maps our MJCF and its leg/foot names onto that reference contract.
The Feetech XML impedance is still a hypothesis, not a measured motor model.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.utils.spec_config import CollisionCfg

from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg


MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v1.xml"
ELBOW_MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v3_elbows_small_head.xml"
ELBOW_PROXY60_MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v3_collision_proxy.xml"
ELBOW_NARROW55_MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v3_collision_proxy_narrow55.xml"
CAD_V8_MODEL = Path(__file__).resolve().parents[3] / "ape_cad" / "ape_spaced_v8_casedynamics.xml"
TASK_ID = "Mjlab-Velocity-Flat-Ape"
BACKWARD_TASK_ID = "Mjlab-Velocity-Flat-Ape-Backward"
SIGNED_TASK_ID = "Mjlab-Velocity-Flat-Ape-Signed"
SIDEWAYS_TASK_ID = "Mjlab-Velocity-Flat-Ape-Sideways"
PLANAR_TASK_ID = "Mjlab-Velocity-Flat-Ape-Planar"
ELBOW_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows"
CAD_V8_WALK_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-CadV8"
ELBOW_BACKWARD_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-Backward"
ELBOW_SIDEWAYS_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-Sideways"
ELBOW_PLANAR_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-Planar"
ELBOW_PLANAR_PROXY60_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-Planar-Proxy60"
ELBOW_PLANAR_NARROW55_TASK_ID = "Mjlab-Velocity-Flat-Ape-Elbows-Planar-Narrow55"
ELBOW_SQUAT_TASK_ID = "Mjlab-Posture-Flat-Ape-Elbows"
ELBOW_STAND_TASK_ID = "Mjlab-Posture-Flat-Ape-Elbows-Stand"
ELBOW_SQUAT_LOW_NOISE_TASK_ID = "Mjlab-Posture-Flat-Ape-Elbows-LowNoise"
ELBOW_SQUAT_HEADING_TASK_ID = "Mjlab-Posture-Flat-Ape-Elbows-Heading"
ELBOW_SQUAT_HEADING_OBS_TASK_ID = "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved"
ELBOW_TRANSFER_TASK_ID = "Mjlab-Stand-Flat-Ape-Elbows-Transfer"
ELBOW_TRANSFER_PARTIAL_TASK_ID = "Mjlab-Stand-Flat-Ape-Elbows-TransferPartial"
ELBOW_ARM_TASK_ID = "Mjlab-Stand-Flat-Ape-Elbows-Arms"
ELBOW_ARM_INDEPENDENT_TASK_ID = "Mjlab-Stand-Flat-Ape-Elbows-ArmsIndependent"
ELBOW_FLAMINGO_HOLD_TASK_ID = "Mjlab-FlamingoHold-Flat-Ape-Elbows"
ELBOW_FLAMINGO_EFFORT_TASK_ID = "Mjlab-FlamingoHoldEffort-Flat-Ape-Elbows"
ELBOW_FLAMINGO_STRONG_EFFORT_TASK_ID = "Mjlab-FlamingoHoldStrongEffort-Flat-Ape-Elbows"
ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID = "Mjlab-FlamingoHoldCommandLimit-Flat-Ape-Elbows"
ELBOW_FLAMINGO_LEFT_TASK_ID = "Mjlab-FlamingoHoldCommandLimitLeft-Flat-Ape-Elbows"
ELBOW_FLAMINGO_CYCLE_TASK_ID = "Mjlab-FlamingoCycle-Flat-Ape-Elbows"
ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID = "Mjlab-FlamingoCycleArms-Flat-Ape-Elbows"
SIDES = ("left", "right")
LEG_JOINTS = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
UPPER_JOINTS = (
    "shoulder_yaw", "shoulder_roll", "shoulder_pitch", "gripper"
)


def _ape_spec(path: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(path))
    # mjlab supplies its own terrain and contact sensors; retain the same ape
    # inertias, joints, actuator gains and force limits as the source MJCF.
    for name in (
        "left_sole_floor_found", "right_sole_floor_found",
        "left_hand_floor_found", "right_hand_floor_found",
    ):
        spec.delete(spec.sensor(name))
    spec.delete(spec.geom("floor"))
    spec.sensor("gyro").name = "imu_ang_vel"
    spec.sensor("local_linvel").name = "imu_lin_vel"
    return spec


def ape_spec() -> mujoco.MjSpec:
    return _ape_spec(MODEL)


def elbow_ape_spec() -> mujoco.MjSpec:
    return _ape_spec(ELBOW_MODEL)


def robot_cfg(elbowed: bool = False) -> EntityCfg:
    pose = {
        r".*hip_pitch.*": -0.2,
        r".*knee.*": 0.4,
        r".*ankle_pitch.*": -0.2,
        r".*shoulder_pitch.*": -0.6,
        r".*": 0.0,
    }
    if elbowed:
        pose = {r".*elbow_pitch.*": -1.1, **pose}
    return EntityCfg(
        spec_fn=elbow_ape_spec if elbowed else ape_spec,
        # MJLab creates its own initial-state keyframe rather than inheriting
        # the source MJCF's 0.181 m free-root qpos. Leaving pos at its default
        # puts the pelvis at z=0 and buries the soles/hands in the terrain.
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.181), joint_pos=pose),
        collisions=(CollisionCfg(
            geom_names_expr=(
                r"^(left_sole|right_sole|left_hand|right_hand|left_finger_tip|right_finger_tip)$",
            ),
            contype=1,
            conaffinity=1,
            condim=3,
            friction=(1.0,),
        ),),
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=(r".*",)),),
            soft_joint_pos_limit_factor=0.9,
        ),
    )


def env_cfg(play: bool = False, elbowed: bool = False, motion: str = "forward"):
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": robot_cfg(elbowed=elbowed)}
    cfg.viewer.body_name = "pelvis"
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    foot_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom", pattern=r"^(left_sole|right_sole)$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    hand_contact = ContactSensorCfg(
        name="hand_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_hand|right_hand|left_finger_tip|right_finger_tip)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
    )
    foot_height = TerrainHeightSensorCfg(
        name="foot_height_scan",
        frame=tuple(
            ObjRef(type="site", name=f"{side}_sole_center", entity="robot")
            for side in SIDES
        ),
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=2),
        ray_alignment="yaw",
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=False,
    )
    cfg.scene.sensors = (foot_contact, hand_contact, foot_height)

    # The 2 ms dynamic-gait run learned clean alternating steps; the 4 ms
    # static screen had not predicted its gait difference. Both keep 50 Hz control.
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 8.0
    cfg.scene.extent = 2.0
    del cfg.observations["actor"].terms["base_lin_vel"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    actor = cfg.observations["actor"]
    actor.terms["base_ang_vel"] = deepcopy(actor.terms["base_ang_vel"])
    actor.terms["projected_gravity"] = deepcopy(actor.terms["projected_gravity"])
    actor.terms["joint_pos"] = deepcopy(actor.terms["joint_pos"])
    actor.terms["joint_vel"] = deepcopy(actor.terms["joint_vel"])
    actor.terms["base_ang_vel"].noise = UniformNoiseCfg(n_min=-0.03, n_max=0.03)
    actor.terms["projected_gravity"].noise = UniformNoiseCfg(n_min=-0.01, n_max=0.01)
    actor.terms["joint_pos"].noise = UniformNoiseCfg(n_min=-0.001, n_max=0.001)
    actor.terms["joint_vel"].noise = UniformNoiseCfg(n_min=-0.25, n_max=0.25)

    # The upper body stays physically present, but its servo targets are held
    # at the settled tucked home pose. The first gait actor controls 10 legs.
    scale = {
        **{f"^{side}_{joint}$": 0.3 for side in SIDES for joint in LEG_JOINTS},
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in (
            UPPER_JOINTS + (("elbow_pitch",) if elbowed else ())
        )},
        "^head_yaw$": 0.0,
        "^head_pitch$": 0.0,
    }
    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = scale
    action.actuator_names = (r".*",)

    command = deepcopy(cfg.commands["twist"])
    assert isinstance(command, UniformVelocityCommandCfg)
    command.heading_command = False
    command.ranges.heading = None
    command.rel_heading_envs = 0.0
    command.rel_forward_envs = 0.0
    command.rel_standing_envs = 0.02
    if motion == "forward":
        command.ranges.lin_vel_x = (0.25, 0.40)
        command.ranges.lin_vel_y = (0.0, 0.0)
    elif motion == "backward":
        command.ranges.lin_vel_x = (-0.40, -0.25)
        command.ranges.lin_vel_y = (0.0, 0.0)
    elif motion == "signed":
        command.ranges.lin_vel_x = (-0.40, 0.40)
        command.ranges.lin_vel_y = (0.0, 0.0)
    elif motion == "sideways":
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (-0.30, 0.30)
    elif motion == "planar":
        # Pollen's published x and yaw ranges. Keep y fixed while teaching
        # turns and curves; the separate lateral lesson is not learned yet.
        command.ranges.lin_vel_x = (-0.40, 0.40)
        command.ranges.lin_vel_y = (0.0, 0.0)
    else:
        raise ValueError(f"Unknown motion: {motion}")
    command.ranges.ang_vel_z = (-1.0, 1.0) if motion == "planar" else (0.0, 0.0)
    if motion == "planar":
        # Pollen's explicit pure-turn bucket: independent x/yaw sampling almost
        # never presents zero translation with a meaningful turn command.
        cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(
            **vars(command), rel_turn_in_place_envs=0.15
        )
    else:
        cfg.commands["twist"] = command

    # Pollen's small-biped velocity recipe, with its ten leg joint names.
    leg_cfg = SceneEntityCfg("robot", joint_names=(
        r"^(left|right)_(hip_yaw|hip_roll|hip_pitch|knee|ankle_pitch)$",
    ))
    cfg.rewards["pose"].params["asset_cfg"] = leg_cfg
    cfg.rewards["pose"].params["std_standing"] = {
        r".*hip_yaw.*": 0.1, r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.15, r".*knee.*": 0.15, r".*ankle.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*hip_yaw.*": 0.3, r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.4, r".*knee.*": 0.4, r".*ankle.*": 0.25,
    }
    cfg.rewards["pose"].params["std_running"] = cfg.rewards["pose"].params["std_walking"]
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["upright"].params["std"] = math.sqrt(0.05)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards.pop("angular_momentum", None)  # The ape has no root_angmom XML sensor.
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)
    cfg.rewards["track_linear_velocity"].weight = 2.0
    cfg.rewards["track_angular_velocity"].weight = 2.0
    if motion == "planar":
        cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
    if motion == "signed":
        # Pollen's existing zero-command term gives the gait actor a direct
        # reason to settle after walking. Keep its scale level with velocity
        # tracking so this run changes one training signal at a time.
        cfg.rewards["stillness_at_zero"] = RewardTermCfg(
            func=microduck_mdp.stillness_at_zero_command,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "twist", "vel_std": 0.1},
        )
    cfg.rewards["air_time"].weight = 3.0
    cfg.rewards["air_time"].params.update({
        "threshold_min": 0.125, "threshold_max": 0.300,
        "command_threshold": 0.01,
    })
    for name in ("foot_clearance", "foot_slip"):
        cfg.rewards[name].params["asset_cfg"].site_names = (
            "left_sole_center", "right_sole_center"
        )
        cfg.rewards[name].params["command_threshold"] = 0.01
    cfg.rewards["foot_clearance"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.02
    cfg.rewards["foot_swing_height"].params["command_threshold"] = 0.01
    cfg.rewards["foot_slip"].weight = -0.1
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.terminations["hand_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "hand_ground_contact", "force_threshold": 1.0},
    )
    cfg.terminations.pop("out_of_terrain_bounds", None)

    # No guessed motor/contact DR: identify nominal hardware before widening.
    for name in ("push_robot", "foot_friction", "encoder_bias", "base_com"):
        cfg.events.pop(name, None)
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.004),
        "roll": (-0.035, 0.035), "pitch": (-0.035, 0.035), "yaw": (0.0, 0.0),
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.02, 0.02)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-0.1, 0.1)
    cfg.curriculum.clear()
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 500 * 24, "weight": -0.2},
                {"step": 750 * 24, "weight": -0.4},
                {"step": 1000 * 24, "weight": -0.6},
                {"step": 1250 * 24, "weight": -0.8},
                {"step": 1500 * 24, "weight": -1.0},
            ],
        },
    )
    if motion in ("signed", "planar"):
        # The upstream velocity lesson increases zero-command worlds only
        # after gait discovery. Keep its exact schedule for walk-to-stop.
        cfg.curriculum["standing_envs"] = CurriculumTermCfg(
            func=microduck_mdp.standing_envs_curriculum,
            params={
                "command_name": "twist",
                "standing_stages": [
                    {"step": 0, "rel_standing_envs": 0.02},
                    {"step": 500 * 24, "rel_standing_envs": 0.05},
                    {"step": 750 * 24, "rel_standing_envs": 0.1},
                    {"step": 1000 * 24, "rel_standing_envs": 0.15},
                    {"step": 1500 * 24, "rel_standing_envs": 0.2},
                    {"step": 2000 * 24, "rel_standing_envs": 0.25},
                ],
            },
        )
    if play:
        cfg.observations["actor"].enable_corruption = False
        cfg.curriculum.clear()
        cfg.episode_length_s = 20.0
    return cfg


def squat_env_cfg(play: bool = False):
    """Pollen's commanded sit/stand lesson, scaled to measured ape squat poses."""
    cfg = env_cfg(play=play, elbowed=True)
    cfg.episode_length_s = 12.0
    cfg.actions["joint_pos"].scale = {
        **{f"^{side}_{joint}$": 1.0 for side in SIDES for joint in LEG_JOINTS},
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in (
            UPPER_JOINTS + ("elbow_pitch",)
        )},
        "^head_yaw$": 0.0,
        "^head_pitch$": 0.0,
    }
    # The static v3 model settles at z≈0.178 with knees 0.4 rad and z≈0.117
    # with knees 1.9 rad. Both poses hold for four seconds without hand support.
    stand_z, squat_z = 0.178, 0.117
    squat_pose = {
        i: angle for i, angle in enumerate((
            0.0, 0.0, -0.95, 1.9, -0.95,
            0.0, 0.0, -0.95, 1.9, -0.95,
        ))
    }
    command = cfg.commands["twist"]
    command.resampling_time_range = (3.5, 6.5)
    cfg.commands["twist"] = microduck_mdp.SitStandCommandCfg(
        **{
            **vars(command),
            "sit_prob": 0.5,
            "ramp_s": 2.0,
            "sit_z": squat_z,
            "stand_z": stand_z,
        }
    )
    for name in (
        "track_linear_velocity", "track_angular_velocity", "air_time", "pose",
        "foot_clearance", "foot_swing_height", "foot_slip",
    ):
        cfg.rewards.pop(name, None)
    legs = list(range(10))
    posture = {"command_name": "twist", "sit_overrides": squat_pose, "joint_indices": legs}
    height = {"command_name": "twist", "sit_z": squat_z, "stand_z": stand_z}
    cfg.rewards["posture_pose_legs"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match, weight=4.0,
        params={**posture, "std": 0.5},
    )
    cfg.rewards["posture_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1, weight=1.0, params=posture,
    )
    cfg.rewards["posture_height"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian, weight=1.0,
        params={**height, "std": 0.04},
    )
    cfg.rewards["posture_height_sharp"] = RewardTermCfg(
        func=microduck_mdp.posture_height_gaussian, weight=1.0,
        params={**height, "std": 0.015},
    )
    cfg.rewards["posture_height_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_height_l1, weight=6.0, params=height,
    )
    cfg.rewards["posture_stillness"] = RewardTermCfg(
        func=microduck_mdp.posture_stillness, weight=2.0,
        params={**height, "band_full": 0.012, "band_zero": 0.03, "vel_std": 0.05},
    )
    cfg.rewards["posture_composite"] = RewardTermCfg(
        func=microduck_mdp.posture_composite, weight=3.0,
        params={**posture, "sit_z": squat_z, "stand_z": stand_z,
                "height_std": 0.03, "upright_std": 0.40, "pose_std": 0.40},
    )
    cfg.rewards["descent_speed"] = RewardTermCfg(
        func=microduck_mdp.trunk_downward_velocity_penalty, weight=10.0,
        params={"max_down_vel": 0.05,
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.rewards["upright_linear"] = RewardTermCfg(
        func=microduck_mdp.body_upright_linear, weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.terminations.pop("fell_over", None)
    cfg.terminations.pop("hand_contact", None)
    cfg.events["set_ground_state"] = EventTermCfg(
        func=microduck_mdp.set_random_ground_state, mode="reset",
        params={
            "face_down_prob": 0.0, "face_up_prob": 0.0,
            "sitting_prob": 0.5, "standing_prob": 0.5,
            "sitting_joint_overrides": squat_pose,
            "sitting_joint_noise_std": 0.03,
            "sitting_tilt_max": math.radians(3),
            "sitting_z_min": squat_z,
            "sitting_z_max": squat_z + 0.003,
            "standing_z_min": stand_z,
            "standing_z_max": stand_z + 0.003,
        },
    )
    return cfg


def stand_env_cfg(play: bool = False):
    """Same posture contract as squat, with only the standing lesson sampled."""
    cfg = squat_env_cfg(play=play)
    cfg.episode_length_s = 8.0
    cfg.commands["twist"].sit_prob = 0.0
    cfg.events["set_ground_state"].params["sitting_prob"] = 0.0
    cfg.events["set_ground_state"].params["standing_prob"] = 1.0
    return cfg


def squat_heading_env_cfg(play: bool = False):
    cfg = squat_env_cfg(play=play)
    # Pollen's heading-hold term pays returning to spawn yaw, allowing active
    # correction; the first learned squat cycles rotated ~90 degrees in 10 s.
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=1.5,
        params={"std": 0.4, "asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg


class ApeSitStandHeadingCommand(microduck_mdp.SitStandCommand):
    """Pollen sit/stand command with its heading error exposed in slot 2.

    The original heading-hold reward sees spawn yaw, but the posture actor
    otherwise cannot observe yaw error. Keep the 3D command and checkpoint
    shape unchanged while making the corrective signal observable.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._spawn_heading = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float) -> None:
        super().compute(dt)
        quat = self.robot.data.root_link_quat_w
        w, x, y, z = quat.unbind(dim=1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        fresh = self._env_ref.episode_length_buf <= 1
        self._spawn_heading = torch.where(fresh, yaw, self._spawn_heading)
        error = self._spawn_heading - yaw
        self.vel_command_b[:, 2] = torch.atan2(torch.sin(error), torch.cos(error))


@dataclass(kw_only=True)
class ApeSitStandHeadingCommandCfg(microduck_mdp.SitStandCommandCfg):
    class_type: type = ApeSitStandHeadingCommand

    def build(self, env):
        return ApeSitStandHeadingCommand(self, env)


def squat_heading_observed_env_cfg(play: bool = False):
    cfg = squat_heading_env_cfg(play=play)
    cfg.commands["twist"] = ApeSitStandHeadingCommandCfg(
        **{**vars(cfg.commands["twist"]), "class_type": ApeSitStandHeadingCommand}
    )
    return cfg


class ApeStandTransferCommand(ApeSitStandHeadingCommand):
    """Pollen's posture command, with y=+1/-1 selecting the support foot."""

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        n = len(env_ids)
        if n:
            self.vel_command_b[env_ids, 1] = self.cfg.support_magnitude * torch.where(
                torch.rand(n, device=self.device) < 0.5, -1.0, 1.0
            )


@dataclass(kw_only=True)
class ApeStandTransferCommandCfg(ApeSitStandHeadingCommandCfg):
    class_type: type = ApeStandTransferCommand
    support_magnitude: float = 0.2

    def build(self, env):
        return ApeStandTransferCommand(self, env)


def support_com_match(
    env, asset_cfg: SceneEntityCfg, command_name: str = "twist", std: float = 0.04,
    target_fraction: float = 1.0,
) -> torch.Tensor:
    """Whole-body COM over the requested support foot, following Pollen Flamingo."""
    asset = env.scene[asset_cfg.name]
    left, right = asset.data.site_pos_w[:, asset_cfg.site_ids, :2].unbind(dim=1)
    side = env.command_manager.get_command(command_name)[:, 1].sign()
    target = (left + right) / 2.0 + target_fraction * side[:, None] * (left - right) / 2.0
    # root_com_pos_w is the root body COM, not the robot's aggregate COM.
    whole_com_xy = env.sim.data.subtree_com[:, asset.indexing.root_body_id, :2]
    error2 = ((whole_com_xy - target) ** 2).sum(dim=1)
    return torch.exp(-error2 / std**2)


def requested_support_grounded(env, command_name: str = "twist") -> torch.Tensor:
    """Command-conditioned form of Pollen's grounded support-foot reward."""
    found = env.scene["feet_ground_contact"].data.found
    left = env.command_manager.get_command(command_name)[:, 1] >= 0.0
    return torch.where(left, found[:, 0], found[:, 1]).float()


def transfer_env_cfg(play: bool = False):
    cfg = squat_heading_observed_env_cfg(play=play)
    cfg.episode_length_s = 8.0
    cfg.commands["twist"].sit_prob = 0.0
    cfg.commands["twist"].resampling_time_range = (3.5, 6.5)
    cfg.commands["twist"] = ApeStandTransferCommandCfg(
        **{**vars(cfg.commands["twist"]), "class_type": ApeStandTransferCommand}
    )
    cfg.events["set_ground_state"].params["sitting_prob"] = 0.0
    cfg.events["set_ground_state"].params["standing_prob"] = 1.0
    cfg.rewards["support_com"] = RewardTermCfg(
        func=support_com_match, weight=4.0,
        params={"asset_cfg": SceneEntityCfg(
            "robot", site_names=("left_sole_center", "right_sole_center")
        ), "command_name": "twist", "std": 0.04},
    )
    cfg.rewards["support_foot_grounded"] = RewardTermCfg(
        func=requested_support_grounded, weight=2.0,
        params={"command_name": "twist"},
    )
    cfg.rewards["foot_slip"] = deepcopy(env_cfg(elbowed=True).rewards["foot_slip"])
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.terminations["hand_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "hand_ground_contact", "force_threshold": 1.0},
    )
    return cfg


def partial_transfer_env_cfg(play: bool = False):
    """Same support lesson, half-stance goal inside the statically feasible range."""
    cfg = transfer_env_cfg(play=play)
    cfg.rewards["support_com"].params["target_fraction"] = 0.5
    cfg.rewards["support_com"].params["std"] = 0.02
    return cfg


def arm_raise_env_cfg(play: bool = False):
    """Pollen's ramped posture lesson, isolated to both shoulder pitches.

    Legs stay at their measured stable home pose so this tests arm control
    alone. The first +1.0 rad target made the hand descend before it could
    rise, so use the nearer -1.2 rad forward/upward target first.
    """
    cfg = squat_heading_observed_env_cfg(play=play)
    cfg.episode_length_s = 8.0
    cfg.commands["twist"].sit_prob = 0.5
    cfg.commands["twist"].sit_z = cfg.commands["twist"].stand_z
    cfg.commands["twist"].ramp_s = 1.5
    cfg.events["set_ground_state"].params["sitting_prob"] = 0.0
    cfg.events["set_ground_state"].params["standing_prob"] = 1.0
    cfg.actions["joint_pos"].scale = {
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in LEG_JOINTS},
        **{f"^{side}_shoulder_pitch$": 1.0 for side in SIDES},
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in (
            "shoulder_yaw", "shoulder_roll", "elbow_pitch", "gripper"
        )},
        "^head_yaw$": 0.0,
        "^head_pitch$": 0.0,
    }
    for name in ("posture_pose_legs", "posture_pose_l1", "posture_composite"):
        cfg.rewards.pop(name)
    # Pollen's native pose tracking supplies the commanded, slewed joint goal.
    # MJLab's compiled joint order places head (10-11) before the arms.
    arm_pose = {"command_name": "twist", "sit_overrides": {14: -1.2, 19: -1.2},
                "joint_indices": [14, 19]}
    cfg.rewards["arm_pose"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_match, weight=4.0,
        params={**arm_pose, "std": 0.5},
    )
    cfg.rewards["arm_pose_l1"] = RewardTermCfg(
        func=microduck_mdp.posture_pose_l1, weight=1.0, params=arm_pose,
    )
    cfg.terminations["hand_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "hand_ground_contact", "force_threshold": 1.0},
    )
    return cfg


def independent_arm_raise_env_cfg(play: bool = False):
    """The established arm lesson with independent left/right posture flags."""
    cfg = arm_raise_env_cfg(play=play)
    cfg.commands["right_arm"] = microduck_mdp.SitStandCommandCfg(
        **{**vars(cfg.commands["twist"]), "class_type": microduck_mdp.SitStandCommand}
    )
    for group in ("actor", "critic"):
        cfg.observations[group].terms["right_arm_command"] = ObservationTermCfg(
            func=mdp.generated_commands, params={"command_name": "right_arm"}
        )
    cfg.rewards.pop("arm_pose")
    cfg.rewards.pop("arm_pose_l1")
    for side, command_name, joint_idx in (
        ("left", "twist", 14), ("right", "right_arm", 19)
    ):
        pose = {"command_name": command_name,
                "sit_overrides": {joint_idx: -1.2}, "joint_indices": [joint_idx]}
        cfg.rewards[f"{side}_arm_pose"] = RewardTermCfg(
            func=microduck_mdp.posture_pose_match, weight=2.0,
            params={**pose, "std": 0.5},
        )
        cfg.rewards[f"{side}_arm_pose_l1"] = RewardTermCfg(
            func=microduck_mdp.posture_pose_l1, weight=0.5, params=pose,
        )
    return cfg


# Right-stance pose from find_flamingo_pose.py and flamingo-pose-screen.json.
# Order is the model's 22 servo joints: left leg, right leg, head, left arm,
# right arm. The tucked arms are essential: straight-down arms collide with
# the soles at this trunk lean. This is a RESET pose, never an action script.
APE_FLAMINGO_RIGHT_POSE = (
    -0.022245, -0.700000, -0.631011, 0.050001, 0.314258,
     0.090021, -0.406406, -0.259267, 0.559223, -0.266818,
     0.0, 0.0,
     0.0, 0.0, -0.6, -1.1, 0.0,
     0.0, 0.0, -0.6, -1.1, 0.0,
)
APE_FLAMINGO_LEFT_POSE = (
    -APE_FLAMINGO_RIGHT_POSE[5], -APE_FLAMINGO_RIGHT_POSE[6],
    *APE_FLAMINGO_RIGHT_POSE[7:10],
    -APE_FLAMINGO_RIGHT_POSE[0], -APE_FLAMINGO_RIGHT_POSE[1],
    *APE_FLAMINGO_RIGHT_POSE[2:5],
    *APE_FLAMINGO_RIGHT_POSE[10:],
)
APE_FLAMINGO_ROLL = 0.468577
APE_FLAMINGO_PITCH = 0.007528
APE_FLAMINGO_Z = 0.19158
APE_FLAMINGO_GRAVITY = (
    math.sin(APE_FLAMINGO_PITCH),
    -math.sin(APE_FLAMINGO_ROLL) * math.cos(APE_FLAMINGO_PITCH),
    -math.cos(APE_FLAMINGO_ROLL) * math.cos(APE_FLAMINGO_PITCH),
)


def flamingo_hold_env_cfg(play: bool = False):
    """Pollen Flamingo stage 1 on the ape: learn balance from an in-pose reset.

    The published Microduck lesson is pinned at commit 0bf9897. Our geometry,
    pose, sole names, and motor contract change; its reward/command structure
    and native MJLab PPO trainer remain intact. No stand-to-lift claim here.
    """
    cfg = env_cfg(play=play, elbowed=True)
    cfg.episode_length_s = 6.0
    cfg.actions["joint_pos"].scale = {
        **{f"^{side}_{joint}$": 1.0 for side in SIDES for joint in LEG_JOINTS},
        **{f"^{side}_{joint}$": 0.0 for side in SIDES for joint in (
            UPPER_JOINTS + ("elbow_pitch",)
        )},
        "^head_yaw$": 0.0, "^head_pitch$": 0.0,
    }
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (6.0, 12.0)
    command.ranges.lin_vel_x = (-0.01, 0.01)
    command.ranges.lin_vel_y = (-0.01, 0.01)
    command.ranges.ang_vel_z = (-0.05, 0.05)
    cfg.commands["twist"] = microduck_mdp.VelocityCommandCommandOnlyCfg(**vars(command))

    for name in (
        "track_linear_velocity", "track_angular_velocity", "air_time",
        "foot_clearance", "foot_swing_height", "pose", "upright",
        "dof_pos_limits",
    ):
        cfg.rewards.pop(name, None)
    stance = SceneEntityCfg("robot", site_names=("right_sole_center",))
    swing = SceneEntityCfg("robot", site_names=("left_sole_center",))
    feet = "feet_ground_contact"
    cfg.rewards["com_over_stance_foot"] = RewardTermCfg(
        func=microduck_mdp.com_over_foot, weight=3.0,
        params={"asset_cfg": stance, "std": 0.03},
    )
    cfg.rewards["stance_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.foot_contact_reward, weight=1.0,
        params={"sensor_name": feet, "slot": 1},
    )
    cfg.rewards["swing_foot_clear"] = RewardTermCfg(
        func=microduck_mdp.foot_height_gaussian, weight=1.5,
        params={"asset_cfg": swing, "target": 0.06, "std": 0.03,
                "gate_sensor_name": feet, "gate_slot": 1},
    )
    cfg.rewards["swing_foot_touch"] = RewardTermCfg(
        func=microduck_mdp.foot_contact_penalty, weight=0.5,
        params={"sensor_name": feet, "slot": 0},
    )
    cfg.rewards["pose_flamingo"] = RewardTermCfg(
        func=microduck_mdp.pose_target_match, weight=1.5,
        params={"target_overrides": dict(enumerate(APE_FLAMINGO_RIGHT_POSE)), "std": 0.5},
    )
    cfg.rewards["gravity_flamingo"] = RewardTermCfg(
        func=microduck_mdp.projected_gravity_match, weight=2.0,
        params={"target": APE_FLAMINGO_GRAVITY, "std": 0.15},
    )
    cfg.rewards["stillness"] = RewardTermCfg(
        func=microduck_mdp.joint_vel_gaussian, weight=1.0,
        params={"std": 2.0, "gate_sensor_name": feet, "gate_slot": 1},
    )
    cfg.rewards["stance_side_tilt"] = RewardTermCfg(
        func=microduck_mdp.lateral_tilt_penalty, weight=4.0,
        params={"threshold": 0.55, "direction": -1.0},
    )
    cfg.rewards["joint_limit_proximity"] = RewardTermCfg(
        func=microduck_mdp.joint_pos_limit_proximity, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "margin": 0.10},
    )
    cfg.rewards["foot_slip"].params["command_threshold"] = 0.0
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis",)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.2,
                "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))},
    )
    cfg.events["set_flamingo_state"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_state, mode="reset",
        params={
            "joint_pose": list(APE_FLAMINGO_RIGHT_POSE),
            "base_roll": APE_FLAMINGO_ROLL,
            "base_pitch": APE_FLAMINGO_PITCH,
            "z_min": APE_FLAMINGO_Z,
            "z_max": APE_FLAMINGO_Z + 0.004,
            "tilt_noise": math.radians(2.0),
            "joint_noise_std": 0.03,
            "standing_prob": 0.0,
        },
    )
    if not play:
        cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
            func=microduck_mdp.reward_weight,
            params={"reward_name": "action_rate_l2", "weight_stages": [
                {"step": 0, "weight": -0.1},
                {"step": 400 * 24, "weight": -0.5},
                {"step": 800 * 24, "weight": -1.0},
            ]},
        )
        cfg.curriculum["flamingo_com_std"] = CurriculumTermCfg(
            func=microduck_mdp.reward_param_curriculum,
            params={"reward_name": "com_over_stance_foot", "param_stages": [
                {"step": 0, "params": {"std": 0.03}},
                {"step": 500 * 24, "params": {"std": 0.02}},
            ]},
        )
    return cfg


def flamingo_hold_effort_env_cfg(play: bool = False):
    """One-change continuation: penalize sustained motor effort on the hold."""
    cfg = flamingo_hold_env_cfg(play=play)
    # Checkpoint 200 holds the pose but continuously saturates leg motors.
    # Pollen's native actuator-force L2 term gives this a gradient. Its
    # measured leg-force-squared sum is 11.42 N²m² per step, so -0.1 costs
    # about 6.85 over a six-second episode before the small upper-body share.
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    return cfg


def flamingo_hold_strong_effort_env_cfg(play: bool = False):
    """Same hold and torque penalty, fourfold weight to test motor headroom."""
    cfg = flamingo_hold_effort_env_cfg(play=play)
    cfg.rewards["joint_torques_l2"].weight = -0.4
    return cfg


def flamingo_hold_command_limit_env_cfg(play: bool = False):
    """Pollen command-limit term deters actuator-limit jamming by the policy."""
    cfg = flamingo_hold_strong_effort_env_cfg(play=play)
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-2.0,
        params={"action_name": "joint_pos", "overshoot": 0.1},
    )
    return cfg


def flamingo_hold_command_limit_left_env_cfg(play: bool = False):
    """The same right-foot lesson mirrored onto the left support side."""
    cfg = flamingo_hold_command_limit_env_cfg(play=play)
    cfg.rewards["com_over_stance_foot"].params["asset_cfg"] = SceneEntityCfg(
        "robot", site_names=("left_sole_center",)
    )
    cfg.rewards["stance_foot_grounded"].params["slot"] = 0
    cfg.rewards["swing_foot_clear"].params.update({
        "asset_cfg": SceneEntityCfg("robot", site_names=("right_sole_center",)),
        "gate_slot": 0,
    })
    cfg.rewards["swing_foot_touch"].params["slot"] = 1
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(
        enumerate(APE_FLAMINGO_LEFT_POSE)
    )
    cfg.rewards["gravity_flamingo"].params["target"] = (
        APE_FLAMINGO_GRAVITY[0], -APE_FLAMINGO_GRAVITY[1],
        APE_FLAMINGO_GRAVITY[2],
    )
    cfg.rewards["stillness"].params["gate_slot"] = 0
    cfg.rewards["stance_side_tilt"].params["direction"] = 1.0
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": list(APE_FLAMINGO_LEFT_POSE),
        "base_roll": -APE_FLAMINGO_ROLL,
    })
    return cfg


def flamingo_cycle_env_cfg(play: bool = False):
    """Pollen Flamingo Cycle stage 2 on the ape: stand ↔ either one-foot pose.

    The command, reset, reward gating and curriculum are upstream Pollen's.
    Ape-specific changes are its two validated poses, 0.178 m standing height,
    sole sites, 0.06 m swing target and the successful stage-one motor penalties.
    """
    cfg = flamingo_hold_command_limit_env_cfg(play=play)
    cfg.episode_length_s = 10.0
    for name in (
        "com_over_stance_foot", "stance_foot_grounded", "swing_foot_clear",
        "swing_foot_touch", "pose_flamingo", "gravity_flamingo", "stillness",
        "stance_side_tilt",
    ):
        cfg.rewards.pop(name, None)
    left = SceneEntityCfg("robot", site_names=("left_sole_center",))
    right = SceneEntityCfg("robot", site_names=("right_sole_center",))
    shared = {"command_name": "twist"}
    cfg.rewards["com_target"] = RewardTermCfg(
        func=microduck_mdp.fl_com_target, weight=3.0,
        params={**shared, "left_cfg": left, "right_cfg": right, "std": 0.03},
    )
    cfg.rewards["stance_foot_grounded"] = RewardTermCfg(
        func=microduck_mdp.fl_stance_foot_grounded, weight=1.0,
        params={**shared, "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards["swing_foot_contact"] = RewardTermCfg(
        func=microduck_mdp.fl_swing_foot_contact, weight=0.5,
        params={**shared, "sensor_name": "feet_ground_contact", "lo": 0.4, "hi": 0.9},
    )
    cfg.rewards["swing_foot_clear"] = RewardTermCfg(
        func=microduck_mdp.fl_swing_foot_clear, weight=1.5,
        params={**shared, "left_cfg": left, "right_cfg": right,
                "sensor_name": "feet_ground_contact", "target": 0.06,
                "std": 0.04, "lo": 0.6, "hi": 1.0},
    )
    cfg.rewards["pose_track"] = RewardTermCfg(
        func=microduck_mdp.fl_pose_track, weight=1.5,
        params={**shared, "pose_right": APE_FLAMINGO_RIGHT_POSE,
                "pose_left": APE_FLAMINGO_LEFT_POSE, "std": 0.5,
                "mid_attenuation": 0.75},
    )
    cfg.rewards["gravity_track"] = RewardTermCfg(
        func=microduck_mdp.fl_gravity_track, weight=2.0,
        params={**shared, "gravity_right": APE_FLAMINGO_GRAVITY,
                "gravity_left": (APE_FLAMINGO_GRAVITY[0],
                                 -APE_FLAMINGO_GRAVITY[1],
                                 APE_FLAMINGO_GRAVITY[2]), "std": 0.15},
    )
    cfg.rewards["stillness"] = RewardTermCfg(
        func=microduck_mdp.fl_stillness, weight=1.0,
        params={**shared, "sensor_name": "feet_ground_contact", "std": 2.0},
    )
    cfg.rewards["stance_side_tilt"] = RewardTermCfg(
        func=microduck_mdp.fl_stance_side_tilt, weight=4.0,
        params={**shared, "threshold": 0.55},
    )
    cfg.rewards["commanded_support"] = RewardTermCfg(
        func=microduck_mdp.fl_single_support_success, weight=1e-3,
        params={**shared, "sensor_name": "feet_ground_contact"},
    )

    command = cfg.commands["twist"]
    command.resampling_time_range = (2.5, 5.0)
    cfg.commands["twist"] = microduck_mdp.FlamingoCommandCfg(
        **{**{k: v for k, v in vars(command).items()
             if k != "rel_turn_in_place_envs"},
           "flamingo_prob": 0.6, "ramp_s": 1.5, "zero_side_prob": 0.5},
    )

    cfg.events.pop("set_flamingo_state")
    cfg.events["set_flamingo_cycle_state"] = EventTermCfg(
        func=microduck_mdp.set_flamingo_cycle_state,
        mode="reset",
        params={
            "pose_right": APE_FLAMINGO_RIGHT_POSE,
            "pose_left": APE_FLAMINGO_LEFT_POSE,
            "base_roll": APE_FLAMINGO_ROLL,
            "base_pitch": APE_FLAMINGO_PITCH,
            "z_min": APE_FLAMINGO_Z,
            "z_max": APE_FLAMINGO_Z + 0.004,
            "command_name": "twist",
            "tilt_noise": math.radians(2.0),
            "joint_noise_std": 0.03,
            "in_pose_prob": 0.6,
            "standing_z_min": 0.178,
            "standing_z_max": 0.182,
        },
    )
    if not play:
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            {"step": 0, "weight": -0.1},
            {"step": 600 * 24, "weight": -0.5},
        ]
        cfg.curriculum["flamingo_com_std"].params.update({
            "reward_name": "com_target",
            "param_stages": [
                {"step": 0, "params": {"std": 0.03}},
                {"step": 700 * 24, "params": {"std": 0.02}},
            ],
        })
        cfg.curriculum["in_pose_prob"] = CurriculumTermCfg(
            func=microduck_mdp.event_param_curriculum,
            params={"event_name": "set_flamingo_cycle_state",
                    "param_stages": [
                        {"step": 0, "params": {"in_pose_prob": 0.6}},
                        {"step": 700 * 24, "params": {"in_pose_prob": 0.5}},
                        {"step": 1400 * 24, "params": {"in_pose_prob": 0.4}},
                        {"step": 2100 * 24, "params": {"in_pose_prob": 0.3}},
                    ]},
        )
    return cfg


def flamingo_cycle_arms_env_cfg(play: bool = False):
    """Same contact-corrected cycle, with shoulder and elbow balance actions."""
    cfg = flamingo_cycle_env_cfg(play=play)
    cfg.rewards["swing_foot_contact"].weight = 1.5
    scale = dict(cfg.actions["joint_pos"].scale)
    for side in SIDES:
        scale[f"^{side}_shoulder_roll$"] = 1.0
        scale[f"^{side}_shoulder_pitch$"] = 1.0
        scale[f"^{side}_elbow_pitch$"] = 0.5
    cfg.actions["joint_pos"].scale = scale
    return cfg


ape_rl_cfg = deepcopy(MicroduckRlCfg)
ape_rl_cfg.experiment_name = "ape_velocity"
ape_rl_cfg.run_name = "ape_forward"
ape_rl_cfg.max_iterations = 4_000

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=env_cfg(),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

backward_ape_rl_cfg = deepcopy(ape_rl_cfg)
backward_ape_rl_cfg.run_name = "ape_backward_from_forward"
register_mjlab_task(
    task_id=BACKWARD_TASK_ID,
    env_cfg=env_cfg(motion="backward"),
    play_env_cfg=env_cfg(play=True, motion="backward"),
    rl_cfg=backward_ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

signed_ape_rl_cfg = deepcopy(ape_rl_cfg)
signed_ape_rl_cfg.run_name = "ape_signed_velocity"
register_mjlab_task(
    task_id=SIGNED_TASK_ID,
    env_cfg=env_cfg(motion="signed"),
    play_env_cfg=env_cfg(play=True, motion="signed"),
    rl_cfg=signed_ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

sideways_ape_rl_cfg = deepcopy(ape_rl_cfg)
sideways_ape_rl_cfg.run_name = "ape_sideways"
register_mjlab_task(
    task_id=SIDEWAYS_TASK_ID,
    env_cfg=env_cfg(motion="sideways"),
    play_env_cfg=env_cfg(play=True, motion="sideways"),
    rl_cfg=sideways_ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

planar_ape_rl_cfg = deepcopy(ape_rl_cfg)
planar_ape_rl_cfg.run_name = "ape_planar_velocity"
register_mjlab_task(
    task_id=PLANAR_TASK_ID,
    env_cfg=env_cfg(motion="planar"),
    play_env_cfg=env_cfg(play=True, motion="planar"),
    rl_cfg=planar_ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

elbow_ape_rl_cfg = deepcopy(ape_rl_cfg)
elbow_ape_rl_cfg.experiment_name = "ape_elbows_velocity"
elbow_ape_rl_cfg.run_name = "ape_elbows_forward"

register_mjlab_task(
    task_id=ELBOW_TASK_ID,
    env_cfg=env_cfg(elbowed=True),
    play_env_cfg=env_cfg(play=True, elbowed=True),
    rl_cfg=elbow_ape_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)


def cad_v8_walk_env_cfg(play: bool = False):
    """The unchanged upstream velocity lesson on the case-dynamics ape body."""
    cfg = env_cfg(play=play, elbowed=True)
    cfg.scene.entities["robot"].spec_fn = partial(_ape_spec, CAD_V8_MODEL)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    return cfg


cad_v8_walk_rl_cfg = deepcopy(elbow_ape_rl_cfg)
cad_v8_walk_rl_cfg.experiment_name = "ape_cad_v8_velocity"
cad_v8_walk_rl_cfg.run_name = "ape_cad_v8_forward"
register_mjlab_task(
    task_id=CAD_V8_WALK_TASK_ID,
    env_cfg=cad_v8_walk_env_cfg(),
    play_env_cfg=cad_v8_walk_env_cfg(play=True),
    rl_cfg=cad_v8_walk_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

for task_id, motion in (
    (ELBOW_BACKWARD_TASK_ID, "backward"),
    (ELBOW_SIDEWAYS_TASK_ID, "sideways"),
    (ELBOW_PLANAR_TASK_ID, "planar"),
):
    motion_rl_cfg = deepcopy(elbow_ape_rl_cfg)
    motion_rl_cfg.run_name = f"ape_elbows_{motion}"
    register_mjlab_task(
        task_id=task_id,
        env_cfg=env_cfg(elbowed=True, motion=motion),
        play_env_cfg=env_cfg(play=True, elbowed=True, motion=motion),
        rl_cfg=motion_rl_cfg,
        runner_cls=VelocityOnPolicyRunner,
    )


def planar_sole_variant_env_cfg(model: Path, play: bool = False):
    """Same native Pollen planar lesson; only the measured sole width differs."""
    cfg = env_cfg(play=play, elbowed=True, motion="planar")
    cfg.scene.entities["robot"].spec_fn = partial(_ape_spec, model)
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    return cfg


for task_id, model, run_name in (
    (ELBOW_PLANAR_PROXY60_TASK_ID, ELBOW_PROXY60_MODEL, "ape_turn_proxy60"),
    (ELBOW_PLANAR_NARROW55_TASK_ID, ELBOW_NARROW55_MODEL, "ape_turn_narrow55"),
):
    variant_rl_cfg = deepcopy(elbow_ape_rl_cfg)
    variant_rl_cfg.experiment_name = "ape_elbows_turn_sole_width"
    variant_rl_cfg.run_name = run_name
    register_mjlab_task(
        task_id=task_id,
        env_cfg=planar_sole_variant_env_cfg(model),
        play_env_cfg=planar_sole_variant_env_cfg(model, play=True),
        rl_cfg=variant_rl_cfg,
        runner_cls=VelocityOnPolicyRunner,
    )

squat_rl_cfg = deepcopy(elbow_ape_rl_cfg)
squat_rl_cfg.experiment_name = "ape_elbows_posture"
squat_rl_cfg.run_name = "ape_elbows_squat"
register_mjlab_task(
    task_id=ELBOW_SQUAT_TASK_ID,
    env_cfg=squat_env_cfg(),
    play_env_cfg=squat_env_cfg(play=True),
    rl_cfg=squat_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

stand_rl_cfg = deepcopy(squat_rl_cfg)
stand_rl_cfg.run_name = "ape_elbows_stand"
register_mjlab_task(
    task_id=ELBOW_STAND_TASK_ID,
    env_cfg=stand_env_cfg(),
    play_env_cfg=stand_env_cfg(play=True),
    rl_cfg=stand_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

low_noise_squat_rl_cfg = deepcopy(squat_rl_cfg)
low_noise_squat_rl_cfg.experiment_name = "ape_elbows_posture_low_noise"
low_noise_squat_rl_cfg.run_name = "ape_elbows_squat_low_noise"
low_noise_squat_rl_cfg.actor.distribution_cfg["init_std"] = 0.2
register_mjlab_task(
    task_id=ELBOW_SQUAT_LOW_NOISE_TASK_ID,
    env_cfg=squat_env_cfg(),
    play_env_cfg=squat_env_cfg(play=True),
    rl_cfg=low_noise_squat_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

heading_squat_rl_cfg = deepcopy(low_noise_squat_rl_cfg)
heading_squat_rl_cfg.run_name = "ape_elbows_squat_heading"
register_mjlab_task(
    task_id=ELBOW_SQUAT_HEADING_TASK_ID,
    env_cfg=squat_heading_env_cfg(),
    play_env_cfg=squat_heading_env_cfg(play=True),
    rl_cfg=heading_squat_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

observed_heading_rl_cfg = deepcopy(heading_squat_rl_cfg)
observed_heading_rl_cfg.run_name = "ape_elbows_squat_heading_observed"
register_mjlab_task(
    task_id=ELBOW_SQUAT_HEADING_OBS_TASK_ID,
    env_cfg=squat_heading_observed_env_cfg(),
    play_env_cfg=squat_heading_observed_env_cfg(play=True),
    rl_cfg=observed_heading_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

transfer_rl_cfg = deepcopy(observed_heading_rl_cfg)
transfer_rl_cfg.run_name = "ape_elbows_support_transfer"
register_mjlab_task(
    task_id=ELBOW_TRANSFER_TASK_ID,
    env_cfg=transfer_env_cfg(),
    play_env_cfg=transfer_env_cfg(play=True),
    rl_cfg=transfer_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

partial_transfer_rl_cfg = deepcopy(transfer_rl_cfg)
partial_transfer_rl_cfg.run_name = "ape_elbows_support_partial"
register_mjlab_task(
    task_id=ELBOW_TRANSFER_PARTIAL_TASK_ID,
    env_cfg=partial_transfer_env_cfg(),
    play_env_cfg=partial_transfer_env_cfg(play=True),
    rl_cfg=partial_transfer_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

arm_rl_cfg = deepcopy(observed_heading_rl_cfg)
arm_rl_cfg.run_name = "ape_elbows_arm_raise"
register_mjlab_task(
    task_id=ELBOW_ARM_TASK_ID,
    env_cfg=arm_raise_env_cfg(),
    play_env_cfg=arm_raise_env_cfg(play=True),
    rl_cfg=arm_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

independent_arm_rl_cfg = deepcopy(arm_rl_cfg)
independent_arm_rl_cfg.run_name = "ape_elbows_arms_independent"
register_mjlab_task(
    task_id=ELBOW_ARM_INDEPENDENT_TASK_ID,
    env_cfg=independent_arm_raise_env_cfg(),
    play_env_cfg=independent_arm_raise_env_cfg(play=True),
    rl_cfg=independent_arm_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_hold_rl_cfg = deepcopy(observed_heading_rl_cfg)
flamingo_hold_rl_cfg.experiment_name = "ape_elbows_flamingo"
flamingo_hold_rl_cfg.run_name = "ape_elbows_flamingo_right_hold"
flamingo_hold_rl_cfg.max_iterations = 500
flamingo_hold_rl_cfg.save_interval = 50
flamingo_hold_rl_cfg.actor.distribution_cfg["init_std"] = 1.0
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_HOLD_TASK_ID,
    env_cfg=flamingo_hold_env_cfg(),
    play_env_cfg=flamingo_hold_env_cfg(play=True),
    rl_cfg=flamingo_hold_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_effort_rl_cfg = deepcopy(flamingo_hold_rl_cfg)
flamingo_effort_rl_cfg.run_name = "ape_elbows_flamingo_right_effort"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_EFFORT_TASK_ID,
    env_cfg=flamingo_hold_effort_env_cfg(),
    play_env_cfg=flamingo_hold_effort_env_cfg(play=True),
    rl_cfg=flamingo_effort_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_strong_effort_rl_cfg = deepcopy(flamingo_effort_rl_cfg)
flamingo_strong_effort_rl_cfg.run_name = "ape_elbows_flamingo_right_strong_effort"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_STRONG_EFFORT_TASK_ID,
    env_cfg=flamingo_hold_strong_effort_env_cfg(),
    play_env_cfg=flamingo_hold_strong_effort_env_cfg(play=True),
    rl_cfg=flamingo_strong_effort_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_command_limit_rl_cfg = deepcopy(flamingo_strong_effort_rl_cfg)
flamingo_command_limit_rl_cfg.run_name = "ape_elbows_flamingo_right_command_limit"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_COMMAND_LIMIT_TASK_ID,
    env_cfg=flamingo_hold_command_limit_env_cfg(),
    play_env_cfg=flamingo_hold_command_limit_env_cfg(play=True),
    rl_cfg=flamingo_command_limit_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_left_rl_cfg = deepcopy(flamingo_command_limit_rl_cfg)
flamingo_left_rl_cfg.run_name = "ape_elbows_flamingo_left_command_limit"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_LEFT_TASK_ID,
    env_cfg=flamingo_hold_command_limit_left_env_cfg(),
    play_env_cfg=flamingo_hold_command_limit_left_env_cfg(play=True),
    rl_cfg=flamingo_left_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_cycle_rl_cfg = deepcopy(flamingo_command_limit_rl_cfg)
flamingo_cycle_rl_cfg.experiment_name = "ape_elbows_flamingo_cycle"
flamingo_cycle_rl_cfg.run_name = "ape_elbows_flamingo_cycle"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_CYCLE_TASK_ID,
    env_cfg=flamingo_cycle_env_cfg(),
    play_env_cfg=flamingo_cycle_env_cfg(play=True),
    rl_cfg=flamingo_cycle_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

flamingo_cycle_arms_rl_cfg = deepcopy(flamingo_cycle_rl_cfg)
flamingo_cycle_arms_rl_cfg.run_name = "ape_elbows_flamingo_cycle_arms"
register_mjlab_task(
    task_id=ELBOW_FLAMINGO_CYCLE_ARMS_TASK_ID,
    env_cfg=flamingo_cycle_arms_env_cfg(),
    play_env_cfg=flamingo_cycle_arms_env_cfg(play=True),
    rl_cfg=flamingo_cycle_arms_rl_cfg,
    runner_cls=VelocityOnPolicyRunner,
)

# Keep the reference manipulation lesson on this package's existing task
# entrypoint, so the native MJLab CLI discovers it with the walking tasks.
from . import reach as _reach  # noqa: E402,F401
from . import grasp as _grasp  # noqa: E402,F401
from . import grasp_tracking as _grasp_tracking  # noqa: E402,F401
from . import place_tracking as _place_tracking  # noqa: E402,F401
from . import place_stages as _place_stages  # noqa: E402,F401
from . import foot_upright as _foot_upright  # noqa: E402,F401
