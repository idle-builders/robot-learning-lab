"""Berkeley Lite Arena playroom: native physics, learned skills, and composition."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=250)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--velocity", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--train-iterations", type=int, default=0)
parser.add_argument("--training-dir", type=Path)
parser.add_argument("--resume-checkpoint", type=Path)
parser.add_argument("--policy-checkpoint", type=Path)
parser.add_argument("--eval-baseline", action="store_true")
parser.add_argument("--collect-demos", type=int, default=0)
parser.add_argument("--collect-full-state", action="store_true")
parser.add_argument("--max-attempts", type=int, default=40)
parser.add_argument("--dataset-dir", type=Path)
parser.add_argument("--groot-host")
parser.add_argument("--groot-model-id")
parser.add_argument("--groot-port", type=int, default=5555)
parser.add_argument("--groot-execution-horizon", type=int, default=4)
parser.add_argument("--groot-applied-action-contract", action="store_true")
parser.add_argument("--groot-goal-state", action="store_true")
parser.add_argument("--groot-camera-warmup-renders", type=int, default=0)
parser.add_argument("--forward-prior", action="store_true")
parser.add_argument("--task", choices=("ball", "pad", "button", "combo"), default="ball")
parser.add_argument("--layout", choices=("train", "transition_train", "heldout"), default="train")
parser.add_argument("--instruction", default="Score the orange ball, then step on the red floor button.")
parser.add_argument("--ball-checkpoint", type=Path)
parser.add_argument("--pad-checkpoint", type=Path)
parser.add_argument("--button-checkpoint", type=Path)
parser.add_argument("--combo-second", choices=("pad", "button"), default="pad")
parser.add_argument("--combo-demo-dir", type=Path)
parser.add_argument("--cosmos-model", type=Path)
parser.add_argument("--hand-usd", type=Path)
parser.add_argument("--button-position", nargs=3, type=float)
parser.add_argument("--trace-env-index", type=int, default=0)
parser.add_argument("--control", choices=("joint", "velocity"), default="joint")
parser.add_argument("--video-dir", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output = args.output.resolve()
if args.video_dir:
    args.video_dir = args.video_dir.resolve()
if args.video_dir or args.task == "combo" or args.collect_demos or args.groot_host:
    args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np
import onnxruntime as ort
import torch
import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import BerkeleyHumanoidLiteEnvCfg
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.agents.rsl_rl_ppo_cfg import BerkeleyHumanoidLitePPORunnerCfg
from berkeley_velocity_action import BerkeleyVelocityActionCfg, BerkeleyVelocityLeftArmActionCfg
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_ARM_JOINTS, HUMANOID_LITE_LEG_JOINTS


if args.layout == "heldout":
    BALL_START = (0.65, -0.20, 0.055)
    GOAL = (1.15, -0.20, 0.0)
    PAD = (0.70, 0.30, 0.025)
elif args.layout == "transition_train":
    BALL_START = (0.60, -0.16, 0.055)
    GOAL = (1.10, -0.16, 0.0)
    PAD = (0.70, 0.26, 0.025)
else:
    BALL_START = (0.55, 0.0, 0.055)
    GOAL = (1.15, 0.0, 0.0)
    PAD = (0.70, -0.35, 0.025)
BUTTON = tuple(args.button_position) if args.button_position else (
    (0.55, 0.15, 0.55) if args.layout == "heldout" else (0.50, 0.25, 0.55)
)
LEG_BODY_NAMES = tuple(name.removesuffix("_joint") for name in HUMANOID_LITE_LEG_JOINTS)


def relative_to_robot(env, position):
    robot = env.scene["robot"].data
    return quat_apply_inverse(robot.root_quat_w, position - robot.root_pos_w)[:, :2]


def ball_relative(env):
    return relative_to_robot(env, env.scene["ball"].data.root_pos_w)


def goal_relative(env):
    center = torch.tensor(GOAL, device=env.device) + env.scene.env_origins
    return relative_to_robot(env, center)


def ball_velocity_relative(env):
    robot = env.scene["robot"].data
    return quat_apply_inverse(robot.root_quat_w, env.scene["ball"].data.root_lin_vel_w)[:, :2]


def robot_velocity_xy(env):
    return env.scene["robot"].data.root_lin_vel_b[:, :2]


def robot_yaw_rate(env):
    return env.scene["robot"].data.root_ang_vel_b[:, 2:3]


def projected_gravity(env):
    return env.scene["robot"].data.projected_gravity_b


def last_command(env):
    return env.action_manager.action


@configclass
class BallObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        ball: ObservationTermCfg = ObservationTermCfg(func=ball_relative)
        goal: ObservationTermCfg = ObservationTermCfg(func=goal_relative)
        ball_velocity: ObservationTermCfg = ObservationTermCfg(func=ball_velocity_relative)
        robot_velocity: ObservationTermCfg = ObservationTermCfg(func=robot_velocity_xy)
        yaw_rate: ObservationTermCfg = ObservationTermCfg(func=robot_yaw_rate)
        gravity: ObservationTermCfg = ObservationTermCfg(func=projected_gravity)
        previous_command: ObservationTermCfg = ObservationTermCfg(func=last_command)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


def pad_relative(env):
    center = torch.tensor(PAD, device=env.device) + env.scene.env_origins
    return relative_to_robot(env, center)


@configclass
class PadObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        pad: ObservationTermCfg = ObservationTermCfg(func=pad_relative)
        robot_velocity: ObservationTermCfg = ObservationTermCfg(func=robot_velocity_xy)
        yaw_rate: ObservationTermCfg = ObservationTermCfg(func=robot_yaw_rate)
        gravity: ObservationTermCfg = ObservationTermCfg(func=projected_gravity)
        previous_command: ObservationTermCfg = ObservationTermCfg(func=last_command)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


def pad_force(env):
    forces = env.scene["pad_contact"].data.force_matrix_w
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))


def foot_ball_forces(env):
    left = env.scene["left_ball_contact"].data.force_matrix_w_history
    right = env.scene["right_ball_contact"].data.force_matrix_w_history
    return torch.stack((
        torch.linalg.vector_norm(left, dim=-1).amax(dim=(1, 2, 3)),
        torch.linalg.vector_norm(right, dim=-1).amax(dim=(1, 2, 3)),
    ), dim=1)


def ball_leg_forces(env):
    forces = env.scene["ball_leg_contact"].data.force_matrix_w_history
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))


def ball_total_force(env):
    forces = env.scene["ball_total_contact"].data.net_forces_w_history
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))


def pad_pressed(env):
    return pad_force(env) > 3.0


def hand_button_force(env):
    forces = env.scene["hand_button_contact"].data.force_matrix_w
    return torch.linalg.vector_norm(forces, dim=-1).amax(dim=(1, 2))


def button_pressed(env):
    return hand_button_force(env) > 3.0


def button_relative(env):
    center = torch.tensor(BUTTON, device=env.device) + env.scene.env_origins
    robot = env.scene["robot"].data
    return quat_apply_inverse(robot.root_quat_w, center - robot.root_pos_w)


def hand_button_relative(env):
    robot = env.scene["robot"]
    hand_ids, _ = robot.find_bodies("arm_left_hand_link")
    center = torch.tensor(BUTTON, device=env.device) + env.scene.env_origins
    return center - robot.data.body_pos_w[:, hand_ids[0]]


def left_arm_positions(env):
    robot = env.scene["robot"]
    arm_ids, _ = robot.find_joints(HUMANOID_LITE_ARM_JOINTS[:5], preserve_order=True)
    return robot.data.joint_pos[:, arm_ids]


def hand_button_proximity(env):
    return torch.exp(-5.0 * torch.linalg.vector_norm(hand_button_relative(env), dim=1))


@configclass
class ButtonObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        button: ObservationTermCfg = ObservationTermCfg(func=button_relative)
        hand_to_button: ObservationTermCfg = ObservationTermCfg(func=hand_button_relative)
        robot_velocity: ObservationTermCfg = ObservationTermCfg(func=robot_velocity_xy)
        yaw_rate: ObservationTermCfg = ObservationTermCfg(func=robot_yaw_rate)
        gravity: ObservationTermCfg = ObservationTermCfg(func=projected_gravity)
        arm_positions: ObservationTermCfg = ObservationTermCfg(func=left_arm_positions)
        previous_command: ObservationTermCfg = ObservationTermCfg(func=last_command)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: PolicyCfg = PolicyCfg()


@configclass
class ButtonRewardsCfg:
    approach: RewardTermCfg = RewardTermCfg(func=hand_button_proximity, weight=0.5)
    pressed: RewardTermCfg = RewardTermCfg(func=lambda env: button_pressed(env).float(), weight=500.0)


def pad_proximity(env):
    return torch.exp(-3.0 * torch.linalg.vector_norm(pad_relative(env), dim=1))


@configclass
class PadRewardsCfg:
    approach: RewardTermCfg = RewardTermCfg(func=pad_proximity, weight=2.0)
    pressed: RewardTermCfg = RewardTermCfg(func=lambda env: pad_pressed(env).float(), weight=100.0)


def ball_goal_progress(env):
    ball = env.scene["ball"].data.root_pos_w[:, :2]
    goal = torch.tensor(GOAL[:2], device=env.device) + env.scene.env_origins[:, :2]
    distance = torch.linalg.vector_norm(goal - ball, dim=1)
    return torch.clamp(0.6 - distance, min=-0.2, max=0.6)


def robot_ball_proximity(env):
    distance = torch.linalg.vector_norm(ball_relative(env), dim=1)
    return torch.exp(-4.0 * distance)


def robot_toward_ball(env):
    robot = env.scene["robot"].data
    displacement = env.scene["ball"].data.root_pos_w[:, :2] - robot.root_pos_w[:, :2]
    direction = displacement / torch.clamp(torch.linalg.vector_norm(displacement, dim=1, keepdim=True), min=0.01)
    return torch.clamp(torch.sum(robot.root_lin_vel_w[:, :2] * direction, dim=1), min=-0.5, max=0.5)


def ball_goal_velocity(env):
    ball = env.scene["ball"].data
    goal = torch.tensor(GOAL[:2], device=env.device) + env.scene.env_origins[:, :2]
    to_goal = goal - ball.root_pos_w[:, :2]
    direction = to_goal / torch.clamp(torch.linalg.vector_norm(to_goal, dim=1, keepdim=True), min=0.01)
    return torch.clamp(torch.sum(ball.root_lin_vel_w[:, :2] * direction, dim=1), min=0.0, max=1.0)


@configclass
class BallRewardsCfg:
    ball_progress: RewardTermCfg = RewardTermCfg(func=ball_goal_progress, weight=6.0)
    approach_ball: RewardTermCfg = RewardTermCfg(
        func=robot_ball_proximity, weight=2.0 if args.layout == "transition_train" else 0.1
    )
    toward_ball: RewardTermCfg = RewardTermCfg(
        func=robot_toward_ball, weight=1.0 if args.layout == "transition_train" else 0.0
    )
    ball_toward_goal: RewardTermCfg = RewardTermCfg(func=ball_goal_velocity, weight=2.0)
    success: RewardTermCfg = RewardTermCfg(func=lambda env: ball_in_goal(env).float(), weight=100.0)


@configclass
class VelocityActionsCfg:
    walk: BerkeleyVelocityActionCfg = BerkeleyVelocityActionCfg()


@configclass
class ButtonActionsCfg:
    walk_and_reach: BerkeleyVelocityLeftArmActionCfg = BerkeleyVelocityLeftArmActionCfg()


class PlayroomObjects(Asset):
    def __init__(self):
        super().__init__(name="playroom_objects")

    def get_object_cfg(self):
        ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ball",
            spawn=sim_utils.SphereCfg(
                radius=0.055,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.35, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_START),
        )
        goal = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/goal",
            spawn=sim_utils.CylinderCfg(
                radius=0.13,
                height=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.3)),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=GOAL),
        )
        pad = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/pad",
            spawn=sim_utils.CuboidCfg(
                size=(0.25, 0.25, 0.05),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.08, 0.12)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=PAD),
        )
        contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/pad",
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/robot/leg_left_ankle_roll",
                "{ENV_REGEX_NS}/robot/leg_right_ankle_roll",
            ],
            update_period=0.0,
            history_length=1,
        )
        left_ball_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/leg_left_ankle_roll",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ball"],
            update_period=0.0, history_length=8,
        )
        right_ball_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/leg_right_ankle_roll",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/ball"],
            update_period=0.0, history_length=8,
        )
        ball_leg_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ball",
            filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/robot/{name}" for name in LEG_BODY_NAMES],
            update_period=0.0, history_length=8,
        )
        ball_total_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/ball", update_period=0.0, history_length=8,
        )
        objects = {
            "ball": ball, "goal": goal, "pad": pad, "pad_contact": contact,
            "left_ball_contact": left_ball_contact, "right_ball_contact": right_ball_contact,
            "ball_leg_contact": ball_leg_contact,
            "ball_total_contact": ball_total_contact,
        }
        if args.task == "button" or (args.task == "combo" and args.combo_second == "button"):
            objects["button"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/button",
                spawn=sim_utils.CuboidCfg(
                    size=(0.08, 0.10, 0.10),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    activate_contact_sensors=True,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.98, 0.75, 0.08)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=BUTTON),
            )
            objects["hand_button_contact"] = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/robot/arm_left_hand_link",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/button"],
                update_period=0.0, history_length=1,
            )
        return objects


class BerkeleyEmbodiment(EmbodimentBase):
    name = "berkeley_humanoid_lite"

    def __init__(self):
        super().__init__()
        native = BerkeleyHumanoidLiteEnvCfg()
        native.scene.num_envs = args.num_envs
        native.scene.env_spacing = 3.0
        native.commands.base_velocity.resampling_time_range = (1000.0, 1000.0)
        native.commands.base_velocity.heading_command = False
        native.commands.base_velocity.rel_standing_envs = 0.0
        native.commands.base_velocity.ranges.lin_vel_x = (args.velocity, args.velocity)
        native.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        native.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        native.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # The published velocity task resets yaw anywhere in [-pi, pi]. A room
        # skill needs to start facing its objects, with small pose variation.
        native.events.reset_base.params["pose_range"] = (
            {"x": (0.22, 0.48), "y": (0.14, 0.36), "yaw": (-0.30, 0.30)}
            if args.layout == "transition_train" else
            {"x": (-0.15, 0.15), "y": (-0.10, 0.10), "yaw": (-0.15, 0.15)}
        )
        self.native = native
        self.scene_config = native.scene
        if args.task == "button" or (args.task == "combo" and args.combo_second == "button"):
            if not args.hand_usd or not args.hand_usd.exists():
                raise ValueError("Hand-button task needs --hand-usd with palm colliders")
            self.scene_config.robot.spawn.usd_path = str(args.hand_usd.resolve())
        self.action_config = (
            native.actions if args.control == "joint" else
            ButtonActionsCfg() if args.task == "button" or (args.task == "combo" and args.combo_second == "button")
            else VelocityActionsCfg()
        )
        learned_ball_mode = bool(args.train_iterations or args.policy_checkpoint or args.eval_baseline or args.task in ("button", "combo"))
        self.observation_config = (
            (ButtonObservationsCfg() if args.task == "button" else
             PadObservationsCfg() if args.task == "pad" else BallObservationsCfg())
            if learned_ball_mode else native.observations
        )
        self.event_config = native.events
        self.reward_config = (
            (ButtonRewardsCfg() if args.task == "button" else
             PadRewardsCfg() if args.task == "pad" else BallRewardsCfg())
            if learned_ball_mode else native.rewards
        )
        self.command_config = native.commands
        self.termination_cfg = native.terminations

    def modify_env_cfg(self, env_cfg):
        env_cfg.sim = self.native.sim
        env_cfg.decimation = self.native.decimation
        return env_cfg


def ball_in_goal(env):
    ball = env.scene["ball"].data.root_pos_w
    center = torch.tensor(GOAL, device=ball.device) + env.scene.env_origins
    return torch.linalg.vector_norm(ball[:, :2] - center[:, :2], dim=1) < 0.13


@configclass
class BallTerminationsCfg:
    success: TerminationTermCfg = TerminationTermCfg(func=ball_in_goal)


@configclass
class BallEventsCfg:
    reset_ball: EventTermCfg = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {"x": (-0.06, 0.06), "y": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )


class BallGoalTask(TaskBase):
    def __init__(self):
        super().__init__(episode_length_s=16.0)

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return BallTerminationsCfg()

    def get_events_cfg(self):
        return BallEventsCfg()

    def get_prompt(self):
        return "Nudge the orange ball onto the green goal."

    def get_mimic_env_cfg(self, embodiment_name):
        return None

    def get_metrics(self):
        return [SuccessRateMetric()]

    def get_viewer_cfg(self):
        return ViewerCfg(eye=(1.8, -1.8, 1.4), lookat=(0.6, 0.0, 0.4), origin_type="env", env_index=0)


@configclass
class PadTerminationsCfg:
    success: TerminationTermCfg = TerminationTermCfg(func=pad_pressed)


class PadPressTask(BallGoalTask):
    def get_termination_cfg(self):
        return PadTerminationsCfg()

    def get_events_cfg(self):
        return None

    def get_prompt(self):
        return "Step on the red floor button."


@configclass
class ButtonTerminationsCfg:
    success: TerminationTermCfg = TerminationTermCfg(func=button_pressed)


class HandButtonTask(BallGoalTask):
    def get_termination_cfg(self):
        return ButtonTerminationsCfg()

    def get_events_cfg(self):
        return None

    def get_prompt(self):
        return "Press the yellow toy button with the left hand."


class ComboTask(BallGoalTask):
    def __init__(self):
        TaskBase.__init__(self, episode_length_s=32.0)

    def get_termination_cfg(self):
        return None

    def get_prompt(self):
        return args.instruction

    def get_metrics(self):
        return []


def load_skill_actor(checkpoint, width, num_actions, device):
    from rsl_rl.modules import ActorCritic

    dummy = {"policy": torch.zeros((1, width), device=device), "critic": torch.zeros((1, width), device=device)}
    actor = ActorCritic(
        obs=dummy, obs_groups={"policy": ["policy"], "critic": ["critic"]},
        num_actions=num_actions, actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128], activation="elu",
    ).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    actor.load_state_dict(saved["model_state_dict"])
    return actor.eval()


def combo_observation(env, skill):
    shared = [robot_velocity_xy(env), robot_yaw_rate(env), projected_gravity(env)]
    if skill == "hand_press":
        return torch.cat([
            button_relative(env), hand_button_relative(env), *shared,
            left_arm_positions(env), last_command(env),
        ], dim=1)
    shared.append(last_command(env)[:, :3])
    if skill == "ball_nudge":
        pieces = [ball_relative(env), goal_relative(env), ball_velocity_relative(env), *shared]
    else:
        pieces = [pad_relative(env), *shared]
    return torch.cat(pieces, dim=1)


def run_combo(arena, name, cfg):
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    second_skill = "hand_press" if args.combo_second == "button" else "pad_press"
    use_groot_button = second_skill == "hand_press" and bool(args.groot_host)
    second_checkpoint = args.button_checkpoint if args.combo_second == "button" else args.pad_checkpoint
    if not all((args.ball_checkpoint, args.cosmos_model)) or (not use_groot_button and not second_checkpoint):
        raise ValueError("Combo needs --ball-checkpoint, the second skill checkpoint, and --cosmos-model")
    if use_groot_button and not (args.groot_goal_state and args.groot_applied_action_contract):
        raise ValueError("GR00T combo button needs --groot-goal-state and --groot-applied-action-contract")
    env = gym.make(name, cfg=cfg, render_mode="rgb_array")
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(args.video_dir), step_trigger=lambda step: step == 0,
            video_length=args.steps, disable_logger=True, name_prefix="berkeley-combo",
        )
    core = env.unwrapped
    obs, _ = env.reset()
    actors = {"ball_nudge": load_skill_actor(args.ball_checkpoint, 15, 3, core.device)}
    if not use_groot_button:
        actors[second_skill] = load_skill_actor(
            second_checkpoint, 25 if second_skill == "hand_press" else 11,
            8 if second_skill == "hand_press" else 3, core.device,
        )
    groot_client = None
    groot_calls = []
    groot_chunk = np.empty((0, 8), dtype=np.float32)
    hand_steps = 0
    hand_demo_frames, hand_demo_states, hand_demo_actions = [], [], []
    action_lower = np.array([-0.5, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    action_upper = np.array([0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    if use_groot_button:
        import os
        from gr00t.policy.server_client import PolicyClient

        token = os.environ.get("GROOT_API_TOKEN") or Path("/workspace/groot-api-token").read_text().strip()
        groot_client = PolicyClient(
            host=args.groot_host, port=args.groot_port, timeout_ms=120000,
            api_token=token, strict=False,
        )
        if not groot_client.ping():
            raise RuntimeError("GR00T policy server did not respond")
        modalities = groot_client.get_modality_config()
        if (modalities["state"].modality_keys !=
                ["button_relative", "hand_to_button", "body_motion", "left_arm", "previous_action"]):
            raise ValueError("GR00T combo button state contract differs")
        groot_client.reset()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.cosmos_model, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.cosmos_model)
    selections = []
    trace = []
    ball_done = False
    second_done = False
    ball_contact_during_skill = False
    ball_outside_seen_during_skill = False

    def select(step):
        valid = [skill for skill, done in (("ball_nudge", ball_done), (second_skill, second_done)) if not done]
        if not valid:
            valid = ["stop"]
        second_state = "hand_button_pressed" if second_skill == "hand_press" else "pad_pressed"
        state = f"ball_in_goal={str(ball_done).lower()}; {second_state}={str(second_done).lower()}"
        second_description = (
            "hand_press = press the yellow toy button with the left hand"
            if second_skill == "hand_press" else "pad_press = step on the red floor button"
        )
        prompt = (
            "You select one robot skill for the next step from the instruction, camera image, "
            "and authoritative measured completion state. Follow the instruction order. "
            "Never repeat a completed subgoal. Skills: ball_nudge = nudge the orange ball "
            f"into the green goal; {second_description}; stop = all "
            "requested subgoals complete. Do not emit joint commands.\n"
            f"Instruction: {args.instruction}\nMeasured state: {state}\n"
            f"Valid next IDs: {', '.join(valid)}. Answer with exactly one valid ID."
        )
        frame = Image.fromarray(env.render()).convert("RGB")
        frame.thumbnail((640, 480))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            frame_path = Path(handle.name)
        try:
            frame.save(frame_path, quality=85)
            digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You reason about physical robot scenes."}]},
                {"role": "user", "content": [
                    {"type": "image", "image": str(frame_path)},
                    {"type": "text", "text": prompt},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=24, do_sample=False)
            raw = processor.batch_decode(
                [generated[0, inputs.input_ids.shape[-1]:]], skip_special_tokens=True
            )[0].strip()
            constrained = None
            if raw not in valid:
                # Keep the failed free-form answer as evidence. On a second
                # pass, use Transformers' prefix constraint to restrict the
                # *model's* output tokens to measured-state-valid skill IDs.
                sequences = [processor.tokenizer.encode(skill, add_special_tokens=False) for skill in valid]
                prefix_length = inputs.input_ids.shape[-1]
                eos = processor.tokenizer.eos_token_id

                def allowed_tokens(_batch, token_ids):
                    suffix = token_ids[prefix_length:].tolist()
                    allowed = []
                    for sequence in sequences:
                        if sequence[:len(suffix)] == suffix:
                            allowed.append(sequence[len(suffix)] if len(suffix) < len(sequence) else eos)
                    return sorted(set(allowed)) or [eos]

                with torch.inference_mode():
                    restricted = model.generate(
                        **inputs, max_new_tokens=max(map(len, sequences)) + 1,
                        do_sample=False, prefix_allowed_tokens_fn=allowed_tokens,
                    )
                constrained = processor.batch_decode(
                    [restricted[0, prefix_length:]], skip_special_tokens=True
                )[0].strip()
        finally:
            frame_path.unlink(missing_ok=True)
        chosen = raw if raw in valid else (constrained if constrained in valid else None)
        selection = {
            "step": step, "prompt": prompt, "measured_state": state,
            "valid_next_ids": valid, "raw_prediction": raw,
            "constrained_prediction": constrained, "chosen_skill": chosen,
            "image_sha256": digest,
        }
        selections.append(selection)
        print(json.dumps({k: v for k, v in selection.items() if k != "prompt"}), flush=True)
        return chosen

    skill = select(0)
    failure = None
    try:
        for step in range(args.steps):
            if skill is None:
                failure = "invalid_selector_output"
                break
            if skill == "stop":
                break
            policy_obs = combo_observation(core, skill)
            if use_groot_button and skill == "hand_press":
                if hand_steps % args.groot_execution_horizon == 0:
                    image = np.asarray(
                        Image.fromarray(env.render()).convert("RGB").resize((256, 256)),
                        dtype=np.uint8,
                    )
                    state = policy_obs[0].detach().cpu().numpy().astype(np.float32)
                    state[17:25] = np.clip(state[17:25], action_lower, action_upper)
                    observation = {
                        "video": {"external_view": image[None, None]},
                        "state": {
                            "button_relative": state[None, None, :3],
                            "hand_to_button": state[None, None, 3:6],
                            "body_motion": state[None, None, 6:12],
                            "left_arm": state[None, None, 12:17],
                            "previous_action": state[None, None, 17:25],
                        },
                        "language": {"annotation.human.task_description": [["Press the yellow toy button with the left hand."]]},
                    }
                    prediction, _ = groot_client.get_action(observation)
                    groot_chunk = np.concatenate((
                        np.asarray(prediction["walk"][0], dtype=np.float32),
                        np.asarray(prediction["left_arm"][0], dtype=np.float32),
                    ), axis=-1)
                    groot_calls.append({
                        "step": step, "input_state": state.tolist(),
                        "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                        "action_chunk": groot_chunk.tolist(),
                    })
                command = np.clip(groot_chunk[hand_steps % args.groot_execution_horizon], action_lower, action_upper)
                action = torch.from_numpy(command[None]).to(core.device)
                hand_steps += 1
            else:
                with torch.inference_mode():
                    action = actors[skill].act_inference({"policy": policy_obs})
                if args.combo_demo_dir and skill == "hand_press":
                    hand_demo_frames.append(np.asarray(
                        Image.fromarray(env.render()).convert("RGB").resize((256, 256)),
                        dtype=np.uint8,
                    ))
                    hand_demo_states.append(policy_obs[0].detach().cpu().numpy().astype(np.float32))
                    hand_demo_actions.append(action[0].detach().cpu().numpy().astype(np.float32))
            if args.combo_second == "button" and skill == "ball_nudge":
                action = torch.nn.functional.pad(action, (0, 5))
            if args.combo_second == "button":
                core.action_manager.get_term("walk_and_reach").arm_override_enabled = skill == "hand_press"
            obs, reward, terminated, truncated, _ = env.step(action)
            current_ball = bool(ball_in_goal(core)[0].item())
            current_pad = bool(pad_pressed(core)[0].item())
            current_button = bool(button_pressed(core)[0].item()) if args.combo_second == "button" else False
            contact = foot_ball_forces(core)[0].detach().cpu().tolist()
            leg_contact = ball_leg_forces(core)[0].detach().cpu().tolist()
            reciprocal_contact = max(leg_contact)
            # A contact made while pursuing a different subgoal is evidence,
            # but it does not satisfy a later "then" instruction.
            if skill == "ball_nudge":
                ball_outside_seen_during_skill = ball_outside_seen_during_skill or not current_ball
                ball_contact_during_skill = ball_contact_during_skill or max(*contact, reciprocal_contact) > 1.0
                ball_done = ball_done or (
                    current_ball and ball_outside_seen_during_skill and ball_contact_during_skill
                )
            if skill == second_skill:
                second_done = second_done or (current_button if second_skill == "hand_press" else current_pad)
            robot_xyz = core.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            ball_xyz = core.scene["ball"].data.root_pos_w[0].detach().cpu().tolist()
            robot = core.scene["robot"]
            body_distances = torch.linalg.vector_norm(
                robot.data.body_pos_w[0] - core.scene["ball"].data.root_pos_w[0], dim=1
            )
            nearest_body_index = int(torch.argmin(body_distances).item())
            entry = {
                "step": step, "skill": skill, "action": action[0].detach().cpu().tolist(),
                "robot_xyz": robot_xyz, "ball_xyz": ball_xyz,
                "ball_in_goal": current_ball, "pad_pressed": current_pad,
                "hand_button_pressed": current_button,
                "ball_contact_during_skill": ball_contact_during_skill,
                "ball_outside_seen_during_skill": ball_outside_seen_during_skill,
                "ball_done_latched": ball_done,
                "pad_done_latched": second_done if second_skill == "pad_press" else False,
                "hand_button_done_latched": second_done if second_skill == "hand_press" else False,
                "pad_force_n": float(pad_force(core)[0].item()),
                "hand_button_force_n": float(hand_button_force(core)[0].item()) if args.combo_second == "button" else None,
                "hand_xyz": (
                    core.scene["robot"].data.body_pos_w[
                        0, core.scene["robot"].find_bodies("arm_left_hand_link")[0][0]
                    ].detach().cpu().tolist() if args.combo_second == "button" else None
                ),
                "foot_ball_force_n": contact,
                "ball_leg_force_n": leg_contact,
                "ball_total_force_n": float(ball_total_force(core)[0].item()),
                "nearest_body_to_ball": robot.body_names[nearest_body_index],
                "nearest_body_center_distance_m": float(body_distances[nearest_body_index].item()),
                "reward": float(reward[0].item()),
                "terminated": bool(terminated[0].item()), "truncated": bool(truncated[0].item()),
            }
            trace.append(entry)
            if entry["terminated"] or entry["truncated"]:
                failure = "fall_or_timeout"
                break
            if (skill == "ball_nudge" and ball_done) or (skill == second_skill and second_done):
                skill = select(step + 1)
    finally:
        result = {
            "embodiment": "Berkeley Humanoid Lite V1", "task": arena.task.get_prompt(),
            "selector": "Cosmos-Reason2-2B zero-shot", "seed": args.seed,
            "hand_controller": "GR00T N1.7" if use_groot_button else "RSL-RL PPO",
            "groot_model_id": args.groot_model_id if use_groot_button else None,
            "groot_model_calls": groot_calls,
            "layout": args.layout, "ball_start_local_xyz": BALL_START,
            "goal_local_xyz": GOAL, "pad_local_xyz": PAD, "button_local_xyz": BUTTON,
            "num_envs": args.num_envs, "evaluated_env": 0,
            "env0_origin_xyz": core.scene.env_origins[0].detach().cpu().tolist(),
            "ball_checkpoint": str(args.ball_checkpoint),
            "pad_checkpoint": str(args.pad_checkpoint) if args.pad_checkpoint else None,
            "button_checkpoint": str(args.button_checkpoint) if args.button_checkpoint else None,
            "cosmos_model": str(args.cosmos_model),
            "ball_done_latched": ball_done,
            "pad_done_latched": second_done if second_skill == "pad_press" else False,
            "hand_button_done_latched": second_done if second_skill == "hand_press" else False,
            "ball_contact_during_skill": ball_contact_during_skill,
            "ball_outside_seen_during_skill": ball_outside_seen_during_skill,
            "last_skill": skill, "failure": failure,
            "success": ball_done and second_done and skill == "stop" and failure is None,
            "selections": selections, "trace_env0": trace,
            "video_dir": str(args.video_dir) if args.video_dir else None,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "trace_env0"}, indent=2), flush=True)
        if args.combo_demo_dir and result["success"] and hand_demo_actions:
            import imageio.v2 as imageio

            output_dir = args.combo_demo_dir.resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            arrays_path = output_dir / "episode_000000.npz"
            video_path = output_dir / "episode_000000.mp4"
            np.savez_compressed(
                arrays_path, state=np.stack(hand_demo_states), action=np.stack(hand_demo_actions),
            )
            imageio.mimwrite(video_path, hand_demo_frames, fps=25)
            attempt = {
                "attempt": 0, "success": True, "length": len(hand_demo_actions),
                "episode": "episode_000000",
                "state_action_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
                "video_sha256": hashlib.sha256(video_path.read_bytes()).hexdigest(),
            }
            (output_dir / "manifest.json").write_text(json.dumps({
                "source": "successful PPO hand stage after Cosmos-selected ball-to-button transition",
                "robot": "Berkeley Humanoid Lite V1",
                "task": "Press the yellow toy button with the left hand.",
                "instruction": args.instruction, "layout": args.layout, "seed": args.seed,
                "fps": 25, "image_size": [256, 256], "state_dim": 25,
                "state": "full 25-D button observation at each post-ball hand step",
                "action": "3 walking commands and 5 left-arm targets",
                "checkpoint_sha256": hashlib.sha256(args.button_checkpoint.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "attempts": [attempt], "successful_episodes": 1,
            }, indent=2))
            print(json.dumps({"combo_demo_dir": str(output_dir), "frames": len(hand_demo_actions)}), flush=True)
        env.close()


def collect_button_demos(name, cfg):
    """Distill successful PPO rollouts into compact image/state/action episodes."""
    import imageio.v2 as imageio
    from PIL import Image
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    if args.task != "button" or args.num_envs != 1 or not args.policy_checkpoint or not args.dataset_dir:
        raise ValueError("Demo collection needs --task button, one env, --policy-checkpoint, --dataset-dir")
    output_dir = args.dataset_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gym_env = gym.make(name, cfg=cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(gym_env)
    core = env.unwrapped
    agent_cfg = BerkeleyHumanoidLitePPORunnerCfg()
    agent_cfg.device = args.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(str(args.policy_checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=args.device)
    checkpoint_hash = hashlib.sha256(args.policy_checkpoint.read_bytes()).hexdigest()
    attempts = []
    kept = 0
    try:
        for attempt_index in range(args.max_attempts):
            if kept >= args.collect_demos:
                break
            obs, _ = env.reset()
            robot = core.scene["robot"].data
            start = {
                "root_state": robot.root_state_w[0].detach().cpu().tolist(),
                "joint_pos": robot.joint_pos[0].detach().cpu().tolist(),
                "joint_vel": robot.joint_vel[0].detach().cpu().tolist(),
            }
            frames, states, actions = [], [], []
            success = False
            max_force = 0.0
            for _step in range(args.steps):
                frame = Image.fromarray(gym_env.render()).convert("RGB").resize((256, 256))
                frames.append(np.asarray(frame, dtype=np.uint8))
                state = obs["policy"][0] if args.collect_full_state else obs["policy"][0, 6:]
                states.append(state.detach().cpu().numpy().astype(np.float32))
                with torch.inference_mode():
                    action = policy(obs)
                actions.append(action[0].detach().cpu().numpy().astype(np.float32))
                obs, _reward, done, _info = env.step(action)
                success = bool(core.termination_manager.get_term("success")[0].item())
                max_force = max(max_force, float(hand_button_force(core)[0].item()))
                if bool(done[0]):
                    break
            attempt = {
                "attempt": attempt_index, "success": success, "length": len(actions),
                "max_hand_button_force_n": max_force, "start": start,
            }
            if success:
                episode_name = f"episode_{kept:06d}"
                np.savez_compressed(
                    output_dir / f"{episode_name}.npz",
                    state=np.stack(states), action=np.stack(actions),
                )
                imageio.mimwrite(output_dir / f"{episode_name}.mp4", frames, fps=25)
                attempt["episode"] = episode_name
                attempt["state_action_sha256"] = hashlib.sha256(
                    (output_dir / f"{episode_name}.npz").read_bytes()
                ).hexdigest()
                attempt["video_sha256"] = hashlib.sha256(
                    (output_dir / f"{episode_name}.mp4").read_bytes()
                ).hexdigest()
                kept += 1
            attempts.append(attempt)
            print(json.dumps({k: v for k, v in attempt.items() if k != "start"}), flush=True)
    finally:
        env.close()
    manifest = {
        "source": "successful RSL-RL policy rollouts, not human demonstrations",
        "robot": "Berkeley Humanoid Lite V1", "task": "Press the yellow toy button with the left hand.",
        "layout": args.layout, "seed": args.seed, "fps": 25, "image_size": [256, 256],
        "state": (
            "full 25-D button policy observation: target-relative vectors, robot motion, left arm, previous action"
            if args.collect_full_state else
            "button policy observation indices 6:25: robot velocity/yaw/gravity, left arm, previous action"
        ),
        "state_dim": 25 if args.collect_full_state else 19,
        "action": "3 walking commands and 5 left-arm targets", "checkpoint_sha256": checkpoint_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attempts": attempts, "successful_episodes": kept,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"dataset_dir": str(output_dir), "successful_episodes": kept, "attempts": len(attempts)}))


def evaluate_groot_button(name, cfg):
    """Run NVIDIA's PolicyClient against one fresh button episode in Arena."""
    import os
    import time
    from PIL import Image
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from gr00t.policy.server_client import PolicyClient

    if args.task != "button" or args.num_envs != 1 or args.control != "velocity":
        raise ValueError("GR00T button rollout needs --task button --num-envs 1 --control velocity")
    if args.groot_execution_horizon < 1:
        raise ValueError("GR00T execution horizon must be positive")
    token = os.environ.get("GROOT_API_TOKEN") or Path("/workspace/groot-api-token").read_text().strip()
    client = PolicyClient(
        host=args.groot_host, port=args.groot_port, timeout_ms=120000,
        api_token=token, strict=False,
    )
    if not client.ping():
        raise RuntimeError("GR00T policy server did not respond")
    modalities = client.get_modality_config()
    state_keys = (
        ["button_relative", "hand_to_button", "body_motion", "left_arm", "previous_action"]
        if args.groot_goal_state else ["body_motion", "left_arm", "previous_action"]
    )
    expected = {
        "video": ["external_view"],
        "state": state_keys,
        "action": ["walk", "left_arm"],
    }
    for modality, keys in expected.items():
        if modalities[modality].modality_keys != keys:
            raise ValueError(f"GR00T {modality} keys differ: {modalities[modality].modality_keys}")
    client.reset()
    gym_env = gym.make(name, cfg=cfg, render_mode="rgb_array")
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        gym_env = gym.wrappers.RecordVideo(
            gym_env, video_folder=str(args.video_dir), step_trigger=lambda step: step == 0,
            video_length=args.steps, disable_logger=True, name_prefix="berkeley-groot-button",
        )
    env = RslRlVecEnvWrapper(gym_env)
    obs, _ = env.reset()
    core = env.unwrapped
    warmup_frame_means = []
    for _ in range(args.groot_camera_warmup_renders):
        warmup_frame_means.append(float(np.asarray(gym_env.render()).mean()))
    origin = core.scene.env_origins[0].detach().cpu().tolist()
    chunk = np.empty((0, 8), dtype=np.float32)
    action_lower = np.array([-0.5, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    action_upper = np.array([0.5, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    model_calls, trace = [], []
    first_outcome = None
    try:
        for step in range(args.steps):
            if step % args.groot_execution_horizon == 0 or len(chunk) == 0:
                image = np.asarray(
                    Image.fromarray(gym_env.render()).convert("RGB").resize((256, 256)),
                    dtype=np.uint8,
                )
                state = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
                if args.groot_applied_action_contract:
                    state[17:25] = np.clip(state[17:25], action_lower, action_upper)
                if not args.groot_goal_state:
                    state = state[6:]
                state_fields = {
                    "body_motion": state[None, None, 6:12] if args.groot_goal_state else state[None, None, :6],
                    "left_arm": state[None, None, 12:17] if args.groot_goal_state else state[None, None, 6:11],
                    "previous_action": state[None, None, 17:25] if args.groot_goal_state else state[None, None, 11:19],
                }
                if args.groot_goal_state:
                    state_fields = {
                        "button_relative": state[None, None, :3],
                        "hand_to_button": state[None, None, 3:6],
                        **state_fields,
                    }
                observation = {
                    "video": {"external_view": image[None, None]},
                    "state": state_fields,
                    "language": {"annotation.human.task_description": [["Press the yellow toy button with the left hand."]]},
                }
                started = time.monotonic()
                prediction, _info = client.get_action(observation)
                latency = time.monotonic() - started
                walk = np.asarray(prediction["walk"][0], dtype=np.float32)
                arm = np.asarray(prediction["left_arm"][0], dtype=np.float32)
                chunk = np.concatenate((walk, arm), axis=-1)
                if chunk.ndim != 2 or chunk.shape[1] != 8 or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid GR00T action chunk: {chunk.shape}")
                model_calls.append({
                    "step": step, "latency_s": latency, "input_state": state.tolist(),
                    "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "image_mean": float(image.mean()),
                    "action_chunk": chunk.tolist(),
                })
            offset = step % args.groot_execution_horizon
            action_np = chunk[min(offset, len(chunk) - 1)]
            if args.groot_applied_action_contract:
                action_np = np.clip(action_np, action_lower, action_upper)
            action = torch.from_numpy(action_np[None]).to(core.device)
            obs, reward, done, _ = env.step(action)
            success = bool(core.termination_manager.get_term("success")[0].item())
            fall = bool(core.termination_manager.get_term("base_orientation")[0].item())
            timeout = bool(core.termination_manager.get_term("time_out")[0].item())
            trace.append({
                "step": step, "action": action_np.tolist(),
                "model_call_index": len(model_calls) - 1, "action_chunk_offset": offset,
                "robot_xyz": core.scene["robot"].data.root_pos_w[0].detach().cpu().tolist(),
                "ball_xyz": core.scene["ball"].data.root_pos_w[0].detach().cpu().tolist(),
                "success": success, "fall": fall, "timeout": timeout,
                "reward": float(reward[0].item()),
                "hand_button_force_n": float(hand_button_force(core)[0].item()),
                "hand_xyz": core.scene["robot"].data.body_pos_w[
                    0, core.scene["robot"].find_bodies("arm_left_hand_link")[0][0]
                ].detach().cpu().tolist(),
            })
            if bool(done[0]):
                first_outcome = {"step": step, "success": success, "fall": fall, "timeout": timeout}
                break
    finally:
        env.close()
    result = {
        "embodiment": "Berkeley Humanoid Lite V1",
        "task": "Press the yellow toy button with the left hand.",
        "controller": "NVIDIA GR00T N1.7 custom embodiment",
        "groot_model_id": args.groot_model_id,
        "seed": args.seed, "layout": args.layout, "first_outcomes": [first_outcome],
        "first_episode_success_count": int(bool(first_outcome and first_outcome["success"])),
        "execution_horizon": args.groot_execution_horizon,
        "camera_warmup_frame_means": warmup_frame_means,
        "action_contract": "applied" if args.groot_applied_action_contract else "raw",
        "observation_contract": "goal_state_25d" if args.groot_goal_state else "visual_19d",
        "model_calls": model_calls, "trace_env0": trace,
        "trace_origin_xyz": origin, "trace_env_index": 0,
        "goal_local_xyz": BUTTON,
        "video_dir": str(args.video_dir) if args.video_dir else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("model_calls", "trace_env0")}, indent=2))


def main():
    if (args.train_iterations or args.policy_checkpoint or args.eval_baseline or args.collect_demos or args.groot_host or args.task == "combo") and args.control != "velocity":
        raise ValueError("Learned skills need --control velocity")
    arena = IsaacLabArenaEnvironment(
        name=f"Berkeley-Lite-Playroom-{args.task}-v0",
        embodiment=BerkeleyEmbodiment(),
        scene=Scene(assets=[PlayroomObjects()]),
        task={"ball": BallGoalTask, "pad": PadPressTask, "button": HandButtonTask, "combo": ComboTask}[args.task](),
    )
    builder = ArenaEnvBuilder(
        arena,
        SimpleNamespace(device=args.device, num_envs=args.num_envs, disable_fabric=False, mimic=False),
    )
    name, cfg = builder.build_registered()
    cfg.seed = args.seed
    if args.task == "combo":
        cfg.scene.lazy_sensor_update = False
    if args.trace_env_index >= args.num_envs:
        raise ValueError("--trace-env-index must be less than --num-envs")
    cfg.viewer.env_index = args.trace_env_index
    if args.task == "combo":
        run_combo(arena, name, cfg)
        return
    if args.collect_demos:
        collect_button_demos(name, cfg)
        return
    if args.groot_host:
        evaluate_groot_button(name, cfg)
        return
    if args.train_iterations:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = BerkeleyHumanoidLitePPORunnerCfg()
        agent_cfg.seed = args.seed
        agent_cfg.device = args.device
        agent_cfg.max_iterations = args.train_iterations
        agent_cfg.save_interval = max(1, min(20, args.train_iterations))
        if args.forward_prior:
            # Start PPO near the published walker's already-successful forward
            # command, then let PPO learn navigation corrections from reward.
            agent_cfg.policy.init_noise_std = 0.15
        output_dir = (args.training_dir or args.output.parent / "ball-training").resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        env = RslRlVecEnvWrapper(gym.make(name, cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(output_dir), device=args.device)
        if args.resume_checkpoint:
            runner.load(str(args.resume_checkpoint), load_optimizer=False)
        elif args.forward_prior:
            actor_head = runner.alg.policy.actor[-1]
            actor_head.weight.data *= 0.1
            actor_head.bias.data.zero_()
            actor_head.bias.data[0] = args.velocity
            if args.task == "button":
                runner.alg.policy.std.data[3:] = 0.35
        runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=True)
        env.close()
        print(json.dumps({
            "training_dir": str(output_dir), "iterations": args.train_iterations,
            "num_envs": args.num_envs, "layout": args.layout,
            "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
        }))
        return
    if args.policy_checkpoint or args.eval_baseline:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        print("evaluation setup: creating gym environment", flush=True)
        gym_env = gym.make(name, cfg=cfg, render_mode="rgb_array" if args.video_dir else None)
        print("evaluation setup: environment created", flush=True)
        if args.video_dir:
            args.video_dir.mkdir(parents=True, exist_ok=True)
            gym_env = gym.wrappers.RecordVideo(
                gym_env, video_folder=str(args.video_dir), step_trigger=lambda step: step == 0,
                video_length=args.steps, disable_logger=True, name_prefix="berkeley-ball-policy",
            )
        env = RslRlVecEnvWrapper(gym_env)
        print("evaluation setup: vector wrapper created", flush=True)
        agent_cfg = BerkeleyHumanoidLitePPORunnerCfg()
        agent_cfg.device = args.device
        if args.policy_checkpoint:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
            print("evaluation setup: runner created", flush=True)
            runner.load(str(args.policy_checkpoint), load_optimizer=False)
            print("evaluation setup: checkpoint loaded", flush=True)
            policy = runner.get_inference_policy(device=args.device)
            print("evaluation setup: inference policy ready", flush=True)
        obs, _ = env.reset()
        print("evaluation setup: environment reset", flush=True)
        core = env.unwrapped
        trace_index = args.trace_env_index
        first_outcomes = [None] * args.num_envs
        trace = []
        try:
            for step in range(args.steps):
                if step % 50 == 0:
                    print(f"evaluation step {step}/{args.steps}", flush=True)
                if args.policy_checkpoint:
                    with torch.inference_mode():
                        action = policy(obs)
                else:
                    action = torch.zeros((args.num_envs, 8 if args.task == "button" else 3), device=core.device)
                    action[:, 0] = args.velocity
                obs, reward, done, _ = env.step(action)
                success = core.termination_manager.get_term("success").detach().cpu().tolist()
                pressure = pad_force(core).detach().cpu().tolist()
                ball_contact = foot_ball_forces(core).detach().cpu().tolist()
                fall = core.termination_manager.get_term("base_orientation").detach().cpu().tolist()
                timeout = core.termination_manager.get_term("time_out").detach().cpu().tolist()
                robot = core.scene["robot"].data.root_pos_w.detach().cpu().tolist()
                ball = core.scene["ball"].data.root_pos_w.detach().cpu().tolist()
                trace.append({
                    "step": step, "action": action[trace_index].detach().cpu().tolist(),
                    "robot_xyz": robot[trace_index], "ball_xyz": ball[trace_index],
                    "success": bool(success[trace_index]), "fall": bool(fall[trace_index]),
                    "timeout": bool(timeout[trace_index]), "reward": float(reward[trace_index].detach().cpu()),
                    "pad_force_n": pressure[trace_index],
                    "ball_foot_force_n": ball_contact[trace_index],
                    "hand_button_force_n": float(hand_button_force(core)[trace_index].item()) if args.task == "button" else None,
                    "hand_xyz": (
                        core.scene["robot"].data.body_pos_w[:, core.scene["robot"].find_bodies("arm_left_hand_link")[0][0]][trace_index]
                        .detach().cpu().tolist() if args.task == "button" else None
                    ),
                })
                for index in range(args.num_envs):
                    if first_outcomes[index] is None and bool(done[index]):
                        first_outcomes[index] = {
                            "step": step, "success": bool(success[index]),
                            "fall": bool(fall[index]), "timeout": bool(timeout[index]),
                        }
                if all(outcome is not None for outcome in first_outcomes):
                    break
        finally:
            pass
        result = {
            "embodiment": "Berkeley Humanoid Lite V1", "task": arena.task.get_prompt(),
            "checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else "constant-forward-baseline",
            "num_envs": args.num_envs,
            "seed": args.seed, "first_outcomes": first_outcomes,
            "layout": args.layout,
            "first_episode_success_count": sum(bool(x and x["success"]) for x in first_outcomes),
            "trace_env_index": trace_index,
            "trace_origin_xyz": core.scene.env_origins[trace_index].detach().cpu().tolist(),
            "env0_origin_xyz": core.scene.env_origins[trace_index].detach().cpu().tolist(),
            "goal_local_xyz": GOAL if args.task == "ball" else BUTTON if args.task == "button" else PAD,
            "video_dir": str(args.video_dir) if args.video_dir else None,
            "trace_env0": trace,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "trace_env0"}, indent=2))
        env.close()
        return
    env = gym.make(name, cfg=cfg, render_mode="rgb_array" if args.video_dir else None)
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(args.video_dir),
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            disable_logger=True,
            name_prefix="berkeley-ball-baseline",
        )
    core = env.unwrapped
    checkpoint = Path(__file__).resolve().parents[2] / ".cache/upstream/berkeley-humanoid-lite/checkpoints/policy_humanoid.onnx"
    if not checkpoint.exists():
        checkpoint = Path.cwd() / "checkpoints/policy_humanoid.onnx"
    if args.control == "joint":
        policy = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        input_name = policy.get_inputs()[0].name
    obs, _ = env.reset()
    trace = []
    try:
        for step in range(args.steps):
            if args.control == "joint":
                state = obs["policy"].detach().cpu().numpy().astype(np.float32)
                action = policy.run(None, {input_name: state})[0]
            else:
                action = np.array([[args.velocity, 0.0, 0.0] + ([0.0] * 5 if args.task == "button" else [])], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(torch.from_numpy(action).to(core.device))
            robot_xyz = core.scene["robot"].data.root_pos_w[0].detach().cpu().tolist()
            robot_quat = core.scene["robot"].data.root_quat_w[0].detach().cpu().tolist()
            ball_xyz = core.scene["ball"].data.root_pos_w[0].detach().cpu().tolist()
            ball_velocity = core.scene["ball"].data.root_lin_vel_w[0].detach().cpu().tolist()
            step_success = bool(core.termination_manager.get_term("success")[0].detach().cpu())
            trace.append({
                "step": step,
                "robot_xyz": robot_xyz,
                "robot_quat_wxyz": robot_quat,
                "ball_xyz": ball_xyz,
                "ball_velocity_w": ball_velocity,
                "ball_to_goal_xy": float(np.linalg.norm(np.asarray(ball_xyz[:2]) - np.asarray(GOAL[:2]))),
                "success": step_success,
                "terminated": bool(terminated[0].detach().cpu()),
                "truncated": bool(truncated[0].detach().cpu()),
                "reward": float(reward[0].detach().cpu()),
            })
            if trace[-1]["terminated"] or trace[-1]["truncated"]:
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "embodiment": "Berkeley Humanoid Lite V1",
        "arena": "IsaacLab-Arena release/0.1.1@f479817431f6a02a6e40e4ea76d89203b59146d4",
        "task": arena.task.get_prompt(),
        "controller": "released ONNX locomotion policy; authored constant forward command baseline",
        "control_interface": args.control,
        "seed": args.seed,
        "video_dir": str(args.video_dir) if args.video_dir else None,
        "success": any(item["success"] for item in trace),
        "trace": trace,
    }
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "trace"} | {"steps": len(trace), "last": trace[-1]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
