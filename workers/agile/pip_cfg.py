"""Pip embodiment configuration using AGILE commands and native Isaac managers.

The policy owns four leg-position and two wheel-velocity targets. Upper-body
targets are sampled by AGILE during motor training, as in its loco-manipulation
recipe. This stage does not implement a manipulation policy or task planner.
"""
import json
import os
from pathlib import Path

import isaaclab.sim as sim
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (EventTermCfg as Event, ObservationGroupCfg as Group,
    ObservationTermCfg as Obs, RewardTermCfg as Reward, TerminationTermCfg as Done, SceneEntityCfg)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from agile.rl_env import mdp
from agile.rl_env.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from electromechanical import ElectromechanicalDCMotor, ElectromechanicalDCMotorCfg

C = json.loads(Path(os.environ['ROBOTLAB_CONTRACT']).read_text())
E = json.loads(Path(os.environ['ROBOTLAB_ELECTRICAL']).read_text())
ORDER = C['joint_order']
LEGS = [n for n in ORDER if n.endswith(('_thigh_joint', '_shin_joint'))]
WHEELS = [n for n in ORDER if n.endswith('_wheel_joint')]
LOWER = LEGS + WHEELS


def actuator():
    return ElectromechanicalDCMotorCfg(
        class_type=ElectromechanicalDCMotor,
        joint_names_expr=ORDER,
        stiffness={j['name']: j['kp'] for j in C['joints']},
        damping={j['name']: j['kd'] for j in C['joints']},
        effort_limit={j['name']: j['effort_limit_nm'] for j in C['joints']},
        effort_limit_sim={j['name']: j['effort_limit_nm'] for j in C['joints']},
        velocity_limit=E['servo_proxy']['no_load_speed_rad_s'],
        saturation_effort=E['servo_proxy']['stall_effort_nm'],
        delay_steps=E['delay_steps'], dt_s=1 / C['physics_hz'],
        source_voltage_v=E['pack_rc']['source_voltage_v'],
        source_resistance_ohm=E['pack_rc']['source_resistance_ohm'],
        bulk_capacitance_f=E['pack_rc']['bulk_capacitance_f'],
        no_load_speed_rad_s=E['servo_proxy']['no_load_speed_rad_s'],
        stall_current_a=E['servo_proxy']['stall_current_a'],
        idle_current_a=E['servo_proxy']['idle_current_a'])


@configclass
class Scene(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path='/World/ground', spawn=sim.GroundPlaneCfg())
    robot = ArticulationCfg(prim_path='{ENV_REGEX_NS}/Robot',
        spawn=sim.UsdFileCfg(usd_path=os.environ['ROBOTLAB_USD'],
            articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(C['root_position_m']), joint_pos=C['rest_rad']),
        actuators={'shared_pack': actuator()})
    # Native MJCF import places the link hierarchy under Geometry (as in the
    # existing converted robot/camera assets), not directly below Robot.
    height = RayCasterCfg(prim_path='{ENV_REGEX_NS}/Robot/Geometry/' + C['root_body'],
        offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 2.)), ray_alignment='yaw',
        pattern_cfg=patterns.GridPatternCfg(resolution=.05, size=(0., 0.)),
        mesh_prim_paths=['/World/ground'], debug_vis=False)


@configclass
class Actions:
    legs = mdp.JointPositionActionCfg(asset_name='robot', joint_names=LEGS,
        scale=.4, use_default_offset=True, preserve_order=True,
        clip={j['name']: tuple(j['limits']) for j in C['joints'] if j['name'] in LEGS})
    wheels = mdp.JointVelocityActionCfg(asset_name='robot', joint_names=WHEELS,
        scale={j['name']: j['velocity_limit_rad_s'] for j in C['joints'] if j['continuous']},
        use_default_offset=False, preserve_order=True,
        clip={j['name']: (-j['velocity_limit_rad_s'], j['velocity_limit_rad_s'])
              for j in C['joints'] if j['name'] in WHEELS})
    upper = mdp.RandomActionCfg(asset_name='robot', joint_names_exclude=LOWER,
        sample_range=(.5, 2.), preserve_order=True,
        joint_pos_limits={j['name']: (max(j['limits'][0], C['rest_rad'][j['name']] - .35),
                                     min(j['limits'][1], C['rest_rad'][j['name']] + .35))
                          for j in C['joints'] if j['name'] not in LOWER},
        command_name='base_velocity', no_random_when_walking=False)


@configclass
class Commands:
    base_velocity = mdp.UniformVelocityBaseHeightCommandCfg(asset_name='robot',
        resampling_time_range=(1., 3.), rel_standing_envs=.3, heading_command=False,
        default_height=C['root_position_m'][2], min_walk_height=.17,
        squatting_threshold=.18, min_vel_norm=0., random_height_during_walking=True,
        ema_smoothing_param=.5, height_sensor='height', root_name='torso',
        ranges=mdp.UniformVelocityBaseHeightCommandCfg.Ranges(
            lin_vel_x=(-.04, .04), lin_vel_y=(0., 0.), ang_vel_z=(-.1, .1),
            base_height=(.18, C['root_position_m'][2])))


@configclass
class Observations:
    @configclass
    class Policy(Group):
        command = Obs(func=mdp.generated_commands, params={'command_name': 'base_velocity'})
        gyro = Obs(func=mdp.base_ang_vel)
        gravity = Obs(func=mdp.projected_gravity)
        position = Obs(func=mdp.joint_pos_rel, params={'asset_cfg': SceneEntityCfg(
            'robot', joint_names=[n for n in ORDER if n not in WHEELS], preserve_order=True)})
        velocity = Obs(func=mdp.joint_vel_rel, scale=.1)
        action = Obs(func=mdp.last_action)
        def __post_init__(self):
            self.history_length = 5
            self.flatten_history_dim = True
            self.concatenate_terms = True
            self.enable_corruption = False
    policy = Policy()


@configclass
class Events:
    reset = Event(func=mdp.reset_scene_to_default, mode='reset')


@configclass
class Rewards:
    alive = Reward(func=mdp.is_alive, weight=1.)
    velocity = Reward(func=mdp.track_lin_vel_xy_exp, weight=2., params={'command_name': 'base_velocity', 'std': .04})
    yaw = Reward(func=mdp.track_ang_vel_z_exp, weight=.5, params={'command_name': 'base_velocity', 'std': .1})
    height = Reward(func=mdp.track_base_height_exp_smooth, weight=1., params={'command_name': 'base_velocity', 'std': .025})
    upright = Reward(func=mdp.flat_orientation_l2, weight=-2.)
    action_rate = Reward(func=mdp.action_rate_l2, weight=-.01)
    torque = Reward(func=mdp.joint_torques_l2, weight=-.005)
    failed = Reward(func=mdp.is_terminated, weight=-5.)


@configclass
class Terminations:
    timeout = Done(func=mdp.time_out, time_out=True)
    fallen = Done(func=mdp.bad_orientation, params={'limit_angle': .28})


@configclass
class PipEnvCfg(ManagerBasedRLEnvCfg):
    scene = Scene(num_envs=64, env_spacing=2.)
    actions = Actions()
    observations = Observations()
    commands = Commands()
    events = Events()
    rewards = Rewards()
    terminations = Terminations()
    def __post_init__(self):
        self.controller_freq = C['controller_hz']
        self.physics_freq = C['physics_hz']
        self.decimation = self.physics_freq // self.controller_freq
        self.episode_length_s = 8.
        self.sim.dt = 1 / self.physics_freq
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg()

    def eval(self):
        # Native AGILE evaluation observations are recorder-only; the actor's
        # sensor observation group remains unchanged.
        self.observations.eval = mdp.EvaluationObservationsCfg()
        self.actions.upper.randomize = False


@configclass
class PipRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100
    save_interval = 25
    experiment_name = 'pip_agile_velocity_height'
    empirical_normalization = True
    # Native action terms bound physical targets in both train and eval. A
    # trainer-only +/-1 clamp restricted knees to 0.4 rad and blocked crouching.
    # Keep scale/offset unchanged so retained checkpoint outputs keep their units.
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(init_noise_std=.3, actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64], activation='elu')
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1., use_clipped_value_loss=True,
        clip_param=.2, entropy_coef=.01, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1e-3, schedule='adaptive', gamma=.99, lam=.95, desired_kl=.01, max_grad_norm=1.)
