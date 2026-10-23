"""Pip configuration derived from its sealed interface, using native Isaac managers."""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
import isaaclab.sim as sim
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_tasks.core.cartpole.agents.rsl_rl_ppo_cfg import CartpolePPORunnerCfg
from . import terms

BUNDLE = Path(os.environ["ROBOTLAB_BUNDLE"])
INTERFACE = json.loads((BUNDLE / "model/interfaces.json").read_text())
ROBOT_CONFIG = json.loads((BUNDLE / "model/resolved-config.json").read_text())
SETTINGS = json.loads(Path(os.environ["ROBOTLAB_SETTINGS"]).read_text())
JOINTS = INTERFACE["joints"]
ORDER = INTERFACE["joint_order"]
ROOT_POSITION = tuple(map(float, ET.parse(BUNDLE / "model/robot.xml").find("worldbody/body").get("pos").split()))
REST = {name: SETTINGS["arm_rest_targets_rad"].get(name, 0.0) for name in ORDER}


def actuators():
    if os.environ.get('ROBOTLAB_ELECTROMECHANICAL'):
        from .electromechanical import ElectromechanicalDCMotorCfg, ElectromechanicalDCMotor
        settings=json.loads(os.environ['ROBOTLAB_ELECTROMECHANICAL'])
        cfg=ElectromechanicalDCMotorCfg(
            joint_names_expr=ORDER,
            stiffness={j['name']:0.0 if j['continuous'] else ROBOT_CONFIG['control']['kp'] for j in JOINTS},
            damping={j['name']:SETTINGS['balance']['wheel_velocity_gain_nm_s'] if j['continuous'] else ROBOT_CONFIG['control']['kd'] for j in JOINTS},
            actuator_effort_limit={j['name']:j['effort_limit_nm'] for j in JOINTS},
            joint_effort_limit={j['name']:j['effort_limit_nm'] for j in JOINTS},
            actuator_velocity_limit=settings['servo_proxy']['no_load_speed_rad_s'],
            saturation_effort=settings['servo_proxy']['stall_effort_nm'],
            delay_steps=settings['delay_steps'],dt_s=1/settings['physics_hz'],
            source_voltage_v=settings['pack_rc']['source_voltage_v'],
            source_resistance_ohm=settings['pack_rc']['source_resistance_ohm'],
            bulk_capacitance_f=settings['pack_rc']['bulk_capacitance_f'],
            no_load_speed_rad_s=settings['servo_proxy']['no_load_speed_rad_s'],
            stall_current_a=settings['servo_proxy']['stall_current_a'],
            idle_current_a=settings['servo_proxy']['idle_current_a'],
        )
        cfg.class_type=ElectromechanicalDCMotor
        return {'shared_pack':cfg}
    result = {}
    for family in sorted({j["family"] for j in JOINTS}):
        names = [j["name"] for j in JOINTS if j["family"] == family]
        limits = {j["name"]: j["effort_limit_nm"] for j in JOINTS if j["family"] == family}
        speeds = {j["name"]: j["velocity_limit_rad_s"] for j in JOINTS if j["family"] == family}
        result[family] = IdealPDActuatorCfg(
            joint_names_expr=names, stiffness=0.0 if family == "wheel" else ROBOT_CONFIG["control"]["kp"],
            damping=SETTINGS["balance"]["wheel_velocity_gain_nm_s"] if family == "wheel" else ROBOT_CONFIG["control"]["kd"],
            actuator_effort_limit=limits, joint_effort_limit=limits, actuator_velocity_limit=speeds,
        )
    return result


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim.GroundPlaneCfg(size=(20.0, 20.0)))
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim.UsdFileCfg(usd_path=os.environ["ROBOTLAB_USD"], articulation_props=sim.ArticulationRootPropertiesCfg(enabled_self_collisions=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=ROOT_POSITION, joint_pos=REST),
        joint_ordering=ORDER, body_ordering=[b["name"] for b in INTERFACE["links"]], actuators=actuators(),
    )


@configclass
class ActionsCfg:
    motion = terms.ResidualMotionCfg(asset_name="robot")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        proprioception = ObservationTermCfg(func=terms.proprioception)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy = PolicyCfg()


@configclass
class CommandsCfg:
    motion = mdp.UniformVelocityCommandCfg(
        asset_name="robot", resampling_time_range=(2.0, 4.0), rel_standing_envs=0.3,
        heading_command=False, ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-SETTINGS["max_forward_m_s"], SETTINGS["max_forward_m_s"]), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-SETTINGS["max_yaw_rad_s"], SETTINGS["max_yaw_rad_s"]),
        ),
    )


@configclass
class EventsCfg:
    reset = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    alive = RewardTermCfg(func=mdp.is_alive, weight=1.0)
    track = RewardTermCfg(func=terms.tracking_reward, weight=2.0)
    upright = RewardTermCfg(func=terms.upright_reward, weight=1.0)
    effort = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.005)
    smooth = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    failure = RewardTermCfg(func=mdp.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    fallen = TerminationTermCfg(func=terms.fallen)
    overspeed = TerminationTermCfg(func=terms.overspeed)


@configclass
class PipEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg(num_envs=64, env_spacing=3.0)
    actions = ActionsCfg()
    observations = ObservationsCfg()
    commands = CommandsCfg()
    events = EventsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 8.0
        self.sim.dt = 0.002
        self.sim.render_interval = 10
        # Upstream Newton's built-in actuator kernel only accepts its native
        # models. Use Isaac's supported Python actuator path for shared-pack DC.
        if os.environ.get('ROBOTLAB_ELECTROMECHANICAL'):
            self.sim.use_newton_actuators = False
        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(njmax=512, nconmax=128, cone="pyramidal", integrator="implicitfast"), num_substeps=1, use_cuda_graph=True)


@configclass
class PipRunnerCfg(CartpolePPORunnerCfg):
    experiment_name = "pip_residual_drive"
    num_steps_per_env = 24
    max_iterations = 100
    save_interval = 25
    clip_actions = 1.0
    def __post_init__(self):
        self.actor.hidden_dims = [64, 64]
        self.critic.hidden_dims = [64, 64]
        self.actor.distribution_cfg.init_std = 0.2
