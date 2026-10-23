"""MJLab BeyondMimic tracking of a physics-replayed ape foot transition.

The teacher is an authored Robot Keyframe Kit motion. PPO must still learn a
closed-loop policy; the teacher replay itself never counts as learned success.
"""

from pathlib import Path
from functools import partial

from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from . import task


TASK_ID = "Mjlab-Tracking-Ape-Foot-Transition"
ROOT = Path(__file__).resolve().parents[5]
MOTION = ROOT / "experiments/compact_morphology/motions/ape_foot_transition_v6.npz"
BODY = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_collision_proxy.xml"


def tracking_env_cfg(play: bool = False):
    cfg = make_tracking_env_cfg()
    robot = task.robot_cfg(elbowed=True)
    robot.spec_fn = partial(task._ape_spec, BODY)
    # Preserve XML torso/limb masks; otherwise MJLab's selective CollisionCfg
    # silently zeroes the omitted body geoms and lets arms cross the pelvis.
    robot.collisions[0].disable_other_geoms = False
    cfg.scene.entities = {"robot": robot}
    self_sensor = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1,
        history_length=4,
    )
    reference_sensors = task.env_cfg(play=True, elbowed=True).scene.sensors
    cfg.scene.sensors = (self_sensor, reference_sensors[0], reference_sensors[1])
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionCommandCfg)
    motion.motion_file = str(MOTION)
    motion.anchor_body_name = "pelvis"
    motion.body_names = (
        "pelvis", "left_thigh", "left_shin", "left_foot",
        "right_thigh", "right_shin", "right_foot", "torso",
        "left_arm", "left_forearm", "right_arm", "right_forearm",
    )
    # Scale G1's metre-valued tolerances to a 35 cm robot. Angular terms
    # remain in radians. Keep the first lesson nominal until it can track.
    cfg.rewards["motion_global_root_pos"].params["std"] = 0.08
    cfg.rewards["motion_body_pos"].params["std"] = 0.08
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.3
    cfg.terminations["anchor_pos"].params["threshold"] = 0.07
    cfg.terminations["ee_body_pos"].params.update({
        "body_names": ("left_foot", "right_foot", "left_forearm", "right_forearm"),
        "threshold": 0.07,
    })
    cfg.events.clear()
    motion.pose_range = {}
    motion.velocity_range = {}
    motion.joint_position_range = (0.0, 0.0)
    motion.sampling_mode = "start" if play else "adaptive"
    motion.debug_vis = play
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10
    cfg.episode_length_s = 10.5
    cfg.scene.extent = 2.0
    cfg.viewer.body_name = "pelvis"
    cfg.viewer.distance = 0.8
    if play:
        cfg.observations["actor"].enable_corruption = False
    return cfg


runner_cfg = unitree_g1_tracking_ppo_runner_cfg()
runner_cfg.experiment_name = "ape_foot_transition_tracking"
runner_cfg.run_name = "stand_load_lift_setdown_v6"
runner_cfg.max_iterations = 300
runner_cfg.save_interval = 50
runner_cfg.actor.distribution_cfg["init_std"] = 0.2
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=tracking_env_cfg(),
    play_env_cfg=tracking_env_cfg(play=True),
    rl_cfg=runner_cfg,
    runner_cls=MotionTrackingOnPolicyRunner,
)
