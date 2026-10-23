"""Pollen Flamingo hold on a physics-held, compact ape one-foot pose."""

from copy import deepcopy
from functools import partial
import json
import math
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from . import task


TASK_ID = "Mjlab-FlamingoHoldUpright-Flat-Ape-Elbows"
CLEAR_TASK_ID = "Mjlab-FlamingoHoldUprightClear-Flat-Ape-Elbows"
LEFT_CLEAR_TASK_ID = "Mjlab-FlamingoHoldUprightClearLeft-Flat-Ape-Elbows"
RIGHT_CONTACT_SAFE_TASK_ID = "Mjlab-FlamingoHoldFullCollisionRight-Flat-Ape-Elbows"
LEFT_CONTACT_SAFE_TASK_ID = "Mjlab-FlamingoHoldFullCollisionLeft-Flat-Ape-Elbows"
ROOT = Path(__file__).resolve().parents[5]
ENDPOINT = ROOT / "artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json"
LEFT_ENDPOINT = ROOT / "artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json"
BODY = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_collision_proxy.xml"
JOINT_NAMES = tuple(f"{side}_{name}" for side in ("left", "right")
                    for name in task.LEG_JOINTS) + ("head_yaw", "head_pitch") + tuple(
                    f"{side}_{name}" for side in ("left", "right")
                    for name in ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"))


def env_cfg(play: bool = False, side: str = "right"):
    cfg = (task.flamingo_hold_command_limit_left_env_cfg(play=play) if side == "left"
           else task.flamingo_hold_command_limit_env_cfg(play=play))
    pose_spec = json.loads((LEFT_ENDPOINT if side == "left" else ENDPOINT).read_text())
    joint_pose = [pose_spec["joints"].get(name, 0.0) for name in JOINT_NAMES]
    roll, pitch = pose_spec["root_xyz_roll_pitch"][3:]
    z = pose_spec["root_xyz_roll_pitch"][2]
    cfg.scene.entities["robot"].spec_fn = partial(task._ape_spec, BODY)
    scale = dict(cfg.actions["joint_pos"].scale)
    for limb in task.SIDES:
        scale[f"^{limb}_shoulder_roll$"] = 1.0
        scale[f"^{limb}_shoulder_pitch$"] = 1.0
        scale[f"^{limb}_elbow_pitch$"] = 0.5
    cfg.actions["joint_pos"].scale = scale
    cfg.rewards["swing_foot_clear"].params["target"] = 0.04
    cfg.rewards["pose_flamingo"].params["target_overrides"] = dict(enumerate(joint_pose))
    cfg.rewards["gravity_flamingo"].params["target"] = (
        math.sin(pitch), -math.sin(roll) * math.cos(pitch),
        -math.cos(roll) * math.cos(pitch),
    )
    cfg.events["set_flamingo_state"].params.update({
        "joint_pose": joint_pose, "base_roll": roll, "base_pitch": pitch,
        "z_min": z, "z_max": z + 0.004,
    })
    if side == "left":
        # The mirror holds under 0.5°/0.01-rad perturbations in native MJLab,
        # but the reference 2°/0.03-rad reset ejects it before feedback acts.
        cfg.events["set_flamingo_state"].params.update({
            "tilt_noise": math.radians(0.5), "joint_noise_std": 0.01,
        })
    return cfg


def swing_clear_while_airborne(env, asset_cfg, target, std, gate_sensor_name, gate_slot):
    """Pollen height reward, valid only while the swing sole is off the floor.

    On this ape a tilted sole can have its center at the target height while
    the toe is still grounded; height alone rewarded exactly that failure.
    """
    height = task.microduck_mdp.foot_height_gaussian(
        env, asset_cfg, target, std, gate_sensor_name, gate_slot,
    )
    swing_grounded = env.scene.sensors["feet_ground_contact"].data.found[:, 1 - gate_slot].float()
    return height * (1.0 - swing_grounded)


def clear_env_cfg(play: bool = False, side: str = "right"):
    cfg = env_cfg(play=play, side=side)
    cfg.rewards["swing_foot_clear"].func = swing_clear_while_airborne
    return cfg


def contact_safe_env_cfg(play: bool = False, side: str = "right"):
    """Keep source body masks active and use MJLab G1's self-collision cost."""
    cfg = clear_env_cfg(play=play, side=side)
    # The default MJLab CollisionCfg disables all unlisted geoms. The base ape
    # task lists only soles and hands, which silently removed torso/limb contact
    # from native dynamics even when the imported XML enabled those pairs.
    cfg.scene.entities["robot"].collisions[0].disable_other_geoms = False
    sensor = task.ContactSensorCfg(
        name="self_body_contact",
        primary=task.ContactMatch(
            mode="subtree", pattern="pelvis", entity="robot",
        ),
        secondary=task.ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"), reduce="none", num_slots=1,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor,)
    cfg.rewards["self_body_collision"] = task.RewardTermCfg(
        func=task.mdp.self_collision_cost, weight=-1.0,
        params={"sensor_name": sensor.name},
    )
    return cfg


runner_cfg = deepcopy(task.flamingo_command_limit_rl_cfg)
runner_cfg.experiment_name = "ape_elbows_foot_upright"
runner_cfg.run_name = "upright_right_hold"
runner_cfg.max_iterations = 250
runner_cfg.save_interval = 50
runner_cfg.actor.distribution_cfg["init_std"] = 0.2
register_mjlab_task(
    task_id=TASK_ID, env_cfg=env_cfg(), play_env_cfg=env_cfg(play=True),
    rl_cfg=runner_cfg, runner_cls=VelocityOnPolicyRunner,
)
clear_runner_cfg = deepcopy(runner_cfg)
clear_runner_cfg.experiment_name = "ape_elbows_foot_upright_clear"
register_mjlab_task(
    task_id=CLEAR_TASK_ID, env_cfg=clear_env_cfg(), play_env_cfg=clear_env_cfg(play=True),
    rl_cfg=clear_runner_cfg, runner_cls=VelocityOnPolicyRunner,
)
left_runner_cfg = deepcopy(clear_runner_cfg)
left_runner_cfg.experiment_name = "ape_elbows_foot_upright_clear_left"
left_runner_cfg.run_name = "upright_left_hold"
register_mjlab_task(
    task_id=LEFT_CLEAR_TASK_ID,
    env_cfg=clear_env_cfg(side="left"),
    play_env_cfg=clear_env_cfg(play=True, side="left"),
    rl_cfg=left_runner_cfg, runner_cls=VelocityOnPolicyRunner,
)
right_safe_runner_cfg = deepcopy(clear_runner_cfg)
right_safe_runner_cfg.experiment_name = "ape_elbows_foot_full_collision_right"
right_safe_runner_cfg.run_name = "right_from_hold249"
register_mjlab_task(
    task_id=RIGHT_CONTACT_SAFE_TASK_ID,
    env_cfg=contact_safe_env_cfg(side="right"),
    play_env_cfg=contact_safe_env_cfg(play=True, side="right"),
    rl_cfg=right_safe_runner_cfg, runner_cls=VelocityOnPolicyRunner,
)
left_safe_runner_cfg = deepcopy(left_runner_cfg)
left_safe_runner_cfg.experiment_name = "ape_elbows_foot_full_collision_left"
left_safe_runner_cfg.run_name = "left_from_hold50"
register_mjlab_task(
    task_id=LEFT_CONTACT_SAFE_TASK_ID,
    env_cfg=contact_safe_env_cfg(side="left"),
    play_env_cfg=contact_safe_env_cfg(play=True, side="left"),
    rl_cfg=left_safe_runner_cfg, runner_cls=VelocityOnPolicyRunner,
)
