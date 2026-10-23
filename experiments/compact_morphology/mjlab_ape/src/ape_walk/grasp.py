"""The same native MJLab cube lesson with the ape's opposed-jaw hand."""

from copy import deepcopy
from functools import partial
from pathlib import Path

from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .reach import reach_env_cfg, runner_cfg
from .task import _ape_spec


TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Opposed-Jaws-Right"
PREGRASP_TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Jaw-Pregrasp-Right"
HELD_TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Jaw-Held-Right"
HELD_COLLISION_TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Jaw-Held-Collision-Right"
ACQUIRE_COLLISION_TASK_ID = "Mjlab-Lift-Floor-Toy-Ape-Jaw-Acquire-Collision-Right"
MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v4_opposed_jaws.xml"
COLLISION_MODEL = Path(__file__).resolve().parents[3] / "playground_models" / "ape_sourced_v5_collision_proxy.xml"


def object_is_held_lifted(env, object_name: str, minimal_height: float):
    """Lifted-object reward only while both physical jaws contact the toy."""
    elevated = env.scene[object_name].data.root_link_pos_w[:, 2] > minimal_height
    fixed = env.scene.sensors["right_hand_cube_contact"].data.found.any(dim=-1)
    moving = env.scene.sensors["right_finger_tip_cube_contact"].data.found.any(dim=-1)
    return (elevated & fixed & moving).float()


def grasp_env_cfg(play: bool = False, model: Path = MODEL):
    cfg = reach_env_cfg(play=play)
    cfg.scene.entities["robot"].spec_fn = partial(_ape_spec, model)
    return cfg


def pregrasp_env_cfg(play: bool = False, model: Path = MODEL):
    cfg = grasp_env_cfg(play=play, model=model)
    pose = cfg.scene.entities["robot"].init_state.joint_pos
    # Use the post-reset MJLab pose, not the MJCF's pre-reset keyframe. The
    # right jaw is around x≈0.080 m after env.reset(); the tiny toy starts
    # between its opposed pads with yaw fixed for this first lesson.
    broad_pose = pose.copy()
    pose.clear()
    pose.update({
        r"^right_shoulder_yaw$": 0.0,
        r"^right_shoulder_roll$": 0.0,
        r"^right_shoulder_pitch$": -0.035,
        r"^right_elbow_pitch$": -0.986,
        r"^right_gripper$": 0.0,
    })
    pose.update(broad_pose)
    # MJLab's default action offset already resolves to this task's reset pose.
    # The first PPO run failed to lift while varying all five right arm joints;
    # train only the three joints used by the free-cube feasibility probe.
    action = cfg.actions["joint_pos"]
    action.scale.update({
        "^right_shoulder_yaw$": 0.0,
        "^right_shoulder_roll$": 0.0,
        "^right_shoulder_pitch$": 0.35,
        "^right_elbow_pitch$": 0.9,
    })
    command = cfg.commands["lift_height"]
    command.object_pose_range.x = (0.079, 0.081)
    command.object_pose_range.y = (-0.085, -0.083)
    command.object_pose_range.yaw = (0.0, 0.0)
    command.target_position_range.x = (0.087, 0.093)
    command.target_position_range.y = (-0.089, -0.087)
    command.target_position_range.z = (0.055, 0.065)
    return cfg


def held_env_cfg(play: bool = False, model: Path = MODEL):
    """First lift lesson: hold the cube with physical jaws, then move it."""
    cfg = pregrasp_env_cfg(play=play, model=model)
    cfg.scene.entities["robot"].init_state.joint_pos[r"^right_gripper$"] = 0.35
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.0
    cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
        ContactSensorCfg(
            name=f"{name}_cube_contact",
            primary=ContactMatch(mode="geom", pattern=name, entity="robot"),
            secondary=ContactMatch(mode="geom", pattern="cube_geom", entity="cube"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        )
        for name in ("right_hand", "right_finger_tip")
    )
    # The tiny 0.015 m precise kernel was effectively zero at floor height;
    # use MJLab YAM's 0.05 m kernel for this 5 cm lift curriculum. Keep the
    # ape's upright term: the free body can still tip when its arms move.
    cfg.rewards["lift_precise"].params["std"] = 0.05
    # Arena pays each step above 4 cm. The first trial exploited that by
    # launching the toy 76 cm high, so require both physical jaw contacts too.
    cfg.rewards["lifted"] = RewardTermCfg(
        func=object_is_held_lifted,
        weight=15.0,
        params={"object_name": "cube", "minimal_height": 0.04},
    )
    return cfg


def acquire_env_cfg(play: bool = False):
    """Same collision-aware lift lesson, but the gripper starts open and can close."""
    cfg = held_env_cfg(play=play, model=COLLISION_MODEL)
    cfg.scene.entities["robot"].init_state.joint_pos[r"^right_gripper$"] = 0.0
    cfg.actions["joint_pos"].scale[r"^right_gripper$"] = 0.5
    return cfg


grasp_runner_cfg = deepcopy(runner_cfg)
grasp_runner_cfg.experiment_name = "ape_jaw_floor_toy"
grasp_runner_cfg.run_name = "ape_right_jaw_lift_from_squat"
register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=grasp_env_cfg(),
    play_env_cfg=grasp_env_cfg(play=True),
    rl_cfg=grasp_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)

pregrasp_runner_cfg = deepcopy(grasp_runner_cfg)
pregrasp_runner_cfg.run_name = "ape_right_jaw_pregrasp_lift"
register_mjlab_task(
    task_id=PREGRASP_TASK_ID,
    env_cfg=pregrasp_env_cfg(),
    play_env_cfg=pregrasp_env_cfg(play=True),
    rl_cfg=pregrasp_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)

held_runner_cfg = deepcopy(pregrasp_runner_cfg)
held_runner_cfg.run_name = "ape_right_jaw_lift_held_toy"
register_mjlab_task(
    task_id=HELD_TASK_ID,
    env_cfg=held_env_cfg(),
    play_env_cfg=held_env_cfg(play=True),
    rl_cfg=held_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)

collision_runner_cfg = deepcopy(held_runner_cfg)
collision_runner_cfg.run_name = "ape_right_jaw_lift_collision_proxy"
register_mjlab_task(
    task_id=HELD_COLLISION_TASK_ID,
    env_cfg=held_env_cfg(model=COLLISION_MODEL),
    play_env_cfg=held_env_cfg(play=True, model=COLLISION_MODEL),
    rl_cfg=collision_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)

acquire_runner_cfg = deepcopy(held_runner_cfg)
acquire_runner_cfg.run_name = "ape_right_jaw_acquire_collision_proxy"
register_mjlab_task(
    task_id=ACQUIRE_COLLISION_TASK_ID,
    env_cfg=acquire_env_cfg(),
    play_env_cfg=acquire_env_cfg(play=True),
    rl_cfg=acquire_runner_cfg,
    runner_cls=ManipulationOnPolicyRunner,
)
