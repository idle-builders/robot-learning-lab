"""Catalog-bounded motor sensitivity for the selected 21-axis 50 cm ape.

This keeps the exact frame MJCF, action order, and task. MJLab's upstream
DcMotorActuator substitutes for the XML position servos. Its linear motor
envelope is a sensitivity model, not an identified Feetech/ROBOTIS servo law.
"""

from functools import partial
import hashlib
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import yaml
from mjlab.actuator import DcMotorActuatorCfg


REPO = Path(__file__).resolve().parents[6]
CATALOG = REPO / "robot/actuator-catalog.yaml"
FEETECH_HIP_KNEE = "feetech-sts3250-c001"
FEETECH_ANKLE_ARM = "feetech-st3215-c018"
ROBOTIS_SMALL = "robotis-xl330-m288-t"
DIRECT_FRAME_SHA256 = "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
DIRECT_FRAME_JOINTS = tuple(
    f"{side}_{axis}"
    for side in ("left", "right")
    for axis in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
) + ("head_yaw",) + tuple(
    f"{side}_{axis}"
    for side in ("left", "right")
    for axis in ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper")
)


def require_direct_frame(path: Path) -> mujoco.MjModel:
    """Reject a substituted body or shuffled motor channels in new training."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DIRECT_FRAME_SHA256:
        raise ValueError(f"Direct Frame body SHA mismatch: {digest}; expected {DIRECT_FRAME_SHA256}")
    model = mujoco.MjModel.from_xml_path(str(path))
    if (model.nu, model.nq) != (21, 28):
        raise ValueError(f"Direct Frame dimensions changed: nu={model.nu}, nq={model.nq}")
    joints = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[i, 0]))
        for i in range(model.nu)
    )
    if joints != DIRECT_FRAME_JOINTS:
        raise ValueError(f"Direct Frame ordered actuator joints changed: {joints}")
    return model


def give_soles_contact_priority(cfg) -> None:
    """Let sampled sole grip dominate the default terrain on contact."""
    collisions = cfg.scene.entities["robot"].collisions
    if len(collisions) != 1:
        raise ValueError("Expected one Cove collision configuration")
    collisions[0].priority = {r"^(left|right)_sole$": 1}


def part_for_joint(joint: str) -> str:
    if joint == "head_yaw" or joint.endswith("_gripper"):
        return ROBOTIS_SMALL
    if joint.startswith(("left_", "right_")):
        if "_hip_" in joint or joint.endswith("_knee"):
            return FEETECH_HIP_KNEE
        if joint.endswith("_ankle_pitch") or "_shoulder_" in joint or joint.endswith("_elbow_pitch"):
            return FEETECH_ANKLE_ARM
    raise ValueError(f"No selected 50 cm drive for {joint}")


def _motor_spec(model_path: Path) -> mujoco.MjSpec:
    from ape_walk.task import _ape_spec

    spec = _ape_spec(model_path)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def motor_configs(model_path: Path, *, rail_v: float = 12.0, delay_steps: int = 0,
                  ankle_motor_model: str = "selected"):
    if not 10.0 <= rail_v <= 12.0:
        raise ValueError("Feetech rail sensitivity must be 10–12 V")
    if not 0 <= delay_steps <= 10:
        raise ValueError("Command-delay sensitivity must be 0–10 physics steps")
    if ankle_motor_model not in ("selected", "st3250"):
        raise ValueError("Ankle motor sensitivity must be selected or st3250")
    positions = ET.parse(model_path).getroot().find("actuator")
    if positions is None:
        raise ValueError(f"No actuators in {model_path}")
    acts = list(positions)
    joints = [act.get("joint") for act in acts]
    if len(joints) != 21 or len(set(joints)) != 21 or None in joints or "head_pitch" in joints:
        raise ValueError("Selected frame model must have 21 distinct axes without head pitch")
    catalog = yaml.safe_load(CATALOG.read_text())["parts"]
    configs = []
    for act, joint in zip(acts, joints):
        if act.tag != "position":
            raise ValueError(f"Expected a position servo at {joint}")
        part_name = part_for_joint(joint)
        if ankle_motor_model == "st3250" and joint.endswith("_ankle_pitch"):
            part_name = FEETECH_HIP_KNEE
        part = catalog[part_name]
        if part_name == ROBOTIS_SMALL:
            # No continuous XL330 rating in the official e-manual. Use the
            # existing 20%-of-stall sensitivity only for this light-duty group.
            effort = 0.2 * part["stall_effort_nm_at_5v"]
            peak = part["stall_effort_nm_at_5v"]
            speed = part["no_load_speed_rad_s_at_5v"]
        else:
            scale = rail_v / part["rating_voltage_v"]
            effort = part["rated_effort_nm"] * scale
            peak = part["stall_effort_nm"] * scale
            speed = part["no_load_speed_rad_s"] * scale
        configs.append(DcMotorActuatorCfg(
            target_names_expr=(f"^{re.escape(joint)}$",),
            stiffness=float(act.get("kp")),
            damping=float(act.get("kv")),
            effort_limit=effort,
            saturation_effort=peak,
            velocity_limit=speed,
            delay_min_lag=delay_steps,
            delay_max_lag=delay_steps,
        ))
    return tuple(configs)


def configure_frame50_motors(cfg, model_path: Path, *, rail_v: float = 12.0,
                             delay_s: float = 0.0, ankle_motor_model: str = "selected"):
    """Swap only the servo model; preserve frame/contact geometry and channels."""
    if not 0.0 <= delay_s <= 0.02:
        raise ValueError("Command-delay sensitivity must be 0–20 ms")
    model_path = model_path.resolve()
    delay_steps = round(delay_s / cfg.sim.mujoco.timestep)
    configs = motor_configs(model_path, rail_v=rail_v, delay_steps=delay_steps,
                            ankle_motor_model=ankle_motor_model)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = partial(_motor_spec, model_path)
    robot.sort_actuators = False
    robot.articulation.actuators = configs
    robot.collisions[0].disable_other_geoms = False
    return cfg


def clip_frame50_leg_targets(cfg, model_path: Path, *, margin_rad: float = 0.02,
                             axes: tuple[str, ...] = ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")):
    """Keep selected leg position commands inside the frame's joint travel.

    The retained actor can request ankle targets outside Cove's +/-0.349 rad
    range. MJLab applies this clip after action scale and default-pose offset,
    before the position target reaches the motor model.
    """
    if margin_rad < 0:
        raise ValueError("Joint travel margin must be non-negative")
    if not axes or any(axis not in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch") for axis in axes):
        raise ValueError("Axes must name selected leg joints")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    clip = dict(cfg.actions["joint_pos"].clip or {})
    for side in ("left", "right"):
        for axis in axes:
            joint = f"{side}_{axis}"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id < 0 or not model.jnt_limited[joint_id]:
                raise ValueError(f"Selected frame is missing a limited {joint}")
            lower, upper = model.jnt_range[joint_id]
            if lower + margin_rad >= upper - margin_rad:
                raise ValueError(f"Margin leaves no legal travel for {joint}")
            clip[f"^{re.escape(joint)}$"] = (
                float(lower + margin_rad), float(upper - margin_rad)
            )
    cfg.actions["joint_pos"].clip = clip
    return cfg


def clip_frame50_shoulder_targets(cfg, model_path: Path, *, margin_rad: float = 0.05):
    """Bound shoulder pitch targets to reachable Cove joint travel."""
    if margin_rad < 0:
        raise ValueError("Joint travel margin must be non-negative")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    clip = dict(cfg.actions["joint_pos"].clip or {})
    for side in ("left", "right"):
        joint = f"{side}_shoulder_pitch"
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if joint_id < 0 or not model.jnt_limited[joint_id]:
            raise ValueError(f"Selected frame is missing a limited {joint}")
        lower, upper = model.jnt_range[joint_id]
        if lower + margin_rad >= upper - margin_rad:
            raise ValueError(f"Margin leaves no legal travel for {joint}")
        clip[f"^{re.escape(joint)}$"] = (
            float(lower + margin_rad), float(upper - margin_rad)
        )
    cfg.actions["joint_pos"].clip = clip
    return cfg


def reveal_frame50_proxy_geometry_for_video(model: mujoco.MjModel) -> int:
    """Reveal the selected physics proxies in an in-memory replay model only.

    The training XML hides group-4 geoms with alpha zero. Rendering an
    unmodified scene shows command markers but effectively no robot. RGBA and
    geom group are visual properties; motor and contact behavior stay intact.
    """
    count = 0
    for geom_id in range(model.ngeom):
        if (model.geom_bodyid[geom_id] != 0
                and model.geom_group[geom_id] == 4
                and model.geom_rgba[geom_id, 3] == 0):
            model.geom_rgba[geom_id, 3] = 1
            model.geom_group[geom_id] = 0
            count += 1
    return count
