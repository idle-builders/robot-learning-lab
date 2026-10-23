"""Use MJLab's upstream DC-motor actuator on a separate ape body.

Manufacturer stall/no-load/rated values bound the model. The linear torque-speed
shape is MJLab's generic DC approximation; Feetech impedance, latency, voltage
response and continuous thermal behavior are not identified on our hardware.
"""

from pathlib import Path
from functools import partial
import re
import xml.etree.ElementTree as ET

import mujoco
import yaml
from mjlab.actuator import DcMotorActuatorCfg


REPO = Path(__file__).resolve().parents[6]
MODEL = REPO / "experiments/compact_morphology/playground_models/ape_sourced_v6_candidate_mass.xml"
NARROW55_MODEL = REPO / "experiments/compact_morphology/playground_models/ape_sourced_v7_candidate_narrow55.xml"
LEDGER = REPO / "robot/ape-physics-ledger.yaml"
CATALOG = REPO / "robot/actuator-catalog.yaml"


def _candidate_spec(model_path: Path = MODEL) -> mujoco.MjSpec:
    # Keep the native task's terrain/sensor edits. Replace only XML position
    # actuators with MJLab motor actuators below, preserving the joint topology.
    from ape_walk.task import _ape_spec

    spec = _ape_spec(model_path)
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    return spec


def _actuator_configs(motor_rail_v: float, delay_steps: int):
    ledger = yaml.safe_load(LEDGER.read_text())
    catalog = yaml.safe_load(CATALOG.read_text())["parts"]
    motor_for_joint = {
        joint: group["part"]
        for group in ledger["axes"].values()
        for joint in group["joints"]
    }
    source_positions = ET.parse(MODEL).getroot().find("actuator")
    assert source_positions is not None
    names = [act.get("joint") for act in source_positions]
    assert len(names) == 22 and set(names) == set(motor_for_joint)
    cfgs = []
    for act in source_positions:
        joint = act.get("joint")
        assert joint is not None
        part = catalog[motor_for_joint[joint]]
        if motor_for_joint[joint].startswith("feetech"):
            # A simple rail-voltage sensitivity, not an identified servo law.
            scale = motor_rail_v / part["rating_voltage_v"]
            effort = part["rated_effort_nm"] * scale
            peak = part["stall_effort_nm"] * scale
            speed = part["no_load_speed_rad_s"] * scale
        else:
            # ROBOTIS's selection guide suggests ~20% of stall as a general
            # working range. It is not a certified continuous torque rating.
            effort = 0.2 * part["stall_effort_nm_at_5v"]
            peak = part["stall_effort_nm_at_5v"]
            speed = part["no_load_speed_rad_s_at_5v"]
        cfgs.append(DcMotorActuatorCfg(
            target_names_expr=(f"^{re.escape(joint)}$",),
            stiffness=float(act.get("kp")),
            damping=float(act.get("kv")),
            effort_limit=effort,
            saturation_effort=peak,
            velocity_limit=speed,
            delay_min_lag=delay_steps,
            delay_max_lag=delay_steps,
        ))
    return tuple(cfgs)


def configure_candidate(cfg, *, motor_rail_v: float = 12.0, delay_s: float = 0.0,
                        model_path: Path = MODEL):
    """Keep observation/action shape; swap body and physical actuator envelope.

    The default deterministic case uses source-rated rails and zero *assumed*
    additional delay. Use 11 V or 10 ms only as explicitly labeled sensitivity
    runs until a bench fit supplies those values.
    """
    if not 10.0 <= motor_rail_v <= 12.0:
        raise ValueError("The provisional Feetech rail sensitivity is 10–12 V")
    if not 0.0 <= delay_s <= 0.02:
        raise ValueError("The provisional command-delay sensitivity is 0–20 ms")
    dt = cfg.sim.mujoco.timestep
    delay_steps = round(delay_s / dt)
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = partial(_candidate_spec, model_path)
    robot.sort_actuators = False  # Exactly the source XML's 22-actuator order.
    robot.articulation.actuators = _actuator_configs(motor_rail_v, delay_steps)
    robot.collisions[0].disable_other_geoms = False  # Retain v5 body proxy contact.
    return cfg
