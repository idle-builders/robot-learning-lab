"""Build a source-bounded ape variant; run a neutral MuJoCo rollout.

Motor masses and rated effort come from manufacturer data. The 35 cm geometry,
servo gains and total mass remain design hypotheses, not a measured twin.
"""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "ape_short_torso.xml"
DEST = HERE / "ape_sourced_v1.xml"
CATALOG = ROOT / "robot/actuator-catalog.yaml"
REPORT = ROOT / "artifacts/ape-sourced-model-20260928/nominal.json"


def add_motor(body, name, mass, dimensions):
    # Only inertial placement; case fit/collision is not established.
    ET.SubElement(
        body, "geom", name=name + "_motor_mass", type="box",
        size=" ".join(f"{x / 2:.6f}" for x in dimensions),
        mass=f"{mass:.9f}", contype="0", conaffinity="0", rgba="0 0 0 0",
    )


def build():
    parts = yaml.safe_load(CATALOG.read_text())["parts"]
    leg = parts["feetech-sts3250-c001"]
    arm = parts["feetech-st3215-c018"]
    root = ET.parse(SOURCE).getroot()
    bodies = {body.get("name"): body for body in root.iter("body")}
    actuators = {act.get("joint"): act for act in root.find("actuator")}
    extra_mass = 0.0

    for side in ("left", "right"):
        for axis in ("hip_yaw", "hip_roll"):
            geom = bodies[f"{side}_{axis}"].find("geom")
            extra_mass += leg["mass_kg"] - float(geom.get("mass"))
            geom.set("mass", str(leg["mass_kg"]))
        for link in ("thigh", "shin", "foot"):
            body = bodies[f"{side}_{link}"]
            geom = body.find("geom")
            shell_mass = float(geom.get("mass")) - leg["mass_kg"]
            assert shell_mass > 0
            geom.set("mass", f"{shell_mass:.9f}")
            add_motor(body, f"{side}_{link}", leg["mass_kg"], leg["envelope_m"])
        for link in ("shoulder_yaw_body", "shoulder_roll_body"):
            geom = bodies[f"{side}_{link}"].find("geom")
            extra_mass += arm["mass_kg"] - float(geom.get("mass"))
            geom.set("mass", str(arm["mass_kg"]))
        body = bodies[f"{side}_arm"]
        geom = body.find("geom")
        shell_mass = float(geom.get("mass")) - arm["mass_kg"]
        assert shell_mass > 0
        geom.set("mass", f"{shell_mass:.9f}")
        add_motor(body, f"{side}_shoulder_pitch", arm["mass_kg"], arm["envelope_m"])
        # ROBOTIS XL330-M288-T: 18 g. This hinge/hand packaging is unverified.
        add_motor(bodies[f"{side}_finger"], f"{side}_gripper", 0.018, (0.030, 0.020, 0.015))
        extra_mass += 0.018

    # Keep the old 2.22 kg hypothesis while moving missing servo mass from
    # the torso proxy to the joints. Geometry and total mass stay controlled.
    torso = bodies["torso"].find("geom")
    old_torso_mass = float(torso.get("mass"))
    assert old_torso_mass > extra_mass
    torso.set("mass", f"{old_torso_mass - extra_mass:.9f}")
    for name, actuator in actuators.items():
        if any(part in name for part in ("hip_", "knee", "ankle_")):
            effort = leg["rated_effort_nm"]
        elif "shoulder_" in name:
            effort = arm["rated_effort_nm"]
        else:
            continue  # Head/gripper 0.5 Nm is near XL330 momentary stall, not continuous.
        actuator.set("forcerange", f"-{effort:.6f} {effort:.6f}")
    for geom in root.findall("worldbody/geom"):
        if geom.get("name") == "floor":
            geom.set("friction", "1.0 0.01 0.001")
            geom.set("contype", "1")
            geom.set("conaffinity", "7")
    for body in root.iter("body"):
        internal_joint = body.get("name", "").endswith(
            ("hip_yaw", "hip_roll", "shoulder_yaw_body", "shoulder_roll_body",
             "head_yaw_body")
        )
        for geom in body.findall("geom"):
            if geom.get("contype") == "0" or internal_joint:
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            elif geom.get("name") in (
                "left_sole", "right_sole", "left_hand", "right_hand",
                "left_finger_tip", "right_finger_tip",
            ):
                geom.set("contype", "4")
                geom.set("conaffinity", "7")
                if geom.get("name") in ("left_sole", "right_sole"):
                    geom.set("friction", "1.0 0.01 0.001")
            else:
                # Coarse adjacent-body shells overlap at the concept joints.
                # Keep their world/object/end-effector contacts, not phantom
                # shell-shell forces. End-effectors still collide with shells.
                geom.set("contype", "2")
                geom.set("conaffinity", "5")
    sensors = root.find("sensor")
    for side in ("left", "right"):
        ET.SubElement(
            bodies[f"{side}_foot"], "site", name=f"{side}_sole_touch",
            type="box", pos="0.014 0 -0.032", size="0.045 0.030 0.012",
            rgba="0 0 0 0",
        )
        ET.SubElement(
            sensors, "touch", name=f"{side}_sole_normal_force",
            site=f"{side}_sole_touch",
        )
        ET.SubElement(
            sensors, "contact", name=f"{side}_hand_floor_found",
            geom1=f"{side}_hand", geom2="floor", reduce="mindist",
            num="1", data="found",
        )
    ET.indent(root)
    ET.ElementTree(root).write(DEST, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(DEST))
    assert model.nu == 20
    return model, leg, arm, extra_mass


def main():
    model, leg, arm, extra_mass = build()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.ctrl[:] = data.qpos[7:]
    start_height = float(data.qpos[2])
    lowest = start_height
    for _ in range(round(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        lowest = min(lowest, float(data.qpos[2]))
    nonfloor = []
    floor_normal_n = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        a, b = model.geom(contact.geom1), model.geom(contact.geom2)
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        if a.name == "floor" or b.name == "floor":
            floor_normal_n += float(force[0])
        else:
            nonfloor.append({
                "body_a": model.body(int(a.bodyid[0])).name,
                "body_b": model.body(int(b.bodyid[0])).name,
                "normal_force_n": float(force[0]),
            })
    report = {
        "status": "source_bounded_design_hypothesis_not_measured_robot",
        "model": str(DEST.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
        "source_model_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "compiled": {"actuators": model.nu, "mass_kg": float(model.body_mass.sum())},
        "neutral_2s_rollout": {
            "start_pelvis_z_m": start_height, "end_pelvis_z_m": float(data.qpos[2]),
            "minimum_pelvis_z_m": lowest, "fell_by_height": lowest < 0.65 * start_height,
            "floor_normal_force_n": floor_normal_n,
            "nonfloor_contacts": nonfloor,
        },
        "sourced_motor_proxies": {
            "legs": {"model": "STS3250-C001", "mass_kg_each": leg["mass_kg"],
                     "rated_effort_nm": leg["rated_effort_nm"],
                     "stall_effort_nm": leg["stall_effort_nm"],
                     "no_load_speed_rad_s": leg["no_load_speed_rad_s"],
                     "rating_voltage_v": leg["rating_voltage_v"], "source": leg["source"]},
            "shoulders": {"model": "ST-3215-C018", "mass_kg_each": arm["mass_kg"],
                          "rated_effort_nm": arm["rated_effort_nm"],
                          "stall_effort_nm": arm["stall_effort_nm"],
                          "no_load_speed_rad_s": arm["no_load_speed_rad_s"],
                          "rating_voltage_v": arm["rating_voltage_v"], "source": arm["source"]},
            "head_gripper": {"model": "XL330-M288-T proxy", "mass_kg_each": 0.018,
                             "effort_limit_nm": 0.5, "limit_status": "momentary stall proxy",
                             "source": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/"},
        },
        "still_hypothetical": {
            "total_mass_kg": "old 2.22 kg concept retained, not weighed",
            "mass_reallocated_from_torso_kg": extra_mass,
            "joint_ranges_rad": "concept geometry, not measured stops",
            "floor_friction": "1.0 reference proxy from Microduck, not measured ape sole",
            "position_servo_kp_kv": "unchanged schematic values, not Feetech identification",
            "damping_armature": "unchanged schematic values",
            "self_collision": "coarse shell-shell collisions disabled because source geometry overlaps; end-effector-shell and all floor/object contacts retained",
        },
        "unknown_physical_parameters": {
            "command_delay_s": None,
            "motor_thermal_time_constant_s": None,
            "loaded_torque_speed_curve_nm_per_rad_s": None,
            "joint_backlash_rad": None,
            "sole_floor_contact_stiffness_n_per_m": None,
        },
        "unmodeled": ["torque-speed curve", "voltage sag", "command delay",
                      "backlash", "thermal duty", "case packaging"],
        "candidate_perturbations_not_measured_bounds": {
            "coherent_mass_scale": [0.95, 1.05], "friction_scale": [0.8, 1.2],
            "servo_gain_scale": [0.9, 1.1], "joint_zero_offset_rad": [-0.02, 0.02],
        },
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"model": str(DEST), "report": str(REPORT),
                      "compiled": report["compiled"],
                      "rollout": report["neutral_2s_rollout"]}, indent=2))


if __name__ == "__main__":
    main()
