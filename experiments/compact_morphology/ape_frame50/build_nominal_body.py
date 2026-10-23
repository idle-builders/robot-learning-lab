"""Generate a runnable *provisional* 50 cm ape anatomy body from frame_contract.

Mass allocation and position-servo dynamics here are design assumptions until
the bare frame has real supports and measured actuator response. They are
deliberately not described as CAD-derived or physically qualified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def fmt(values):
    return " ".join(f"{float(x):.8g}" for x in values)


def child(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def geom(parent, name, kind, mass, color, **attrs):
    return child(parent, "geom", name=name, type=kind, mass=f"{mass:.7g}",
                 rgba=fmt(color), **attrs)


def add_joint(parent, name, axis, lower, upper):
    child(parent, "joint", name=name, axis=fmt(axis), range=fmt((lower, upper)),
          damping="0.012", armature="0.001")


def add_motor(parent, name, center_world, body_origin_world, size, mass):
    pos = np.asarray(center_world) - np.asarray(body_origin_world)
    geom(parent, name + "_motor_case", "box", mass, (.28, .46, .64, 1),
         pos=fmt(pos), size=fmt(np.asarray(size) / 2),
         contype="0", conaffinity="0")


def main(leg_candidate=False, supported_leg=False, supported_body=False):
    if sum((leg_candidate, supported_leg, supported_body)) > 1:
        raise ValueError("choose one candidate variant")
    cfg_path = HERE / "frame_contract.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    variant = "supported-body" if supported_body else "supported-leg" if supported_leg else "leg-kinematics" if leg_candidate else "bare"
    candidate_path = HERE / ("leg_kinematics_supported.json" if supported_leg or supported_body else "leg_kinematics_candidate.json")
    candidate = json.loads(candidate_path.read_text()) if variant != "bare" else None
    upper_path = HERE / "upper_kinematics_supported.json"
    upper = json.loads(upper_path.read_text()) if supported_body else None
    mass_path = HERE / "mass_ledger_candidate.json"
    ledger = json.loads(mass_path.read_text()) if supported_body else None
    def frame_mass(*stages):
        return sum(ledger["stages"][name]["frame_mass_kg"] for name in stages)
    def frame_center(stage, sign):
        p = np.asarray(ledger["stages"][stage]["frame_center_left_m"])
        return np.array((p[0], sign*p[1], p[2]))
    def mass_marker(parent, name, mass, center_world, body_origin_world):
        geom(parent, name, "box", mass, (.43,.48,.46,1),
             pos=fmt(np.asarray(center_world)-np.asarray(body_origin_world)),
             size="0.008 0.008 0.008", contype="0", conaffinity="0")
    out = ROOT / f"artifacts/ape-frame50-{variant}"
    x = ET.Element("mujoco", model=f"ape_frame50_{variant.replace('-', '_')}_unqualified")
    child(x, "compiler", angle="radian", autolimits="true")
    child(x, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 -9.81")
    default = child(x, "default")
    child(default, "geom", condim="3", friction="0.8 0.01 0.001")
    child(default, "joint", limited="true")
    world = child(x, "worldbody")
    child(world, "light", pos="0.4 -0.4 1.2", dir="0 0 -1", directional="true")
    child(world, "geom", name="floor", type="plane", size="2 2 0.1",
          rgba="0.7 0.7 0.68 1", friction="0.9 0.01 0.001")

    pelvis_origin = np.array([0., 0., .231])
    pelvis = child(world, "body", name="pelvis", pos=fmt(pelvis_origin))
    child(pelvis, "freejoint", name="root")
    geom(pelvis, "pelvis_frame_budget", "box", frame_mass("pelvis_shared","leg_hip_fixed","leg_hip_fixed") if ledger else .350, (.48, .52, .54, 1),
         size="0.037 0.065 0.022")
    imu = np.asarray(cfg["sensors"]["body_imu"]["frame_position_m"])
    child(pelvis, "site", name="imu", pos=fmt(imu - pelvis_origin), size="0.004")
    # A source-sized candidate battery is separately visible, but its 107 g
    # remains within the 350 g pelvis allocation, rather than double counted.
    battery = cfg["battery"]
    child(pelvis, "site", name="battery_4s_center",
          pos=fmt(np.asarray(battery["provisional_center_m"]) - pelvis_origin),
          size="0.004")
    if ledger:
        geom(pelvis, "battery_4s850", "box", .107, (.38,.47,.39,1),
             pos=fmt(np.asarray(battery["provisional_center_m"])-pelvis_origin),
             size=fmt(np.asarray(battery["envelope_m"])/2),contype="0",conaffinity="0")
        mass_marker(pelvis,"imu_mass_reserve",.005,imu,pelvis_origin)

    cases = cfg["actuators"]["provisional_case_centers_m_left"]
    torque = {}
    gains = {}
    for side, sign in (("left", 1), ("right", -1)):
        point = {k: np.asarray((v[0], sign * v[1], v[2]), dtype=float)
                 for k, v in cfg["geometry"]["left"].items()}
        case = {k: np.asarray((v[0], sign * v[1], v[2]), dtype=float)
                for k, v in cases.items() if k != "head_yaw"}
        hip, knee, ankle = point["hip_m"], point["knee_m"], point["ankle_m"]
        hip_yaw = hip
        motor_sizes = {}
        if candidate:
            for name, value in candidate["left_joint_m"].items():
                point[name] = np.asarray((value[0], sign * value[1], value[2]), dtype=float)
            for name, value in candidate["left_motor_center_m"].items():
                case[name] = np.asarray((value[0], sign * value[1], value[2]), dtype=float)
            motor_sizes = candidate["motor_size_m"]
            hip_yaw, hip, knee = point["hip_yaw"], point["hip_pitch"], point["knee"]
            ankle = point.get("ankle_pitch", ankle)
        shoulder, elbow, hand = (point[k] for k in
                                  ("shoulder_m", "elbow_m", "hand_center_reference_m"))
        hy = child(pelvis, "body", name=f"{side}_hip_yaw_body", pos=fmt(hip_yaw - pelvis_origin))
        add_joint(hy, f"{side}_hip_yaw", (0, 0, 1), -1.2, 1.2)
        if ledger: mass_marker(hy,f"{side}_hip_yaw_frame",frame_mass("leg_hip_yaw"),
                               frame_center("leg_hip_yaw",sign),hip_yaw)
        add_motor(pelvis, f"{side}_hip_yaw", case["hip_yaw"], pelvis_origin,
                  motor_sizes.get("hip_yaw", cfg["actuators"]["leg"]["case_m"]), .0745)
        hr = child(hy, "body", name=f"{side}_hip_roll_body")
        add_joint(hr, f"{side}_hip_roll", (1, 0, 0), -.8, .8)
        if ledger: mass_marker(hr,f"{side}_hip_roll_frame",frame_mass("leg_hip_roll"),
                               frame_center("leg_hip_roll",sign),hip_yaw)
        add_motor(hy, f"{side}_hip_roll", case["hip_roll"], hip_yaw,
                  motor_sizes.get("hip_roll", cfg["actuators"]["leg"]["case_m"]), .0745)
        thigh = child(hr, "body", name=f"{side}_thigh", pos=fmt(hip - hip_yaw))
        add_joint(thigh, f"{side}_hip_pitch", (0, 1, 0), -1.5, 1.5)
        add_motor(hr, f"{side}_hip_pitch", case["hip_pitch"], hip_yaw,
                  motor_sizes.get("hip_pitch", cfg["actuators"]["leg"]["case_m"]), .0745)
        geom(thigh, f"{side}_thigh_frame_budget", "capsule", 0 if ledger else .120, (.55, .63, .64, 1),
             fromto=fmt((0, 0, 0, *(knee - hip))), size="0.012")
        if ledger:
            # Combined hip output pad plus knee drive/thigh stock. These are
            # gross-stock masses; case masses remain on their parent links.
            a,b=(ledger["stages"][s] for s in ("leg_hip_thigh","leg_thigh"))
            ma,mb=a["frame_mass_kg"],b["frame_mass_kg"]
            ca,cb=frame_center("leg_hip_thigh",sign),frame_center("leg_thigh",sign)
            mass_marker(thigh,f"{side}_thigh_stock_mass",ma+mb,(ma*ca+mb*cb)/(ma+mb),hip)
        shin = child(thigh, "body", name=f"{side}_shin", pos=fmt(knee - hip))
        add_joint(shin, f"{side}_knee", (0, 1, 0), -.05,
                  candidate.get("knee_upper_rad", 2.3) if candidate else 2.3)
        add_motor(thigh, f"{side}_knee", case["knee"], hip,
                  motor_sizes.get("knee", cfg["actuators"]["leg"]["case_m"]), .0745)
        geom(shin, f"{side}_shin_frame_budget", "capsule", 0 if ledger else .100, (.55, .63, .64, 1),
             fromto=fmt((0, 0, 0, *(ankle - knee))), size="0.010")
        if ledger:
            a,b=(ledger["stages"][s] for s in ("leg_shin","leg_ankle_fixed"))
            ma,mb=a["frame_mass_kg"],b["frame_mass_kg"]
            ca,cb=frame_center("leg_shin",sign),frame_center("leg_ankle_fixed",sign)
            mass_marker(shin,f"{side}_shin_stock_mass",ma+mb,(ma*ca+mb*cb)/(ma+mb),knee)
        foot = child(shin, "body", name=f"{side}_foot", pos=fmt(ankle - knee))
        ankle_range = candidate.get("ankle_pitch_range_rad", [-1.5, 1.5]) if candidate else [-1.5, 1.5]
        add_joint(foot, f"{side}_ankle_pitch", (0, 1, 0), *ankle_range)
        add_motor(shin, f"{side}_ankle_pitch", case["ankle_pitch"], knee,
                  motor_sizes.get("ankle_pitch", cfg["actuators"]["leg"]["case_m"]),
                  candidate.get("ankle_motor_mass_kg", .0745) if candidate else .0745)
        # The exposed CAD plate is 120 mm fore-aft, 70 mm lateral and 8 mm
        # thick, centered 12 mm forward and 39 mm below the ankle axis. The
        # previous 12 mm box was a walking-contact proxy, not the plate.
        sole_pos = "0.012 0 -0.039" if supported_body else "0.01 0 -0.037"
        sole_size = "0.060 0.035 0.004" if supported_body else "0.060 0.035 0.006"
        geom(foot, f"{side}_sole", "box", frame_mass("leg_foot") if ledger else .130, (.32, .36, .36, 1),
             pos=sole_pos, size=sole_size,
             friction="0.9 0.01 0.001")
        child(foot, "site", name=f"{side}_sole_center",
              pos="0.012 0 -0.043" if supported_body else "0.01 0 -0.043",
              size="0.004")
        for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch"):
            torque[f"{side}_{j}"] = (candidate.get("ankle_rated_effort_nm", 1.569064)
                                     if candidate and j == "ankle_pitch" else 1.569064)
            gains[f"{side}_{j}"] = (30, .8)

    torso = child(pelvis, "body", name="torso")
    geom(torso, "torso_frame_compute_budget", "capsule", frame_mass("upper_torso","upper_fixed","upper_fixed")+.150 if ledger else .450, (.60, .65, .64, 1),
         fromto="0 0 0 0 0 0.166", size="0.022")
    head_yaw_pos = np.asarray(upper["neck_yaw_m"] if upper else cfg["geometry"]["neck_yaw_m"])
    neck = child(torso, "body", name="head_yaw_body", pos=fmt(head_yaw_pos - pelvis_origin))
    add_joint(neck, "head_yaw", (0, 0, 1), -1.5, 1.5)
    add_motor(torso, "head_yaw", upper["head_motor_center_m"] if upper else cases["head_yaw"], pelvis_origin,
              upper["head_motor_size_m"] if upper else cfg["actuators"]["hand_and_head"]["case_m"], .018)
    head_center = np.asarray(cfg["geometry"]["head_center_m"])
    head = child(neck, "body", name="head", pos=fmt(head_center - head_yaw_pos))
    geom(head, "head_frame_sensor_budget", "sphere", frame_mass("upper_head") if ledger else .130, (.74, .76, .72, 1),
         size="0.043", contype="0", conaffinity="0")
    for view in ("upper", "lower"):
        spec = cfg["sensors"]["cameras"][f"{view}_frame"]
        pitch = np.deg2rad(spec["pitch_up_deg"] if view == "upper" else -spec["pitch_down_deg"])
        up = (-np.sin(pitch), 0, np.cos(pitch))
        child(head, "camera", name=f"{view}_rgb", pos=fmt(np.asarray(spec["position_m"]) - head_center),
              xyaxes=fmt((0, -1, 0, *up)), fovy="67")
        if ledger:
            geom(head,f"{view}_camera_module","box",.004,(.20,.24,.25,1),
                 pos=fmt(np.asarray(spec["position_m"])-head_center),
                 size=fmt(np.asarray(cfg["sensors"]["cameras"]["envelope_m_each"])/2),
                 contype="0",conaffinity="0")
    torque["head_yaw"] = .10  # Explicit 20%-of-stall screening heuristic, not rated duty.
    gains["head_yaw"] = (8, .12)

    for side, sign in (("left", 1), ("right", -1)):
        point = {k: np.asarray((v[0], sign * v[1], v[2]), dtype=float)
                 for k, v in cfg["geometry"]["left"].items()}
        case = {k: np.asarray((v[0], sign * v[1], v[2]), dtype=float)
                for k, v in cases.items() if k != "head_yaw"}
        upper_sizes = {}
        ranges = {}
        if upper:
            for name, value in upper["left_joint_m"].items():
                point[name] = np.asarray((value[0], sign * value[1], value[2]), dtype=float)
            for name, value in upper["left_motor_center_m"].items():
                case[name] = np.asarray((value[0], sign * value[1], value[2]), dtype=float)
            upper_sizes = upper["motor_size_m"]
            ranges = upper["candidate_joint_range_rad"]
            shoulder_yaw = point["shoulder_yaw"]
            shoulder_roll = point["shoulder_roll"]
            shoulder_pitch = point["shoulder_pitch"]
            elbow = point["elbow_pitch"]
            hand = point["gripper"]
        else:
            shoulder_yaw, elbow, hand = (point[k] for k in
                                          ("shoulder_m", "elbow_m", "hand_center_reference_m"))
            shoulder_roll = shoulder_yaw
            shoulder_pitch = shoulder_yaw
        sy = child(torso, "body", name=f"{side}_shoulder_yaw_body",
                   pos=fmt(shoulder_yaw - pelvis_origin))
        add_joint(sy, f"{side}_shoulder_yaw", (0, 0, 1), *ranges.get("shoulder_yaw", (-2, 2)))
        if ledger: mass_marker(sy,f"{side}_shoulder_yaw_frame",frame_mass("upper_yaw"),
                               frame_center("upper_yaw",sign),shoulder_yaw)
        add_motor(torso, f"{side}_shoulder_yaw", case["shoulder_yaw"], pelvis_origin,
                  upper_sizes.get("shoulder_yaw", cfg["actuators"]["arm"]["case_m"]), .055)
        sr = child(sy, "body", name=f"{side}_shoulder_roll_body", pos=fmt(shoulder_roll - shoulder_yaw))
        add_joint(sr, f"{side}_shoulder_roll", (1, 0, 0), *ranges.get("shoulder_roll", (-1.6, 1.6)))
        if ledger: mass_marker(sr,f"{side}_shoulder_roll_frame",frame_mass("upper_pitch"),
                               frame_center("upper_pitch",sign),shoulder_roll)
        add_motor(sy, f"{side}_shoulder_roll", case["shoulder_roll"], shoulder_yaw,
                  upper_sizes.get("shoulder_roll", cfg["actuators"]["arm"]["case_m"]), .055)
        arm = child(sr, "body", name=f"{side}_upper_arm", pos=fmt(shoulder_pitch - shoulder_roll))
        add_joint(arm, f"{side}_shoulder_pitch", (0, 1, 0), *ranges.get("shoulder_pitch", (-2.2, 2.2)))
        if ledger: mass_marker(arm,f"{side}_shoulder_pitch_frame",frame_mass("upper_arm"),
                               frame_center("upper_arm",sign),shoulder_pitch)
        add_motor(sr, f"{side}_shoulder_pitch", case["shoulder_pitch"], shoulder_roll,
                  upper_sizes.get("shoulder_pitch", cfg["actuators"]["arm"]["case_m"]), .055)
        geom(arm, f"{side}_upper_arm_frame_budget", "capsule", 0 if ledger else .070, (.67, .62, .57, 1),
             fromto=fmt((0, 0, 0, *(elbow - shoulder_pitch))), size="0.011")
        forearm = child(arm, "body", name=f"{side}_forearm", pos=fmt(elbow - shoulder_pitch))
        add_joint(forearm, f"{side}_elbow_pitch", (0, 1, 0), *ranges.get("elbow_pitch", (-2.2, .15)))
        if ledger: mass_marker(forearm,f"{side}_forearm_stock_mass",frame_mass("upper_forearm"),
                               frame_center("upper_forearm",sign),elbow)
        add_motor(arm, f"{side}_elbow_pitch", case["elbow_pitch"], shoulder_pitch,
                  upper_sizes.get("elbow_pitch", cfg["actuators"]["arm"]["case_m"]), .055)
        geom(forearm, f"{side}_forearm_frame_budget", "capsule", 0 if ledger else .065, (.67, .62, .57, 1),
             fromto=fmt((0, 0, 0, *(hand - elbow))), size="0.010")
        geom(forearm, f"{side}_hand", "box", .002 if ledger else .022, (.75, .64, .49, 1),
             pos=fmt(hand - elbow + np.array([0, -sign * .016, 0])),
             size="0.011 0.003 0.013")
        finger = child(forearm, "body", name=f"{side}_moving_jaw",
                       pos=fmt(hand - elbow + np.array([0, sign * .016, 0])))
        add_joint(finger, f"{side}_gripper", (-1, 0, 0), 0, .5)
        add_motor(forearm, f"{side}_gripper", case["gripper"], elbow,
                  upper_sizes.get("gripper", cfg["actuators"]["hand_and_head"]["case_m"]), .018)
        geom(finger, f"{side}_finger_tip", "box", .005 if ledger else .008, (.75, .64, .49, 1),
             size="0.011 0.003 0.013")
        child(forearm, "site", name=f"{side}_hand_center", pos=fmt(hand - elbow), size="0.004")
        for j in ("shoulder_yaw", "shoulder_roll", "shoulder_pitch", "elbow_pitch"):
            torque[f"{side}_{j}"] = .980665
            gains[f"{side}_{j}"] = (20, .4)
        torque[f"{side}_gripper"] = .10
        gains[f"{side}_gripper"] = (8, .12)

    contacts = child(x, "contact")
    for side in ("left", "right"):
        # The thigh capsule starts within the parent pelvis stock at the hip
        # attachment. These adjacent-link proxies must not repel each other.
        child(contacts, "exclude", body1="pelvis", body2=f"{side}_thigh")

    actuators = child(x, "actuator")
    for name, cap in torque.items():
        kp, kv = gains[name]
        child(actuators, "position", name=name + "_servo", joint=name,
              kp=str(kp), kv=str(kv), forcerange=fmt((-cap, cap)))
    sensors = child(x, "sensor")
    child(sensors, "gyro", name="gyro", site="imu")
    child(sensors, "velocimeter", name="local_linvel", site="imu")
    child(sensors, "accelerometer", name="accelerometer", site="imu")
    child(sensors, "framezaxis", name="upvector", objtype="site", objname="imu")
    child(sensors, "framelinvel", name="global_linvel", objtype="site", objname="imu")
    # The upstream MJLab ape importer strips these scene-specific contact
    # sensors and its own floor, then installs the same reference task sensors.
    for side in ("left", "right"):
        child(sensors, "framelinvel", name=f"{side}_sole_center_global_linvel",
              objtype="site", objname=f"{side}_sole_center")
        for part, contact_geom in (("sole", f"{side}_sole"),
                                   ("hand", f"{side}_hand")):
            child(sensors, "contact", name=f"{side}_{part}_floor_found",
                  geom1=contact_geom, geom2="floor", reduce="mindist",
                  num="1", data="found")

    out.mkdir(parents=True, exist_ok=True)
    path = out / ("ape-frame50-nominal.xml" if variant == "bare" else f"ape-frame50-{variant}.xml")
    ET.indent(x, space="  ")
    ET.ElementTree(x).write(path, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(path))
    report = {"stage": ("runnable CAD-axis and gross-stock-mass candidate; drive dynamics and manufacturing unqualified" if supported_body else
                         "runnable candidate joint geometry, NOT mass/drive-derived or learned-skill-qualified" if candidate else
                         "runnable nominal anatomy, NOT assembled/CAD-derived or learned-skill-qualified"),
              "model": str(path.relative_to(ROOT)),
              "contract_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
              "leg_candidate_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest() if candidate else None,
              "upper_candidate_sha256": hashlib.sha256(upper_path.read_bytes()).hexdigest() if upper else None,
              "mass_ledger_sha256": hashlib.sha256(mass_path.read_bytes()).hexdigest() if ledger else None,
              "mjcf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "powered_axes": model.nu, "joints_including_root": model.njnt,
              "compiled_mass_kg": round(float(model.body_mass.sum()), 5),
              "mass_status": ("gross CAD stock/source component ledger plus 150 g unselected compute/power reserve; manufactured mass, link inertia and wiring not measured" if ledger else
                             "nominal 3.199 kg link allocation hypothesis" +
                             (" with two 55 g ankle motors replacing two 74.5 g nominal cases; frame/compute distribution unmeasured" if supported_leg else
                              "; source motor masses, unmeasured frame/compute distribution")),
              "leg_geometry_status": ("hip/knee/ankle axes, neutral cases and foot plate dimensions/center from CAD; sole friction and link inertia still approximate" if supported_body else
                                      "hip/knee/ankle axes and neutral cases from CAD candidate; foot stock/contact and nonmotor mass/inertia still nominal" if supported_leg else
                                      "hip/knee axes and neutral case envelopes from CAD candidate; ankle, foot, mass/inertia and drive law still nominal") if candidate else "anatomical nominal axes and cases",
              "motor_law_status": "ideal position servos capped at catalog rated Feetech points" +
                                  (" including ST3215 ankle" if supported_leg or supported_body else "") +
                                  "; XL330 0.10 Nm heuristic; no speed/thermal/delay model",
              "upper_geometry_status": "shoulder yaw/roll/pitch and elbow axes, cases and gross-stock frame mass from exposed CAD; link inertia and transmission still approximate" if upper else "nominal arm placeholders",
              "camera_names": ["upper_rgb", "lower_rgb"]}
    (out / "nominal-body-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--leg-candidate", action="store_true", help="use sampled CAD hip/knee axes and neutral cases")
    parser.add_argument("--supported-leg", action="store_true", help="also use sampled knee-to-ankle package")
    parser.add_argument("--supported-body", action="store_true", help="use exposed upper frame and supported leg candidate")
    args = parser.parse_args()
    main(args.leg_candidate, args.supported_leg, args.supported_body)
