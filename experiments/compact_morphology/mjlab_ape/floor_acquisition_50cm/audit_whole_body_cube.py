"""Audit 20 mm cube contacts at the fixed whole-body pregrasp IK poses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    tree = ET.parse(BODY).getroot()
    cube = ET.SubElement(tree.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected the selected 21-axis body plus free cube")
    data = mujoco.MjData(model)
    with np.load(OUT / "candidate-states.npz") as saved:
        candidates = saved["qpos"].copy()
        ids = saved["start"].copy()
        ik_pass = saved["screen_pass"].copy()
        toy = saved["toy_xyz_m"].copy()
    cube_id = model.geom("audit_cube_geom").id
    floor_id = model.geom("floor").id
    palm_id = model.geom("right_hand").id
    finger_id = model.geom("right_finger_tip").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    rows = []
    for index, qpos in enumerate(candidates):
        for height_mm in (10.0, round(float(toy[2] * 1000), 3)):
            for dx_mm in (-2, 0, 2):
                for dy_mm in (-2, 0, 2):
                    for yaw_deg in (0, 15, 30, 45):
                        for jaw_rad in (0.0, 0.5):
                            data.qpos[:28] = qpos
                            data.qpos[jaw_adr] = jaw_rad
                            data.qpos[cube_adr:cube_adr + 3] = (
                                toy[0] + dx_mm / 1000,
                                toy[1] + dy_mm / 1000,
                                height_mm / 1000)
                            yaw = np.deg2rad(yaw_deg)
                            data.qpos[cube_adr + 3:cube_adr + 7] = (
                                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                            mujoco.mj_forward(model, data)
                            jaw = {"palm": [], "finger": []}
                            floor_penetration = 0.0
                            other = {}
                            for cindex in range(data.ncon):
                                contact = data.contact[cindex]
                                if cube_id not in (contact.geom1, contact.geom2):
                                    continue
                                partner = (contact.geom2 if contact.geom1 == cube_id
                                           else contact.geom1)
                                distance_mm = float(contact.dist * 1000)
                                if partner == palm_id:
                                    jaw["palm"].append(distance_mm)
                                elif partner == finger_id:
                                    jaw["finger"].append(distance_mm)
                                elif partner == floor_id:
                                    floor_penetration = max(floor_penetration, -distance_mm)
                                elif distance_mm < -0.1:
                                    name = mujoco.mj_id2name(
                                        model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                                    other[name] = max(other.get(name, 0), -distance_mm)
                            two_jaw = bool(jaw["palm"] and jaw["finger"])
                            clean = floor_penetration <= 0.1 and not other
                            rows.append({
                                "ik_start": int(ids[index]), "ik_screen_pass": bool(ik_pass[index]),
                                "cube_height_mm": height_mm, "xy_offset_mm": [dx_mm, dy_mm],
                                "yaw_deg": yaw_deg, "jaw_rad": jaw_rad,
                                "palm_min_contact_dist_mm": round(min(jaw["palm"]), 3) if jaw["palm"] else None,
                                "finger_min_contact_dist_mm": round(min(jaw["finger"]), 3) if jaw["finger"] else None,
                                "floor_penetration_mm": round(floor_penetration, 3),
                                "other_robot_penetration_mm": {k: round(v, 3) for k, v in other.items()},
                                "two_jaw_contact": two_jaw,
                                "clean_two_jaw": two_jaw and clean,
                            })
    clean_rows = [row for row in rows if row["clean_two_jaw"]]
    jaw_rows = [row for row in rows if row["two_jaw_contact"]]
    passing_jaw_rows = [row for row in jaw_rows if row["ik_screen_pass"]]
    forearm_rows = [row for row in jaw_rows
                    if "right_forearm_frame_budget" in row["other_robot_penetration_mm"]]
    report = {
        "schema": "robot-learning-lab.cove50-whole-body-cube-contact-audit.v1",
        "method": "static selected-MJCF cube contact grid around whole-body IK candidate poses",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "body_sha256": sha(BODY),
        "plan_sha256": sha(OUT / "plan.json"),
        "ik_report_sha256": sha(OUT / "report.json"),
        "candidate_states_sha256": sha(OUT / "candidate-states.npz"),
        "cases": len(rows), "ik_passing_poses": int(np.sum(ik_pass)),
        "two_jaw_contact_cases": len(jaw_rows),
        "two_jaw_contact_cases_in_ik_passing_poses": len(passing_jaw_rows),
        "two_jaw_with_forearm_intersection_cases": len(forearm_rows),
        "two_jaw_with_forearm_intersection_cases_in_ik_passing_poses": sum(
            "right_forearm_frame_budget" in row["other_robot_penetration_mm"]
            for row in passing_jaw_rows),
        "clean_two_jaw_cases": len(clean_rows),
        "two_jaw_by_height_mm": {str(z): sum(row["cube_height_mm"] == z for row in jaw_rows)
                                  for z in sorted(set(row["cube_height_mm"] for row in rows))},
        "least_forearm_intersection_mm_at_two_jaw": min(
            (row["other_robot_penetration_mm"].get("right_forearm_frame_budget", 0)
             for row in jaw_rows), default=None),
        "least_forearm_intersection_mm_at_two_jaw_in_ik_passing_poses": min(
            (row["other_robot_penetration_mm"].get("right_forearm_frame_budget", 0)
             for row in passing_jaw_rows), default=None),
        "clean_examples": clean_rows[:10],
        "limits": [
            "These IK states were not attained by continuous native motor control.",
            "Two-jaw contact in a static pose is not compressive force, lift or possession.",
            "The finite cube offset and yaw grid does not prove geometric impossibility outside the sampled range.",
            "The selected model includes the forearm frame-budget collision capsule; any hand revision requires source-backed redesign and reselection.",
        ],
    }
    (OUT / "cube-contact-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-whole-body-cube-contact-matrix.v1",
        "cases": rows}, indent=2) + "\n")
    summary = []
    for index, start in enumerate(ids):
        subset = [row for row in rows if row["ik_start"] == int(start)]
        two_jaw_subset = [row for row in subset if row["two_jaw_contact"]]
        summary.append({
            "case": f"IK start {int(start)}",
            "ik_screen_pass": bool(ik_pass[index]),
            "two_jaw_cases": len(two_jaw_subset),
            "clean_two_jaw_cases": sum(row["clean_two_jaw"] for row in subset),
            "min_forearm_intersection_mm": round(min((
                row["other_robot_penetration_mm"].get("right_forearm_frame_budget", 0)
                for row in two_jaw_subset), default=0), 3),
            "featured": index < 5,
        })
    (OUT / "cube-contact-summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-whole-body-cube-contact-summary.v1",
        "cases": summary}, indent=2) + "\n")
    (OUT / "cube-contact-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
