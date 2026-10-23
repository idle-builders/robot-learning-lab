"""Static selected-body contact screen for the 20 mm cube after reach IK.

The arm IK pose has not been attained by a policy. Contact geometry here is
diagnostic only; it is not a grasp or motor-feasible teacher demonstration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ENTRY = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/pregrasp-ik-v1/step-feedback-global.npz"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/floor-pinch-screen-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="screen_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="screen_cube_free")
    ET.SubElement(cube, "geom", name="screen_cube_geom", type="box", size="0.01 0.01 0.01",
                  mass="0.00625", friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected selected 21-axis body plus one free cube")
    data = mujoco.MjData(model)
    with np.load(ENTRY) as entry:
        robot_qpos = entry["candidate_qpos"]
        source_toy = entry["toy_xyz_m"]
    data.qpos[:28] = robot_qpos
    jaw = model.joint("right_gripper")
    data.qpos[jaw.qposadr[0]] = jaw.range[1]
    toy_qadr = model.joint("screen_cube_free").qposadr[0]
    ids = {name: model.geom(name).id for name in
           ("right_hand", "right_finger_tip", "screen_cube_geom", "floor")}

    def screen_case(z_mm: float, x_mm: int, y_mm: int, yaw_deg: int) -> dict:
        yaw = np.deg2rad(yaw_deg)
        data.qpos[toy_qadr:toy_qadr + 3] = (
            source_toy[0] + x_mm / 1000, source_toy[1] + y_mm / 1000, z_mm / 1000)
        data.qpos[toy_qadr + 3:toy_qadr + 7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        contact = {"right_hand": [], "right_finger_tip": []}
        floor_penetration = 0.0
        other_robot_penetration = []
        for i in range(data.ncon):
            c = data.contact[i]
            pair = {c.geom1, c.geom2}
            if ids["screen_cube_geom"] not in pair:
                continue
            partner = next(iter(pair - {ids["screen_cube_geom"]}))
            if partner == ids["floor"]:
                floor_penetration = max(floor_penetration, -float(c.dist))
            elif partner in (ids["right_hand"], ids["right_finger_tip"]):
                name = "right_hand" if partner == ids["right_hand"] else "right_finger_tip"
                contact[name].append(round(float(c.dist) * 1000, 4))
            elif c.dist < 0:
                other_robot_penetration.append(mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_GEOM, partner))
        return {
            "cube_center_z_mm": z_mm, "xy_offset_mm": [x_mm, y_mm], "yaw_deg": yaw_deg,
            "fixed_jaw_contact": bool(contact["right_hand"]),
            "moving_jaw_contact": bool(contact["right_finger_tip"]),
            "minimum_fixed_jaw_dist_mm": min(contact["right_hand"], default=None),
            "minimum_moving_jaw_dist_mm": min(contact["right_finger_tip"], default=None),
            "cube_floor_penetration_mm": round(1000 * floor_penetration, 4),
            "other_robot_penetration": sorted(set(other_robot_penetration)),
        }

    heights = (10, 11.34, 15, 20, 25, 30, 35)
    rows = [screen_case(z, x, y, yaw) for z in (10, 11.34)
            for x in range(-8, 9) for y in range(-8, 9)
            for yaw in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)]
    rows.extend(screen_case(z, x, y, yaw) for z in heights[2:]
            for x in range(-20, 21, 2) for y in range(-16, 17, 2)
            for yaw in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45))
    both = [row for row in rows if row["fixed_jaw_contact"] and row["moving_jaw_contact"]]
    clean = [row for row in both if not row["other_robot_penetration"]]
    report = {
        "schema": "robot-learning-lab.cove50-floor-pinch-screen.v1",
        "method": "static MuJoCo contact detection at closed selected jaw, varying cube XY/yaw/height",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": sha(BODY), "ik_entry_sha256": sha(ENTRY),
        "cube_half_size_m": 0.01, "cube_mass_kg": 0.00625,
        "source_toy_xyz_m": source_toy.tolist(), "grid_cases": len(rows),
        "floor_cases": sum(row["cube_center_z_mm"] in (10, 11.34) for row in rows),
        "floor_both_jaw_contact_cases": sum(row["cube_center_z_mm"] in (10, 11.34) for row in both),
        "both_jaw_contact_cases": len(both),
        "both_jaw_no_other_robot_penetration_cases": len(clean),
        "both_jaw_by_center_height_mm": {str(z): sum(row["cube_center_z_mm"] == z for row in both)
                                          for z in heights},
        "best_candidate_rows": sorted(clean, key=lambda row: (
            row["cube_center_z_mm"], max(abs(v) for v in row["xy_offset_mm"]),
            row["yaw_deg"]))[:20],
        "limits": [
            "The robot arm is placed at an unattained static IK pose; no continuous transition or balance is shown.",
            "Cube center z=10 and 11.34 mm represent floor and saved contact heights; higher rows are an alignment diagnostic.",
            "The gripper is set to its closed joint limit before contact testing.",
            "Contact detection does not demonstrate pinch force, lift, or retained possession.",
            "Cube size/mass match the native reach task, but this XML screen does not instantiate full MJLab.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps({"cases": rows}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("grid_cases", "floor_both_jaw_contact_cases",
                       "both_jaw_by_center_height_mm", "best_candidate_rows")}, indent=2))


if __name__ == "__main__":
    main()
