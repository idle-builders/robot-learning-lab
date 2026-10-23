"""Audit fixed floor-cube contacts at arm-only raised Cove hand entries."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_static_contact import mirror_qpos


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v1/candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"
IK = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/ik-report.json"
SOURCE_STATES = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1/candidate-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-axis-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--narrow-bridge", action="store_true")
    args = parser.parse_args()
    if args.narrow_bridge and args.left_axis_sign != -1:
        raise ValueError("Narrow bridge body already contains left +x axis")
    out = (ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/static-contact-v1"
           if args.narrow_bridge else OUT if args.left_axis_sign == -1 else
           ROOT / "artifacts/ape-cove50-hand-clearance-coupon/left-sign-v1")
    out.mkdir(parents=True, exist_ok=True)
    if json.loads(IK.read_text())["passing"] != 100:
        raise ValueError("Fixed arm-only raised-entry IK did not pass all 100 starts")
    body_path = (ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml"
                 if args.narrow_bridge else BODY)
    root = ET.parse(body_path).getroot()
    if args.left_axis_sign == 1:
        next(node for node in root.iter("joint")
             if node.get("name") == "left_gripper").set("axis", "1 0 0")
    cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected candidate 21-axis body plus free cube")
    with np.load(STATES) as saved:
        poses = saved["qpos"].copy()
        starts = saved["start"].copy()
        raise_mm = saved["raise_mm"].copy()
        passed = saved["screen_pass"].copy()
    with np.load(SOURCE_STATES) as saved:
        contact_height_mm = round(float(saved["toy_xyz_m"][2] * 1000), 3)
    if len(poses) != 100 or not np.all(passed):
        raise ValueError("Expected 100 accepted raised states")
    data = mujoco.MjData(model)
    cube_id = model.geom("audit_cube_geom").id
    floor_id = model.geom("floor").id
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    soles = {model.geom(f"{side}_sole").id for side in ("left", "right")}
    rows = []
    for side in ("right", "left"):
        palm_id = model.geom(f"{side}_hand").id
        finger_id = model.geom(f"{side}_finger_tip").id
        site_id = model.site(f"{side}_hand_center").id
        jaw_adr = model.joint(f"{side}_gripper").qposadr[0]
        for index, (start, height, right_pose) in enumerate(zip(starts, raise_mm, poses)):
            pose = right_pose if side == "right" else mirror_qpos(model, right_pose)
            for cube_height in (10.0, contact_height_mm):
                for dx_mm in (-2, 0, 2):
                    for dy_mm in (-2, 0, 2):
                        for yaw_deg in (0, 15, 30, 45):
                            for jaw_rad in (0.0, 0.25, 0.5):
                                data.qpos[:28] = pose
                                data.qpos[jaw_adr] = jaw_rad
                                mujoco.mj_forward(model, data)
                                hand = data.site_xpos[site_id].copy()
                                data.qpos[cube_adr:cube_adr + 3] = (
                                    hand[0] + dx_mm / 1000,
                                    hand[1] + dy_mm / 1000,
                                    cube_height / 1000)
                                yaw = np.deg2rad(yaw_deg)
                                data.qpos[cube_adr + 3:cube_adr + 7] = (
                                    np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                                mujoco.mj_forward(model, data)
                                jaw = {"palm": [], "finger": []}
                                floor_depth = 0.0
                                other = {}
                                robot = {}
                                for cindex in range(data.ncon):
                                    contact = data.contact[cindex]
                                    a, b = contact.geom1, contact.geom2
                                    distance = float(contact.dist * 1000)
                                    if cube_id in (a, b):
                                        partner = b if a == cube_id else a
                                        if partner == palm_id:
                                            jaw["palm"].append(distance)
                                        elif partner == finger_id:
                                            jaw["finger"].append(distance)
                                        elif partner == floor_id:
                                            floor_depth = max(floor_depth, -distance)
                                        elif distance < -0.1:
                                            name = mujoco.mj_id2name(
                                                model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                                            other[name] = max(other.get(name, 0), -distance)
                                    elif distance < -0.1 and not (
                                        floor_id in (a, b) and
                                        (b if a == floor_id else a) in soles
                                    ):
                                        names = sorted(mujoco.mj_id2name(
                                            model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in (a, b))
                                        key = "/".join(names)
                                        robot[key] = max(robot.get(key, 0), -distance)
                                two_jaw = bool(jaw["palm"] and jaw["finger"])
                                clean = two_jaw and floor_depth <= .1 and not other and not robot
                                rows.append({
                                    "case": f"{side}-ik{int(start)}-raise{int(height)}-z{cube_height}-x{dx_mm}-y{dy_mm}-yaw{yaw_deg}-jaw{jaw_rad}",
                                    "side": side, "source_ik_start": int(start),
                                    "raised_state_index": index, "raise_target_mm": int(height),
                                    "cube_height_mm": cube_height,
                                    "xy_offset_mm": [dx_mm, dy_mm],
                                    "yaw_deg": yaw_deg, "jaw_rad": jaw_rad,
                                    "two_jaw_contact": two_jaw, "clean_two_jaw": clean,
                                    "palm_min_contact_dist_mm": round(min(jaw["palm"]), 3) if jaw["palm"] else None,
                                    "finger_min_contact_dist_mm": round(min(jaw["finger"]), 3) if jaw["finger"] else None,
                                    "cube_floor_penetration_mm": round(floor_depth, 3),
                                    "other_cube_robot_penetration_mm": {
                                        k: round(v, 3) for k, v in other.items()},
                                    "nonsole_floor_or_self_penetration_mm": {
                                        k: round(v, 3) for k, v in robot.items()},
                                })
    if len(rows) != 43200:
        raise ValueError("Fixed raised cube case count changed")
    summary = []
    for side in ("right", "left"):
        for height in (6, 10, 12, 15):
            subset = [row for row in rows if row["side"] == side and
                      row["raise_target_mm"] == height]
            clean = [row for row in subset if row["clean_two_jaw"]]
            closed = [row for row in clean if row["jaw_rad"] > 0]
            clean_grid = {}
            for row in clean:
                key = (row["source_ik_start"], row["cube_height_mm"],
                       *row["xy_offset_mm"], row["yaw_deg"])
                clean_grid.setdefault(key, set()).add(row["jaw_rad"])
            full_sweeps = [key for key, angles in clean_grid.items()
                           if {0.0, 0.25, 0.5} <= angles]
            blockers = Counter(key for row in subset for key in
                               (*row["other_cube_robot_penetration_mm"],
                                *row["nonsole_floor_or_self_penetration_mm"]))
            summary.append({
                "case": f"{side} +{height} mm", "side": side,
                "raise_target_mm": height, "static_cases": len(subset),
                "two_jaw_cases": sum(row["two_jaw_contact"] for row in subset),
                "clean_two_jaw_cases": len(clean),
                "clean_closed_jaw_cases": len(closed),
                "clean_floor_closed_jaw_cases": sum(
                    row["cube_height_mm"] == 10.0 for row in closed),
                "source_ik_starts_with_clean_closed_contact": len({
                    row["source_ik_start"] for row in closed}),
                "clean_open_to_closed_triplets": len(full_sweeps),
                "clean_floor_open_to_closed_triplets": sum(
                    key[1] == 10.0 for key in full_sweeps),
                "clean_floor_height_cases": sum(row["cube_height_mm"] == 10.0
                                                for row in clean),
                "source_ik_starts_with_clean_contact": len({row["source_ik_start"]
                                                            for row in clean}),
                "top_blockers": blockers.most_common(5),
            })
    report = {
        "schema": "robot-learning-lab.cove50-hand-coupon-raised-cube-audit.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(body_path),
        "raised_states_sha256": sha(STATES),
        "ik_report_sha256": sha(IK),
        "plan_sha256": sha(out / "plan.json"),
        "left_gripper_axis_sign": args.left_axis_sign,
        "mj_version": mujoco.__version__,
        "cases": len(rows), "summary": summary,
        "clean_examples": [row for row in rows if row["clean_two_jaw"]][:12],
        "limits": [
            "All arm-only raised postures are static IK from prior whole-body static IK, not attained by native motor control.",
            "The cube is placed at each raised candidate hand site, not approached from outside the aperture.",
            "Static contacts are not grasp force, lift, continuous possession or policy qualification.",
            "The source-derived collision proxies and stock-density mass transfer remain unmeasured model assumptions.",
        ],
    }
    (out / "cube-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-coupon-raised-cube-matrix.v1",
        "cases": rows}, indent=2) + "\n")
    (out / "cube-summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-coupon-raised-cube-summary.v1",
        "cases": summary}, indent=2) + "\n")
    (out / "cube-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
