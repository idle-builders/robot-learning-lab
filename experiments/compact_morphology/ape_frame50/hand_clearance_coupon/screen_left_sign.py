"""Isolate positive left-gripper axis reversal on the unselected hand coupon."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_swept_pocket import (
    JAW_ANGLES_DEG, candidate_shapes, intersection_mm3,
)
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.screen_static_contact import mirror_qpos


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/left-sign-v1"
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v1/candidate-training.xml"
STATES = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/raised-entry-v1/raised-states.npz"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cad_rows() -> list[dict]:
    shapes = candidate_shapes("left", 114)
    rows = []
    for axis_sign in (-1, 1):
        axis = bd.Axis(shapes["drive"].pivot, (axis_sign, 0, 0))
        for angle in JAW_ANGLES_DEG:
            moved = shapes["jaw"].rotate(axis, angle)
            rows.append({
                "axis_sign": axis_sign,
                "jaw_rad": round(angle / 57.29577951308232, 5),
                "frame_overlap_mm3": intersection_mm3(moved, shapes["frame"]),
                "case_overlap_mm3": intersection_mm3(moved, shapes["case"]),
                "open_cube_overlap_mm3": intersection_mm3(moved, shapes["cube"]),
            })
    return rows


def physics_rows() -> list[dict]:
    with np.load(STATES) as saved:
        poses = saved["qpos"].copy()
        heights = saved["raise_mm"].copy()
    poses = poses[heights == 15]
    if len(poses) != 25:
        raise ValueError("Expected 25 +15 mm states")
    rows = []
    for axis_sign in (-1, 1):
        root = ET.parse(BODY).getroot()
        joint = next(n for n in root.iter("joint") if n.get("name") == "left_gripper")
        joint.set("axis", f"{axis_sign} 0 0")
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        jaw_adr = model.joint("left_gripper").qposadr[0]
        root_geom = model.geom("left_jaw_root_frame_budget").id
        bridge_geom = model.geom("left_jaw_bridge_frame_budget").id
        floor = model.geom("floor").id
        palm = model.geom("left_hand").id
        finger = model.geom("left_finger_tip").id
        for pose_index, right_pose in enumerate(poses):
            pose = mirror_qpos(model, right_pose)
            for jaw_rad in (0, .25, .5):
                data.qpos[:] = pose
                data.qpos[jaw_adr] = jaw_rad
                mujoco.mj_forward(model, data)
                floor_hits = {}
                for c in data.contact[:data.ncon]:
                    if floor in (c.geom1, c.geom2) and c.dist < -.0001:
                        g = c.geom2 if c.geom1 == floor else c.geom1
                        if g in (root_geom, bridge_geom):
                            floor_hits[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)] = max(
                                floor_hits.get(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g), 0),
                                -1000 * float(c.dist))
                rows.append({
                    "axis_sign": axis_sign, "pose_index": pose_index,
                    "jaw_rad": jaw_rad,
                    "root_or_bridge_floor_penetration_mm": {
                        k: round(v, 3) for k, v in floor_hits.items()},
                    "pad_center_gap_mm": round(float(np.linalg.norm(
                        data.geom_xpos[palm] - data.geom_xpos[finger])) * 1000, 3),
                    "palm_center_z_mm": round(float(data.geom_xpos[palm][2]) * 1000, 3),
                    "finger_center_z_mm": round(float(data.geom_xpos[finger][2]) * 1000, 3),
                })
    return rows


def main() -> None:
    cad = cad_rows()
    physics = physics_rows()
    summary = []
    for sign in (-1, 1):
        c = [r for r in cad if r["axis_sign"] == sign]
        p = [r for r in physics if r["axis_sign"] == sign]
        summary.append({
            "axis_sign": sign,
            "max_cad_frame_overlap_mm3": max(r["frame_overlap_mm3"] for r in c),
            "max_cad_case_overlap_mm3": max(r["case_overlap_mm3"] for r in c),
            "max_cad_cube_overlap_mm3": max(r["open_cube_overlap_mm3"] for r in c),
            "root_or_bridge_floor_hit_cases": sum(bool(r["root_or_bridge_floor_penetration_mm"]) for r in p),
            "max_root_or_bridge_floor_penetration_mm": max(
                [v for r in p for v in r["root_or_bridge_floor_penetration_mm"].values()] or [0]),
            "pad_gap_mm_by_jaw": {str(a): [min(r["pad_center_gap_mm"] for r in p if r["jaw_rad"] == a),
                                            max(r["pad_center_gap_mm"] for r in p if r["jaw_rad"] == a)]
                                    for a in (0, .25, .5)},
        })
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "robot-learning-lab.cove50-left-sign-screen.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "raised_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "summary": summary,
        "limits": [
            "A reversed left joint axis changes the policy action and actuator sign contract.",
            "The 25 poses are mirrored static IK, not motor-attained trajectories.",
            "No cube approach, grasp force, lift, or possession is tested here.",
        ],
    }
    (OUT / "cad-matrix.json").write_text(json.dumps({"cases": cad}, indent=2) + "\n")
    (OUT / "physics-matrix.json").write_text(json.dumps({"cases": physics}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
