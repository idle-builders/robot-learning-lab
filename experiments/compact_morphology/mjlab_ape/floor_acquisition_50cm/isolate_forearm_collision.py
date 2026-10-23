"""Replay the fixed pregrasp grid with one diagnostic collision ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
ENTRY = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1"
OUT = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/forearm-isolation-v1"
VISUAL = ROOT / "artifacts/ape-frame50-direct-skeleton/ape-frame50-direct-visual.xml"
FRAME_MESH = ROOT / "artifacts/ape-frame50-direct-skeleton/meshes/right_forearm_frame.obj"
MOTOR_MESH = ROOT / "artifacts/ape-frame50-direct-skeleton/meshes/right_forearm_right_gripper_XL330_case.obj"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_model(disable_forearm: bool) -> mujoco.MjModel:
    root = ET.parse(BODY).getroot()
    if disable_forearm:
        matches = root.findall(".//geom[@name='right_forearm_frame_budget']")
        if len(matches) != 1:
            raise ValueError("Expected one selected right forearm collision geom")
        matches[0].set("contype", "0")
        matches[0].set("conaffinity", "0")
    cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected selected 21-axis body plus free cube")
    return model


def replay(model: mujoco.MjModel, rows: list[dict], poses: dict[int, np.ndarray],
           toy: np.ndarray) -> list[dict]:
    data = mujoco.MjData(model)
    cube = model.geom("audit_cube_geom").id
    floor = model.geom("floor").id
    palm = model.geom("right_hand").id
    finger = model.geom("right_finger_tip").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    results = []
    for row in rows:
        data.qpos[:28] = poses[row["ik_start"]]
        data.qpos[jaw_adr] = row["jaw_rad"]
        dx, dy = row["xy_offset_mm"]
        data.qpos[cube_adr:cube_adr + 3] = (
            toy[0] + dx / 1000, toy[1] + dy / 1000, row["cube_height_mm"] / 1000)
        yaw = np.deg2rad(row["yaw_deg"])
        data.qpos[cube_adr + 3:cube_adr + 7] = (
            np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        mujoco.mj_forward(model, data)
        jaw = {"palm": [], "finger": []}
        floor_depth = 0.0
        other = {}
        for index in range(data.ncon):
            contact = data.contact[index]
            if cube not in (contact.geom1, contact.geom2):
                continue
            partner = contact.geom2 if contact.geom1 == cube else contact.geom1
            distance = float(contact.dist * 1000)
            if partner == palm:
                jaw["palm"].append(distance)
            elif partner == finger:
                jaw["finger"].append(distance)
            elif partner == floor:
                floor_depth = max(floor_depth, -distance)
            elif distance < -0.1:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                other[name] = max(other.get(name, 0.0), -distance)
        two_jaw = bool(jaw["palm"] and jaw["finger"])
        results.append({
            "ik_start": row["ik_start"],
            "ik_screen_pass": row["ik_screen_pass"],
            "cube_height_mm": row["cube_height_mm"],
            "xy_offset_mm": row["xy_offset_mm"],
            "yaw_deg": row["yaw_deg"],
            "jaw_rad": row["jaw_rad"],
            "two_jaw_contact": two_jaw,
            "clean_two_jaw": two_jaw and floor_depth <= 0.1 and not other,
            "floor_penetration_mm": round(floor_depth, 3),
            "other_robot_penetration_mm": {k: round(v, 3) for k, v in other.items()},
            "palm_min_contact_dist_mm": round(min(jaw["palm"]), 3) if jaw["palm"] else None,
            "finger_min_contact_dist_mm": round(min(jaw["finger"]), 3) if jaw["finger"] else None,
        })
    return results


def probe_visual_frame(rows: list[dict], poses: dict[int, np.ndarray],
                       toy: np.ndarray) -> dict:
    model = mujoco.MjModel.from_xml_path(str(VISUAL))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Visual mesh body differs from the selected 21-axis qpos interface")
    selected_model = mujoco.MjModel.from_xml_path(str(BODY))
    selected_joints = [mujoco.mj_id2name(selected_model, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(selected_model.njnt)]
    visual_joints = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                     for i in range(model.njnt)]
    if selected_joints != visual_joints:
        raise ValueError("Visual mesh body uses a different ordered joint map")
    data = mujoco.MjData(model)
    forearm = model.body("right_forearm").id
    jaw_adr = model.joint("right_gripper").qposadr[0]
    meshes = {"forearm_frame": trimesh.load_mesh(FRAME_MESH),
              "gripper_motor_case": trimesh.load_mesh(MOTOR_MESH)}
    if not all(mesh.is_watertight for mesh in meshes.values()):
        raise ValueError("Selected frame or gripper case mesh is not watertight")
    cube_points = np.array(np.meshgrid(*([np.linspace(-0.009, 0.009, 5)] * 3)))
    cube_points = cube_points.reshape(3, -1).T
    interfering = {name: 0 for name in meshes}
    samples_inside = {name: 0 for name in meshes}
    examples = []
    for row in rows:
        data.qpos[:] = poses[row["ik_start"]]
        data.qpos[jaw_adr] = row["jaw_rad"]
        mujoco.mj_forward(model, data)
        dx, dy = row["xy_offset_mm"]
        center = np.array([toy[0] + dx / 1000, toy[1] + dy / 1000,
                           row["cube_height_mm"] / 1000])
        yaw = np.deg2rad(row["yaw_deg"])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                             [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        world = center + cube_points @ rotation.T
        local = (world - data.xpos[forearm]) @ data.xmat[forearm].reshape(3, 3)
        inside = {name: int(np.sum(mesh.contains(local)))
                  for name, mesh in meshes.items()}
        for name, count in inside.items():
            samples_inside[name] += count
            interfering[name] += count > 0
        if any(inside.values()) and len(examples) < 5:
            examples.append({"ik_start": row["ik_start"],
                             "cube_height_mm": row["cube_height_mm"],
                             "xy_offset_mm": row["xy_offset_mm"],
                             "yaw_deg": row["yaw_deg"], "jaw_rad": row["jaw_rad"],
                             "inside_samples": inside})
    return {
        "visual_body_sha256": digest(VISUAL),
        "visual_frame_mesh_sha256": digest(FRAME_MESH),
        "visual_motor_case_mesh_sha256": digest(MOTOR_MESH),
        "watertight": {name: bool(mesh.is_watertight) for name, mesh in meshes.items()},
        "cube_interior_grid_per_axis_mm": [-9, -4.5, 0, 4.5, 9],
        "diagnostic_clean_ik_passing_cases_sampled": len(rows),
        "cases_with_cube_samples_inside_visual_frame": interfering["forearm_frame"],
        "total_samples_inside_visual_frame": samples_inside["forearm_frame"],
        "cases_with_cube_samples_inside_gripper_motor_case": interfering["gripper_motor_case"],
        "total_samples_inside_gripper_motor_case": samples_inside["gripper_motor_case"],
        "examples": examples,
    }


def main() -> None:
    with np.load(ENTRY / "candidate-states.npz") as saved:
        poses = {int(key): pose for key, pose in zip(saved["start"], saved["qpos"])}
        toy = saved["toy_xyz_m"].copy()
    rows = json.loads((ENTRY / "cube-contact-matrix.json").read_text())["cases"]
    selected = replay(make_model(False), rows, poses, toy)
    if len(rows) != 4608 or any(
        a["two_jaw_contact"] != b["two_jaw_contact"] or
        a["clean_two_jaw"] != b["clean_two_jaw"]
        for a, b in zip(rows, selected)
    ):
        raise ValueError("Selected-body replay did not reproduce the fixed prior grid")
    diagnostic = replay(make_model(True), rows, poses, toy)
    if any(a["two_jaw_contact"] != b["two_jaw_contact"]
           for a, b in zip(selected, diagnostic)):
        raise ValueError("Ablation changed jaw contacts; comparison is not isolated")
    clean = [row for row in diagnostic if row["clean_two_jaw"]]
    visual_probe = probe_visual_frame(
        [row for row in clean if row["ik_screen_pass"]], poses, toy)
    report = {
        "schema": "robot-learning-lab.cove50-forearm-isolation-report.v1",
        "selected_pointer_sha256": digest(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": digest(BODY),
        "actuator_catalog_sha256": digest(ROOT / "robot/actuator-catalog.yaml"),
        "source_candidates_sha256": digest(ENTRY / "candidate-states.npz"),
        "source_grid_sha256": digest(ENTRY / "cube-contact-matrix.json"),
        "plan_sha256": digest(OUT / "plan.json"),
        "cases": len(rows),
        "selected_two_jaw": sum(row["two_jaw_contact"] for row in selected),
        "selected_clean_two_jaw": sum(row["clean_two_jaw"] for row in selected),
        "diagnostic_two_jaw": sum(row["two_jaw_contact"] for row in diagnostic),
        "diagnostic_clean_two_jaw": len(clean),
        "diagnostic_clean_two_jaw_in_ik_passing_poses": sum(
            row["ik_screen_pass"] for row in clean),
        "diagnostic_clean_two_jaw_by_height_mm": {
            str(z): sum(row["cube_height_mm"] == z for row in clean)
            for z in sorted(set(row["cube_height_mm"] for row in rows))},
        "diagnostic_clean_examples": clean[:5],
        "visual_frame_probe": visual_probe,
        "limits": [
            "Ablation disables selected forearm contact for diagnosis only; it is not a physically revised hand.",
            "The visual mesh probe samples cube interiors against the selected frame; it does not prove exact collision-free volume or buildable frame clearance.",
            "Static two-jaw contact does not establish compressive force, controlled lift or possession.",
            "These IK states were not attained through a continuous native policy transition.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnostic-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-forearm-isolation-matrix.v1",
        "cases": diagnostic}, indent=2) + "\n")
    (OUT / "comparison-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-forearm-isolation-comparison.v1",
        "cases": [
            {"case": "Selected collision model", "two_jaw": report["selected_two_jaw"],
             "clean_two_jaw": report["selected_clean_two_jaw"],
             "clean_in_ik_passing": 0, "visual_frame_interference": "not sampled"},
            {"case": "Forearm contact disabled · diagnostic only",
             "two_jaw": report["diagnostic_two_jaw"],
             "clean_two_jaw": report["diagnostic_clean_two_jaw"],
             "clean_in_ik_passing": report["diagnostic_clean_two_jaw_in_ik_passing_poses"],
             "visual_frame_interference": (
                 f"{visual_probe['cases_with_cube_samples_inside_visual_frame']}/"
                 f"{visual_probe['diagnostic_clean_ik_passing_cases_sampled']}")},
        ]}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
