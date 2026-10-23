"""Static collision coupon for a rearward opposed-jaw extension on Cove-50.

This leaves the selected source/body untouched. It does not design a connected
support or demonstrate a motor-controlled acquisition.
"""

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
VISUAL = ROOT / "artifacts/ape-frame50-direct-skeleton/ape-frame50-direct-visual.xml"
FRAME = ROOT / "artifacts/ape-frame50-direct-skeleton/meshes/right_forearm_frame.obj"
MOTOR = ROOT / "artifacts/ape-frame50-direct-skeleton/meshes/right_forearm_right_gripper_XL330_case.obj"
STATES = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1/candidate-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/right-offset-screen-v1"
EXTENSIONS_MM = (25, 30, 35, 40)
INSET_MM = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_model(extension_mm: int) -> mujoco.MjModel:
    root = ET.parse(BODY).getroot()
    hand = root.find(".//geom[@name='right_hand']")
    finger = root.find(".//geom[@name='right_finger_tip']")
    if hand is None or finger is None:
        raise ValueError("Selected jaw geoms missing")
    ext = extension_mm / 1000
    hand.set("pos", f"{0.005 - ext / 2:.6f} 0.010 -0.141")
    hand.set("size", f"{ext / 2:.6f} 0.003 0.013")
    finger.set("pos", f"{-ext / 2:.6f} {INSET_MM / 1000:.6f} 0")
    finger.set("size", f"{ext / 2:.6f} 0.003 0.013")
    cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Candidate changed the selected 21-axis interface")
    return model


def sampled_beam_overlap(model: mujoco.MjModel, pose: np.ndarray,
                         meshes: dict[str, trimesh.Trimesh]) -> dict:
    data = mujoco.MjData(model)
    forearm = model.body("right_forearm").id
    moving = model.body("right_moving_jaw").id
    joint = model.joint("right_gripper").qposadr[0]
    ids = {name: model.geom(name).id for name in ("right_hand", "right_finger_tip")}
    result = {}
    for angle in (0.0, 0.25, 0.5):
        data.qpos[:28] = pose
        data.qpos[joint] = angle
        mujoco.mj_forward(model, data)
        counts = {}
        for jaw, body in (("right_hand", forearm), ("right_finger_tip", moving)):
            geom = ids[jaw]
            extent = model.geom_size[geom] * 0.9
            grid = np.array(np.meshgrid(*[np.linspace(-a, a, 5 if j == 0 else 3)
                                          for j, a in enumerate(extent)]))
            local_geom = grid.reshape(3, -1).T
            local_body = model.geom_pos[geom] + local_geom @ np.eye(3)
            world = data.xpos[body] + local_body @ data.xmat[body].reshape(3, 3).T
            in_forearm = (world - data.xpos[forearm]) @ data.xmat[forearm].reshape(3, 3)
            counts[jaw] = {name: int(mesh.contains(in_forearm).sum())
                           for name, mesh in meshes.items()}
        result[str(angle)] = counts
    return result


def gap_mm(model: mujoco.MjModel, pose: np.ndarray, angle: float) -> float:
    data = mujoco.MjData(model)
    data.qpos[:28] = pose
    data.qpos[model.joint("right_gripper").qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    palm, finger = (model.geom(name).id for name in ("right_hand", "right_finger_tip"))
    axis = data.geom_xpos[finger] - data.geom_xpos[palm]
    span = np.linalg.norm(axis)
    axis /= span
    radii = []
    for geom in (palm, finger):
        rotation = data.geom_xmat[geom].reshape(3, 3)
        radii.append(float(np.abs(axis @ rotation) @ model.geom_size[geom]))
    return round(1000 * (span - sum(radii)), 3)


def main() -> None:
    selected = mujoco.MjModel.from_xml_path(str(BODY))
    visual = mujoco.MjModel.from_xml_path(str(VISUAL))
    for model in (selected, visual):
        if model.nu != 21 or model.nq != 28:
            raise ValueError("Selected or visual body has unexpected joint interface")
    names = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
                       for i in range(m.njnt)]
    if names(selected) != names(visual):
        raise ValueError("Selected and visual joint order differ")
    meshes = {"frame": trimesh.load_mesh(FRAME), "motor": trimesh.load_mesh(MOTOR)}
    if not all(mesh.is_watertight for mesh in meshes.values()):
        raise ValueError("Source frame/case mesh must be watertight")
    with np.load(STATES) as saved:
        poses = saved["qpos"][saved["screen_pass"]].copy()
        starts = saved["start"][saved["screen_pass"]].copy()
        saved_height_mm = round(float(saved["toy_xyz_m"][2] * 1000), 3)
    if len(poses) != 25:
        raise ValueError("Expected the fixed 25 passing static IK poses")
    # Post-screen comparator: the selected proxy itself may already overlap
    # the visual stock, which changes how to interpret candidate beam hits.
    selected_beam_overlap = sampled_beam_overlap(selected, poses[0], meshes)
    cube_points = np.array(np.meshgrid(*([np.linspace(-0.009, 0.009, 5)] * 3)))
    cube_points = cube_points.reshape(3, -1).T
    rows = []
    summaries = []
    for extension_mm in EXTENSIONS_MM:
        model = candidate_model(extension_mm)
        data = mujoco.MjData(model)
        forearm = model.body("right_forearm").id
        jaw_adr = model.joint("right_gripper").qposadr[0]
        cube_adr = model.joint("audit_cube_free").qposadr[0]
        cube_id = model.geom("audit_cube_geom").id
        palm_id = model.geom("right_hand").id
        finger_id = model.geom("right_finger_tip").id
        floor_id = model.geom("floor").id
        beam_overlap = sampled_beam_overlap(model, poses[0], meshes)
        start_row = len(rows)
        for start, pose in zip(starts, poses):
            for height_mm in (10.0, saved_height_mm):
                for dy_mm in (-2, 0, 2):
                    for yaw_deg in (0, 15, 30, 45):
                        for jaw_rad in (0.0, 0.25, 0.5):
                            data.qpos[:28] = pose
                            data.qpos[jaw_adr] = jaw_rad
                            mujoco.mj_forward(model, data)
                            rotation = data.xmat[forearm].reshape(3, 3)
                            center = data.xpos[forearm] + rotation @ np.array([
                                0.005 - extension_mm / 1000,
                                -0.004 + dy_mm / 1000, -0.141])
                            center[2] = height_mm / 1000
                            data.qpos[cube_adr:cube_adr + 3] = center
                            yaw = np.deg2rad(yaw_deg)
                            data.qpos[cube_adr + 3:cube_adr + 7] = (
                                np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                            mujoco.mj_forward(model, data)
                            palm = finger = False
                            floor_depth = 0.0
                            other = {}
                            for index in range(data.ncon):
                                contact = data.contact[index]
                                if cube_id not in (contact.geom1, contact.geom2):
                                    continue
                                partner = contact.geom2 if contact.geom1 == cube_id else contact.geom1
                                depth = float(contact.dist * 1000)
                                if partner == palm_id:
                                    palm = True
                                elif partner == finger_id:
                                    finger = True
                                elif partner == floor_id:
                                    floor_depth = max(floor_depth, -depth)
                                elif depth < -0.1:
                                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                                    other[name] = max(other.get(name, 0.0), -depth)
                            two_jaw = palm and finger
                            clean = two_jaw and floor_depth <= 0.1 and not other
                            visual_hits = None
                            if clean:
                                yaw_rotation = np.array([
                                    [np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                                world = center + cube_points @ yaw_rotation.T
                                local = (world - data.xpos[forearm]) @ rotation
                                visual_hits = {name: int(mesh.contains(local).sum())
                                               for name, mesh in meshes.items()}
                            rows.append({
                                "extension_mm": extension_mm, "ik_start": int(start),
                                "cube_height_mm": height_mm, "local_jaw_y_offset_mm": dy_mm,
                                "yaw_deg": yaw_deg, "jaw_rad": jaw_rad,
                                "two_jaw_contact": two_jaw,
                                "contact_clean": clean,
                                "floor_penetration_mm": round(floor_depth, 3),
                                "other_robot_penetration_mm": {k: round(v, 3) for k, v in other.items()},
                                "visual_cube_inside_samples": visual_hits,
                            })
        subset = rows[start_row:]
        clean_rows = [row for row in subset if row["contact_clean"]]
        visual_clear = [row for row in clean_rows if all(
            count == 0 for count in row["visual_cube_inside_samples"].values())]
        beam_clear = all(count == 0 for angle in beam_overlap.values()
                         for jaw in angle.values() for count in jaw.values())
        summaries.append({
            "case": f"{extension_mm} mm rearward jaw",
            "extension_mm": extension_mm,
            "cases": len(subset),
            "two_jaw_cases": sum(row["two_jaw_contact"] for row in subset),
            "contact_clean_cases": len(clean_rows),
            "cube_visual_clear_sampled_cases": len(visual_clear),
            "extended_beam_visual_clear": beam_clear,
            "beam_inside_samples": beam_overlap,
            "open_surface_gap_mm": gap_mm(model, poses[0], 0.0),
            "closed_surface_gap_mm": gap_mm(model, poses[0], 0.5),
        })
    report = {
        "schema": "robot-learning-lab.cove50-hand-offset-screen.v1",
        "method": "in-memory selected-body jaw-box extension and fixed static floor-cube contact grid",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "selected_body_sha256": sha(BODY),
        "selected_visual_body_sha256": sha(VISUAL),
        "selected_frame_mesh_sha256": sha(FRAME),
        "selected_motor_mesh_sha256": sha(MOTOR),
        "candidate_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "starts": len(poses), "cases": len(rows),
        "post_screen_selected_baseline_proxy_inside_visual_samples": selected_beam_overlap,
        "summaries": summaries,
        "limits": [
            "The extension is an in-memory contact geometry coupon; no connected CAD support or motor-output joint was built.",
            "Interior samples detect interference but are not exact signed clearance or swept motion checks.",
            "The selected fingertip proxy versus visual mesh comparison was added after the fixed coupon screen; it is a diagnostic comparator, not a predeclared candidate gate.",
            "The cube is relocated with the extended jaw at each prior static IK pose; these are not attained motor transitions or the original world toy locations.",
            "Two-jaw contacts are not grasp force, lift or possession; left-hand symmetry is untested.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-offset-matrix.v1",
        "cases": rows}, indent=2) + "\n")
    (OUT / "summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-offset-summary.v1",
        "cases": summaries}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"starts": len(poses), "cases": len(rows), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
