"""Fixed two-hand static floor-cube screen for the unselected hand coupon."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[4]
BODY = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v1/candidate-training.xml"
CAD = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/swept-pocket-v2/report.json"
STATES = ROOT / "artifacts/ape-cove50-acquisition-entry-20261009/whole-body-entry-v1/candidate-states.npz"
OUT = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/static-contact-v1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mirror_qpos(model: mujoco.MjModel, source: np.ndarray) -> np.ndarray:
    reflection = np.diag([1.0, -1.0, 1.0])
    mirrored = source.copy()
    mirrored[1] *= -1
    rotation = Rotation.from_quat(source[3:7], scalar_first=True).as_matrix()
    mirrored[3:7] = Rotation.from_matrix(reflection @ rotation @ reflection).as_quat(
        scalar_first=True)
    for suffix in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch",
                   "shoulder_yaw", "shoulder_roll", "shoulder_pitch",
                   "elbow_pitch", "gripper"):
        left = model.joint(f"left_{suffix}").qposadr[0]
        right = model.joint(f"right_{suffix}").qposadr[0]
        sign = -1 if suffix.endswith(("yaw", "roll")) else 1
        mirrored[left], mirrored[right] = sign * source[right], sign * source[left]
    mirrored[model.joint("head_yaw").qposadr[0]] *= -1
    return mirrored


def gap_mm(model: mujoco.MjModel, data: mujoco.MjData,
           side: str, pose: np.ndarray, angle: float) -> float:
    data.qpos[:28] = pose
    data.qpos[model.joint(f"{side}_gripper").qposadr[0]] = angle
    mujoco.mj_forward(model, data)
    palm, finger = (model.geom(f"{side}_{name}").id for name in
                    ("hand", "finger_tip"))
    axis = data.geom_xpos[finger] - data.geom_xpos[palm]
    span = np.linalg.norm(axis)
    axis /= span
    radii = []
    for geom in (palm, finger):
        rotation = data.geom_xmat[geom].reshape(3, 3)
        radii.append(float(np.abs(axis @ rotation) @ model.geom_size[geom]))
    return round(1000 * (span - sum(radii)), 3)


def main() -> None:
    cad = json.loads(CAD.read_text())
    if cad["sides_passing"] != 2:
        raise ValueError("CAD coupon did not clear both hands")
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="audit_cube", pos="0 0 0.01")
    ET.SubElement(cube, "freejoint", name="audit_cube_free")
    ET.SubElement(cube, "geom", name="audit_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  friction="0.8 0.01 0.001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if model.nu != 21 or model.nq != 35:
        raise ValueError("Expected candidate 21-axis body plus free cube")
    with np.load(STATES) as saved:
        poses = saved["qpos"][saved["screen_pass"]].copy()
        starts = saved["start"][saved["screen_pass"]].copy()
        saved_height = round(float(saved["toy_xyz_m"][2] * 1000), 3)
    if len(poses) != 25:
        raise ValueError("Expected exactly 25 fixed passing IK states")
    data = mujoco.MjData(model)
    cube_id = model.geom("audit_cube_geom").id
    floor_id = model.geom("floor").id
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    sole_ids = {model.geom(f"{side}_sole").id for side in ("left", "right")}
    rows = []
    gaps = {}
    for side in ("right", "left"):
        palm_id = model.geom(f"{side}_hand").id
        finger_id = model.geom(f"{side}_finger_tip").id
        site_id = model.site(f"{side}_hand_center").id
        jaw_adr = model.joint(f"{side}_gripper").qposadr[0]
        pose0 = poses[0] if side == "right" else mirror_qpos(model, poses[0])
        gaps[side] = {str(angle): gap_mm(model, data, side, pose0, angle)
                      for angle in (0.0, 0.25, 0.5)}
        for start, right_pose in zip(starts, poses):
            pose = right_pose if side == "right" else mirror_qpos(model, right_pose)
            for height_mm in (10.0, saved_height):
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
                                    height_mm / 1000)
                                yaw = np.deg2rad(yaw_deg)
                                data.qpos[cube_adr + 3:cube_adr + 7] = (
                                    np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                                mujoco.mj_forward(model, data)
                                palm = finger = False
                                floor_depth = 0.0
                                other = {}
                                nonsole_or_self = {}
                                for index in range(data.ncon):
                                    contact = data.contact[index]
                                    a, b = contact.geom1, contact.geom2
                                    depth = float(contact.dist * 1000)
                                    if cube_id in (a, b):
                                        partner = b if a == cube_id else a
                                        if partner == palm_id:
                                            palm = True
                                        elif partner == finger_id:
                                            finger = True
                                        elif partner == floor_id:
                                            floor_depth = max(floor_depth, -depth)
                                        elif depth < -0.1:
                                            name = mujoco.mj_id2name(
                                                model, mujoco.mjtObj.mjOBJ_GEOM, partner)
                                            other[name] = max(other.get(name, 0), -depth)
                                    elif depth < -0.1 and not (
                                        floor_id in (a, b) and
                                        (b if a == floor_id else a) in sole_ids
                                    ):
                                        names = [mujoco.mj_id2name(
                                            model, mujoco.mjtObj.mjOBJ_GEOM, geom)
                                            for geom in (a, b)]
                                        key = "/".join(sorted(names))
                                        nonsole_or_self[key] = max(nonsole_or_self.get(key, 0), -depth)
                                two_jaw = palm and finger
                                clean = (two_jaw and floor_depth <= 0.1 and not other and
                                         not nonsole_or_self)
                                rows.append({
                                    "case": f"{side}-ik{int(start)}-z{height_mm}-x{dx_mm}-y{dy_mm}-yaw{yaw_deg}-jaw{jaw_rad}",
                                    "side": side, "ik_start": int(start),
                                    "cube_height_mm": height_mm,
                                    "world_xy_offset_mm": [dx_mm, dy_mm],
                                    "yaw_deg": yaw_deg, "jaw_rad": jaw_rad,
                                    "two_jaw_contact": two_jaw,
                                    "clean_two_jaw": clean,
                                    "cube_floor_penetration_mm": round(floor_depth, 3),
                                    "other_cube_robot_penetration_mm": {
                                        k: round(v, 3) for k, v in other.items()},
                                    "nonsole_floor_or_self_penetration_mm": {
                                        k: round(v, 3) for k, v in nonsole_or_self.items()},
                                })
    if len(rows) != 10800:
        raise ValueError("Fixed static screen case count changed")
    summary = []
    for side in ("right", "left"):
        subset = [row for row in rows if row["side"] == side]
        summary.append({
            "case": side,
            "static_cases": len(subset),
            "two_jaw_cases": sum(row["two_jaw_contact"] for row in subset),
            "clean_two_jaw_cases": sum(row["clean_two_jaw"] for row in subset),
            "clean_floor_height_cases": sum(
                row["clean_two_jaw"] and row["cube_height_mm"] == 10.0
                for row in subset),
            "ik_starts_with_clean_contact": len({row["ik_start"] for row in subset
                                                  if row["clean_two_jaw"]}),
            "nonsole_or_self_interference_cases": sum(
                bool(row["nonsole_floor_or_self_penetration_mm"]) for row in subset),
            "open_gap_mm": gaps[side]["0.0"],
            "closed_gap_mm": gaps[side]["0.5"],
        })
    report = {
        "schema": "robot-learning-lab.cove50-hand-coupon-static-contact.v1",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "candidate_body_sha256": sha(BODY),
        "cad_report_sha256": sha(CAD),
        "candidate_states_sha256": sha(STATES),
        "plan_sha256": sha(OUT / "plan.json"),
        "mj_version": mujoco.__version__,
        "cases": len(rows), "source_right_ik_starts": len(poses),
        "jaw_surface_gap_mm": gaps,
        "summary": summary,
        "limits": [
            "All poses are static selected-body IK states, mirrored on the left; none is attained under the candidate mass/contact model.",
            "The cube is placed at the candidate hand site, not the original world toy; approach from outside the aperture is untested.",
            "Static two-jaw contact is not compressive force, controlled lift or possession.",
            "The revised frame uses box/capsule collision proxies; exact CAD swept clearance and bearing support remain separate checks.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-coupon-static-contact-matrix.v1",
        "cases": rows}, indent=2) + "\n")
    (OUT / "summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-hand-coupon-static-contact-summary.v1",
        "cases": summary}, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
