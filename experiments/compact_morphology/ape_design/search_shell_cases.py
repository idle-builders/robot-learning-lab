"""Search shaft-attached case orientations against the v2 shell and 20/22 axes.

Uses the existing source-sized envelope/SAT code. Eight rotations about each
output shaft are a coarse physical orientation menu, not an arbitrary box pack.
Outputs failures as evidence; no shell cutouts or transmissions are invented.
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ape_cad"))
import build as envelope  # noqa: E402
import sketch_shell as shell  # noqa: E402
from shell_aligned_model import OUT, make_model  # noqa: E402


def regions(name: str) -> tuple[str, ...]:
    side = "left" if name.startswith("left_") else "right"
    if "hip_yaw" in name or "hip_roll" in name:
        return ("torso_shell", f"{side}_thigh_shell")
    if "hip_pitch" in name:
        return ("torso_shell", f"{side}_thigh_shell")
    if "knee" in name:
        return (f"{side}_thigh_shell", f"{side}_shin_shell")
    if "ankle" in name:
        return (f"{side}_shin_shell", f"{side}_foot_shell")
    if "shoulder" in name:
        return ("torso_shell", f"{side}_upper_arm_shell")
    if "elbow" in name:
        return (f"{side}_upper_arm_shell", f"{side}_forearm_shell")
    if "gripper" in name:
        return (f"{side}_forearm_shell", f"{side}_hand_shell")
    if name == "head_yaw":
        return ("torso_shell", "head_shell")
    return ("head_shell",)


@dataclass
class Choice:
    name: str
    index: int
    angle_deg: int
    pose: dict
    outside_corners: int


def corners(pose: dict) -> np.ndarray:
    return np.array([
        pose["center"] + pose["rot"] @ (pose["half"] * signs)
        for signs in product((-1, 1), repeat=3)
    ])


def choices(model: mujoco.MjModel, data: mujoco.MjData,
            scale: float = 1.0) -> dict[str, list[Choice]]:
    result = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        bid = int(model.jnt_bodyid[jid])
        parent = int(model.body_parentid[bid])
        axis = np.asarray(data.xaxis[jid]).copy()
        parent_rot = np.asarray(data.xmat[parent]).reshape(3, 3)
        initial = parent_rot @ envelope.tail_for(name)
        initial -= axis * (initial @ axis)
        initial /= np.linalg.norm(initial)
        spec = envelope.CFG["parts"][envelope.family(name)]
        half = np.array(spec["envelope_mm"], dtype=float) / 2000
        group = envelope.family(name)
        item = []
        for index in range(8):
            angle = index * 45
            tail = Rotation.from_rotvec(axis * np.deg2rad(angle)).apply(initial)
            across = np.cross(tail, axis)
            rot = np.column_stack((tail, axis, across))
            center = scale * np.asarray(data.xanchor[jid]) + tail * spec["shaft_to_case_center_mm"] / 1000
            pose = {
                "name": name, "owner": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
                "family": group, "part": spec["catalog"], "center": center,
                "rot": rot, "half": half, "shaft": scale * np.asarray(data.xanchor[jid]).copy(),
                "mass": {"legs": .0745, "arms": .055, "small": .018}[group],
            }
            outer = 0
            for point in corners(pose):
                if not any(shell.INTERIORS[key].is_inside(bd.Vector(*(point * 1000 / scale)))
                           for key in regions(name)):
                    outer += 1
            item.append(Choice(name, index, angle, pose, outer))
        result[name] = item
    return result


def choose_group(names: list[str], menu: dict[str, list[Choice]],
                 beam_width: int = 250) -> list[Choice]:
    # The named joint order preserves the physical parent chain. Search cost
    # favors zero motor clashes, then fewer case corners outside the shell.
    states: list[tuple[int, list[Choice]]] = [(0, [])]
    for name in names:
        next_states = []
        for score, picked in states:
            for candidate in menu[name]:
                hits = sum(not envelope.separated(candidate.pose, old.pose, .001)
                           for old in picked)
                next_states.append((score + hits * 100 + candidate.outside_corners * 2,
                                    picked + [candidate]))
        next_states.sort(key=lambda state: state[0])
        states = next_states[:beam_width]
    return states[0][1]


def serial(choice: Choice) -> dict:
    p = choice.pose
    return {
        "joint": choice.name, "orientation_about_shaft_deg": choice.angle_deg,
        "source_case": p["part"],
        "center_mm": (p["center"] * 1000).round(2).tolist(),
        "shaft_mm": (p["shaft"] * 1000).round(2).tolist(),
        "rotation_matrix": p["rot"].round(6).tolist(),
        "size_mm": (p["half"] * 2000).round(2).tolist(),
        "outside_interior_corners_of_8": choice.outside_corners,
        "candidate_regions": regions(choice.name),
        "mass_g": round(p["mass"] * 1000, 1),
    }


def run(axes: int, path_override: Path | None = None) -> dict:
    path = path_override or make_model(axes)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    menu = choices(model, data)
    groups = []
    for side in ("left", "right"):
        groups.append([f"{side}_{name}" for name in
                       ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")])
        arm = ["shoulder_roll", "shoulder_pitch", "elbow_pitch", "gripper"]
        if axes in (22, 21):
            arm.insert(0, "shoulder_yaw")
        groups.append([f"{side}_{name}" for name in arm])
    groups.append(["head_yaw"] + (["head_pitch"] if axes in (22, 20) else []))
    selected = [choice for group in groups for choice in choose_group(group, menu)]
    battery = {
        "name": "battery_4s_850", "family": "battery",
        "center": np.array([0., 0., .191]),
        "rot": np.eye(3), "half": np.array([.030, .015, .01525]),
    }
    poses = [choice.pose for choice in selected] + [battery]
    case_clashes = envelope.clashes(poses, 1)
    # Full BRep intersection catches enclosure violations missed by corners.
    shell_hits = {}
    for choice in selected:
        p = choice.pose
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected")
            # build123d Rot uses intrinsic XYZ; scipy's lowercase xyz is a
            # different composition, even though both encode the same angles.
            angles = Rotation.from_matrix(p["rot"]).as_euler("XYZ", degrees=True)
        shape = bd.Pos(*(p["center"] * 1000)) * bd.Rot(*angles) * bd.Box(*(p["half"] * 2000))
        hits = {}
        for part in shell.PARTS:
            overlap = shell_volume(shape.intersect(part.shape))
            if overlap > 1:
                hits[part.name] = round(overlap, 1)
        shell_hits[choice.name] = hits
    bounds = np.array([np.abs(p["rot"]) @ p["half"] for p in poses])
    centers = np.array([p["center"] for p in poses])
    motor_mass = sum(p["mass"] for p in (c.pose for c in selected))
    mass_center = (sum((p["mass"] * p["center"] for p in (c.pose for c in selected)),
                       np.zeros(3)) + .107 * battery["center"]) / (motor_mass + .107)
    return {
        "axes": axes,
        "model": str(path.relative_to(ROOT)),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "case_source_layout": str((HERE.parent / "ape_cad/layout.yaml").relative_to(ROOT)),
        "source_size_case_bounds_mm": {
            "min": ((centers - bounds).min(axis=0) * 1000).round(1).tolist(),
            "max": ((centers + bounds).max(axis=0) * 1000).round(1).tolist(),
        },
        "neutral_case_clashes_1mm": case_clashes,
        "outside_interior_corners": sum(c.outside_corners for c in selected),
        "shell_hits_mm3": shell_hits,
        "known_motor_battery_mass_kg": round(motor_mass + .107, 3),
        "known_motor_battery_com_mm": (mass_center * 1000).round(1).tolist(),
        "cases": [serial(c) for c in selected],
        "caveat": "Eight shaft-attached case orientations per joint. Groupwise beam search; not a global optimum. No remote transmission, shell cutout, mount or wire. Unscreened joint travel and policy. Cases that collide in neutral cannot be a buildable layout.",
    }


def shell_volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    if shape is None:
        return 0.
    return sum(part.volume for part in shape) if isinstance(shape, bd.ShapeList) else shape.volume


def main() -> None:
    shell.build()
    OUT.mkdir(parents=True, exist_ok=True)
    for axes in (21, 22, 20):
        report = run(axes)
        (OUT / f"case-screen-{axes}.json").write_text(json.dumps(report, indent=2) + "\n")
        hits = sum(bool(parts) for parts in report["shell_hits_mm3"].values())
        print(f"{axes}: {len(report['neutral_case_clashes_1mm'])} case clashes, "
              f"{report['outside_interior_corners']} outside corners, "
              f"{hits} cases hit shell material")


if __name__ == "__main__":
    main()
