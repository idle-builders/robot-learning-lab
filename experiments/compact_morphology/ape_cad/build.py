"""Source-sized servo envelope CAD and pose clearance screen for the ape.

Run: .venv/bin/python experiments/compact_morphology/ape_cad/build.py
This deliberately reports failures. A box clearing here is not a printable joint.
"""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import warnings
from collections import Counter
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = yaml.safe_load((HERE / "layout.yaml").read_text())
MODEL = Path(os.environ.get("APE_CAD_MODEL", HERE / CFG["model"])).resolve()
OUT = Path(os.environ.get("APE_CAD_OUT", ROOT / "artifacts/ape-cad-source-20260930")).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def family(name: str) -> str:
    if "hip" in name or "knee" in name or "ankle" in name:
        return "legs"
    if "shoulder" in name or "elbow" in name:
        return "arms"
    return "small"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tail_for(name: str) -> np.ndarray:
    stem = name.removeprefix("left_").removeprefix("right_")
    if stem in ("hip_roll", "shoulder_roll"):
        stem += "_left" if name.startswith("left_") else "_right"
    return np.array(CFG["tails"][stem], dtype=float)


def poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    result = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        bid = int(model.jnt_bodyid[jid])
        parent = int(model.body_parentid[bid])
        axis = np.array(data.xaxis[jid], dtype=float)
        parent_rot = np.array(data.xmat[parent]).reshape(3, 3)
        tail = parent_rot @ tail_for(name)
        tail -= axis * np.dot(axis, tail)
        tail /= np.linalg.norm(tail)
        across = np.cross(tail, axis)
        across /= np.linalg.norm(across)
        rot = np.column_stack([tail, axis, across])
        spec = CFG["parts"][family(name)]
        center = np.array(data.xanchor[jid]) + tail * spec["shaft_to_case_center_mm"] / 1000
        result.append({
            "name": name, "owner": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
            "part": spec["catalog"], "family": family(name), "center": center,
            "rot": rot, "axis": axis, "half": np.array(spec["envelope_mm"]) / 2000,
            "shaft": np.array(data.xanchor[jid]),
            "mass": {"legs": 0.0745, "arms": 0.055, "small": 0.018}[family(name)],
        })
    return result


def battery_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "candidate_4s_battery_mass")
    rot = np.array(data.geom_xmat[gid]).reshape(3, 3)
    return {"name": "battery_example", "owner": "pelvis", "part": "Tattu-4S-850mAh",
            "family": "battery", "center": np.array(data.geom_xpos[gid]),
            "rot": rot, "half": np.array(model.geom_size[gid]),
            "mass": 0.107}


def separated(a: dict, b: dict, margin: float = 0.0) -> bool:
    """OBB separating-axis test; True only when an axis clears the boxes."""
    ar, br = a["rot"], b["rot"]
    aa, bb = a["half"] + margin / 2, b["half"] + margin / 2
    r = ar.T @ br
    t = ar.T @ (b["center"] - a["center"])
    abs_r = np.abs(r) + 1e-12
    for i in range(3):
        if abs(t[i]) > aa[i] + bb @ abs_r[i, :]:
            return True
    for j in range(3):
        if abs(t @ r[:, j]) > bb[j] + aa @ abs_r[:, j]:
            return True
    for i in range(3):
        for j in range(3):
            if abs(t[(i + 2) % 3] * r[(i + 1) % 3, j] - t[(i + 1) % 3] * r[(i + 2) % 3, j]) > (
                aa[(i + 1) % 3] * abs_r[(i + 2) % 3, j]
                + aa[(i + 2) % 3] * abs_r[(i + 1) % 3, j]
                + bb[(j + 1) % 3] * abs_r[i, (j + 2) % 3]
                + bb[(j + 2) % 3] * abs_r[i, (j + 1) % 3]
            ):
                return True
    return False


def clashes(parts: list[dict], margin_mm: float = 0.0) -> list[tuple[str, str]]:
    return [(a["name"], b["name"]) for i, a in enumerate(parts)
            for b in parts[i + 1:] if not separated(a, b, margin_mm / 1000)]


def total_com(model: mujoco.MjModel, data: mujoco.MjData, parts: list[dict]) -> dict:
    total = float(sum(model.body_mass))
    original = sum((model.body_mass[i] * data.xipos[i] for i in range(1, model.nbody)), np.zeros(3)) / total
    # V6 places all motor masses at their joint centers. Keep every gram, move
    # only the specified motor masses to envelope centers on fixed-side links.
    shift = sum((p["mass"] * (p["center"] - p["shaft"]) for p in parts), np.zeros(3)) / total
    return {"total_kg": total, "model_com_m": original.tolist(),
            "fixed_side_case_com_m": (original + shift).tolist(),
            "com_shift_mm": (shift * 1000).tolist()}


def recorded_poses(model: mujoco.MjModel):
    yield "neutral", np.array(model.key_qpos[0])
    for stem in ("forward", "turn-left", "squat", "step-left", "acquire-place"):
        path = ROOT / f"artifacts/ape-physics-pair-20260930/{stem}-candidate.npz"
        if not path.exists():
            continue
        with np.load(path) as trace:
            q = trace["qpos"]
        for index in np.unique(np.linspace(0, len(q) - 1, 13).astype(int)):
            yield f"{stem}:{index}", q[index]


def single_joint_ranges(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    """Case-only clearance with each axis varied and every other axis neutral."""
    neutral = np.array(model.key_qpos[0])
    result = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        lo, hi = (float(x) for x in model.jnt_range[jid])
        angles = sorted(set(np.linspace(lo, hi, 101).tolist() + [0.0]))
        valid = []
        for angle in angles:
            data.qpos[:] = neutral
            data.qpos[model.jnt_qposadr[jid]] = angle
            mujoco.mj_forward(model, data)
            valid.append(not clashes(poses(model, data) + [battery_pose(model, data)], CFG["clearance_mm"]))
        zero = angles.index(0.0)
        left, right = zero, zero
        if valid[zero]:
            while left > 0 and valid[left - 1]:
                left -= 1
            while right < len(angles) - 1 and valid[right + 1]:
                right += 1
        result[name] = {"software_range_rad": [lo, hi],
                        "isolated_clear_interval_rad": [round(angles[left], 4), round(angles[right], 4)]
                        if valid[zero] else None,
                        "sample_count": len(angles)}
    return result


def make_cad(parts: list[dict], model: mujoco.MjModel, data: mujoco.MjData):
    cad = []
    scene = trimesh.Scene()
    palette = {"legs": [61, 108, 154, 255], "arms": [219, 143, 75, 255],
               "small": [191, 169, 90, 255]}
    for p in parts:
        center_mm = p["center"] * 1000
        size_mm = p["half"] * 2000
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            angles = Rotation.from_matrix(p["rot"]).as_euler("xyz", degrees=True)
        shape = bd.Pos(*center_mm) * bd.Rot(*angles) * bd.Box(*size_mm)
        shape.label = p["name"]
        cad.append(shape)
        mesh = trimesh.creation.box(extents=p["half"] * 2)
        tf = np.eye(4)
        tf[:3, :3], tf[:3, 3] = p["rot"], p["center"]
        mesh.apply_transform(tf)
        mesh.visual.face_colors = palette[p["family"]]
        scene.add_geometry(mesh, node_name=p["name"], geom_name=p["name"])
    # The central battery is an allocated component, not a selected mounting design.
    battery_part = battery_pose(model, data)
    pelvis_rot = battery_part["rot"]
    battery_center = battery_part["center"]
    battery_dims = battery_part["half"] * 2
    angles = Rotation.from_matrix(pelvis_rot).as_euler("xyz", degrees=True)
    battery = bd.Pos(*(battery_center * 1000)) * bd.Rot(*angles) * bd.Box(*(battery_dims * 1000))
    battery.label = "battery_example"
    cad.append(battery)
    mesh = trimesh.creation.box(extents=battery_dims)
    tf = np.eye(4)
    tf[:3, :3], tf[:3, 3] = pelvis_rot, battery_center
    mesh.apply_transform(tf)
    mesh.visual.face_colors = [69, 169, 114, 255]
    scene.add_geometry(mesh, node_name="battery_example", geom_name="battery_example")
    # Reference link/sole geometry comes from the same MJCF that generated the
    # saved policies. It is included for size and reach, not as printable CAD.
    for side in ("left", "right"):
        for start, end, radius in (("hip_pitch", "knee", .018), ("knee", "ankle_pitch", .016),
                                   ("shoulder_pitch", "elbow_pitch", .013),
                                   ("elbow_pitch", "gripper", .012)):
            anchors = {p["name"]: p["shaft"] for p in parts}
            a, b = anchors[f"{side}_{start}"], anchors[f"{side}_{end}"]
            direction = b - a
            length = np.linalg.norm(direction)
            rot, _ = Rotation.align_vectors([direction], [[0, 0, 1]])
            angles = rot.as_euler("xyz", degrees=True)
            midpoint = (a + b) / 2
            shape = bd.Pos(*(midpoint * 1000)) * bd.Rot(*angles) * bd.Cylinder(radius * 1000, length * 1000)
            shape.label = f"{side}_{start}_to_{end}_link_proxy"
            cad.append(shape)
            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=12)
            tf = np.eye(4)
            tf[:3, :3], tf[:3, 3] = rot.as_matrix(), midpoint
            mesh.apply_transform(tf)
            mesh.visual.face_colors = [183, 200, 202, 255]
            scene.add_geometry(mesh, node_name=shape.label, geom_name=shape.label)
        sole_name = f"{side}_sole"
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, sole_name)
        sole_rot = np.array(data.geom_xmat[gid]).reshape(3, 3)
        sole_center = np.array(data.geom_xpos[gid])
        sole_size = np.array(model.geom_size[gid]) * 2
        shape = bd.Pos(*(sole_center * 1000)) * bd.Box(*(sole_size * 1000))
        shape.label = f"{sole_name}_proxy"
        cad.append(shape)
        mesh = trimesh.creation.box(extents=sole_size)
        tf = np.eye(4)
        tf[:3, :3], tf[:3, 3] = sole_rot, sole_center
        mesh.apply_transform(tf)
        mesh.visual.face_colors = [92, 113, 125, 255]
        scene.add_geometry(mesh, node_name=shape.label, geom_name=shape.label)
    head_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    head_center = np.array(data.xpos[head_bid])
    head = bd.Pos(*(head_center * 1000)) * bd.Sphere(52)
    head.label = "head_proxy"
    cad.append(head)
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=.052)
    mesh.apply_translation(head_center)
    mesh.visual.face_colors = [217, 214, 199, 255]
    scene.add_geometry(mesh, node_name="head_proxy", geom_name="head_proxy")
    bd.export_step(bd.Compound(children=cad), OUT / "ape-envelope-assembly.step")
    scene.export(OUT / "ape-envelope-assembly.glb")


def view_state(model: mujoco.MjModel, data: mujoco.MjData, parts: list[dict], hits):
    pose = {p["name"]: {"center": p["center"].tolist(),
                        "quaternion": Rotation.from_matrix(p["rot"]).as_quat().tolist()}
            for p in parts}
    joints = {p["name"]: p["shaft"].tolist() for p in parts}
    visual = {}
    for name in ("left_sole", "right_sole"):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        visual[name] = {"center": data.geom_xpos[gid].tolist(),
                        "quaternion": Rotation.from_matrix(data.geom_xmat[gid].reshape(3, 3)).as_quat().tolist(),
                        "size": (model.geom_size[gid] * 2).tolist()}
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")
    visual["head"] = {"center": data.xpos[gid].tolist(), "radius": 0.052}
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    visual["pelvis"] = {"center": data.xpos[bid].tolist()}
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    visual["torso"] = {"center": data.xpos[bid].tolist()}
    battery = battery_pose(model, data)
    visual["battery"] = {"center": battery["center"].tolist(),
                         "quaternion": Rotation.from_matrix(battery["rot"]).as_quat().tolist()}
    return {"parts": pose, "joints": joints, "visual": visual, "collisions": hits,
            "com": total_com(model, data, parts)}


def reach_pose(model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    def site(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return np.array(data.site_xpos[sid])

    left_hand, right_hand = site("left_hand_center"), site("right_hand_center")
    left_foot, right_foot = site("left_sole_center"), site("right_sole_center")
    pelvis = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")]
    head = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head")]
    return {"hand_height_mm": [(left_hand[2] * 1000), (right_hand[2] * 1000)],
            "hand_forward_of_pelvis_mm": [(left_hand[0] - pelvis[0]) * 1000,
                                          (right_hand[0] - pelvis[0]) * 1000],
            "sole_center_separation_mm": float(np.linalg.norm(left_foot - right_foot) * 1000),
            "head_top_mm": float((head[2] + .052) * 1000)}


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    rows = []
    count = Counter()
    margin = CFG["clearance_mm"]
    for label, qpos in recorded_poses(model):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        parts = poses(model, data)
        checked = parts + [battery_pose(model, data)]
        hits = clashes(checked)
        too_close = clashes(checked, margin)
        count.update(tuple(sorted(pair)) for pair in hits)
        rows.append({"pose": label, "intersections": len(hits),
                     "under_clearance_mm": len(too_close), "pairs": hits,
                     "reach": reach_pose(model, data),
                     "view": view_state(model, data, parts, hits)})
        if label == "neutral":
            make_cad(parts, model, data)
            mass = total_com(model, data, parts)
            neutral = [{"name": p["name"], "owner": p["owner"], "part": p["part"],
                        "shaft_m": p["shaft"].tolist(), "axis_world": p["axis"].tolist(),
                        "case_center_m": p["center"].tolist(),
                        "dimensions_m": (p["half"] * 2).tolist(), "mass_kg": p["mass"]} for p in parts]
    report = {"status": "failed_envelope_fit" if count else "envelope_fit_only",
              "source_model": str(MODEL.relative_to(ROOT)), "sampled_poses": len(rows),
              "provenance": {
                  "model_sha256": sha256(MODEL), "layout_sha256": sha256(HERE / "layout.yaml"),
                  "builder_sha256": sha256(HERE / "build.py"),
                  "trace_sha256": {stem: sha256(ROOT / f"artifacts/ape-physics-pair-20260930/{stem}-candidate.npz")
                                   for stem in ("forward", "turn-left", "squat", "step-left", "acquire-place")
                                   if (ROOT / f"artifacts/ape-physics-pair-20260930/{stem}-candidate.npz").exists()},
              },
              "clearance_mm": margin, "part_count": len(neutral), "mass": mass,
              "single_joint_case_clearance": single_joint_ranges(model, data),
              "neutral_parts": neutral, "worst_pairs": [{"pair": list(pair), "pose_count": n}
                                              for pair, n in count.most_common(30)], "poses": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(HERE / "viewer.html", OUT / "index.html")
    print(json.dumps({k: report[k] for k in ("status", "sampled_poses", "part_count", "mass", "worst_pairs")}, indent=2))


if __name__ == "__main__":
    main()
