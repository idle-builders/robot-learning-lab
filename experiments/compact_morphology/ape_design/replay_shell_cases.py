"""Geometric stress screen of saved motions on shell-aligned joint trees.

The inputs are saved policy trajectories from the *old* morphology. Mapping
joint names into a new tree is a clearance probe, not a policy evaluation.
"""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "ape_cad"))
import build as envelope  # noqa: E402
import sketch_shell as shell  # noqa: E402
from search_shell_cases import OUT, corners, regions  # noqa: E402
from shell_aligned_model import SOURCE  # noqa: E402


def owner(part: str) -> str:
    side = "left" if part.startswith("left_") else "right"
    if part in ("torso_shell", "chest_light"):
        return "pelvis"
    if part in ("head_shell", "face_mask", "muzzle", "left_eye", "right_eye",
                "left_eye_glint", "right_eye_glint"):
        return "head"
    if part == "neck_shadow":
        return "head_yaw_body"
    if "upper_arm" in part:
        return f"{side}_arm"
    if any(word in part for word in ("forearm", "elbow_shadow", "wrist_shadow", "hand_shell")):
        return f"{side}_forearm"
    if "thigh" in part:
        return f"{side}_thigh"
    if any(word in part for word in ("shin", "knee_shadow")):
        return f"{side}_shin"
    if "foot" in part:
        return f"{side}_foot"
    raise ValueError(part)


def frame(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(name)
    result = np.eye(4)
    result[:3, :3] = np.asarray(data.xmat[bid]).reshape(3, 3)
    result[:3, 3] = data.xpos[bid]
    return result


def case_from_neutral(item: dict, model: mujoco.MjModel,
                      data: mujoco.MjData, neutral_frames: dict[str, np.ndarray]) -> dict:
    joint = item["joint"]
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    bid = int(model.body_parentid[model.jnt_bodyid[jid]])
    parent = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    delta = frame(model, data, parent) @ np.linalg.inv(neutral_frames[parent])
    neutral_center = np.array(item["center_mm"]) / 1000
    center = (delta @ np.r_[neutral_center, 1])[:3]
    return {
        "name": joint, "owner": parent, "center": center,
        "rot": delta[:3, :3] @ np.array(item["rotation_matrix"]),
        "half": np.array(item["size_mm"]) / 2000,
    }


def map_qpos(old: mujoco.MjModel, new: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    mapped = np.array(new.key_qpos[0])
    mapped[:7] = qpos[:7]
    mapped[:2] = 0  # The source policy starts in a translated arena tile.
    for jid in range(1, new.njnt):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_JOINT, jid)
        old_id = mujoco.mj_name2id(old, mujoco.mjtObj.mjOBJ_JOINT, name)
        mapped[new.jnt_qposadr[jid]] = qpos[old.jnt_qposadr[old_id]]
    return mapped


def run(axes: int, suffix: str = "") -> dict:
    model_path = OUT / f"shell-aligned-{axes}{suffix}.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    old = mujoco.MjModel.from_xml_path(str(SOURCE))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    case_report = json.loads((OUT / f"case-screen-{axes}{suffix}.json").read_text())
    owners = {owner(p.name) for p in shell.PARTS}
    owners |= {c["joint"] for c in case_report["cases"]}
    for case in case_report["cases"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, case["joint"])
        bid = model.body_parentid[model.jnt_bodyid[jid]]
        owners.add(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)))
    owners = {o for o in owners if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, o) >= 0}
    neutral = {name: frame(model, data, name) for name in owners}
    pelvis_neutral = neutral["pelvis"]
    battery_center = np.array([0., 0., .191])
    samples = []
    for record_index, (label, old_q) in enumerate(envelope.recorded_poses(old)):
        if record_index and (record_index - 1) % 13 not in (0, 3, 6, 9, 12):
            continue
        data.qpos[:] = map_qpos(old, model, old_q)
        mujoco.mj_forward(model, data)
        deltas = {name: frame(model, data, name) @ np.linalg.inv(neutral[name])
                  for name in owners}
        inverse_deltas = {name: np.linalg.inv(value) for name, value in deltas.items()}
        poses = [case_from_neutral(c, model, data, neutral) for c in case_report["cases"]]
        battery_delta = deltas["pelvis"]
        battery = {"name": "battery_4s_850", "center":
                   (battery_delta @ np.r_[battery_center, 1])[:3],
                   "rot": battery_delta[:3, :3], "half": np.array([.030, .015, .01525])}
        collisions = envelope.clashes(poses + [battery], 1)
        outside = {}
        # A moving shell's nominal free space is evaluated in that shell's
        # neutral CAD frame. This is a corner screen, not a full BRep sweep.
        for pose in poses:
            count = 0
            for point in corners(pose):
                inside = False
                for region in regions(pose["name"]):
                    shell_owner = owner(region)
                    old_point = inverse_deltas[shell_owner] @ np.r_[point, 1]
                    if shell.INTERIORS[region].is_inside(bd.Vector(*(old_point[:3] * 1000))):
                        inside = True
                        break
                count += not inside
            if count:
                outside[pose["name"]] = count
        pose_transforms = {
            part.name: (deltas[owner(part.name)]).round(6).tolist()
            for part in shell.PARTS
        }
        samples.append({
            "label": label,
            "source_root_xy_m": old_q[:2].round(3).tolist(),
            "case_clashes_1mm": collisions,
            "outside_case_corners": outside,
            "part_transforms_m": pose_transforms,
            "battery": {"center_m": battery["center"].round(6).tolist(),
                        "rotation": battery["rot"].round(6).tolist()},
            "cases": [{"joint": p["name"], "center_m": p["center"].round(6).tolist(),
                       "rotation": p["rot"].round(6).tolist(),
                       "size_mm": (p["half"] * 2000).round(2).tolist()}
                      for p in poses],
        })
    task_summary = {}
    for prefix in ("neutral", "forward", "turn-left", "squat", "step-left", "acquire-place"):
        group = [s for s in samples if s["label"].split(":")[0] == prefix]
        if group:
            task_summary[prefix] = {
                "frames": len(group),
                "worst_case_clashes": max(len(s["case_clashes_1mm"]) for s in group),
                "frames_with_case_clashes": sum(bool(s["case_clashes_1mm"]) for s in group),
                "worst_outside_corners": max(sum(s["outside_case_corners"].values()) for s in group),
            }
    return {
        "axes": axes, "source_trajectory_model": str(SOURCE.relative_to(ROOT)),
        "source_model_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "source_trace_sha256": {
            stem: hashlib.sha256(path.read_bytes()).hexdigest()
            for stem in ("forward", "turn-left", "squat", "step-left", "acquire-place")
            if (path := ROOT / f"artifacts/ape-physics-pair-20260930/{stem}-candidate.npz").exists()
        },
        "target_model": str(model_path.relative_to(ROOT)),
        "summary": task_summary, "samples": samples,
        "interpretation": "Saved motion qpos, mapped by joint name onto a new skeleton; not a trained policy replay or dynamic evaluation. Shaft-attached case rotations fixed at neutral. Corner containment and case SAT only in moving frames; no swept BRep or cable/mount checks.",
    }


def main() -> None:
    shell.build()
    variants = [(20, "-spaced-probe")] if "--spaced-only" in sys.argv else [(21, ""), (22, ""), (20, "")]
    for axes, suffix in variants:
        result = run(axes, suffix)
        (OUT / f"motion-screen-{axes}{suffix}.json").write_text(json.dumps(result) + "\n")
        print(axes, suffix, result["summary"])
    shutil.copyfile(HERE / "shell_motion_viewer.html", OUT / "index.html")


if __name__ == "__main__":
    main()
