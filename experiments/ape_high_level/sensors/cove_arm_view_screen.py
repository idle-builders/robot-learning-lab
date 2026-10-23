"""Screen floor and near-paw sight during bounded arm-pose sweeps on Cove-50.

Uses the pinned direct-frame visual body with the saved Cove shell overlay. The
poses are kinematic sensitivity cases, not qualified reaches or a new policy.
Evaluator spheres are not perception inputs; a second rendering with robot
visuals hidden distinguishes a missing ray from self-occlusion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from cove_view_alternatives import DESIGN, ROOT, SHELL_REPORT, SHELL_VISUAL, load_model, sha256


CAMERAS = ("lower_rgb", "chest_low50_x45")
TARGETS = {
    "floor_center_12cm": ((0.12, 0.0, 0.04), 0.04),
    "floor_right_12cm": ((0.12, -0.15, 0.04), 0.04),
    "floor_right_25cm": ((0.25, -0.15, 0.04), 0.04),
    "near_right_paw_3cm": ((0.0, 0.0, 0.0), 0.015),
}
# Yaw, roll, pitch, elbow. These are within the visual model's joint limits;
# they were chosen to test optical sensitivity, not to imply collision-free reach.
ARM_CASES = {
    "neutral": (0.0, 0.0, 0.0, 0.0),
    "forward_low": (0.0, 0.0, -0.55, 0.0),
    "cross_center": (0.75, 0.0, -0.55, 0.0),
    "forward_high": (0.0, 0.0, -1.0, 0.0),
    "bent_cross": (0.4, 0.0, -0.55, -0.7),
}
JOINTS = ("right_shoulder_yaw", "right_shoulder_roll",
          "right_shoulder_pitch", "right_elbow_pitch")


def visible_pixels(renderer: mujoco.Renderer, data: mujoco.MjData,
                   camera: str, gid: int, option: mujoco.MjvOption) -> int:
    renderer.update_scene(data, camera=camera, scene_option=option)
    image = renderer.render()
    return int(np.count_nonzero(
        (image[:, :, 0] == gid) &
        (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))


def run() -> dict:
    design = json.loads(DESIGN.read_text())
    shell = json.loads(SHELL_REPORT.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("Current selected design is not Cove-50")
    if shell["source_visual_body_sha256"] != design["evidence"]["visual_body_sha256"]:
        raise ValueError("Saved shell does not overlay the selected visual body")
    if sha256(SHELL_VISUAL) != shell["shell_visual_body_sha256"]:
        raise ValueError("Saved shell visual body changed")
    frame_report = ROOT / design["evidence"]["report"]
    if sha256(frame_report) != design["evidence"]["report_sha256"]:
        raise ValueError("Selected frame report changed")
    trace_info = json.loads(frame_report.read_text())
    trace = ROOT / trace_info["trace"]
    if sha256(trace) != trace_info["trace_sha256"]:
        raise ValueError("Saved pose trace changed")

    model = load_model(SHELL_VISUAL, targets=TARGETS)
    data = mujoco.MjData(model)
    poses = np.load(trace, allow_pickle=False)["qpos"]
    if poses.shape[1] != model.nq:
        raise ValueError("Saved pose trace does not match shell joint ordering")
    sampled = np.unique(np.linspace(0, len(poses) - 1, 5).round().astype(int))
    pose_cases = [("neutral", -1, model.qpos0.copy())] + [
        ("saved_walk", int(i), poses[i].copy()) for i in sampled]
    joint_addresses = [int(model.jnt_qposadr[model.joint(name).id]) for name in JOINTS]
    for address, name in zip(joint_addresses, JOINTS, strict=True):
        joint = model.joint(name)
        for angles in ARM_CASES.values():
            value = angles[JOINTS.index(name)]
            if not joint.range[0] <= value <= joint.range[1]:
                raise ValueError(f"{name} pose outside model limit")

    full = mujoco.MjvOption()
    bare = mujoco.MjvOption()
    bare.geomgroup[:] = 0
    bare.geomgroup[0] = 1
    renderer = mujoco.Renderer(model, width=640, height=480)
    renderer.enable_segmentation_rendering()
    rows = []
    try:
        for pose_label, pose_index, qpos in pose_cases:
            for arm_label, angles in ARM_CASES.items():
                data.qpos[:] = qpos
                for address, value in zip(joint_addresses, angles, strict=True):
                    data.qpos[address] = value
                mujoco.mj_forward(model, data)
                jaw = data.xpos[model.body("right_moving_jaw").id].copy()
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                basis = rotation.reshape(3, 3)[:2, :2]
                for target, (relative, _) in TARGETS.items():
                    if target == "near_right_paw_3cm":
                        # Proxy object 4 cm forward of the moving jaw, at jaw height.
                        xyz = (jaw + rotation.reshape(3, 3)[:, 0] * 0.04).tolist()
                    else:
                        xy = data.qpos[:2] + basis @ np.asarray(relative[:2])
                        xyz = [float(xy[0]), float(xy[1]), relative[2]]
                    for other in TARGETS:
                        model.body_pos[model.body(f"probe_{other}").id] = (
                            xyz if other == target else (100, 0, 0))
                    mujoco.mj_forward(model, data)
                    gid = model.geom(f"probe_{target}").id
                    for camera in CAMERAS:
                        clear = visible_pixels(renderer, data, camera, gid, bare)
                        with_robot = visible_pixels(renderer, data, camera, gid, full)
                        rows.append({
                            "pose": pose_label, "saved_pose_index": pose_index,
                            "arm_case": arm_label, "target": target, "camera": camera,
                            "target_center_m": xyz,
                            "jaw_center_m": jaw.tolist(),
                            "unoccluded_pixels": clear,
                            "visible_pixels": with_robot,
                            "visible_fraction_of_unoccluded":
                                round(with_robot / clear, 3) if clear else None,
                        })
    finally:
        renderer.close()
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for camera in CAMERAS:
            sample = [r for r in rows if r["target"] == target and r["camera"] == camera]
            summary[target][camera] = {
                "pose_cases": len(sample),
                "unoccluded_visible": sum(r["unoccluded_pixels"] > 0 for r in sample),
                "actual_visible": sum(r["visible_pixels"] > 0 for r in sample),
                "mostly_self_occluded": sum(
                    r["unoccluded_pixels"] > 0 and
                    r["visible_pixels"] < 0.2 * r["unoccluded_pixels"] for r in sample),
                "minimum_visible_pixels": min(r["visible_pixels"] for r in sample),
            }
    return {
        "design_id": design["id"],
        "shell_visual_sha256": shell["shell_visual_body_sha256"],
        "trace_sha256": trace_info["trace_sha256"],
        "camera_names": CAMERAS,
        "arm_joints": JOINTS,
        "arm_cases_rad": ARM_CASES,
        "target_definitions": TARGETS,
        "scope": "Kinematic right-arm sensitivity on selected-body/saved-shell visual MJCF; "
                 "neutral plus five older-policy saved walk qpos, with arm angles overwritten. "
                 "The 3 cm near-paw proxy moves with the jaw and can be above the floor. "
                 "Ideal segmentation pixels, not detector/recognition, collision-free reach, "
                 "physical optics, blur, or a new Cove-50 policy rollout.",
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
