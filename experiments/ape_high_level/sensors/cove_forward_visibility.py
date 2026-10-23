"""Screen fixed-world floor targets through a saved Cove-50 forward walk.

This is a selected-body policy pose trace, not a new control rollout. Ideal
segmentation pixels through a virtual chest camera cannot establish recognition,
blur, physical optics, camera fit, stopping range or a safe approach.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from cove_view_alternatives import (DESIGN, ROOT, SHELL_REPORT, SHELL_VISUAL,
                                    load_model, sha256)


TRANSFER = ROOT / "artifacts/ape-cove50-transfer-20261008"
NATIVE = TRANSFER / "varied150-forward-seed144.json"
TRACE = TRANSFER / "varied150-forward-seed144.npz"
CAMERAS = ("lower_rgb", "chest_low50_x45")
TARGETS = {
    "large_40cm": (0.40, 0.0, 0.04),
    "large_80cm": (0.80, 0.0, 0.04),
    "large_right_80cm": (0.80, -0.15, 0.04),
    "small_80cm": (0.80, 0.0, 0.015),
}


def check_sources() -> tuple[dict, dict, dict]:
    design = json.loads(DESIGN.read_text())
    shell = json.loads(SHELL_REPORT.read_text())
    transfer = json.loads((TRANSFER / "report.json").read_text())
    native = json.loads(NATIVE.read_text())
    if (design["id"] != "cove-50-direct-frame" or
        transfer["selected_design"]["id"] != design["id"] or
        transfer["selected_design"]["pointer_sha256"] != sha256(DESIGN) or
        transfer["selected_design"]["cad_generator_sha256"] !=
            sha256(ROOT / design["generator"]["path"]) or
        shell["source_visual_body_sha256"] != design["evidence"]["visual_body_sha256"] or
        shell["shell_visual_body_sha256"] != sha256(SHELL_VISUAL) or
        transfer["body"]["training_xml_sha256"] != native["body_sha256"] or
        transfer["body"]["training_xml_sha256"] !=
            sha256(ROOT / transfer["body"]["training_xml"]) or
        transfer["forward_continuation"]["checkpoint_sha256"] !=
            sha256(ROOT / transfer["forward_continuation"]["checkpoint"]) or
        native["seed"] != 144 or native["speed_command_m_s"] != 0.35 or
        native["stop_at_s"] is not None or len(native["worlds"]) != 4):
        raise ValueError("Selected body, shell or forward-policy evidence changed")
    return design, transfer, native


def visibility(renderer: mujoco.Renderer, data: mujoco.MjData,
               camera: str, geom_id: int, option: mujoco.MjvOption) -> int:
    renderer.update_scene(data, camera=camera, scene_option=option)
    pixels = renderer.render()
    return int(np.count_nonzero(
        (pixels[:, :, 0] == geom_id) &
        (pixels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))


def project_center(model: mujoco.MjModel, data: mujoco.MjData,
                   camera: str, target_xyz: np.ndarray) -> tuple[float, float, bool]:
    """Project a world point through MuJoCo's local -Z camera at 640 x 480."""
    camera_id = model.camera(camera).id
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    x, y, z = rotation.T @ (target_xyz - data.cam_xpos[camera_id])
    if z >= 0:
        return float("nan"), float("nan"), False
    focal = 480 / (2 * math.tan(math.radians(float(model.cam_fovy[camera_id])) / 2))
    u, v = 320 + focal * x / -z, 240 - focal * y / -z
    return float(u), float(v), bool(0 <= u < 640 and 0 <= v < 480)


def range_band(forward_m: float) -> str:
    if forward_m <= 0:
        return "behind"
    if forward_m < 0.12:
        return "ahead_under_12cm"
    if forward_m < 0.25:
        return "ahead_12_to_25cm"
    if forward_m < 0.40:
        return "ahead_25_to_40cm"
    if forward_m < 0.80:
        return "ahead_40_to_80cm"
    return "ahead_over_80cm"


def run() -> dict:
    design, transfer, native = check_sources()
    model = load_model(SHELL_VISUAL, targets={
        name: ((forward, lateral, radius), radius)
        for name, (forward, lateral, radius) in TARGETS.items()})
    training = mujoco.MjModel.from_xml_path(str(ROOT / transfer["body"]["training_xml"]))
    joints = [(model.joint(i).name, int(model.jnt_qposadr[i]), int(model.jnt_type[i]))
              for i in range(model.njnt)]
    training_joints = [(training.joint(i).name, int(training.jnt_qposadr[i]),
                       int(training.jnt_type[i])) for i in range(training.njnt)]
    if model.nq != training.nq or joints != training_joints:
        raise ValueError("Saved policy pose cannot be transferred to the shell visual body")
    saved = np.load(TRACE, allow_pickle=False)
    qpos = saved["qpos"]
    dt = float(saved["dt"])
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or len(qpos) != 400 or not math.isclose(dt, .02):
        raise ValueError("Expected one world of the saved 8 s / 50 Hz Cove trace")
    if not np.all(np.isfinite(qpos)):
        raise ValueError("Nonfinite saved pose")

    start = qpos[0]
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, start[3:7])
    start_basis = rotation.reshape(3, 3)[:2, :2]
    world_targets = {
        name: (start[:2] + start_basis @ np.array([forward, lateral]), radius)
        for name, (forward, lateral, radius) in TARGETS.items()}
    full = mujoco.MjvOption()
    probe_only = mujoco.MjvOption()
    probe_only.geomgroup[:] = 0
    probe_only.geomgroup[0] = 1
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    renderer.enable_segmentation_rendering()
    rows = []
    projection_sanity = None
    try:
        for index in range(0, len(qpos), 5):  # 100 ms, including near-range crossings.
            data.qpos[:] = qpos[index]
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            basis = rotation.reshape(3, 3)[:2, :2]
            for name, (world_xy, radius) in world_targets.items():
                for other in TARGETS:
                    model.body_pos[model.body(f"probe_{other}").id] = (
                        (float(world_xy[0]), float(world_xy[1]), radius)
                        if other == name else (100.0, 0.0, 0.0))
                mujoco.mj_forward(model, data)
                relative = basis.T @ (world_xy - data.qpos[:2])
                geom_id = model.geom(f"probe_{name}").id
                for camera in CAMERAS:
                    clear = visibility(renderer, data, camera, geom_id, probe_only)
                    observed = visibility(renderer, data, camera, geom_id, full)
                    rows.append({
                        "time_s": round(index * dt, 3), "pose_index": index,
                        "target": name, "camera": camera,
                        "target_world_xy_m": [float(x) for x in world_xy],
                        "forward_in_body_m": round(float(relative[0]), 4),
                        "lateral_in_body_m": round(float(relative[1]), 4),
                        "range_band": range_band(float(relative[0])),
                        "unoccluded_pixels": clear, "visible_pixels": observed,
                    })
        # Check the pinhole projection against a rendered marker before using
        # its frame-to-frame displacement as an optical motion proxy.
        data.qpos[:] = qpos[0]
        target_xy, radius = world_targets["large_40cm"]
        for other in TARGETS:
            model.body_pos[model.body(f"probe_{other}").id] = (
                (float(target_xy[0]), float(target_xy[1]), radius)
                if other == "large_40cm" else (100.0, 0.0, 0.0))
        mujoco.mj_forward(model, data)
        projected = project_center(model, data, "lower_rgb",
                                   np.array([*target_xy, radius]))
        renderer.update_scene(data, camera="lower_rgb", scene_option=full)
        segmented = renderer.render()
        gid = model.geom("probe_large_40cm").id
        pixels = np.argwhere((segmented[:, :, 0] == gid) &
                             (segmented[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))
        if not len(pixels) or not projected[2]:
            raise ValueError("Projection sanity marker is not visible")
        centroid = pixels.mean(axis=0)[::-1]
        error_px = math.hypot(projected[0] - centroid[0], projected[1] - centroid[1])
        if error_px > 2:
            raise ValueError("Camera projection disagrees with rendered segmentation")
        projection_sanity = {"projected_center_px": list(projected[:2]),
                             "segmentation_centroid_px": centroid.tolist(),
                             "center_error_px": round(error_px, 3)}
    finally:
        renderer.close()

    projected_tracks = {(target, camera): [] for target in TARGETS for camera in CAMERAS}
    for index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        mujoco.mju_quat2Mat(rotation, pose[3:7])
        basis = rotation.reshape(3, 3)[:2, :2]
        for target, (world_xy, radius) in world_targets.items():
            forward = float((basis.T @ (world_xy - pose[:2]))[0])
            xyz = np.array([float(world_xy[0]), float(world_xy[1]), radius])
            for camera in CAMERAS:
                u, v, in_frame = project_center(model, data, camera, xyz)
                projected_tracks[target, camera].append(
                    (range_band(forward), u, v, in_frame))

    motion_summary = {}
    for target in TARGETS:
        motion_summary[target] = {}
        for camera in CAMERAS:
            track = projected_tracks[target, camera]
            motion_summary[target][camera] = {}
            for band in ("ahead_over_80cm", "ahead_40_to_80cm", "ahead_25_to_40cm",
                         "ahead_12_to_25cm", "ahead_under_12cm"):
                displacements = [math.hypot(b[1] - a[1], b[2] - a[2])
                                 for a, b in zip(track, track[1:])
                                 if a[0] == band and b[0] == band and a[3] and b[3]]
                motion_summary[target][camera][band] = {
                    "in_frame_consecutive_20ms_pairs": len(displacements),
                    "median_displacement_px_per_20ms":
                        round(float(np.median(displacements)), 2) if displacements else None,
                    "p95_displacement_px_per_20ms":
                        round(float(np.percentile(displacements, 95)), 2)
                        if displacements else None,
                    "max_displacement_px_per_20ms":
                        round(max(displacements), 2) if displacements else None,
                }

    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for camera in CAMERAS:
            sample = [r for r in rows if r["target"] == target and r["camera"] == camera]
            summary[target][camera] = {}
            for band in ("ahead_over_80cm", "ahead_40_to_80cm", "ahead_25_to_40cm",
                         "ahead_12_to_25cm", "ahead_under_12cm", "behind"):
                subset = [r for r in sample if r["range_band"] == band]
                summary[target][camera][band] = {
                    "frames": len(subset),
                    "visible_frames": sum(r["visible_pixels"] > 0 for r in subset),
                    "in_optical_view_frames": sum(r["unoccluded_pixels"] > 0 for r in subset),
                    "mostly_self_occluded_frames": sum(
                        r["unoccluded_pixels"] > 0 and
                        r["visible_pixels"] < .2 * r["unoccluded_pixels"] for r in subset),
                    "median_visible_pixels": float(np.median(
                        [r["visible_pixels"] for r in subset])) if subset else None,
                }
    return {
        "selected_design": design["id"],
        "sources": {
            "design_pointer_sha256": sha256(DESIGN),
            "shell_visual_sha256": sha256(SHELL_VISUAL),
            "body_training_xml_sha256": sha256(ROOT / transfer["body"]["training_xml"]),
            "checkpoint_sha256": transfer["forward_continuation"]["checkpoint_sha256"],
            "native_rollout_sha256": sha256(NATIVE),
            "native_trace_sha256": sha256(TRACE),
        },
        "trace_frames": len(qpos), "sampled_frames": len(range(0, len(qpos), 5)),
        "native_worlds": len(native["worlds"]), "visualized_world": 0,
        "target_definitions": TARGETS,
        "scope": "One saved selected-body forward-policy qpos trace, with each floor target "
                 "held fixed in world coordinates. The chest view is virtual; segmentation "
                 "of non-colliding markers at 640x480 is only a view/occlusion screen. "
                 "Frame-to-frame center displacement is a projection-motion proxy, not measured "
                 "exposure blur or rolling-shutter distortion. No actual motion blur, "
                 "recognition, physical camera, target avoidance, stop decision or owner trial.",
        "projection_sanity": projection_sanity,
        "summary": summary, "projected_motion": motion_summary, "rows": rows,
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
