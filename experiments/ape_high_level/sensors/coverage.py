"""Measure task-target visibility through recorded ape poses.

This is an optical/occlusion screen. The added props are evaluator markers, not
recognized objects; no result from this script establishes semantic perception.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BODY = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_elbows_small_head.xml"
DEFAULT_TRACES = {
    "walk": ROOT / "artifacts/ape-self-contact-replay-20260930/native-forward-walk-v3c.npz",
    "squat": ROOT / "artifacts/ape-self-contact-replay-20260930/native-squat-v3c.npz",
}
WIDTH, HEIGHT = 640, 360
VFOV_DEG = 67.0

# MuJoCo camera local +X points image-right, +Y points image-up, -Z looks out.
# Angles are elevation above the horizontal in the head's neutral coordinates.
CAMERAS = {"down25": -25.0, "down40": -40.0, "down55": -55.0,
           "up20": 20.0, "up35": 35.0}


def _xyaxes(elevation_deg: float) -> str:
    angle = np.deg2rad(elevation_deg)
    return f"0 -1 0 {-np.sin(angle):.9f} 0 {np.cos(angle):.9f}"


def make_model(body: Path) -> mujoco.MjModel:
    root = ET.parse(body).getroot()
    head = next((b for b in root.iter("body") if b.get("name") == "head"), None)
    if head is None:
        raise ValueError(f"No body named 'head' in {body}")
    for name, elevation in CAMERAS.items():
        ET.SubElement(head, "camera", name=name, pos="0.063 0 0.012",
                      xyaxes=_xyaxes(elevation), fovy=str(VFOV_DEG))
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Model has no worldbody")
    for name, geom_type, size, rgba in (
        ("floor_object", "box", "0.01 0.01 0.01", "0.78 0.61 0.4 1"),
        ("person_face", "sphere", "0.095", "0.73 0.48 0.36 1"),
    ):
        body_node = ET.SubElement(world, "body", name=name, pos="10 0 0")
        ET.SubElement(body_node, "geom", name=name, type=geom_type,
                      size=size, rgba=rgba, contype="0", conaffinity="0")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def _set_props(model: mujoco.MjModel, trace: str, initial_qpos: np.ndarray) -> None:
    if trace == "walk":
        floor_xyz, face_xyz = (1.24, 0.0, 0.01), (1.7, 0.0, 1.5)
    elif trace == "squat":
        floor_xyz, face_xyz = (0.20, -0.05, 0.01), (1.0, 0.0, 1.5)
    else:
        raise ValueError(trace)
    rotation_flat = np.empty(9)
    mujoco.mju_quat2Mat(rotation_flat, initial_qpos[3:7])
    forward_xy = rotation_flat.reshape(3, 3)[:2, 0]
    forward_xy /= np.linalg.norm(forward_xy)
    left_xy = np.array([-forward_xy[1], forward_xy[0]])
    origin = np.array([initial_qpos[0], initial_qpos[1], 0.0])
    for name, xyz in (("floor_object", floor_xyz), ("person_face", face_xyz)):
        world_xy = origin[:2] + xyz[0] * forward_xy + xyz[1] * left_xy
        model.body_pos[model.body(name).id] = [*world_xy, xyz[2]]


def _sample_indices(length: int, count: int) -> np.ndarray:
    return np.unique(np.linspace(0, length - 1, min(count, length)).round().astype(int))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _project_center(model: mujoco.MjModel, data: mujoco.MjData,
                    camera_name: str, world_xyz: np.ndarray) -> dict:
    camera_id = model.camera(camera_name).id
    pos = data.cam_xpos[camera_id]
    rotation = data.cam_xmat[camera_id].reshape(3, 3)
    local = rotation.T @ (world_xyz - pos)
    if local[2] >= 0:
        return {"in_frustum": False, "forward_m": float(-local[2])}
    focal = HEIGHT / (2 * np.tan(np.deg2rad(VFOV_DEG) / 2))
    u = WIDTH / 2 + focal * local[0] / -local[2]
    v = HEIGHT / 2 - focal * local[1] / -local[2]
    return {"in_frustum": bool(0 <= u < WIDTH and 0 <= v < HEIGHT),
            "u": float(u), "v": float(v), "forward_m": float(-local[2])}


def measure(body: Path, traces: dict[str, Path], samples: int) -> dict:
    model = make_model(body)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.enable_segmentation_rendering()
    geom_ids = {name: model.geom(name).id for name in ("floor_object", "person_face")}
    head_pitch_id = model.joint("head_pitch").id if "head_pitch" in [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ] else None
    head_pitch_addr = int(model.jnt_qposadr[head_pitch_id]) if head_pitch_id is not None else None
    rows = []
    try:
        for trace_name, path in traces.items():
            if trace_name not in ("walk", "squat"):
                raise ValueError(f"Unknown target layout for trace {trace_name}")
            archive = np.load(path, allow_pickle=False)
            poses = archive["qpos"]
            if poses.ndim != 2 or poses.shape[1] != model.nq:
                raise ValueError(f"{path}: qpos shape {poses.shape}, model nq {model.nq}")
            _set_props(model, trace_name, poses[0])
            for frame in _sample_indices(len(poses), samples):
                for camera_name in CAMERAS:
                    data.qpos[:] = poses[frame]
                    # The baseline is a yaw-only camera. Up/down alternatives
                    # are hypothetical optical orientations, not motor actions.
                    if head_pitch_addr is not None:
                        data.qpos[head_pitch_addr] = 0.0
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera_name)
                    segmentation = renderer.render()
                    targets = {}
                    for name, geom_id in geom_ids.items():
                        pixels = int(np.count_nonzero(
                            (segmentation[:, :, 0] == geom_id)
                            & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
                        ))
                        targets[name] = {
                            "visible_pixels": pixels,
                            **_project_center(model, data, camera_name,
                                              data.geom_xpos[geom_id]),
                        }
                    rows.append({"trace": trace_name, "frame": int(frame),
                                 "camera": camera_name,
                                 "camera_z_m": float(data.cam_xpos[model.camera(camera_name).id, 2]),
                                 "root_xyz_m": data.qpos[:3].astype(float).tolist(),
                                 "targets": targets})
    finally:
        renderer.close()
    summary = {}
    for trace_name in traces:
        summary[trace_name] = {}
        for camera_name in CAMERAS:
            subset = [r for r in rows if r["trace"] == trace_name and r["camera"] == camera_name]
            summary[trace_name][camera_name] = {
                name: {"frames_with_pixels": sum(r["targets"][name]["visible_pixels"] > 0 for r in subset),
                       "frames": len(subset),
                       "median_visible_pixels": float(np.median([
                           r["targets"][name]["visible_pixels"] for r in subset])),
                       "minimum_visible_pixels": min(r["targets"][name]["visible_pixels"] for r in subset)}
                for name in geom_ids
            }
    return {"scope": "optical/occlusion screen only; no semantic perception or physical sensor qualification",
            "body": {"path": str(body), "sha256": _sha256(body)},
            "traces": {k: {"path": str(v), "sha256": _sha256(v)} for k, v in traces.items()},
            "image_size": [WIDTH, HEIGHT], "vertical_fov_deg": VFOV_DEG,
            "camera_elevations_deg": CAMERAS,
            "target_size": "20 mm floor cube; 190 mm face sphere proxy",
            "summary": summary, "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    result = measure(args.body, DEFAULT_TRACES, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
