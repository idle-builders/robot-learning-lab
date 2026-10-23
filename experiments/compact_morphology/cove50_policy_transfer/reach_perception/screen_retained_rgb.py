"""Offline selected-camera visibility on retained native squat-to-reach states.

Cube truth is used to score projection and choose an optimistic legal head-yaw
upper bound. No cube truth is exposed as a controller goal and no policy runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-camera-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
POINTER = ROOT / "robot/ape-current-design.json"
CHECKPOINT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v7/actor.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_with_visual_cube() -> mujoco.MjModel:
    root = ET.parse(BODY).getroot()
    cube = ET.SubElement(root.find("worldbody"), "body", name="offline_visual_cube")
    ET.SubElement(cube, "freejoint", name="offline_visual_cube_free")
    ET.SubElement(cube, "geom", name="offline_visual_cube_geom", type="box",
                  size="0.01 0.01 0.01", mass="0.00625",
                  rgba="0.95 0.36 0.05 1", group="0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    assert model.nq == 35 and model.ncam == 2
    # The training XML hides physics proxies. Make them visible in this
    # in-memory camera replay so head/body/cases can occlude the cube.
    for geom in range(model.ngeom):
        if (model.geom_bodyid[geom] != 0 and model.geom_group[geom] == 4
                and model.geom_rgba[geom, 3] == 0):
            model.geom_rgba[geom, 3] = 1
            model.geom_group[geom] = 0
    return model


def project(model: mujoco.MjModel, data: mujoco.MjData, camera_name: str,
            point_w: np.ndarray, width: int, height: int) -> dict:
    camera = model.camera(camera_name).id
    # MuJoCo camera +X is image right, +Y image up, -Z optical forward.
    # p_C = R_W_C.T (p_W - t_W_C), with R_W_C at image capture time.
    rotation_w_c = data.cam_xmat[camera].reshape(3, 3)
    point_c = rotation_w_c.T @ (point_w - data.cam_xpos[camera])
    depth = -float(point_c[2])
    focal = (height / 2) / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    if depth <= 0:
        return {"in_front": False, "center_in_frame": False,
                "camera_xyz_m": np.round(point_c, 5).tolist(),
                "depth_m": round(depth, 5), "uv_px": None,
                "pixel_edge_margin": None}
    u = (width - 1) / 2 + focal * point_c[0] / depth
    v = (height - 1) / 2 - focal * point_c[1] / depth
    margin = min(u, width - 1 - u, v, height - 1 - v)
    return {"in_front": True, "center_in_frame": bool(margin >= 0),
            "camera_xyz_m": np.round(point_c, 5).tolist(),
            "depth_m": round(depth, 5),
            "uv_px": np.round([u, v], 2).tolist(),
            "pixel_edge_margin": round(float(margin), 2)}


def orange_bbox(rgb: np.ndarray) -> dict | None:
    """Known synthetic orange proxy only; no semantic recognition claim."""
    red = rgb[:, :, 0].astype(int)
    green = rgb[:, :, 1].astype(int)
    blue = rgb[:, :, 2].astype(int)
    mask = (red > 100) & (red > green * 1.25) & (green > blue * 1.25)
    yy, xx = np.nonzero(mask)
    if len(xx) < 10:
        return None
    return {"xywh_px": [int(xx.min()), int(yy.min()),
                        int(xx.max() - xx.min() + 1), int(yy.max() - yy.min() + 1)],
            "centroid_uv_px": [round(float(xx.mean()), 2), round(float(yy.mean()), 2)],
            "pixels": int(len(xx))}


def estimate_on_known_plane(bbox: dict | None, model: mujoco.MjModel,
                            data: mujoco.MjData, camera_name: str,
                            width: int, height: int, plane_z: float,
                            pose_healthy: bool = True) -> list[float] | None:
    if bbox is None or not pose_healthy:
        return None
    camera = model.camera(camera_name).id
    focal = (height / 2) / np.tan(np.deg2rad(model.cam_fovy[camera]) / 2)
    u, v = bbox["centroid_uv_px"]
    ray_c = np.array([(u - (width - 1) / 2) / focal,
                      -((v - (height - 1) / 2) / focal), -1.0])
    ray_w = data.cam_xmat[camera].reshape(3, 3) @ ray_c
    origin_w = data.cam_xpos[camera]
    if ray_w[2] >= -0.05:
        return None
    scale = (plane_z - origin_w[2]) / ray_w[2]
    if scale <= 0:
        return None
    return np.round(origin_w + scale * ray_w, 5).tolist()


def main() -> None:
    plan_path = OUT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert sha(POINTER) == plan["selected_pointer_sha256"]
    assert sha(BODY) == plan["selected_body_sha256"]
    assert sha(CHECKPOINT) == plan["saved_reach_checkpoint_sha256"]
    width, height = plan["image_width_px"], plan["image_height_px"]
    model = model_with_visual_cube()
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[:] = 1
    head = model.joint("head_yaw").qposadr[0]
    yaw_lo, yaw_hi = model.joint("head_yaw").range
    yaw_grid = np.linspace(yaw_lo, yaw_hi, 61)
    cameras = ("upper_rgb", "lower_rgb")
    cases = []
    images = []
    for trace_name in plan["development_traces"]:
        trace_path = ROOT / trace_name
        report_path = trace_path.with_suffix(".json")
        with np.load(trace_path) as trace:
            qpos = trace["qpos"].astype(float)
            phases = trace["phase"].tolist()
            dt = float(trace["step_dt_s"])
        assert qpos.shape[1] == model.nq and abs(dt - 0.02) < 1e-8
        report = json.loads(report_path.read_text())
        assert report["body_sha256"] == plan["selected_body_sha256"]
        seed = report["seed"]
        rows = []
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for time_s in plan["sample_times_s"]:
                frame = round(time_s / dt)
                data.qpos[:] = qpos[frame]
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                saved_head_yaw = float(data.qpos[head])
                cube_w = data.qpos[28:31].copy()
                direct = {camera: project(model, data, camera, cube_w, width, height)
                          for camera in cameras}
                rendered = {}
                for camera in cameras:
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    rgb = renderer.render().copy()
                    bbox = orange_bbox(rgb)
                    estimated = estimate_on_known_plane(
                        bbox, model, data, camera, width, height,
                        plan["known_plane_cube_center_z_m"])
                    error_mm = None if estimated is None else round(float(
                        np.linalg.norm((np.asarray(estimated) - cube_w)[:2]) * 1000), 3)
                    rendered[camera] = {
                        "orange_bbox": bbox,
                        "estimated_cube_xyz_m": estimated,
                        "xy_error_mm": error_mm,
                        "blank_image_returns_no_goal": estimate_on_known_plane(
                            None, model, data, camera, width, height,
                            plan["known_plane_cube_center_z_m"]) is None,
                        "pose_invalid_returns_no_goal": estimate_on_known_plane(
                            bbox, model, data, camera, width, height,
                            plan["known_plane_cube_center_z_m"], pose_healthy=False) is None,
                    }
                    if time_s in (7.5, 10.0) and seed in (9921, 9941, 9942):
                        image_path = OUT / f"seed-{seed}-{camera}-{time_s:g}s-saved-head.png"
                        imageio.imwrite(image_path, rgb)
                        images.append(str(image_path.relative_to(ROOT)))
                sweep = {}
                for camera in cameras:
                    visible = []
                    for yaw in yaw_grid:
                        data.qpos[head] = yaw
                        mujoco.mj_forward(model, data)
                        result = project(model, data, camera, cube_w, width, height)
                        if result["center_in_frame"]:
                            visible.append((float(yaw), result["pixel_edge_margin"], result["uv_px"]))
                    best = max(visible, key=lambda x: x[1]) if visible else None
                    sweep[camera] = {
                        "visible_yaw_count": len(visible),
                        "best_yaw_rad_offline_truth_upper_bound": None if best is None else round(best[0], 3),
                        "best_pixel_margin": None if best is None else best[1],
                        "best_uv_px": None if best is None else best[2],
                    }
                data.qpos[head] = saved_head_yaw
                mujoco.mj_forward(model, data)
                rows.append({"time_s": time_s, "frame": frame, "phase": phases[frame],
                             "saved_head_yaw_rad": round(saved_head_yaw, 5),
                             "cube_truth_world_xyz_m": np.round(cube_w, 5).tolist(),
                             "camera_capture_pose": {c: {
                                 "world_position_m": np.round(data.cam_xpos[model.camera(c).id], 5).tolist(),
                                 "rotation_world_from_camera_rowmajor": np.round(
                                     data.cam_xmat[model.camera(c).id], 6).tolist()}
                                 for c in cameras},
                             "direct_projection": direct,
                             "direct_rgb": rendered,
                             "legal_yaw_sweep_truth_upper_bound": sweep})
        entry = next(row for row in rows if row["time_s"] == 7.5)
        reach = [row for row in rows if row["phase"] == "reach"]
        cases.append({
            "case": f"seed-{seed}", "seed": seed,
            "sources": {label: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                        for label, path in (("trace", trace_path), ("native_report", report_path))},
            "direct_entry_center_visible": any(
                entry["direct_projection"][c]["center_in_frame"] for c in cameras),
            "direct_reach_visible_fraction": round(sum(any(
                row["direct_projection"][c]["center_in_frame"] for c in cameras)
                for row in reach) / len(reach), 3),
            "legal_yaw_entry_visible_upper_bound": any(
                entry["legal_yaw_sweep_truth_upper_bound"][c]["visible_yaw_count"] > 0
                for c in cameras),
            "legal_yaw_reach_visible_fraction_upper_bound": round(sum(any(
                row["legal_yaw_sweep_truth_upper_bound"][c]["visible_yaw_count"] > 0
                for c in cameras) for row in reach) / len(reach), 3),
            "rows": rows,
        })
    direct_gate = all(c["direct_entry_center_visible"] and
                      c["direct_reach_visible_fraction"] >= 0.8 for c in cases)
    yaw_upper_bound = all(c["legal_yaw_entry_visible_upper_bound"] for c in cases)
    result = {
        "schema": "robot-learning-lab.cove50-retained-reach-rgb-screen.v1",
        "classification": "Retained native-state camera rendering and truth-scored offline projection; not image-driven control",
        "method": "Append only a 20 mm orange cube visual proxy to exact selected XML in saved free-joint order, reveal the hidden body proxies for occlusion, render upper/lower 320x240 RGB at retained 50 Hz qpos snapshots, use MuJoCo camera pose at that snapshot, and separately scan legal head yaw using evaluator cube truth as an optimistic visibility upper bound. Projection uses p_C = R_W_C^T (p_W-t_W_C), depth=-p_C.z; blank/invalid-pose inputs return no metric goal. No actor runs and no future/cube truth is a deployed goal.",
        "plan": str(plan_path.relative_to(ROOT)),
        "selected_pointer_sha256": plan["selected_pointer_sha256"],
        "selected_body_sha256": plan["selected_body_sha256"],
        "saved_reach_checkpoint_sha256": plan["saved_reach_checkpoint_sha256"],
        "direct_visibility_gate_pass": direct_gate,
        "legal_yaw_entry_visibility_upper_bound_pass": yaw_upper_bound,
        "images": images,
        "cases": cases,
    }
    (OUT / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for case in cases:
        print(case["case"], "direct entry", case["direct_entry_center_visible"],
              "direct reach", case["direct_reach_visible_fraction"],
              "yaw entry", case["legal_yaw_entry_visible_upper_bound"],
              "yaw reach", case["legal_yaw_reach_visible_fraction_upper_bound"])
    print("direct gate", direct_gate, "legal yaw entry upper bound", yaw_upper_bound)


if __name__ == "__main__":
    main()
