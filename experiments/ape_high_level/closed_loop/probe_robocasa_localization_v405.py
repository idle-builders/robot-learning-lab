"""Compare command odometry, RGB-D odometry and planar range in RoboCasa.

The eye rig follows the same commanded aisle route under paired zero/cumulative
slip. Only rendered RGB/depth and limited-range rays reach the estimators;
actual x is evaluator-only. Fixed heading is a known mount/IMU assumption, not
a measured ape gait. This is a localization sensor comparison, not a physical
navigation or collision-safety result.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import math
import random
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
from probe_robocasa_eye_height import WIDTH, HEIGHT, scene  # noqa: E402
from localization import ScanLocalizer  # noqa: E402
from rgbd_odometry import RGBDOdometry  # noqa: E402


OUT = ROOT / "artifacts/ape-robocasa-localization-v405"
START_X, END_X, Y = 1.05, 4.35, -2.30
STEPS = 48
DELTA_T_S = .4
LIDAR_Z_M = .30
LIDAR_BEAMS = 72
LIDAR_RANGE_PROFILES_M = (2.2, 3.0, 5.0)
LIDAR_NOISE_STD_M = .005
# Camera axes fixed by the MJCF mount: +x right, +z up, +y forward.
CAMERA_ROTATION = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RangeLocalizer(ScanLocalizer):
    def __init__(self, max_range_m: float) -> None:
        super().__init__()
        self.max_range_m = max_range_m

    def endpoints(self, scan: list[float | None], pose: np.ndarray) -> np.ndarray:
        angles = np.linspace(-math.pi, math.pi, len(scan), endpoint=False)
        return np.array([
            pose[:2] + distance * np.array([
                math.cos(pose[2] + angle), math.sin(pose[2] + angle)])
            for angle, distance in zip(angles, scan)
            if distance is not None and .14 < distance < self.max_range_m
        ], dtype=float).reshape(-1, 2)


def render_rgbd(renderer: mujoco.Renderer, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    renderer.disable_segmentation_rendering()
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera="head")
    rgb = renderer.render().copy()
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="head")
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    return rgb, depth


def scan(model: mujoco.MjModel, data: mujoco.MjData, x: float,
         max_range_m: float, rng: np.random.Generator) -> list[float | None]:
    origin = np.array([x, Y, LIDAR_Z_M], dtype=np.float64)
    hits = []
    for angle in np.linspace(-np.pi, np.pi, LIDAR_BEAMS, endpoint=False):
        vector = np.array([np.cos(angle), np.sin(angle), 0.], dtype=np.float64)
        geom_id = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, vector, None, 1, -1, geom_id)
        noise = rng.normal(0., LIDAR_NOISE_STD_M)
        hits.append(float(max(.01, distance + noise))
                    if .14 < distance < max_range_m else None)
    return hits


def visual(rgb: np.ndarray, row: dict, style: int, slip_m: float,
           lidar_max_m: float) -> np.ndarray:
    frame = Image.new("RGB", (WIDTH, HEIGHT + 96), "#12191c")
    frame.paste(Image.fromarray(rgb), (0, 0))
    draw = ImageDraw.Draw(frame)
    draw.text((12, HEIGHT + 8), f"style {style}; slip {slip_m:.2f} m; lidar {lidar_max_m:.1f} m; t={row['time_s']:.1f} s",
              fill="white")
    labels = (("true", row["actual_x_m"], "#46e58b"),
              ("command", row["command_x_m"], "#f4a444"),
              ("RGB-D", row["rgbd_x_m"], "#48bcff"),
              ("range", row["range_x_m"], "#caa4ff"))
    for i, (name, value, color) in enumerate(labels):
        x = 15 + 150 * i
        draw.text((x, HEIGHT + 31), f"{name}: {value:.3f} m", fill=color)
        px = 15 + (value - .9) / 3.7 * 610
        draw.ellipse((px - 4, HEIGHT + 63, px + 4, HEIGHT + 71), fill=color)
    draw.line((15, HEIGHT + 67, 625, HEIGHT + 67), fill="#66757d", width=2)
    return np.asarray(frame)


def episode(style: int, slip_m: float, lidar_max_m: float) -> dict:
    # RoboCasa fixtures use some global random draws in addition to its
    # supplied Generator. Reset both sources before each paired scene.
    scene_seed = 3100 + style
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    model = scene(style_id=style)
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera_id = model.camera("head").id
    visual_odom = RGBDOdometry(width=WIDTH, height=HEIGHT, fovy_deg=70.)
    range_odom = RangeLocalizer(lidar_max_m)
    rng = np.random.default_rng(1000 + style * 100 + int(slip_m * 1000))
    command_est = np.array([START_X, Y, 0.], dtype=float)
    rgbd_est = command_est.copy()
    range_est = command_est.copy()
    trace: list[dict] = []
    frames: list[np.ndarray] = []
    previous_command_x = START_X
    for step in range(STEPS):
        fraction = step / (STEPS - 1)
        command_x = START_X + fraction * (END_X - START_X)
        actual_x = command_x + slip_m * fraction ** 1.4
        model.cam_pos[camera_id, 0] = actual_x
        mujoco.mj_forward(model, data)
        rgb, depth = render_rgbd(renderer, data)
        observed_scan = scan(model, data, actual_x, lidar_max_m, rng)
        commanded_delta = command_x - previous_command_x
        previous_command_x = command_x
        command_est[0] += commanded_delta
        rgbd_predicted = rgbd_est.copy()
        rgbd_predicted[0] += commanded_delta
        range_predicted = range_est.copy()
        range_predicted[0] += commanded_delta
        rgbd_est, visual_status = visual_odom.correct(
            Image.fromarray(rgb), depth, rgbd_predicted, CAMERA_ROTATION)
        range_est, range_status = range_odom.correct(observed_scan, range_predicted)
        row = {"step": step, "time_s": round(step * DELTA_T_S, 3),
               "command_x_m": float(command_est[0]), "actual_x_m": float(actual_x),
               "rgbd_x_m": float(rgbd_est[0]), "range_x_m": float(range_est[0]),
               "command_error_m": float(abs(command_est[0] - actual_x)),
               "rgbd_error_m": float(abs(rgbd_est[0] - actual_x)),
               "range_error_m": float(abs(range_est[0] - actual_x)),
               "rgbd_status": visual_status, "range_status": range_status,
               "range_returns": sum(x is not None for x in observed_scan),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "depth_valid_pixels": int(np.count_nonzero(np.isfinite(depth) & (depth > 0) & (depth < 2.8)))}
        trace.append(row)
        frames.append(visual(rgb, row, style, slip_m, lidar_max_m))
    renderer.close()
    video = OUT / f"style{style}-slip{int(slip_m * 100):02d}cm-range{lidar_max_m:.1f}m.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {"style": style, "slip_m": slip_m, "lidar_max_m": lidar_max_m,
            "scene_seed": scene_seed, "scene_fingerprint": scene_fingerprint,
            "camera_mount_rotation": CAMERA_ROTATION.tolist(),
            "final_error_m": {source: trace[-1][f"{source}_error_m"]
                              for source in ("command", "rgbd", "range")},
            "mean_error_m": {source: float(np.mean([r[f"{source}_error_m"] for r in trace]))
                             for source in ("command", "rgbd", "range")},
            "registered_frames": {
                "rgbd": sum(r["rgbd_status"]["status"] == "registered" for r in trace),
                "range": sum(r["range_status"]["status"] == "registered" for r in trace)},
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "trace": trace}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    for style in (1, 2, 3):
        for slip in (0., .12):
            for lidar_max in LIDAR_RANGE_PROFILES_M:
                result = episode(style, slip, lidar_max)
                episodes.append(result)
                print(style, slip, lidar_max, result["final_error_m"],
                      result["registered_frames"], flush=True)
                (OUT / "report.partial.json").write_text(json.dumps(episodes, indent=2) + "\n")
    report = {"scope": "paired kinematic camera-pose localization screen; no robot gait or action gate",
              "source_sha256": sha256(Path(__file__)),
              "scene_source_sha256": sha256(Path(__file__).with_name("probe_robocasa_eye_height.py")),
              "configuration": {"steps": STEPS, "dt_s": DELTA_T_S,
                                "lidar_beams": LIDAR_BEAMS, "lidar_height_m": LIDAR_Z_M,
                                "lidar_range_profiles_m": LIDAR_RANGE_PROFILES_M,
                                "lidar_noise_std_m": LIDAR_NOISE_STD_M,
                                "fovy_deg": 70, "camera_pose_from": "fixed calibrated mount"},
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
