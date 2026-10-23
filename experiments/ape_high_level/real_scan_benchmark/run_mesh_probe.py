"""Run a bounded visual-route probe in a captured ARKit room.

This is a kinematic/sensor wiring experiment, not a photoreal or body-qualified
owner episode. The actor sees only rendered RGB/depth, simulated range, and
its integrated commands. The nav raster and true pose stay in the environment.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import open3d as o3d

from prepare_arkit_corpus import ASSETS


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "realistic_people"))
from fan_view_route import FanViewRoute  # noqa: E402


def xy_at(nav: dict, cell: np.ndarray) -> np.ndarray:
    return nav["xy_min_m"] + cell[::-1] * float(nav["resolution_m"])


def cell_at(nav: dict, xy: np.ndarray) -> np.ndarray:
    return np.rint((xy - nav["xy_min_m"]) / float(nav["resolution_m"]))[::-1].astype(int)


def in_free(nav: dict, xy: np.ndarray) -> bool:
    row, col = cell_at(nav, xy)
    free = nav["main_component"]
    return bool(0 <= row < free.shape[0] and 0 <= col < free.shape[1] and free[row, col])


def segment_free(nav: dict, start: np.ndarray, end: np.ndarray) -> bool:
    length = float(np.linalg.norm(end - start))
    return all(in_free(nav, start + fraction * (end - start))
               for fraction in np.linspace(0, 1, max(2, math.ceil(length / .025) + 1)))


def route_pair(nav: dict) -> tuple[np.ndarray, np.ndarray]:
    """Pick a repeatable line-of-sight pair before observing actor outcomes."""
    free = nav["main_component"].astype(np.uint8)
    interior = cv2.distanceTransform(free, cv2.DIST_L2, 5)
    cells = np.argwhere(interior >= 3)
    if len(cells) < 2:
        cells = np.argwhere(free)
    candidates = cells[::max(1, len(cells) // 100)]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for i, first in enumerate(candidates):
        a = xy_at(nav, first)
        for second in candidates[i + 1:]:
            b = xy_at(nav, second)
            distance = float(np.linalg.norm(b - a))
            if distance < 1.35 or distance > 2.45:
                continue
            if (best is None or distance > best[0]) and segment_free(nav, a, b):
                best = (distance, a, b)
    if best is None:
        raise ValueError("No 1.35–2.45 m provisional clear segment in room")
    _, a, b = best
    # Fixed order provides the same initial view on every rerun.
    return a, b


def sensing_scene(mesh_path: Path, target_xy: np.ndarray, floor_z: float):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    target = o3d.geometry.TriangleMesh.create_sphere(radius=.12, resolution=12)
    target.translate([float(target_xy[0]), float(target_xy[1]), floor_z + .14])
    target.paint_uniform_color([1., .08, .55])
    renderer = o3d.visualization.rendering.OffscreenRenderer(480, 360)
    renderer.scene.set_background([0., 0., 0., 1.])
    surface = o3d.visualization.rendering.MaterialRecord()
    surface.shader = "defaultUnlit"
    renderer.scene.add_geometry("capture_mesh", mesh, surface)
    marker = o3d.visualization.rendering.MaterialRecord()
    marker.shader = "defaultUnlit"
    renderer.scene.add_geometry("taught_object_proxy", target, marker)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(target))
    return renderer, ray_scene


def camera(renderer, ray_scene, pose_xy: np.ndarray, heading: float, floor_z: float):
    forward = np.array([math.sin(heading), math.cos(heading), 0.])
    eye = np.array([pose_xy[0], pose_xy[1], floor_z + .42])
    renderer.setup_camera(75., eye + forward, eye, np.array([0., 0., 1.]))
    rgb = np.asarray(renderer.render_to_image())[:, :, :3].copy()
    # Metal's offscreen depth readback can fail for this multi-million-face
    # mesh. Cast calibrated camera rays against the same measured triangles.
    width, height = rgb.shape[1], rgb.shape[0]
    sample_w, sample_h = width // 2, height // 2
    focal = height / (2 * math.tan(math.radians(75.) / 2))
    u, v = np.meshgrid(np.arange(sample_w) * 2 + .5,
                       np.arange(sample_h) * 2 + .5)
    right = np.array([math.cos(heading), -math.sin(heading), 0.])
    up = np.array([0., 0., 1.])
    directions = (forward[None, None, :] +
                  ((u - (width - 1) / 2) / focal)[..., None] * right +
                  (((height - 1) / 2 - v) / focal)[..., None] * up)
    origins = np.broadcast_to(eye, directions.shape)
    rays = np.concatenate((origins, directions), axis=-1).astype(np.float32)
    depth_small = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    depth = cv2.resize(depth_small, (width, height), interpolation=cv2.INTER_NEAREST)
    return rgb, depth


def range_fan(ray_scene, pose_xy: np.ndarray, heading: float, floor_z: float):
    angles = np.arange(-180., 180., 10., dtype=np.float32)
    radians = heading + np.deg2rad(angles)
    directions = np.column_stack((np.sin(radians), np.cos(radians),
                                  np.zeros_like(radians))).astype(np.float32)
    readings = []
    for height in (.12, .32):
        origin = np.array([pose_xy[0], pose_xy[1], floor_z + height], np.float32)
        rays = np.column_stack((np.repeat(origin[None, :], len(angles), axis=0),
                                directions)).astype(np.float32)
        hits = ray_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(float)
        hits[hits > 2.] = np.inf
        readings.append(hits)
    return np.stack(readings), angles


def visual_target(rgb: np.ndarray, depth: np.ndarray, pose: np.ndarray,
                  heading: float) -> dict:
    red, green, blue = [rgb[:, :, i].astype(np.int16) for i in range(3)]
    mask = ((red > 150) & (green < 130) & (blue > 65) &
            (red > 1.7 * green) & (blue > 1.25 * green) &
            np.isfinite(depth) & (depth > .1) & (depth < 5.))
    if int(mask.sum()) < 12:
        return {"seen": False, "pixels": int(mask.sum())}
    ys, xs = np.nonzero(mask)
    u = float(np.median(xs))
    v = float(np.median(ys))
    z = float(np.median(depth[mask]))
    fx = rgb.shape[0] / (2 * math.tan(math.radians(75.) / 2))
    right_m = (u - (rgb.shape[1] - 1) / 2) * z / fx
    forward = np.array([math.sin(heading), math.cos(heading)])
    right = np.array([math.cos(heading), -math.sin(heading)])
    target = pose + z * forward + right_m * right
    return {"seen": True, "pixels": len(xs), "pixel_uv": [u, v],
            "depth_m": z, "bearing_deg": math.degrees(math.atan2(right_m, z)),
            "estimated_xy_m": target.tolist()}


def run(scene_id: str, output: Path, steps: int,
        proprio_feedback: bool = True) -> dict:
    scene_dir = ASSETS / scene_id
    nav = dict(np.load(scene_dir / "nav_candidate.npz"))
    floor_z = float(nav["floor_z_m"])
    start_xy, target_xy = route_pair(nav)
    initial_heading = math.atan2(target_xy[0] - start_xy[0],
                                  target_xy[1] - start_xy[1]) + math.radians(30)
    renderer, ray_scene = sensing_scene(scene_dir / "mesh.ply", target_xy, floor_z)
    actual_xy = start_xy.copy()
    actual_heading = initial_heading
    # Dead reckoning starts in an arbitrary local frame. World pose is never
    # passed to the actor or its planner.
    actor_xy = np.zeros(2)
    actor_heading_deg = 0.
    planner = FanViewRoute(proxy_radius_m=.17, required_free_planes=(0, 1))
    last_target: np.ndarray | None = None
    last_command_step = 0.
    last_measured_step = 0.
    trace = []
    output.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output / "first_person.mp4"), fps=5,
                                codec="libx264", quality=7, macro_block_size=8)
    try:
        for index in range(steps):
            rgb, depth = camera(renderer, ray_scene, actual_xy, actual_heading, floor_z)
            ranges, angles = range_fan(ray_scene, actual_xy, actual_heading, floor_z)
            visual = visual_target(rgb, depth, actor_xy,
                                   math.radians(actor_heading_deg))
            if visual["seen"]:
                last_target = np.asarray(visual["estimated_xy_m"])
            no_progress = (proprio_feedback and last_command_step > .02 and
                           last_measured_step < .5 * last_command_step)
            if no_progress:
                decision = {"intent": "hold_request_help",
                            "reason": "proprioception_reports_blocked_motion"}
            elif last_target is None:
                decision = {"intent": "search_turn", "yaw_delta_deg": 10.,
                            "reason": "taught_proxy_not_visible"}
            else:
                decision = planner.plan(ranges, angles, actor_xy,
                                        actor_heading_deg, last_target,
                                        index * .2)
            command_yaw = float(decision.get("yaw_delta_deg", 0.))
            command_step = float(decision.get("forward_step_m", 0.))
            if command_step:
                candidate = actual_xy + command_step * np.array([
                    math.sin(actual_heading), math.cos(actual_heading)])
                accepted = segment_free(nav, actual_xy, candidate)
                if accepted:
                    actual_xy = candidate
                    actor_xy += command_step * np.array([
                        math.sin(math.radians(actor_heading_deg)),
                        math.cos(math.radians(actor_heading_deg))])
            else:
                accepted = True
            # Idealized joint/IMU motion feedback for this kinematic proxy.
            # The actor receives only executed distance on its next cycle.
            measured_step = command_step if accepted else 0.
            actual_heading += math.radians(command_yaw)
            actor_heading_deg += command_yaw
            view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(view, f"{index:03d} {decision['intent']} | object: {'seen' if visual['seen'] else 'unseen'}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            writer.append_data(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            trace.append({"frame": index, "time_s": round(index * .2, 2),
                          "observation": {"visual": visual,
                                          "range_low_m": ranges[0].tolist(),
                                          "range_mid_m": ranges[1].tolist(),
                                          "angles_deg": angles.tolist(),
                                          "valid_depth_fraction": float(np.mean(
                                              np.isfinite(depth) & (depth > 0) & (depth < 10)))},
                          "actor_state": {"odometry_xy_m": actor_xy.tolist(),
                                          "heading_deg": actor_heading_deg},
                          "proprioception": {"available": proprio_feedback,
                                             "previous_command_step_m": last_command_step,
                                             "previous_measured_step_m": last_measured_step},
                          "decision": decision, "motion_accepted": accepted,
                          "environment_truth": {"xy_m": actual_xy.tolist(),
                                                "heading_rad": actual_heading,
                                                "target_xy_m": target_xy.tolist()}})
            last_command_step, last_measured_step = command_step, measured_step
            if decision["intent"] in ("hold_viewpoint", "hold_request_help"):
                break
    finally:
        writer.close()
    true_surface_gap = (float(np.linalg.norm(actual_xy - target_xy)) - .12)
    actor_declared = bool(trace and trace[-1]["decision"]["intent"] == "hold_viewpoint")
    result = {"schema": "ape.real-scan-mesh-probe.v1", "scene_id": scene_id,
              "status": "provisional_kinematic_probe_not_photoreal_or_body_qualified",
              "start_xy_m": start_xy.tolist(), "target_xy_m": target_xy.tolist(),
              "frames": len(trace), "actor_declared_standoff": actor_declared,
              "actor_outcome": (trace[-1]["decision"]["intent"] if trace else "no_frames"),
              "proprioception_feedback": proprio_feedback,
              "true_target_surface_gap_m": true_surface_gap,
              "truth_standoff_within_band": bool(actor_declared and
                                                  1.20 <= true_surface_gap <= 1.35),
              "move_rejections": sum(not row["motion_accepted"] for row in trace),
              "visual_detection_frames": sum(row["observation"]["visual"]["seen"] for row in trace),
              "mean_valid_depth_fraction": float(np.mean([
                  row["observation"]["valid_depth_fraction"] for row in trace])),
              "limitations": ["vertex-colored mesh has nonphotographic novel views and holes",
                              "virtual color-distinct object is a perception wiring target",
                              "measured-mesh ray fan plus conservative raster, no dynamics",
                              "idealized joint/IMU motion feedback when enabled",
                              "kinematic proxy, not articulated 50 cm body"],
              "trace": trace}
    (output / "trace.json").write_text(json.dumps(result, indent=2) + "\n")
    return {key: value for key, value in result.items() if key != "trace"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="47670315")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--no-proprio-feedback", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ASSETS / args.scene / "mesh_probe"
    print(json.dumps(run(args.scene, output, args.steps,
                         not args.no_proprio_feedback), indent=2), flush=True)


if __name__ == "__main__":
    main()
