"""Screen a moving low prop during range-planned ape-proxy motion.

The planner gets 0.4 s snapshots of its 72-beam planar range and estimated
pose. A paired variant also runs a 20 ms raw-range proximity stop during each
bounded kinematic move. Exact prop distance is evaluator-only. This is not a
physical ape or a person model.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_robocasa_layout2_route_v407 import (
    GOAL, START, SensorGrid, frame_with_map, model_with_layout2,
    planar_scan, sha256,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-moving-prop-v423"
OBSTACLE_Y = -3.34
START_X = 4.95
END_X = 4.60
MOVE_BEGIN_S = 14.0
MOVE_END_S = 14.4
ACTION_S = .4
SUBSTEP_S = .02
BODY_RADIUS_M = .17
PROP_RADIUS_M = .24
REQUIRED_GAP_M = .05
GUARD_CENTER_RANGE_M = .30


def obstacle_xy(time_s: float) -> np.ndarray:
    fraction = min(1., max(0., (time_s - MOVE_BEGIN_S) / (MOVE_END_S - MOVE_BEGIN_S)))
    return np.array([START_X + fraction * (END_X - START_X), OBSTACLE_Y])


def raw_nearest_range(model: mujoco.MjModel, data: mujoco.MjData,
                      pose: np.ndarray) -> float:
    origin = np.array([pose[0], pose[1], .30], dtype=np.float64)
    nearest = math.inf
    for angle in np.linspace(-math.pi, math.pi, 72, endpoint=False):
        ray = np.array([math.cos(angle), math.sin(angle), 0.], dtype=np.float64)
        geom_id = np.array([-1], np.int32)
        distance = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geom_id)
        if distance > 0:
            nearest = min(nearest, float(distance))
    return nearest


def image_frame(rgb: np.ndarray, grid: SensorGrid, pose: np.ndarray,
                path: list[tuple[int, int]], label: str, step: int,
                time_s: float, prop_xy: np.ndarray, true_gap: float) -> np.ndarray:
    frame = Image.fromarray(frame_with_map(rgb, grid, pose, path, label, step, 5.))
    draw = ImageDraw.Draw(frame)
    draw.text((648, 510), f"t {time_s:.2f}s; prop {prop_xy[0]:.2f},{prop_xy[1]:.2f}; eval gap {true_gap:.2f}m",
              fill="white")
    return np.asarray(frame)


def episode(monitor: bool) -> dict:
    model = model_with_layout2(obstacle_xy=(START_X, OBSTACLE_Y))
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = model.camera("head").id
    prop_body = model.body("route_obstacle").id
    rng = np.random.default_rng(423)
    localizer = RangeLocalizer(5., grow=True)
    grid = SensorGrid()
    actual = START.copy()
    estimate = np.r_[START, 0.]
    t_s = 0.
    trace = []
    frames = []
    outcome = "step_budget_exhausted"
    min_true_gap = math.inf
    for step in range(110):
        prop = obstacle_xy(t_s)
        model.body_pos[prop_body, :2] = prop
        model.cam_pos[camera_id, :2] = actual
        mujoco.mj_forward(model, data)
        ranges = planar_scan(model, data, actual, 5., rng)
        estimate, localization = localizer.correct(ranges, estimate)
        grid.update(ranges, estimate[:2], 5.)
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        requested, plan_status, path = grid.plan(estimate[:2], GOAL)
        gap = float(np.linalg.norm(actual - prop) - BODY_RADIUS_M - PROP_RADIUS_M)
        min_true_gap = min(min_true_gap, gap)
        row = {"step": step, "time_s": round(t_s, 4),
               "actual_xy_m": actual.tolist(), "estimated_xy_m": estimate[:2].tolist(),
               "pose_error_m": float(np.linalg.norm(actual - estimate[:2])),
               "prop_xy_m": prop.tolist(), "true_prop_gap_m": gap,
               "localization": localization, "plan_status": plan_status,
               "requested_xy_m": requested.tolist() if requested is not None else None,
               "range_returns": sum(v is not None for v in ranges),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "motion_substeps": []}
        trace.append(row)
        frames.append(image_frame(rgb, grid, estimate[:2], path, plan_status,
                                  step, t_s, prop, gap))
        if np.linalg.norm(estimate[:2] - GOAL) <= .15:
            outcome = "goal_viewpoint_reached"
            break
        if localization["status"] not in ("mapped", "registered"):
            outcome = "localization_uncertain_stop"
            break
        if requested is None:
            outcome = plan_status
            break
        commanded_delta = requested - estimate[:2]
        true_delta = commanded_delta * (1. + .30 / 5.89)
        start_pose = actual.copy()
        for substep in range(1, int(ACTION_S / SUBSTEP_S) + 1):
            next_time = t_s + substep * SUBSTEP_S
            next_prop = obstacle_xy(next_time)
            model.body_pos[prop_body, :2] = next_prop
            current_pose = start_pose + (substep - 1) / 20 * true_delta
            proposed_pose = start_pose + substep / 20 * true_delta
            model.cam_pos[camera_id, :2] = current_pose
            mujoco.mj_forward(model, data)
            guard_range = raw_nearest_range(model, data, current_pose) if monitor else None
            if monitor and guard_range < GUARD_CENTER_RANGE_M:
                actual = current_pose
                t_s = next_time
                true_gap = float(np.linalg.norm(actual - next_prop) - BODY_RADIUS_M - PROP_RADIUS_M)
                min_true_gap = min(min_true_gap, true_gap)
                row["motion_substeps"].append({"time_s": round(next_time, 4),
                                               "actor_guard_range_m": guard_range,
                                               "true_prop_gap_m": true_gap,
                                               "stopped_before_move": True})
                outcome = "proximity_stop"
                break
            actual = proposed_pose
            true_gap = float(np.linalg.norm(actual - next_prop) - BODY_RADIUS_M - PROP_RADIUS_M)
            min_true_gap = min(min_true_gap, true_gap)
            row["motion_substeps"].append({"time_s": round(next_time, 4),
                                           "actor_guard_range_m": guard_range,
                                           "true_prop_gap_m": true_gap,
                                           "stopped_before_move": False})
            if true_gap < 0:
                t_s = next_time
                outcome = "moving_prop_contact"
                break
            if true_gap < REQUIRED_GAP_M and not monitor:
                t_s = next_time
                outcome = "moving_prop_margin_violation"
                break
        if outcome != "step_budget_exhausted":
            break
        grid.visits[grid.cell(estimate[:2])] += 1
        estimate[:2] += commanded_delta
        t_s += ACTION_S
    # Retain the physical state at the actual stop, not only the preceding
    # 0.4 s capture. The marked gap is evaluator text, never actor input.
    terminal_prop = obstacle_xy(t_s)
    model.body_pos[prop_body, :2] = terminal_prop
    model.cam_pos[camera_id, :2] = actual
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="head")
    terminal_rgb = renderer.render().copy()
    terminal_gap = float(np.linalg.norm(actual - terminal_prop) - BODY_RADIUS_M - PROP_RADIUS_M)
    frames.append(image_frame(terminal_rgb, grid, estimate[:2], [], outcome,
                              len(trace), t_s, terminal_prop, terminal_gap))
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / ("moving-prop-monitored.mp4" if monitor else "moving-prop-unmonitored.mp4")
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {"monitor": monitor, "outcome": outcome, "steps": len(trace),
            "scene_fingerprint": scene_fingerprint,
            "end_true_xy_m": actual.tolist(), "end_estimated_xy_m": estimate[:2].tolist(),
            "end_time_s": t_s, "min_true_prop_gap_m": min_true_gap,
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "trace": trace}


def main() -> None:
    episodes = [episode(False), episode(True)]
    for episode_row in episodes:
        print("monitored" if episode_row["monitor"] else "unmonitored",
              episode_row["outcome"], episode_row["steps"],
              round(episode_row["min_true_prop_gap_m"], 3), flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "room_source_sha256": sha256(Path(__file__).with_name("run_robocasa_layout2_route_v407.py")),
              "localizer_source_sha256": sha256(Path(__file__).with_name("run_robocasa_layout2_localized_route_v408.py")),
              "scope": "kinematic eye rig, moving low cylinder, 20ms range guard, no human or physical stop",
              "obstacle_motion": {"from_xy_m": [START_X, OBSTACLE_Y],
                                  "to_xy_m": [END_X, OBSTACLE_Y],
                                  "begin_s": MOVE_BEGIN_S, "end_s": MOVE_END_S},
              "body_radius_m": BODY_RADIUS_M, "prop_radius_m": PROP_RADIUS_M,
              "required_gap_m": REQUIRED_GAP_M,
              "guard_center_range_threshold_m": GUARD_CENTER_RANGE_M,
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
