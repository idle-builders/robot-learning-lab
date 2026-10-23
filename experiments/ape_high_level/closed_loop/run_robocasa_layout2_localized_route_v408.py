"""Close range-based route planning through estimated pose under motion slip.

The actor receives an initial surveyed pose, command increments and its own
range returns. True pose and MuJoCo geometry are evaluator-only. This remains
a kinematic eye-rig proxy, not a physically controlled ape.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from localization import ScanLocalizer
from run_robocasa_layout2_route_v407 import (
    GOAL, START, SensorGrid, evaluator_clear, frame_with_map, model_with_layout2,
    planar_scan, sha256,
)


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-localized-route-v408"


class RangeLocalizer(ScanLocalizer):
    def __init__(self, max_range_m: float, *, grow: bool = False) -> None:
        super().__init__()
        self.max_range_m = max_range_m
        self.grow = grow

    def endpoints(self, scan: list[float | None], pose: np.ndarray) -> np.ndarray:
        angles = np.linspace(-np.pi, np.pi, len(scan), endpoint=False)
        return np.array([
            pose[:2] + distance * np.array([
                np.cos(pose[2] + angle), np.sin(pose[2] + angle)])
            for angle, distance in zip(angles, scan)
            if distance is not None and .14 < distance < self.max_range_m
        ], dtype=float).reshape(-1, 2)

    def correct(self, scan: list[float | None], predicted: np.ndarray) -> tuple[np.ndarray, dict]:
        corrected, status = super().correct(scan, predicted)
        if self.grow and status["status"] == "registered" and status["inlier_fraction"] >= .65:
            points = self.endpoints(scan, corrected)
            self.points = np.vstack((self.points, points))
            # One point per 4 cm bin keeps nearest-neighbor work bounded.
            bins = np.floor(self.points / .04).astype(int)
            _, indices = np.unique(bins, axis=0, return_index=True)
            self.points = self.points[np.sort(indices)]
            status = {**status, "map_points": len(self.points)}
        return corrected, status


def episode(max_range_m: float, slip_m: float, source: str,
            obstacle_xy: tuple[float, float] | None = None,
            dropout_step: int | None = None) -> dict:
    model = model_with_layout2(obstacle_xy=obstacle_xy)
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = model.camera("head").id
    rng = np.random.default_rng(408)
    grid = SensorGrid()
    localizer = RangeLocalizer(max_range_m, grow=source == "range_growing")
    actual = START.copy()
    estimate = np.r_[START, 0.]
    trace = []
    frames = []
    outcome = "step_budget_exhausted"
    commanded_distance = 0.
    unregistered_streak = 0
    for step in range(120):
        model.cam_pos[camera_id, :2] = actual
        mujoco.mj_forward(model, data)
        # A missing packet is a transport/health fault, distinct from a valid
        # no-return beam. The actor must not turn missing data into free space.
        ranges = (None if step == dropout_step else
                  planar_scan(model, data, actual, max_range_m, rng))
        if ranges is None:
            localization = {"status": "range_packet_missing"}
            unregistered_streak += 1
        elif source.startswith("range"):
            estimate, localization = localizer.correct(ranges, estimate)
            unregistered_streak = (0 if localization["status"] in ("mapped", "registered")
                                   else unregistered_streak + 1)
        else:
            localization = {"status": "command_only"}
        if ranges is not None:
            grid.update(ranges, estimate[:2], max_range_m)
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        requested, plan_status, path = grid.plan(estimate[:2], GOAL)
        if ranges is None:
            requested, plan_status, path = None, "sensor_missing_stop", []
        elif source == "range_gated" and unregistered_streak >= 5:
            requested, plan_status, path = None, "localization_uncertain_stop", []
        actual_clear, actual_hit = evaluator_clear(model, data, actual)
        error = float(np.linalg.norm(estimate[:2] - actual))
        row = {"step": step, "actual_xy_m": actual.tolist(),
               "estimated_xy_m": estimate[:2].tolist(), "pose_error_m": error,
               "range_returns": sum(v is not None for v in ranges) if ranges is not None else None,
               "range_sha256": (hashlib.sha256(json.dumps(ranges).encode()).hexdigest()
                                if ranges is not None else None),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "localization": localization, "plan_status": plan_status,
               "unregistered_streak": unregistered_streak,
               "requested_xy_m": requested.tolist() if requested is not None else None,
               "known_free_cells": int(grid.free.sum()),
               "occupied_cells": int(grid.occupied.sum()),
               "evaluator_current_clear": actual_clear,
               "evaluator_current_hit": actual_hit}
        if requested is not None:
            command_delta = requested - estimate[:2]
            # Paired slip grows with commanded path length, independent of
            # layout and sensor. The actor never reads the true endpoint.
            true_delta = command_delta * (1. + slip_m / 5.89)
            checks = [evaluator_clear(model, data, actual + f * true_delta)
                      for f in (.25, .5, .75, 1.)]
            row["evaluator_segment_clear"] = all(ok for ok, _ in checks)
            row["evaluator_segment_hits"] = [hit for ok, hit in checks if not ok]
            row["command_delta_m"] = command_delta.tolist()
        trace.append(row)
        frame = Image.fromarray(frame_with_map(rgb, grid, estimate[:2], path,
                                               f"{source} / {plan_status}", step,
                                               max_range_m))
        draw = ImageDraw.Draw(frame)
        draw.text((650, 515), f"evaluator pose error {error:.2f} m; slip {slip_m:.2f} m",
                  fill="#ffffff")
        frames.append(np.asarray(frame))
        if plan_status in ("sensor_missing_stop", "localization_uncertain_stop"):
            outcome = plan_status
            break
        if np.linalg.norm(estimate[:2] - GOAL) <= .15:
            outcome = ("goal_viewpoint_reached" if np.linalg.norm(actual - GOAL) <= .25
                       else "false_goal_due_to_pose_error")
            break
        if requested is None:
            outcome = plan_status
            break
        if not row["evaluator_segment_clear"]:
            outcome = "actor_proposed_geometry_overlap"
            break
        grid.visits[grid.cell(estimate[:2])] += 1
        estimate[:2] += command_delta
        commanded_distance += float(np.linalg.norm(command_delta))
        actual += true_delta
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    obstacle_label = "-obstacle" if obstacle_xy is not None else ""
    outage_label = "-outage" if dropout_step is not None else ""
    video = OUT / f"layout2-{source}-range{max_range_m:.1f}m-slip{int(slip_m*100):02d}cm{obstacle_label}{outage_label}.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {"source": source, "max_range_m": max_range_m, "slip_m": slip_m,
            "scene_fingerprint": scene_fingerprint,
            "obstacle_xy_m": obstacle_xy,
            "dropout_step": dropout_step,
            "outcome": outcome, "steps": len(trace),
            "end_actual_xy_m": actual.tolist(),
            "end_estimated_xy_m": estimate[:2].tolist(),
            "end_pose_error_m": float(np.linalg.norm(estimate[:2] - actual)),
            "max_pose_error_m": max(row["pose_error_m"] for row in trace),
            "commanded_distance_m": commanded_distance,
            "unsafe_requested_steps": sum(row.get("evaluator_segment_clear") is False for row in trace),
            "registered_frames": sum(row["localization"]["status"] == "registered" for row in trace),
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "trace": trace}


def main() -> None:
    episodes = []
    for source, max_range, slip, obstacle, dropout in (
            ("command", 2.2, .12, None, None),
            ("range", 2.2, .12, None, None),
            ("command", 5.0, .12, None, None),
            ("range", 5.0, .12, None, None),
            ("command", 2.2, .30, None, None),
            ("range", 2.2, .30, None, None),
            ("range", 5.0, .30, None, None),
            ("range_growing", 2.2, .30, None, None),
            ("range_growing", 5.0, .30, None, None),
            ("range_growing", 2.2, .30, (4.4, -2.9), None),
            ("range_growing", 5.0, .30, (4.4, -2.9), None),
            ("range_gated", 2.2, .30, None, None),
            ("range_gated", 5.0, .30, None, None),
            ("range_growing", 5.0, .30, None, 25)):
        result = episode(max_range, slip, source, obstacle, dropout)
        episodes.append(result)
        print(source, max_range, result["outcome"],
              result["steps"], round(result["end_pose_error_m"], 3), flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "base_scene_source_sha256": sha256(Path(__file__).with_name("run_robocasa_layout2_route_v407.py")),
              "scope": "layout2 sensor-built map and pose estimate; kinematic eye rig",
              "known_heading": "fixed camera mount / ideal IMU assumption",
              "initial_pose": START.tolist(), "goal_viewpoint": GOAL.tolist(),
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
