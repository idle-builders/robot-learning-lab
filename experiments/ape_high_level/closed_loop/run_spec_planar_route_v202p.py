"""Use A1/A2-sized ideal planar packets in the existing island route loop.

This is a navigation-role comparison on the RoboCasa eye rig. The virtual
scanner is not a mounted device and cannot clear the ape's 3-D swept body.
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

from run import ROOT, sha
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer
from run_robocasa_layout2_route_v407 import (
    GOAL, GRID_M, LIDAR_Z_M, MARGIN_M, RADIUS_M, START, SensorGrid,
    evaluator_clear, frame_with_map, model_with_layout2,
)


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-spec-planar-route-v202p"
DEVICES = {
    "legacy_72": {"minimum_m": .14, "scan_hz": 10., "beams": 72},
    "a1_spec_proxy": {"minimum_m": .15, "scan_hz": 5.5, "beams": 360},
    "a2_spec_proxy": {"minimum_m": .20, "scan_hz": 10., "beams": 800},
}
MAX_RANGE_M = 5.
COMMAND_PERIOD_S = .4
MAX_CAPTURE_AGE_S = .2
NOISE_STD_M = .005  # Matched V408 stress, not a manufacturer noise claim.
STATUSES = {0: "valid_hit", 1: "too_near", 2: "no_target"}


class StatusGrid(SensorGrid):
    """Only a calibrated valid ray clears cells; invalid rays stay unknown."""

    def update_packet(self, ranges: np.ndarray, statuses: np.ndarray,
                      pose: np.ndarray) -> None:
        valid = statuses == 0
        if not np.any(valid):
            return
        lengths = np.where(valid, ranges, 0.)
        dx, dy = self.xx - pose[0], self.yy - pose[1]
        distance = np.hypot(dx, dy)
        angle_index = ((np.arctan2(dy, dx) + math.pi) /
                       (2 * math.pi) * len(ranges)) % len(ranges)
        lower = np.floor(angle_index).astype(int)
        upper = (lower + 1) % len(ranges)
        clear_radius = (np.minimum(lengths[lower], lengths[upper]) -
                        RADIUS_M - MARGIN_M)
        observed = valid[lower] & valid[upper]
        self.free |= observed & (distance < clear_radius)
        for index in np.flatnonzero(valid):
            angle = -math.pi + 2 * math.pi * index / len(ranges)
            hit_x = pose[0] + ranges[index] * math.cos(angle)
            hit_y = pose[1] + ranges[index] * math.sin(angle)
            near = ((self.xx - hit_x) ** 2 + (self.yy - hit_y) ** 2 <=
                    (RADIUS_M + MARGIN_M) ** 2)
            self.occupied |= near
        self.free &= ~self.occupied
        # The occupied current cell is a proprioceptive starting assertion,
        # not a claim that the adjacent body-height corridor is observed.
        self.free[self.cell(pose)] = True


def packet(model: mujoco.MjModel, data: mujoco.MjData,
           xy: np.ndarray, profile: dict, rng: np.random.Generator
           ) -> tuple[np.ndarray, np.ndarray]:
    origin = np.array((xy[0], xy[1], LIDAR_Z_M), dtype=np.float64)
    count = profile["beams"]
    ranges = np.full(count, np.nan, dtype=np.float32)
    status = np.full(count, 2, dtype=np.uint8)
    for i in range(count):
        angle = -math.pi + 2 * math.pi * i / count
        ray = np.array((math.cos(angle), math.sin(angle), 0.), dtype=np.float64)
        geom = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, origin, ray, None, 1, -1, geom)
        if distance < 0 or distance > MAX_RANGE_M:
            continue
        if distance < profile["minimum_m"]:
            status[i] = 1
            continue
        status[i] = 0
        ranges[i] = max(profile["minimum_m"],
                        distance + rng.normal(0., NOISE_STD_M))
    return ranges, status


def episode(device: str, obstacle_xy: tuple[float, float] | None,
            slip_m: float = .30) -> dict:
    profile = DEVICES[device]
    model = model_with_layout2(obstacle_xy=obstacle_xy)
    scene_hash = hashlib.sha256(b"".join(a.tobytes() for a in (
        model.body_pos, model.geom_pos, model.geom_quat, model.geom_size,
        model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    camera_id = model.camera("head").id
    rng = np.random.default_rng(202)
    grid = StatusGrid()
    localizer = RangeLocalizer(MAX_RANGE_M, grow=True)
    actual = START.copy()
    estimate = np.r_[START, 0.]
    trace = []
    frames = []
    outcome = "step_budget_exhausted"
    branch = OUT / f"{device}-{'column' if obstacle_xy else 'clear'}"
    branch.mkdir(parents=True, exist_ok=True)
    try:
        for step in range(120):
            now_s = step * COMMAND_PERIOD_S
            scan_tick = math.floor((now_s + 1e-9) * profile["scan_hz"])
            capture_s = scan_tick / profile["scan_hz"]
            age_s = now_s - capture_s
            model.cam_pos[camera_id, :2] = actual
            mujoco.mj_forward(model, data)
            ranges, statuses = packet(model, data, actual, profile, rng)
            packet_path = branch / f"range-{step:03d}.npz"
            np.savez_compressed(packet_path, ranges_m=ranges,
                                status=statuses,
                                estimated_pose_xyyaw=estimate.copy(),
                                capture_time_s=capture_s,
                                decision_time_s=now_s)
            # The matcher consumes a fixed ~90-beam decimation. The full
            # packet remains available to map coverage and the saved audit.
            stride = {"legacy_72": 1, "a1_spec_proxy": 4,
                      "a2_spec_proxy": 8}[device]
            local_ranges = [float(v) if s == 0 else None
                            for v, s in zip(ranges[::stride],
                                            statuses[::stride])]
            estimate, localization = localizer.correct(local_ranges, estimate)
            health = (localization["status"] in ("mapped", "registered")
                      and age_s <= MAX_CAPTURE_AGE_S)
            if health:
                grid.update_packet(ranges, statuses, estimate[:2])
            renderer.update_scene(data, camera="head")
            rgb = renderer.render().copy()
            requested, plan_status, path = grid.plan(estimate[:2], GOAL)
            if not health:
                requested = None
                plan_status = ("stale_scan_stop" if age_s > MAX_CAPTURE_AGE_S
                               else "localization_uncertain_stop")
                path = []
            actual_clear, actual_hit = evaluator_clear(model, data, actual)
            error = float(np.linalg.norm(estimate[:2] - actual))
            row = {
                "step": step, "capture_time_s": round(capture_s, 6),
                "decision_time_s": round(now_s, 6),
                "capture_age_s": round(age_s, 6),
                "packet_path": str(packet_path.relative_to(ROOT)),
                "packet_sha256": sha(packet_path),
                "status_counts": {name: int(np.count_nonzero(statuses == code))
                                  for code, name in STATUSES.items()},
                "actual_xy_m": actual.tolist(),
                "estimated_xy_m": estimate[:2].tolist(),
                "pose_error_m": error,
                "localization": localization,
                "plan_status": plan_status,
                "requested_xy_m": requested.tolist()
                if requested is not None else None,
                "known_free_cells": int(grid.free.sum()),
                "occupied_cells": int(grid.occupied.sum()),
                "evaluator_current_clear": actual_clear,
                "evaluator_current_hit": actual_hit,
            }
            if requested is not None:
                command_delta = requested - estimate[:2]
                true_delta = command_delta * (1. + slip_m / 5.89)
                checks = [evaluator_clear(model, data,
                          actual + f * true_delta) for f in (.25, .5, .75, 1.)]
                row["evaluator_segment_clear"] = all(ok for ok, _ in checks)
                row["evaluator_segment_hits"] = [hit for ok, hit in checks
                                                  if not ok]
                row["command_delta_m"] = command_delta.tolist()
            trace.append(row)
            frame = Image.fromarray(frame_with_map(
                rgb, grid, estimate[:2], path,
                f"{device} / {plan_status}", step, MAX_RANGE_M))
            ImageDraw.Draw(frame).text(
                (650, 515), f"pose error {error:.2f}m; age {age_s:.2f}s",
                fill="#ffffff")
            frames.append(np.asarray(frame))
            if plan_status.endswith("_stop"):
                outcome = plan_status
                break
            if np.linalg.norm(estimate[:2] - GOAL) <= .15:
                outcome = ("goal_viewpoint_reached" if
                           np.linalg.norm(actual - GOAL) <= .25 else
                           "false_goal_due_to_pose_error")
                break
            if requested is None:
                outcome = plan_status
                break
            if not row["evaluator_segment_clear"]:
                outcome = "actor_proposed_geometry_overlap"
                break
            grid.visits[grid.cell(estimate[:2])] += 1
            estimate[:2] += command_delta
            actual += true_delta
    finally:
        renderer.close()
    video = branch / "route.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {
        "device": device, "obstacle_xy_m": obstacle_xy,
        "slip_m": slip_m, "scene_fingerprint": scene_hash,
        "outcome": outcome, "steps": len(trace),
        "end_pose_error_m": float(np.linalg.norm(estimate[:2] - actual)),
        "max_pose_error_m": max(r["pose_error_m"] for r in trace),
        "registered_frames": sum(r["localization"]["status"] ==
                                 "registered" for r in trace),
        "video": {"path": str(video.relative_to(ROOT)),
                  "sha256": sha(video), "frames": len(frames)},
        "trace": trace,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer["status"] != "rejected_no_selected_mechanical_design":
        raise AssertionError("Historical/reference body status changed")
    episodes = []
    for obstacle in (None, (4.4, -2.9)):
        for device in DEVICES:
            result = episode(device, obstacle)
            episodes.append(result)
            print(device, obstacle, result["outcome"], result["steps"],
                  round(result["end_pose_error_m"], 3), flush=True)
    report = {
        "schema": "robot-learning-lab.ape-spec-planar-route.v1",
        "source_sha256": sha(SOURCE),
        "v408_source_sha256": sha(Path(__file__).with_name(
            "run_robocasa_layout2_localized_route_v408.py")),
        "current_design_status": pointer["status"],
        "device_profiles": DEVICES,
        "status_codes": STATUSES,
        "max_range_m": MAX_RANGE_M,
        "command_period_s": COMMAND_PERIOD_S,
        "maximum_capture_age_s": MAX_CAPTURE_AGE_S,
        "noise_std_m": NOISE_STD_M,
        "scope": "Seeded RoboCasa layout2 kinematic eye-rig route with .30m "
                 "cumulative commanded-slip stress, ideal horizontal rays at "
                 "30cm, A1/A2 angular/min-range/cadence proxies and V408 "
                 "matched 5mm Gaussian range noise. Valid hits alone clear "
                 "planar map cells; too-near/no-target stay unknown. Initial "
                 "pose and heading are ideal; commands teleport the eye rig. "
                 "No device reflectance, scan distortion, self occlusion, "
                 "mount fit, body-height observability or physical safety claim.",
        "manufacturer_sources": [
            "https://www.slamtec.com/en/lidar/a1spec",
            "https://www.slamtec.com/en/lidar/a2spec"],
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
