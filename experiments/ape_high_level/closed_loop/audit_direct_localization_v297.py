"""V297: paired Direct Frame localization replay with saved timed sensor evidence.

This tests pose *measurement* on a safe scripted route. It does not claim a
walking controller, a packaged depth camera, or a deployable 360-degree lidar.
The simulator's actual pose is used only for clearance preflight and scoring.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from localization import ScanLocalizer
from rgbd_odometry import RGBDOdometry
from run import ROOT, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v297"
# Frozen before observations. Both mode and room structure vary; these seeds
# have not appeared in the earlier structural/low-clutter probes.
CASES = ((159, "open", "present"), (160, "doorway", "absent"),
         (161, "screen", "present"), (162, "raised", "absent"),
         (163, "open", "absent"), (164, "doorway", "present"),
         (165, "screen", "absent"), (166, "raised", "present"))
# All motions are <= 5 cm and remain near the clear start area. The lateral
# detour and return generate translation in two axes, including a loop closure.
WAYPOINTS = ((0., 0.), (.04, 0.), (.08, 0.), (.12, 0.), (.16, 0.),
             (.16, .05), (.12, .05), (.08, .05), (.04, .05), (0., .05))
SLIPS = (.58, .94, .72, .88, .55, .92, .68, .85, .62)
STEP_S = .4
MIN_START_CLEARANCE_M = .05
SHORT_DEPTH_M = (.07, .50)
LONG_DEPTH_M = (.08, 2.50)


def depth_variant(depth: np.ndarray, mode: str, seed: int, step: int) -> np.ndarray:
    result = depth.copy()
    low, high = SHORT_DEPTH_M if mode == "short" else LONG_DEPTH_M
    result[(result < low) | (result > high)] = np.nan
    if mode == "noisy_long":
        rng = np.random.default_rng(seed * 1009 + step)
        result += rng.normal(0., .01, result.shape).astype(result.dtype)
        result[rng.random(result.shape) < .15] = np.nan
    return result


def score_pose(pose: np.ndarray, actual: np.ndarray) -> float:
    return float(np.linalg.norm(pose[:2] - actual[:2]))


def panel(external: Image.Image, rgb: Image.Image, title: str,
          estimates: dict[str, np.ndarray], actual: np.ndarray) -> np.ndarray:
    left = external.resize((480, 360))
    right = rgb.resize((480, 360))
    canvas = Image.new("RGB", (960, 410), (245, 243, 237))
    canvas.paste(left, (0, 50))
    canvas.paste(right, (480, 50))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, fill=(20, 25, 30))
    draw.text((12, 27), "external evaluator view", fill=(55, 60, 65))
    draw.text((492, 27), "actor head RGB (depth saved separately)",
              fill=(55, 60, 65))
    draw.text((12, 386), "pose error cm: " + " | ".join(
        f"{name} {100*score_pose(pose, actual):.1f}"
        for name, pose in estimates.items()), fill=(20, 25, 30))
    return np.asarray(canvas)


def run_case(seed: int, kind: str, mode: str) -> dict:
    side = 1 if seed % 2 else -1
    room = DirectStructuralRoom(kind=kind, mode=mode, side=side)
    branch = OUT / f"seed{seed}-{kind}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    visual_modes = ("long", "short", "noisy_long")
    visual = {name: RGBDOdometry() for name in visual_modes}
    pose = {name: np.zeros(3) for name in
            ("command", "long", "short", "noisy_long", "virtual_360_11cm")}
    scan = ScanLocalizer()
    rows: list[dict] = []
    video = imageio.get_writer(branch / "episode.mp4", fps=2, codec="libx264",
                               quality=7, macro_block_size=2)
    try:
        room.layout(seed, absent=mode != "present", furnished=True)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        # Evaluator preflight is deliberately separate from actor estimates.
        preflight = []
        for xy in WAYPOINTS:
            room.actual[:] = (xy[0], side * xy[1], 0.)
            room._forward()
            preflight.append(float(full_body_clearance(room, obstacles, bodies)[0]))
        if min(preflight) < MIN_START_CLEARANCE_M:
            raise ValueError(f"Unsafe scripted route seed {seed}: {min(preflight):.3f}m")
        room.reset()
        room.layout(seed, absent=mode != "present", furnished=True)
        for step in range(len(WAYPOINTS)):
            delta = np.zeros(2)
            if step:
                delta = np.subtract(WAYPOINTS[step], WAYPOINTS[step - 1])
                delta[1] *= side
                room.move(float(delta[0]), float(delta[1]), slip=SLIPS[step - 1])
                for candidate in pose.values():
                    candidate[:2] += delta
            capture_s = float(room.time_s)
            rgb, depth = room.capture("upper_rgb", depth=True)
            if depth is None:
                raise AssertionError("RGB-D simulator view unavailable")
            # The camera orientation comes from the estimated kinematic model.
            # All paths share IMU heading here; no simulator XY pose enters it.
            camera_rotation = room.camera_pose_estimate("upper_rgb")[1]
            visual_status = {}
            for name, estimator in visual.items():
                candidate_depth = depth_variant(depth, name, seed, step)
                if step == 0:
                    visual_status[name] = estimator.initialize(
                        rgb, candidate_depth, pose[name], camera_rotation)
                else:
                    pose[name], visual_status[name] = estimator.correct(
                        rgb, candidate_depth, pose[name], camera_rotation)
            ranges = sensed_range(room, seed, step, height_m=.11)
            if step == 0:
                range_status = scan.initialize(ranges, pose["virtual_360_11cm"])
            else:
                pose["virtual_360_11cm"], range_status = scan.correct(
                    ranges, pose["virtual_360_11cm"])
            # Save the actual captured arrays and timestamps, so the inputs can
            # be replayed with latency, noise, or a replacement localizer.
            rgb_path = branch / f"step{step:02d}-rgb.png"
            depth_path = branch / f"step{step:02d}-depth.npz"
            rgb.save(rgb_path)
            np.savez_compressed(depth_path, depth_m=depth)
            actual = room.actual.copy()
            errors = {name: round(score_pose(value, actual), 5)
                      for name, value in pose.items()}
            row = {
                "step": step, "capture_time_s": round(capture_s, 3),
                "motion_command_xy_m": delta.tolist(),
                "actor_imu_heading_rad": round(float(room.estimated[2]), 6),
                "actor_head_encoder_rad": round(float(room.head), 6),
                "rgb_path": str(rgb_path.relative_to(ROOT)),
                "depth_path": str(depth_path.relative_to(ROOT)),
                "rgb_sha256": sha(rgb_path), "depth_sha256": sha(depth_path),
                "virtual_range_11cm_m": ranges,
                "visual_status": visual_status, "range_status": range_status,
                "actor_pose_xyyaw": {name: value.round(5).tolist()
                                      for name, value in pose.items()},
                "evaluator_actual_xyyaw": actual.round(5).tolist(),
                "evaluator_pose_error_m": errors,
                "evaluator_clearance_m": round(float(full_body_clearance(
                    room, obstacles, bodies)[0]), 5),
            }
            rows.append(row)
            video.append_data(panel(room.capture_external(), rgb,
                                    f"seed {seed} {kind} {mode} | t={capture_s:.1f}s",
                                    pose, actual))
        trace_path = branch / "trace.json"
        trace_path.write_text(json.dumps({
            "case": [seed, kind, mode], "waypoints": WAYPOINTS,
            "slips_evaluator_only": SLIPS, "preflight_clearance_m": preflight,
            "rows": rows}, indent=2) + "\n")
        video.close()
        return {"seed": seed, "kind": kind, "mode": mode,
                "minimum_route_clearance_m": round(min(preflight), 5),
                "minimum_executed_clearance_m": round(min(
                    row["evaluator_clearance_m"] for row in rows), 5),
                "final_pose_error_m": rows[-1]["evaluator_pose_error_m"],
                "maximum_pose_error_m": {name: max(
                    row["evaluator_pose_error_m"][name] for row in rows)
                    for name in pose},
                "visual_registered_steps": {name: sum(
                    row["visual_status"][name]["status"] == "registered"
                    for row in rows[1:]) for name in visual_modes},
                "range_registered_steps": sum(
                    row["range_status"]["status"] == "registered"
                    for row in rows[1:]),
                "trace_path": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
                "video_path": str((branch / "episode.mp4").relative_to(ROOT)),
                "video_sha256": sha(branch / "episode.mp4")}
    finally:
        video.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for seed, kind, mode in CASES:
        case = run_case(seed, kind, mode)
        cases.append(case)
        print(seed, kind, mode, case["final_pose_error_m"], flush=True)
    methods = tuple(cases[0]["final_pose_error_m"])
    summary = {method: {
        "mean_final_error_m": round(float(np.mean([
            case["final_pose_error_m"][method] for case in cases])), 5),
        "worst_final_error_m": round(max(
            case["final_pose_error_m"][method] for case in cases), 5),
        "worst_at_any_step_m": round(max(
            case["maximum_pose_error_m"][method] for case in cases), 5)}
        for method in methods}
    report = {"scope": "V297 selected Direct Frame eight paired open/doorway/screen/raised rooms with present/absent object, safe scripted 2-D route, variable imposed translation slip, 0.4s steps and timestamped ideal MuJoCo upper-head RGB/depth. Compare command+IMU heading against ORB RGB-D PnP with ideal 8-250cm depth, short 7-50cm depth, 1cm depth noise/15% holes, and non-deployable 72-beam 11cm full-circle range upper bound. All actor estimates use only commands and rendered measurements; hidden true pose and clearance score outcomes. RGB-D camera is hypothetical, 360 range ignores packaging and occlusion, joints do not move, IMU acceleration is not physically generated by the kinematic surrogate. This is pose measurement only, not a closed owner-task or native stop qualification.",
              "source_sha256": sha(SOURCE), "selected_visual_sha256": sha(selected_visual()),
              "case_specs": CASES, "waypoints_m": WAYPOINTS,
              "imposed_slip_evaluator_only": SLIPS,
              "step_duration_s": STEP_S, "cases": cases, "summary": summary}
    (OUT / "localization-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
