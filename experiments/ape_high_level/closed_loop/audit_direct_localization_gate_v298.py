"""V298: motion-check visual odometry on V297 development and fresh rooms.

The gate uses commanded motion and RGB-D only. Ground truth is scoring-only.
This is deliberately a small pose-filter experiment, not a finished VIO stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from audit_direct_localization_v297 import (MIN_START_CLEARANCE_M, OUT as DEV,
                                            WAYPOINTS, depth_variant, panel)
from build_scene_pairs_v133 import body_geoms, full_body_clearance, scene_geoms
from direct_structural_rooms_v270 import DirectStructuralRoom, selected_visual
from localization import ScanLocalizer
from rgbd_odometry import RGBDOdometry
from run import ROOT, sensed_range, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v298"
CASES = ((167, "open", "present"), (168, "doorway", "absent"),
         (169, "screen", "present"), (170, "raised", "absent"),
         (171, "open", "absent"), (172, "doorway", "present"),
         (173, "screen", "absent"), (174, "raised", "present"))
SLIPS = (.62, .85, .68, .92, .55, .88, .72, .94, .58)
MOTION_SCALE_UPPER = 1.2
CROSS_TRACK_MAX_M = .02
VISUAL_WEIGHT = .5


def motion_checked(command: np.ndarray, measured: np.ndarray | None
                   ) -> tuple[np.ndarray, str]:
    if measured is None:
        return command, "visual_unavailable"
    length = float(np.linalg.norm(command))
    along = float(np.dot(command, measured) / length)
    across = float(abs(np.linalg.det(np.stack((command, measured)))) / length)
    if not (0. <= along <= MOTION_SCALE_UPPER * length and
            across <= CROSS_TRACK_MAX_M):
        return command, "motion_rejected"
    return (1. - VISUAL_WEIGHT) * command + VISUAL_WEIGHT * measured, "fused"


def development_replay() -> dict:
    prior = json.loads((DEV / "localization-audit.json").read_text())
    cases = []
    for case in prior["cases"]:
        rows = json.loads((ROOT / case["trace_path"]).read_text())["rows"]
        fused = np.zeros(2)
        gate_counts = {"fused": 0, "motion_rejected": 0,
                       "visual_unavailable": 0}
        errors = []
        for row in rows[1:]:
            command = np.asarray(row["motion_command_xy_m"])
            visual = row["visual_status"]["long"]
            measured = (np.asarray(visual["translation_m"])
                        if visual["status"] == "registered" else None)
            delta, gate = motion_checked(command, measured)
            gate_counts[gate] += 1
            fused += delta
            errors.append(float(np.linalg.norm(
                fused - np.asarray(row["evaluator_actual_xyyaw"])[:2])))
        cases.append({"seed": case["seed"], "kind": case["kind"],
                      "final_error_m": round(errors[-1], 5),
                      "max_error_m": round(max(errors), 5),
                      "gate_counts": gate_counts})
    return {"cases": cases,
            "mean_final_error_m": round(float(np.mean(
                [case["final_error_m"] for case in cases])), 5),
            "worst_final_error_m": max(case["final_error_m"] for case in cases),
            "v297_sha256": sha(DEV / "localization-audit.json")}


def heldout_case(seed: int, kind: str, mode: str) -> dict:
    side = 1 if seed % 2 else -1
    room = DirectStructuralRoom(kind=kind, mode=mode, side=side)
    branch = OUT / f"seed{seed}-{kind}-{mode}"
    branch.mkdir(parents=True, exist_ok=True)
    poses = {name: np.zeros(3) for name in
             ("command", "upper_long", "motion_checked_upper_long",
              "lower_short", "virtual_360_11cm")}
    odom = {name: RGBDOdometry() for name in
            ("upper_long", "motion_checked_upper_long", "lower_short")}
    range_odom = ScanLocalizer()
    rows = []
    video_path = branch / "episode.mp4"
    video = imageio.get_writer(video_path, fps=2, codec="libx264",
                               quality=7, macro_block_size=2)
    try:
        room.layout(seed, absent=mode != "present", furnished=True)
        obstacles = scene_geoms(room)
        bodies = body_geoms(room, obstacles)
        preflight = []
        for xy in WAYPOINTS:
            room.actual[:] = (xy[0], side * xy[1], 0.)
            room._forward()
            preflight.append(float(full_body_clearance(room, obstacles, bodies)[0]))
        if min(preflight) < MIN_START_CLEARANCE_M:
            raise ValueError(f"Unsafe heldout route seed {seed}: {min(preflight):.3f}m")
        room.reset()
        room.layout(seed, absent=mode != "present", furnished=True)
        for step in range(len(WAYPOINTS)):
            command = np.zeros(2)
            if step:
                command = np.subtract(WAYPOINTS[step], WAYPOINTS[step - 1])
                command[1] *= side
                room.move(float(command[0]), float(command[1]),
                          slip=SLIPS[step - 1])
                for pose in poses.values():
                    pose[:2] += command
            capture_time_s = float(room.time_s)
            upper_rgb, upper_depth = room.capture("upper_rgb", depth=True)
            lower_rgb, lower_depth = room.capture("lower_rgb", depth=True)
            if upper_depth is None or lower_depth is None:
                raise AssertionError("Simulator depth capture missing")
            upper_rot = room.camera_pose_estimate("upper_rgb")[1]
            lower_rot = room.camera_pose_estimate("lower_rgb")[1]
            depths = {
                "upper_long": depth_variant(upper_depth, "long", seed, step),
                "motion_checked_upper_long": depth_variant(
                    upper_depth, "long", seed, step),
                "lower_short": depth_variant(lower_depth, "short", seed, step)}
            status = {}
            gate = "reference"
            for name, estimator in odom.items():
                rgb = lower_rgb if name == "lower_short" else upper_rgb
                rotation = lower_rot if name == "lower_short" else upper_rot
                if step == 0:
                    status[name] = estimator.initialize(
                        rgb, depths[name], poses[name], rotation)
                    continue
                predicted = poses[name].copy()
                corrected, status[name] = estimator.correct(
                    rgb, depths[name], predicted, rotation)
                if name == "motion_checked_upper_long":
                    measured = (corrected[:2] - (predicted[:2] - command)
                                if status[name]["status"] == "registered"
                                else None)
                    delta, gate = motion_checked(command, measured)
                    poses[name][:2] = predicted[:2] - command + delta
                    # Keep the odometer's image reference paired with the
                    # filtered pose at that frame, including rejected updates.
                    keys, descriptors, depth, _, old_rotation = estimator.previous
                    estimator.previous = (keys, descriptors, depth,
                                          poses[name].copy(), old_rotation)
                else:
                    poses[name] = corrected
            ranges = sensed_range(room, seed, step, height_m=.11)
            if step == 0:
                range_status = range_odom.initialize(
                    ranges, poses["virtual_360_11cm"])
            else:
                poses["virtual_360_11cm"], range_status = range_odom.correct(
                    ranges, poses["virtual_360_11cm"])
            rgb_upper_path = branch / f"step{step:02d}-upper-rgb.png"
            rgb_lower_path = branch / f"step{step:02d}-lower-rgb.png"
            depth_path = branch / f"step{step:02d}-depths.npz"
            upper_rgb.save(rgb_upper_path)
            lower_rgb.save(rgb_lower_path)
            np.savez_compressed(depth_path, upper_m=upper_depth,
                                lower_m=lower_depth)
            actual = room.actual.copy()
            errors = {name: round(float(np.linalg.norm(
                estimate[:2] - actual[:2])), 5)
                for name, estimate in poses.items()}
            rows.append({
                "step": step, "capture_time_s": round(capture_time_s, 3),
                "motion_command_xy_m": command.tolist(),
                "actor_imu_heading_rad": round(float(room.estimated[2]), 6),
                "actor_head_encoder_rad": round(float(room.head), 6),
                "upper_rgb_path": str(rgb_upper_path.relative_to(ROOT)),
                "lower_rgb_path": str(rgb_lower_path.relative_to(ROOT)),
                "depth_path": str(depth_path.relative_to(ROOT)),
                "input_sha256": {"upper_rgb": sha(rgb_upper_path),
                                 "lower_rgb": sha(rgb_lower_path),
                                 "depth": sha(depth_path)},
                "virtual_range_11cm_m": ranges,
                "rgbd_status": status, "motion_gate": gate,
                "range_status": range_status,
                "actor_pose_xyyaw": {name: estimate.round(5).tolist()
                                      for name, estimate in poses.items()},
                "evaluator_actual_xyyaw": actual.round(5).tolist(),
                "evaluator_pose_error_m": errors,
                "evaluator_clearance_m": round(float(full_body_clearance(
                    room, obstacles, bodies)[0]), 5)})
            video.append_data(panel(room.capture_external(), upper_rgb,
                                    f"heldout seed {seed} {kind} | t={capture_time_s:.1f}s",
                                    poses, actual))
        video.close()
        trace_path = branch / "trace.json"
        trace_path.write_text(json.dumps({
            "case": [seed, kind, mode], "waypoints": WAYPOINTS,
            "slips_evaluator_only": SLIPS,
            "preflight_clearance_m": preflight, "rows": rows}, indent=2) + "\n")
        return {"seed": seed, "kind": kind, "mode": mode,
                "minimum_executed_clearance_m": round(min(
                    row["evaluator_clearance_m"] for row in rows), 5),
                "final_pose_error_m": rows[-1]["evaluator_pose_error_m"],
                "maximum_pose_error_m": {name: max(
                    row["evaluator_pose_error_m"][name] for row in rows)
                    for name in poses},
                "upper_registered_steps": sum(row["rgbd_status"]["upper_long"]
                                              ["status"] == "registered"
                                              for row in rows[1:]),
                "lower_short_registered_steps": sum(
                    row["rgbd_status"]["lower_short"]["status"] == "registered"
                    for row in rows[1:]),
                "motion_gate_counts": {name: sum(row["motion_gate"] == name
                                                 for row in rows[1:])
                                       for name in ("fused", "motion_rejected",
                                                    "visual_unavailable")},
                "trace_path": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
                "video_path": str(video_path.relative_to(ROOT)),
                "video_sha256": sha(video_path)}
    finally:
        video.close()
        room.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    development = development_replay()
    print("development", development["mean_final_error_m"],
          development["worst_final_error_m"], flush=True)
    cases = []
    for seed, kind, mode in CASES:
        row = heldout_case(seed, kind, mode)
        cases.append(row)
        print(seed, kind, mode, row["final_pose_error_m"], flush=True)
    methods = tuple(cases[0]["final_pose_error_m"])
    summary = {name: {
        "mean_final_error_m": round(float(np.mean([
            case["final_pose_error_m"][name] for case in cases])), 5),
        "worst_final_error_m": max(
            case["final_pose_error_m"][name] for case in cases),
        "worst_at_any_step_m": max(
            case["maximum_pose_error_m"][name] for case in cases)}
        for name in methods}
    report = {"scope": "V298 frozen motion-consistency gate and half-weight RGB-D fusion replayed on V297 development, then eight fresh Direct Frame open/doorway/screen/raised rooms seeds167-174. Same 0.4s kinematic 2-D route with new slip schedule, simultaneous ideal rendered upper and lower RGB/depth, saved timestamps and inputs. Gate accepts a visual delta only if forward, <=1.2x command and <=2cm cross-track. It uses no true pose. Compare raw upper long RGB-D, gated fusion, lower 7-50cm RGB-D, command+IMU heading, and hypothetical full-circle 11cm range. Evaluator true pose/clearance score only. Real IMU acceleration, walking, sensor package, synchronization defects and delayed capture are untested.",
              "source_sha256": sha(SOURCE),
              "selected_visual_sha256": sha(selected_visual()),
              "development": development, "gate": {
                  "motion_scale_upper": MOTION_SCALE_UPPER,
                  "cross_track_max_m": CROSS_TRACK_MAX_M,
                  "visual_weight": VISUAL_WEIGHT},
              "case_specs": CASES, "slips_evaluator_only": SLIPS,
              "heldout_cases": cases, "heldout_summary": summary}
    (OUT / "localization-gate-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
