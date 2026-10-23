"""Audit V646 and counterfactually reopen only freshly cleared track-trail cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import held_pose_screen
import run_robocasa_layout2_route_v407 as route
import run_robocasa_memory_map_snapshot_v646 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def analyze_map(ep: dict) -> dict:
    decision = next(row for row in ep["trace"]
                    if row["stage"] == "memory_held_hazard_decision")
    snapshot = decision["actor_map_snapshot"]
    assert snapshot is not None
    grid = route.SensorGrid()
    assert np.allclose(grid.xs, snapshot["xs_m"])
    assert np.allclose(grid.ys, snapshot["ys_m"])
    grid.free[:] = False
    grid.occupied[:] = False
    for y, x in snapshot["free_indices_yx"]:
        grid.free[y, x] = True
    for y, x in snapshot["occupied_indices_yx"]:
        grid.occupied[y, x] = True
    original_free = int(np.count_nonzero(grid.free))
    original_occupied = int(np.count_nonzero(grid.occupied))
    scan = next(row for row in ep["trace"]
                if row["stage"] == "memory_post_stop_range_scan" and
                row["time_s"] == decision["time_s"])
    pose = np.asarray(scan["actor_estimated_xy_m"])
    ranges = np.asarray([np.nan if value is None else value
                         for value in scan["range_72_beams_m"]])
    angle = ((np.arctan2(grid.yy - pose[1], grid.xx - pose[0]) + np.pi) /
             (2. * np.pi) * len(ranges)) % len(ranges)
    low = np.floor(angle).astype(int)
    high = (low + 1) % len(ranges)
    distance = np.hypot(grid.xx - pose[0], grid.yy - pose[1])
    ray_limit = np.minimum(ranges[low], ranges[high]) - .32
    freshly_clear = (np.isfinite(ray_limit) & (distance < ray_limit) &
                     (distance < 5. - .32))
    track = decision["actor_track"]
    stale_trail = np.zeros_like(grid.occupied)
    for history in ep["multi_history"]:
        if history["capture_time_s"] > decision["time_s"] - .101:
            continue
        for old in history["tracks"]:
            if (old["id"] == track["id"] and
                    old["status"] == "confirmed_now" and
                    old["last_seen_s"] == history["capture_time_s"]):
                center = np.asarray(old["center_xy_m"])
                stale_trail |= np.hypot(grid.xx - center[0],
                                        grid.yy - center[1]) < track["radius_m"] + .32
    center = np.asarray(track["center_xy_m"])
    current = np.hypot(grid.xx - center[0],
                       grid.yy - center[1]) < track["radius_m"] + .32
    reclaim = grid.occupied & freshly_clear & stale_trail & ~current
    options = decision["actor_candidate_goals"]
    original = [grid.plan(pose, np.asarray(option["goal_xy_m"]))[1]
                for option in options]
    assert all(status == option["plan_status"] for status, option in
               zip(original, options))
    grid.occupied[reclaim] = False
    grid.free[reclaim] = True
    changed = []
    supported = 0
    after = []
    for option, previous in zip(options, original):
        goal = np.asarray(option["goal_xy_m"])
        request, status, path = grid.plan(pose, goal)
        after.append(status)
        if status != previous:
            changed.append({"goal_xy_m": option["goal_xy_m"],
                            "before": previous, "after": status})
        if status == "known_goal" and request is not None:
            path_xy = [pose.tolist()] + [grid.xy(cell).tolist()
                                          for cell in path[1:]]
            forecast = held_pose_screen.screen(
                path_xy_m=path_xy,
                track_center_xy_m=track["center_xy_m"],
                track_velocity_xy_m_s=track["velocity_xy_m_s"],
                track_radius_m=track["radius_m"],
                track_capture_time_s=track["last_seen_s"],
                decision_time_s=decision["time_s"],
                hold_until_s=trial.POST_STOP_UNTIL_S,
                pose_error_m=(.05 if ep["memory_return"]["source"] == "range"
                              else .12))
            supported += int(forecast["supported_under_models"])
    return {"source": ep["memory_return"]["source"],
            "snapshot_free_cells": original_free,
            "snapshot_occupied_cells": original_occupied,
            "fresh_finite_ray_clear_cells": int(np.count_nonzero(freshly_clear)),
            "old_track_trail_cells": int(np.count_nonzero(stale_trail)),
            "reclaimed_observed_clear_cells": int(np.count_nonzero(reclaim)),
            "baseline_known_goals": original.count("known_goal"),
            "reopened_known_goals": after.count("known_goal"),
            "reopened_model_supported_goals": supported,
            "changed_candidate_statuses": changed}


def main() -> None:
    declaration = OUT / "predeclared-map-snapshot.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    assert declared["source_sha256"] == base_audit.sha(OUT / trial.SOURCE.name)
    assert declared["memory_source_sha256"] == base_audit.sha(
        OUT / memory.SOURCE.name)
    assert len(report["episodes"]) == 2
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.prop_motion = lambda _: lambda time_s: old_motion(
            "moving_across_return")(time_s - .4)
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        old_margin = route.MARGIN_M
        route.MARGIN_M = .15
        for ep in report["episodes"]:
            rows.append({**base_audit.audit_episode(ep, "late"),
                         **analyze_map(ep)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
        route.MARGIN_M = old_margin
    result = {
        "schema": "robot-learning-lab.robocasa-memory-map-snapshot-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Posthoc actor-evidence map counterfactual, not a dispatched map policy; provisional body",
        "videos_decoded": len(rows),
        "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
