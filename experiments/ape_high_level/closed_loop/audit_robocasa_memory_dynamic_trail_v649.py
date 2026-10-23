"""Replay V648 owner outcomes and verify each reopened cell has actor ray support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_future_hold_v642 as screen_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_dynamic_trail_v648 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def actor_masks(ep: dict, row: dict, scan: dict, xs, ys) -> tuple:
    xx, yy = np.meshgrid(xs, ys)
    readings = np.asarray([np.nan if value is None else value
                           for value in scan["range_72_beams_m"]])
    pose = np.asarray(scan["actor_estimated_xy_m"])
    distance = np.hypot(xx - pose[0], yy - pose[1])
    angles = ((np.arctan2(yy - pose[1], xx - pose[0]) + np.pi) /
              (2. * np.pi) * len(readings)) % len(readings)
    lower = np.floor(angles).astype(int)
    upper = (lower + 1) % len(readings)
    ray_limit = np.minimum(readings[lower], readings[upper]) - .32
    fresh = np.isfinite(ray_limit) & (distance < ray_limit) & (distance < 4.68)
    trail = np.zeros_like(fresh)
    current = np.zeros_like(fresh)
    eligible = []
    for track in scan["actor_multi_tracks"]:
        velocity = track["velocity_xy_m_s"]
        if (track["status"] != "confirmed_now" or track["hits"] < 3 or
                velocity is None or np.linalg.norm(velocity) < .1 or
                track["radius_m"] < .18):
            continue
        old_centers = []
        for history in ep["multi_history"]:
            if history["capture_time_s"] > row["capture_time_s"] - .1001:
                continue
            for old in history["tracks"]:
                if (old["id"] == track["id"] and old["status"] == "confirmed_now" and
                        abs(old["last_seen_s"] - history["capture_time_s"]) < .0001):
                    old_centers.append(old["center_xy_m"])
        if not old_centers:
            continue
        eligible.append(track["id"])
        radius = track["radius_m"] + .32
        x, y = track["center_xy_m"]
        current |= np.hypot(xx - x, yy - y) < radius
        for x, y in old_centers:
            trail |= np.hypot(xx - x, yy - y) < radius
    assert row["eligible_track_ids"] == eligible
    if row["fresh_clear_cell_count"] != int(np.count_nonzero(fresh)):
        # Actor pose is logged to 5 decimals. A grid cell lying exactly on a
        # beam-sector boundary can change its supporting beam under that
        # rounding, so replay the entire saved-pose rounding box, not an
        # arbitrary count tolerance. Reclaimed cells are checked below.
        possible_counts = set()
        for offset_x in (-.000005, 0., .000005):
            for offset_y in (-.000005, 0., .000005):
                dx = xx - pose[0] - offset_x
                dy = yy - pose[1] - offset_y
                radius = np.hypot(dx, dy)
                beam = ((np.arctan2(dy, dx) + np.pi) /
                        (2. * np.pi) * len(readings)) % len(readings)
                index = np.floor(beam).astype(int)
                limit = np.minimum(readings[index],
                                   readings[(index + 1) % len(readings)]) - .32
                possible_counts.add(int(np.count_nonzero(
                    np.isfinite(limit) & (radius < limit) & (radius < 4.68))))
        assert row["fresh_clear_cell_count"] in possible_counts
    assert row["stale_trail_cell_count"] == int(np.count_nonzero(trail))
    return fresh, trail, current


def audit_trail(ep: dict, enabled: bool) -> dict:
    summary = ep["memory_return"]
    assert summary["dynamic_trail_reopen_requested"] == enabled
    rows = [row for row in ep["trace"]
            if row["stage"] == "memory_dynamic_trail_reopen"]
    if not enabled:
        assert not rows and summary["dynamic_trail_reclaimed_cells"] == 0
        return {"reopen_packets": 0, "reclaimed_events": 0,
                "distinct_reclaimed_cells": 0}
    scans = {row["time_s"]: row for row in ep["trace"]
             if row["stage"] == "memory_post_stop_range_scan"}
    assert len(rows) == len(scans)
    grid_xs = np.arange(-.70, 5.11, .10)
    grid_ys = np.arange(-4.75, -.24, .10)
    distinct = set()
    total = 0
    for row in rows:
        scan = scans[row["capture_time_s"]]
        fresh, trail, current = actor_masks(
            ep, row, scan, grid_xs, grid_ys)
        seen = set()
        for y, x in row["reclaimed_indices_yx"]:
            assert 0 <= y < len(grid_ys) and 0 <= x < len(grid_xs)
            assert (y, x) not in seen
            assert fresh[y, x] and trail[y, x] and not current[y, x]
            seen.add((y, x))
            distinct.add((y, x))
        assert len(seen) == row["reclaimed_cell_count"]
        total += len(seen)
    assert total == summary["dynamic_trail_reclaimed_cells"]
    return {"reopen_packets": len(rows),
            "reclaimed_events": total,
            "distinct_reclaimed_cells": len(distinct)}


def snapshot(ep: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    decision = next(row for row in ep["trace"]
                    if row["stage"] == "memory_held_hazard_decision")
    saved = decision["actor_map_snapshot"]
    assert saved is not None
    shape = (len(saved["ys_m"]), len(saved["xs_m"]))
    free = np.zeros(shape, dtype=bool)
    occupied = np.zeros(shape, dtype=bool)
    for y, x in saved["free_indices_yx"]:
        free[y, x] = True
    for y, x in saved["occupied_indices_yx"]:
        occupied[y, x] = True
    assert not np.any(free & occupied)
    return free, occupied, decision


def compare_pair(base: dict, changed: dict) -> dict:
    packets = screen_audit.compare_predecision(base, changed)
    free_a, occupied_a, decision_a = snapshot(base)
    free_b, occupied_b, decision_b = snapshot(changed)
    assert decision_a["time_s"] == decision_b["time_s"]
    newly_free = free_b & ~free_a
    no_longer_occupied = occupied_a & ~occupied_b
    return {**packets,
            "baseline_known_goals": sum(option["plan_status"] == "known_goal"
                                        for option in decision_a["actor_candidate_goals"]),
            "reopened_known_goals": sum(option["plan_status"] == "known_goal"
                                        for option in decision_b["actor_candidate_goals"]),
            "baseline_supported_goals": sum(option.get("future_hold_screen", {}).get(
                "supported_under_models", False)
                for option in decision_a["actor_candidate_goals"]),
            "reopened_supported_goals": sum(option.get("future_hold_screen", {}).get(
                "supported_under_models", False)
                for option in decision_b["actor_candidate_goals"]),
            "newly_free_cells_at_decision": int(np.count_nonzero(newly_free)),
            "no_longer_occupied_at_decision": int(np.count_nonzero(
                no_longer_occupied)),
            "other_occupied_map_changes": int(np.count_nonzero(
                occupied_b & ~occupied_a))}


def main() -> None:
    declaration = OUT / "predeclared-dynamic-trail.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declared[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.prop_motion = lambda label: (
            old_motion("stationary") if label == "stationary" else
            lambda time_s: old_motion("moving_across_return")(
                time_s + ({"late": -.4, "early": .4}[label])))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (source, label, seed, enabled) in zip(
                report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{label}_seed{seed}_{'reopen' if enabled else 'permanent'}_held")
            rows.append({"paired_scan_seed": seed, "trail_reopen": enabled,
                         **base_audit.audit_episode(ep, label),
                         **audit_trail(ep, enabled)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in (0, 2, 4, 6):
        source, _, seed, _ = trial.PROFILES[index]
        pairs[f"{source}_seed{seed}"] = compare_pair(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-dynamic-trail-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Fresh finite-range evidence can reopen only tracked old occupied trail cells; full owner-task provisional-body replay",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
        "reclaimed_events": sum(row["reclaimed_events"] for row in rows),
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
