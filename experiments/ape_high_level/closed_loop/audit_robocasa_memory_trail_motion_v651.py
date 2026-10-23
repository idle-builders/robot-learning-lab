"""Audit V650's held-to-motion dynamic-map handoff and full owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_dynamic_trail_v649 as trail_audit
import audit_robocasa_memory_future_hold_v642 as screen_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_trail_motion_v650 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_map_evidence(ep: dict, motion_reopen: bool) -> dict:
    summary = ep["memory_return"]
    assert summary["dynamic_trail_reopen_requested"]
    assert summary["dynamic_trail_reopen_during_reactive_requested"] == motion_reopen
    rows = [row for row in ep["trace"]
            if row["stage"] == "memory_dynamic_trail_reopen"]
    held_scans = {row["time_s"]: row for row in ep["trace"]
                  if row["stage"] == "memory_post_stop_range_scan"}
    held_rows = [row for row in rows if row.get("phase") is None]
    motion_rows = [row for row in rows if row.get("phase") == "evade_reactive"]
    assert len(held_rows) == len(held_scans)
    if not motion_reopen:
        assert not motion_rows
    xs = np.arange(-.70, 5.11, .10)
    ys = np.arange(-4.75, -.24, .10)
    total = 0
    for row in rows:
        packet = (held_scans[row["capture_time_s"]]
                  if row.get("phase") is None else row)
        fresh, trail, current = trail_audit.actor_masks(ep, row, packet, xs, ys)
        seen = set()
        for y, x in row["reclaimed_indices_yx"]:
            assert (y, x) not in seen
            assert fresh[y, x] and trail[y, x] and not current[y, x]
            seen.add((y, x))
        assert len(seen) == row["reclaimed_cell_count"]
        total += len(seen)
    assert total == summary["dynamic_trail_reclaimed_cells"]
    reactive = [row for row in ep["trace"]
                if row["stage"] == "memory_route_decision" and
                row["phase"] == "evade_reactive"]
    return {"held_reopen_packets": len(held_rows),
            "reactive_reopen_packets": len(motion_rows),
            "reclaimed_events": total,
            "reactive_route_decisions": len(reactive),
            "first_reactive_plan_status": (
                None if not reactive else reactive[0]["plan_status"]),
            "reactive_motion_samples_20ms": sum(
                len(row["motion_substeps"]) for row in reactive)}


def compare_pair(first: dict, second: dict) -> dict:
    packets = screen_audit.compare_predecision(first, second)
    first_decision = next(row for row in first["trace"]
                          if row["stage"] == "memory_held_hazard_decision")
    second_decision = next(row for row in second["trace"]
                           if row["stage"] == "memory_held_hazard_decision")
    assert first_decision["time_s"] == second_decision["time_s"]
    assert first_decision["actor_map_snapshot"] == second_decision["actor_map_snapshot"]
    return {**packets, "same_map_at_warning": True,
            "same_selected_goal": (first_decision["actor_selected_goal_xy_m"] ==
                                   second_decision["actor_selected_goal_xy_m"])}


def main() -> None:
    declaration = OUT / "predeclared-trail-motion.json"
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
        for ep, (source, label, seed, motion_reopen) in zip(
                report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{label}_seed{seed}_{'motion' if motion_reopen else 'held_only'}_trail")
            rows.append({"paired_scan_seed": seed,
                         "motion_reopen": motion_reopen,
                         **base_audit.audit_episode(ep, label),
                         **audit_map_evidence(ep, motion_reopen)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {
        "rgbd_early": compare_pair(report["episodes"][0], report["episodes"][1]),
        "range_late": compare_pair(report["episodes"][2], report["episodes"][3]),
        "rgbd_late": compare_pair(report["episodes"][4], report["episodes"][5]),
    }
    result = {
        "schema": "robot-learning-lab.robocasa-memory-trail-motion-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Observed-clear map semantics at held and reactive-motion scans; full owner-task provisional-body replay",
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
