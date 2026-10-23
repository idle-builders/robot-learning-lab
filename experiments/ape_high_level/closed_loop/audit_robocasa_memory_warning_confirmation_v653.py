"""Audit V652 warning persistence and the complete owner outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_trail_motion_v651 as trail_audit
import audit_robocasa_memory_future_hold_v642 as screen_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_warning_confirmation_v652 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def projected_gap(scan: dict, track: dict) -> float:
    pose = np.asarray(scan["actor_estimated_xy_m"])
    center = np.asarray(track["center_xy_m"])
    velocity = np.asarray(track["velocity_xy_m_s"])
    return min(float(np.linalg.norm(pose - (center + velocity * dt)) -
                     track["radius_m"] - .17 - .03 - .02 * dt)
               for dt in np.arange(0., 1.201, .1))


def audit_warnings(ep: dict, required: int) -> dict:
    summary = ep["memory_return"]
    assert summary["held_warning_confirmations_requested"] == required
    scans = {row["time_s"]: row for row in ep["trace"]
             if row["stage"] == "memory_post_stop_range_scan"}
    warnings = [row for row in ep["trace"]
                if row["stage"] == "memory_held_hazard_warning"]
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    assert len(decisions) <= 1
    if required == 1:
        assert not warnings
    previous = {}
    for row in warnings:
        scan = scans[row["time_s"]]
        track = row["actor_track"]
        assert track in scan["actor_multi_tracks"]
        assert track["status"] == "confirmed_now"
        assert track["radius_m"] >= .18 and track["velocity_xy_m_s"][0] < -.30
        assert abs(projected_gap(scan, track) -
                   row["actor_projected_gap_m"]) < .00002
        assert row["actor_projected_gap_m"] < .05
        last = previous.get(track["id"])
        expected = (last[1] + 1 if last is not None and
                    abs(row["time_s"] - last[0] - .1) < .001 else 1)
        assert row["actor_warning_streak"] == expected
        assert row["actor_confirmations_required"] == required
        previous[track["id"]] = (row["time_s"], expected)
    if decisions:
        decision = decisions[0]
        scan = scans[decision["time_s"]]
        track = decision["actor_track"]
        assert track in scan["actor_multi_tracks"]
        assert track["status"] == "confirmed_now"
        assert track["velocity_xy_m_s"][0] < -.30
        assert abs(projected_gap(scan, track) -
                   decision["actor_projected_gap_m"]) < .00002
        assert decision["actor_projected_gap_m"] < .05
        if required == 2:
            matched = [row for row in warnings
                       if row["time_s"] == decision["time_s"] and
                       row["actor_track"]["id"] == track["id"]]
            assert len(matched) == 1 and matched[0]["actor_warning_streak"] >= 2
    elif required == 2:
        assert not any(row["actor_warning_streak"] >= 2 for row in warnings)
    reactive = [row for row in ep["trace"]
                if row["stage"] == "memory_route_decision" and
                row["phase"] == "evade_reactive"]
    if not decisions:
        assert not reactive
    return {"preliminary_warning_count": len(warnings),
            "first_preliminary_warning_s": (
                None if not warnings else warnings[0]["time_s"]),
            "decision_time_s": None if not decisions else decisions[0]["time_s"],
            "reactive_motion_samples_20ms": sum(len(row["motion_substeps"])
                                                 for row in reactive)}


def main() -> None:
    declaration = OUT / "predeclared-warning-confirmation.json"
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
        for ep, (source, label, seed, required) in zip(
                report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{label}_seed{seed}_{required}warning_held")
            rows.append({"paired_scan_seed": seed,
                         "confirmations_required": required,
                         **base_audit.audit_episode(ep, label),
                         **trail_audit.audit_map_evidence(ep, True),
                         **audit_warnings(ep, required)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in (2, 4, 6, 8, 10):
        source, label, seed, _ = trial.PROFILES[index]
        pairs[f"{source}_{label}_seed{seed}"] = screen_audit.compare_predecision(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-warning-confirmation-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "One versus two fresh consecutive held range warnings with dynamic map consistency; full owner-task provisional-body replay",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
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
