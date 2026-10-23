"""Audit held range packets and prospective warnings in V623 owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_memory_timed_monitor_v623 as trial
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT


def projected_gap(row: dict, track: dict) -> float:
    pose = np.asarray(row["actor_estimated_xy_m"], dtype=float)
    center = np.asarray(track["center_xy_m"], dtype=float)
    velocity = np.asarray(track["velocity_xy_m_s"], dtype=float)
    return min(float(np.linalg.norm(pose - (center + velocity * dt)) -
                     track["radius_m"] - .17 - .03 - .02 * dt)
               for dt in np.arange(0., 1.201, .1))


def audit_monitor(ep: dict, prior, interval_s: float) -> dict:
    summary = ep["memory_return"]
    assert summary["hold_monitor_requested"]
    assert abs(summary["hold_scan_interval_s_requested"] - interval_s) < 1.e-8
    packets = [row for row in ep["trace"] if row["stage"] in (
        "predeparture_range_scan", "memory_route_decision",
        "memory_post_stop_range_scan")]
    held = [row for row in packets if row["stage"] == "memory_post_stop_range_scan"]
    assert held
    independent = MultiNovelRangeObjects(
        prior["xs"], prior["ys"], prior["free"], prior["occupied"])
    warnings = []
    checked = 0
    for row in packets:
        scan = row["range_72_beams_m"]
        assert len(scan) == 72
        replay = independent.update(scan,
                                    np.asarray(row["actor_estimated_xy_m"]),
                                    row["time_s"])
        if row["stage"] != "memory_post_stop_range_scan":
            continue
        recorded = row["actor_multi_tracks"]
        # Match current compact surfaces by location, not by tracker ID: the
        # original tracker also saw the earlier owner-route packets.
        fresh = [track for track in recorded if track["status"] == "confirmed_now"
                 and track["center_xy_m"][1] < -3.0]
        for track in fresh:
            assert abs(track["last_seen_s"] - row["time_s"]) < .0001
        replayed = [track for track in replay["tracks"]
                    if track["status"] == "confirmed_now" and
                    track["center_xy_m"][1] < -3.0]
        if fresh and replayed:
            original = min(fresh, key=lambda track: np.linalg.norm(
                np.asarray(track["center_xy_m"]) - row["actor_estimated_xy_m"]))
            match = min(replayed, key=lambda track: np.linalg.norm(
                np.asarray(track["center_xy_m"]) - original["center_xy_m"]))
            assert np.linalg.norm(np.asarray(match["center_xy_m"]) -
                                  original["center_xy_m"]) < .085
            checked += 1
        for track in fresh:
            velocity = track["velocity_xy_m_s"]
            if velocity is None or velocity[0] >= -.3:
                continue
            gap = projected_gap(row, track)
            if gap < .05:
                warnings.append({"time_s": row["time_s"],
                                 "track_id": track["id"],
                                 "predicted_gap_m": round(gap, 5),
                                 "track_velocity_xy_m_s": velocity})
    first_breach = next((row["time_s"] for row in ep["trace"]
                         if row["stage"] == "memory_post_stop_hold" and
                         row["evaluator_proxy_gap_m"] < .05), None)
    first_warning = warnings[0]["time_s"] if warnings else None
    return {"held_scan_interval_s": interval_s,
            "held_range_packets": len(held),
            "fresh_dynamic_surfaces_independently_matched": checked,
            "first_actor_prediction_warning_s": first_warning,
            "first_evaluator_5cm_breach_s": first_breach,
            "warning_lead_before_breach_s": (None if first_warning is None or
                                               first_breach is None else
                                               round(first_breach - first_warning, 4)),
            "actor_prediction_warnings": warnings}


def main() -> None:
    declaration_path = OUT / "predeclared-timed-monitor.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    prior = np.load(joined.PRIOR)
    rows = []
    original_motion = crossing.prop_motion
    original_horizon = crossing.POST_STOP_UNTIL_S
    try:
        crossing.prop_motion = lambda label: (
            lambda time_s: original_motion("moving_across_return")(
                time_s + trial.SHIFTS_S[label]))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (source, label, interval_s) in zip(report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{label}_{int(interval_s * 1000)}ms_timed"
            rows.append({**base_audit.audit_episode(ep, label),
                         **audit_monitor(ep, prior, interval_s)})
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    result = {"schema": "robot-learning-lab.robocasa-memory-timed-monitor-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "scope": "Held-source and forecast warning diagnosis; no second action executed",
              "warning_rule": "Current compact south-aisle track has west velocity <-0.3 m/s and constant-velocity 1.2s projected 17cm proxy gap including 3cm base plus 2cm/s uncertainty <5cm",
              "videos_decoded": len(rows),
              "held_range_packets": sum(row["held_range_packets"] for row in rows),
              "moving_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
