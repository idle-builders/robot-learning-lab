"""Audit V635's paired depth handoff and non-dispatched route sweep."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_depth_handoff_v634 as handoff_audit
import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_depth_candidates_v635 as trial
import run_robocasa_memory_return_v599 as memory
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_probe(ep: dict) -> dict:
    rows = [row for row in ep["trace"]
            if row["stage"] == "memory_depth_candidate_probe"]
    if not rows:
        return {"probe_count": 0, "candidate_count": 0,
                "currently_reachable": 0,
                "reachable_with_straight_forecast_gap_at_least_5cm": 0}
    assert len(rows) == 1 and rows[0]["probe_only"]
    decision = next(row for row in ep["trace"]
                    if row["stage"] == "memory_held_hazard_decision" and
                    row.get("actor_source") == "head_depth_after_range_loss")
    capture = next(row for row in ep["trace"]
                   if row["stage"] == "memory_head_depth_fallback" and
                   row["time_s"] == decision["time_s"])
    pose = np.asarray(capture["actor_estimated_xy_m"])
    center = np.asarray(decision["actor_track"]["center_xy_m"])
    velocity = np.asarray(decision["actor_track"]["velocity_xy_m_s"])
    radius = decision["actor_track"]["radius_m"]
    candidates = rows[0]["actor_candidates"]
    assert len(candidates) == 16 * 7
    reachable = supported = 0
    for option in candidates:
        distance = option["distance_m"]
        bearing = option["bearing_rad"]
        goal = np.asarray(option["goal_xy_m"])
        assert np.linalg.norm(goal - (pose + distance * np.array([
            np.cos(bearing), np.sin(bearing)]))) < .00005
        arrival = distance / .16
        horizon = min(2.4, trial.POST_STOP_UNTIL_S - decision["time_s"])
        gap = min(float(np.linalg.norm(
            pose + min(1., dt / arrival) * (goal - pose) -
            (center + velocity * dt)) - radius - .17 - .03 - .02 * dt)
            for dt in np.arange(0., horizon + .001, .1))
        assert abs(gap - option["straight_constant_velocity_gap_m"]) < .00004
        if option["plan_status"] == "known_goal" and option["has_request"]:
            reachable += 1
            supported += gap >= .05
    return {"probe_count": len(rows), "candidate_count": len(candidates),
            "currently_reachable": reachable,
            "reachable_with_straight_forecast_gap_at_least_5cm": supported}


def main() -> None:
    declaration_path = OUT / "predeclared-depth-candidates.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    assert declaration["paired_scan_seed"] == trial.PAIRED_SCAN_SEED
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    prior = np.load(joined.PRIOR)
    camera_model = joined.scene_with_owner_and_prop(crossing.START)
    rows = []
    for ep, (source, label, dropout, depth_fallback) in zip(
            report["episodes"], trial.PROFILES):
        assert ep["memory_return"]["paired_scan_seed_requested"] == (
            trial.PAIRED_SCAN_SEED)
        assert ep["memory_return"]["depth_candidate_probe_requested"] == (
            depth_fallback)
        rows.append({**handoff_audit.audit_episode(
            ep, source, label, dropout, depth_fallback, prior, camera_model),
                     **audit_probe(ep)})
    pairs = {source: handoff_audit.compare_before_loss(
                 report["episodes"][i], report["episodes"][i + 1])
             for source, i in (("range", 2), ("rgbd", 4))}
    assert all(row["identical"] == row["compared"] for row in pairs.values())
    result = {"schema": "robot-learning-lab.robocasa-memory-depth-candidates-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "videos_decoded": len(rows),
              "pre_loss_pair_comparison": pairs,
              "head_depth_captures_replayed": sum(
                  len(row["head_depth_captures_checked"]) for row in rows),
              "movement_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
