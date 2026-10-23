"""Audit V617's actor-only alternate-pose inventory and screen its uncertainty."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_memory_future_probe_v617 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined


OUT = trial.OUT
TURN_STARTS_S = (1.2, 1.6, 2.0, 2.4, 2.8, 5.0)
PROJECTION_END_S = trial.POST_STOP_UNTIL_S - 15.8


def projected_minimum(option: dict, pose: np.ndarray,
                      dynamic: dict, static: list[dict], turn_starts=TURN_STARTS_S) -> float:
    goal = np.asarray(option["goal_xy_m"], dtype=float)
    center = np.asarray(dynamic["center_xy_m"], dtype=float)
    velocity = np.asarray(dynamic["velocity_xy_m_s"], dtype=float)
    arrival_s = option["distance_m"] / .16  # diagnostic motion-rate assumption
    minimum = .5
    for turn_start in turn_starts:
        for side in (-1., 1.):
            turned = np.array([-side * velocity[1], side * velocity[0]])
            for dt in np.arange(0., PROJECTION_END_S + .001, .1):
                object_xy = (center + velocity * min(dt, turn_start) +
                             turned * max(0., dt - turn_start))
                ape_xy = pose + (goal - pose) * min(1., dt / arrival_s)
                # Circle fit, provisional ape proxy, and growing center uncertainty.
                gap = (float(np.linalg.norm(ape_xy - object_xy)) -
                       dynamic["radius_m"] - .17 - .03 - .02 * dt)
                minimum = min(minimum, gap)
    for other in static:
        gap = (float(np.linalg.norm(goal - other["center_xy_m"])) -
               other["radius_m"] - .17 - .03)
        minimum = min(minimum, gap)
    return minimum


def audit_probe(ep: dict, prior) -> dict:
    scans = [row for row in ep["trace"] if row["stage"] == "predeparture_range_scan"]
    assert scans and ep["memory_return"]["future_probe_requested"]
    independent = MultiNovelRangeObjects(
        prior["xs"], prior["ys"], prior["free"], prior["occupied"])
    decision_row = None
    for row in scans:
        replay = independent.update(row["range_72_beams_m"],
                                    np.asarray(row["actor_estimated_xy_m"]),
                                    row["time_s"])
        recorded = row["actor_multi_tracks"]
        assert len(replay["tracks"]) == len(recorded)
        for track in replay["tracks"]:
            match = min(recorded, key=lambda other: np.linalg.norm(
                np.asarray(track["center_xy_m"]) - other["center_xy_m"]))
            assert np.linalg.norm(np.asarray(track["center_xy_m"]) -
                                  match["center_xy_m"]) < .035
            assert abs(track["radius_m"] - match["radius_m"]) < .035
        if "actor_known_free_alternatives" in row:
            assert decision_row is None
            decision_row = row
    assert decision_row is not None
    assert decision_row["time_s"] == 15.8
    alternatives = decision_row["actor_known_free_alternatives"]
    assert alternatives
    pose = np.asarray(decision_row["actor_estimated_xy_m"])
    for option in alternatives:
        angle = 2 * np.pi * option["direction_index"] / 16
        expected = pose + option["distance_m"] * np.array([
            np.cos(angle), np.sin(angle)])
        assert np.linalg.norm(np.asarray(option["goal_xy_m"]) - expected) < .00002
        assert option["path_cells"] >= 2
    tracks = decision_row["actor_multi_tracks"]
    dynamic = max(tracks, key=lambda track: np.linalg.norm(
        track["velocity_xy_m_s"] or (0., 0.)))
    static = [track for track in tracks if track["id"] != dynamic["id"]]
    assert dynamic["status"] == "confirmed_now"
    assert .3 <= np.linalg.norm(dynamic["velocity_xy_m_s"]) <= .7
    broad = [(projected_minimum(option, pose, dynamic, static), option)
             for option in alternatives]
    straight = [(projected_minimum(option, pose, dynamic, static, (5.,)), option)
                for option in alternatives]
    broad.sort(key=lambda pair: pair[0], reverse=True)
    straight.sort(key=lambda pair: pair[0], reverse=True)
    return {"source": ep["memory_return"]["source"],
            "watch_scans_replayed": len(scans),
            "actor_known_free_alternatives": len(alternatives),
            "dynamic_track_center_xy_m": dynamic["center_xy_m"],
            "dynamic_track_velocity_xy_m_s": dynamic["velocity_xy_m_s"],
            "dynamic_track_radius_m": dynamic["radius_m"],
            "straight_model_candidates_above_5cm": int(sum(
                minimum >= .05 for minimum, _ in straight)),
            "turn_uncertainty_candidates_above_5cm": int(sum(
                minimum >= .05 for minimum, _ in broad)),
            "best_straight_model_gap_m": round(straight[0][0], 4),
            "best_turn_uncertainty_gap_m": round(broad[0][0], 4),
            "best_turn_uncertainty_goal_xy_m": broad[0][1]["goal_xy_m"]}


def main() -> None:
    declaration_path = OUT / "predeclared-future-probe.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__))):
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
        for ep, (source, label) in zip(report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{label}_future_probe"
            rows.append({**base_audit.audit_episode(ep, label),
                         **audit_probe(ep, prior)})
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    result = {"schema": "robot-learning-lab.robocasa-memory-future-probe-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "scope": "Actor-track and alternate-goal diagnosis only; no forecast policy dispatched",
              "forecast_assumptions": {
                  "turn_after_s": TURN_STARTS_S,
                  "turn_directions": "instant 90-degree left/right or straight",
                  "projection_horizon_s": PROJECTION_END_S,
                  "ape_diagnostic_rate_m_s": .16,
                  "object_center_uncertainty_m": "0.03 + 0.02 * horizon_s",
                  "ape_proxy_radius_m": .17,
                  "candidate_gap_threshold_m": .05},
              "videos_decoded": len(rows),
              "watch_samples_20ms": sum(row["predeparture_samples_20ms"] for row in rows),
              "moving_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
