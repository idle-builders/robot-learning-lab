"""Replay V641 owner outcomes and independently calculate held-pose screen gaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import run_robocasa_memory_future_hold_v641 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker
import held_pose_screen


OUT = trial.OUT


def independent_screen(option: dict, track: dict, now_s: float,
                       source: str) -> dict:
    saved = option["future_hold_screen"]
    path = np.asarray(saved["path_xy_m"], dtype=float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    distance_axis = np.r_[0., np.cumsum(lengths)]
    center = np.asarray(track["center_xy_m"])
    velocity = np.asarray(track["velocity_xy_m_s"])
    pose_error = .05 if source == "range" else .12
    worst = np.inf
    worst_label = None
    for speed in (.145, .235):
        for delay in (0., .2):
            for turn_s, turn_angle in ((None, 0.),
                                       (.6, -np.pi / 2.), (.6, np.pi / 2.),
                                       (1.2, -np.pi / 2.), (1.2, np.pi / 2.)):
                c, s = np.cos(turn_angle), np.sin(turn_angle)
                new_velocity = np.array([c * velocity[0] - s * velocity[1],
                                         s * velocity[0] + c * velocity[1]])
                for dt in np.arange(0., trial.POST_STOP_UNTIL_S - now_s + .051, .1):
                    along = np.clip((dt - delay) * speed, 0., distance_axis[-1])
                    ape = np.array([np.interp(along, distance_axis, path[:, axis])
                                    for axis in (0, 1)])
                    age = now_s + dt - track["last_seen_s"]
                    if turn_s is None or age <= turn_s:
                        obj = center + velocity * age
                    else:
                        obj = center + velocity * turn_s + new_velocity * (age - turn_s)
                    gap = (np.linalg.norm(ape - obj) - .17 - track["radius_m"] -
                           pose_error - .03 - .02 * age)
                    if gap < worst:
                        worst = gap
                        worst_label = (dt, speed, delay, turn_s, turn_angle)
    slow_arrival = .2 + distance_axis[-1] / .145
    modeled_hold = trial.POST_STOP_UNTIL_S - now_s - slow_arrival
    supported = bool(worst >= .05 and modeled_hold >= .4)
    # Actor paths are retained at five decimals, so replay differs slightly
    # from the unrounded in-process path used at decision time.
    assert abs(round(worst, 5) - saved["worst_modeled_gap_m"]) < .0002
    assert abs(round(slow_arrival, 5) - saved["slow_arrival_from_decision_s"]) < .0001
    assert abs(round(modeled_hold, 5) - saved["modeled_hold_duration_s"]) < .0001
    assert supported == saved["supported_under_models"]
    assert abs(saved["pose_error_assumption_m"] - pose_error) < 1.e-8
    return {"worst_gap_m": round(float(worst), 5),
            "supported": supported, "worst_case": worst_label}


def audit_screen(ep: dict, source: str, label: str, enabled: bool) -> dict:
    summary = ep["memory_return"]
    assert summary["future_hold_screen_requested"] == enabled
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    if not decisions:
        assert not decisions
        return {"screened_candidates": 0, "supported_candidates": 0,
                "decision_time_s": None, "selected_goal_xy_m": None}
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision["actor_future_hold_screen_requested"] == enabled
    scan = next(row for row in ep["trace"]
                if row["stage"] == "memory_post_stop_range_scan" and
                row["time_s"] == decision["time_s"])
    track = decision["actor_track"]
    assert track in scan["actor_multi_tracks"]
    assert track["status"] == "confirmed_now" and track["velocity_xy_m_s"][0] < -.3
    options = decision["actor_candidate_goals"]
    if not enabled:
        assert not any("future_hold_screen" in option for option in options)
        return {"screened_candidates": 0, "supported_candidates": 0,
                "decision_time_s": decision["time_s"],
                "selected_goal_xy_m": decision["actor_selected_goal_xy_m"]}
    assert len(options) == 24
    checked = []
    for option in options:
        if option["plan_status"] != "known_goal" or not option["has_request"]:
            assert "future_hold_screen" not in option
            continue
        saved = option["future_hold_screen"]
        path = np.asarray(saved["path_xy_m"])
        assert np.linalg.norm(path[0] - scan["actor_estimated_xy_m"]) < .00002
        assert np.linalg.norm(path[-1] - option["goal_xy_m"]) <= .071
        assert np.linalg.norm(path[1] - path[0]) <= .215
        if len(path) > 2:
            assert np.max(np.linalg.norm(np.diff(path[1:], axis=0), axis=1)) <= .143
        checked.append((option, independent_screen(
            option, track, decision["time_s"], source)))
    viable = [option for option, result in checked if result["supported"]]
    if viable:
        selected = max(viable, key=lambda option: (
            option["future_hold_screen"]["worst_modeled_gap_m"],
            -option["future_hold_screen"]["path_length_m"]))
        assert np.linalg.norm(np.asarray(selected["goal_xy_m"]) -
                              decision["actor_selected_goal_xy_m"]) < .00002
    else:
        assert decision["actor_selected_goal_xy_m"] is None
        assert summary["route_status"] == "no_actor_supported_held_pose"
        assert any(row.get("speech_intent") for row in ep["trace"]
                   if row["stage"] == "memory_held_reaction_result")
    return {"screened_candidates": len(checked),
            "supported_candidates": len(viable),
            "best_modeled_gap_m": (None if not checked else
                max(result["worst_gap_m"] for _, result in checked)),
            "decision_time_s": decision["time_s"],
            "selected_goal_xy_m": decision["actor_selected_goal_xy_m"]}


def compare_predecision(first: dict, second: dict) -> dict:
    decision = next(row for row in first["trace"]
                    if row["stage"] == "memory_held_hazard_decision")
    cutoff = decision["time_s"]
    def packets(ep):
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                    "memory_route_decision",
                                    "memory_post_stop_range_scan") and
                row["time_s"] <= cutoff and row.get("phase") != "evade_reactive"]
    a, b = packets(first), packets(second)
    matched = sum(x["stage"] == y["stage"] and x["time_s"] == y["time_s"] and
                  x["range_72_beams_m"] == y["range_72_beams_m"]
                  for x, y in zip(a, b))
    return {"first_packets": len(a), "second_packets": len(b),
            "exact_prefix_packets": matched,
            "exact_pair": len(a) == len(b) == matched}


def main() -> None:
    declaration_path = OUT / "predeclared-future-hold.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration_path)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declaration[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    original_motion = crossing.prop_motion
    original_horizon = crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.prop_motion = lambda label: (
            original_motion("stationary") if label == "stationary" else
            lambda time_s: original_motion("moving_across_return")(
                time_s + trial.SHIFTS_S[label]))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (source, label, enabled) in zip(report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{label}_{'screen' if enabled else 'baseline'}_held")
            rows.append({**base_audit.audit_episode(ep, label),
                         **audit_screen(ep, source, label, enabled),
                         "future_hold_screen": enabled})
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    pairs = {
        "range_late": compare_predecision(report["episodes"][4],
                                          report["episodes"][5]),
        "rgbd_late": compare_predecision(report["episodes"][6],
                                         report["episodes"][7]),
    }
    result = {
        "schema": "robot-learning-lab.robocasa-memory-future-hold-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Finite actor forecast and full owner-task proxy replay; no selected-body certificate",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
        "screened_known_goals": sum(row["screened_candidates"] for row in rows),
        "supported_goals": sum(row["supported_candidates"] for row in rows),
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
