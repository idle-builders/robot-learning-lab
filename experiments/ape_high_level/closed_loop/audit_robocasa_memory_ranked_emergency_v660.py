"""Audit V659 owner videos, geometry, source packets and emergency ranking."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_trail_motion_v651 as trail_audit
import audit_robocasa_memory_warning_confirmation_v653 as warning_audit
import dynamic_trail_map
import held_pose_screen
import nominal_emergency_rank
import run_robocasa_memory_ranked_emergency_v659 as trial
import run_robocasa_memory_bearing_v656 as bearing_trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def independent_score(candidate: dict, track: dict, decision_s: float,
                      until_s: float, pose_error_m: float) -> dict:
    path = np.asarray(candidate["future_hold_screen"]["path_xy_m"])
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    arrival = cumulative[-1] / .235
    route_gap, hold_gap = 1., 1.
    center = np.asarray(track["center_xy_m"])
    velocity = np.asarray(track["velocity_xy_m_s"])
    for elapsed in np.arange(0., until_s - decision_s + .0001, .1):
        distance = np.clip(elapsed * .235, 0., cumulative[-1])
        robot = np.array([np.interp(distance, cumulative, path[:, axis])
                          for axis in (0, 1)])
        object_age = decision_s + elapsed - track["last_seen_s"]
        object_xy = center + object_age * velocity
        gap = (np.linalg.norm(robot - object_xy) - .17 -
               track["radius_m"] - pose_error_m - .03 -
               .02 * object_age)
        if elapsed < arrival:
            route_gap = min(route_gap, float(gap))
        else:
            hold_gap = min(hold_gap, float(gap))
    if hold_gap == 1.:
        hold_gap = route_gap
    return {"nominal_route_min_gap_m": route_gap,
            "nominal_hold_min_gap_m": hold_gap,
            "nominal_joint_min_gap_m": min(route_gap, hold_gap),
            "nominal_arrival_s": arrival}


def audit_rank(ep: dict, source: str, strategy: str) -> dict:
    summary = ep["memory_return"]
    assert summary["emergency_candidate_strategy_requested"] == strategy
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    if not decisions:
        return {"ranked_candidates": 0, "selected_support_status": None}
    assert len(decisions) == 1
    decision = decisions[0]
    options = decision["actor_candidate_goals"]
    assert len(options) == 24
    assert not any(row.get("future_hold_screen", {}).get(
        "supported_under_models") for row in options)
    if strategy == "nominal_ranked":
        ranking = decision["actor_emergency_candidates"]
        assert len(ranking) == sum(row["plan_status"] == "known_goal" and
                                   row["has_request"] for row in options)
        for row in ranking:
            actual = independent_score(row, decision["actor_track"],
                                       decision["time_s"],
                                       trial.POST_STOP_UNTIL_S,
                                       .05 if source == "range" else .12)
            for key, value in actual.items():
                assert abs(row["nominal_emergency_score"][key] - value) < .0002
        assert all((a["nominal_emergency_score"]["nominal_joint_min_gap_m"],
                    a["nominal_emergency_score"]["nominal_hold_min_gap_m"],
                    -a["nominal_emergency_score"]["nominal_arrival_s"]) >=
                   (b["nominal_emergency_score"]["nominal_joint_min_gap_m"],
                    b["nominal_emergency_score"]["nominal_hold_min_gap_m"],
                    -b["nominal_emergency_score"]["nominal_arrival_s"])
                   for a, b in zip(ranking, ranking[1:]))
        assert ranking[0]["goal_xy_m"] == (
            decision["actor_selected_goal_xy_m"])
        assert decision["actor_selected_goal_support_status"] == (
            "unqualified_nominal_ranked_known_free")
    else:
        assert decision["actor_selected_goal_support_status"] == (
            "unqualified_emergency_known_free")
        ranking = decision["actor_emergency_candidates"]
    return {"ranked_candidates": len(ranking),
            "selected_support_status":
                decision["actor_selected_goal_support_status"],
            "selected_goal_xy_m": decision["actor_selected_goal_xy_m"]}


def compare_predecision(first: dict, second: dict) -> dict:
    decisions = [row for row in first["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    cutoff = decisions[0]["time_s"] if decisions else trial.POST_STOP_UNTIL_S

    def packets(ep: dict) -> list[dict]:
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                    "memory_route_decision",
                                    "memory_post_stop_range_scan") and
                row["time_s"] <= cutoff and row.get("phase") != "evade_reactive"]

    a, b = packets(first), packets(second)
    exact = sum(x["stage"] == y["stage"] and x["time_s"] == y["time_s"] and
                x["range_72_beams_m"] == y["range_72_beams_m"]
                for x, y in zip(a, b))
    return {"cutoff_s": cutoff, "first_packets": len(a),
            "second_packets": len(b), "exact_packets": exact,
            "exact_pair": len(a) == len(b) == exact}


def main() -> None:
    declaration = OUT / "predeclared-ranked-emergency.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("rank_source_sha256", Path(nominal_emergency_rank.__file__)),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("bearing_source_sha256", bearing_trial.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declared[key] == base_audit.sha(OUT / source.name)
    assert len(report["episodes"]) == len(trial.PROFILES)
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (direction, source, strategy) in zip(report["episodes"],
                                                      trial.PROFILES):
            if direction == "stationary":
                independent_motion = old_motion("stationary")
            else:
                original = old_motion("moving_across_return")
                stage = crossing.STAGE
                end = bearing_trial.ENDS[direction]

                def independent_motion(time_s: float):
                    shifted = time_s - .4
                    if shifted <= crossing.ENTRY_END_S:
                        return original(shifted)
                    fraction = min(1., max(0.,
                        (shifted - crossing.ENTRY_END_S) /
                        (crossing.CROSS_END_S - crossing.ENTRY_END_S)))
                    return [stage[i] + fraction * (end[i] - stage[i])
                            for i in (0, 1)]

            crossing.prop_motion = lambda _, fn=independent_motion: fn
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == (
                f"{direction}_seed{trial.SEED}_{strategy}")
            rows.append({"direction": direction, "source": source,
                         "strategy": strategy,
                         **base_audit.audit_episode(ep, direction),
                         **trail_audit.audit_map_evidence(ep, True),
                         **warning_audit.audit_warnings(ep, 1),
                         **audit_rank(ep, source, strategy)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 2):
        direction, source, _ = trial.PROFILES[index]
        pairs[f"{direction}_{source}"] = compare_predecision(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-ranked-emergency-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Matched emergency route ordering in full owner task; independent packet, rank arithmetic, fixture/body, and video replay",
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
