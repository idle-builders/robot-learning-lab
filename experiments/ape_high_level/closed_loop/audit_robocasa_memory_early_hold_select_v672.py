"""Independent route/hold and full-owner audit for the V671 early choice."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_head_upper_latency_v669 as prior_audit
import audit_robocasa_memory_moving_return_v604 as owner_audit
import audit_robocasa_memory_warning_confirmation_v653 as warning_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_early_hold_select_v671 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def path_at_distance(path: np.ndarray, travel_m: float) -> np.ndarray:
    remaining = travel_m
    for start, end in zip(path, path[1:]):
        segment = end - start
        length = float(np.linalg.norm(segment))
        if length <= 1.e-8:
            continue
        if remaining <= length:
            return start + segment * (remaining / length)
        remaining -= length
    return path[-1]


def independent_screen(saved: dict, track: dict, decision_s: float,
                       hold_until_s: float, pose_error_m: float) -> tuple[float, bool]:
    path = np.asarray(saved["path_xy_m"], dtype=float)
    center = np.asarray(track["center_xy_m"], dtype=float)
    velocity = np.asarray(track["velocity_xy_m_s"], dtype=float)
    age = decision_s - track["last_seen_s"]
    duration = hold_until_s - decision_s
    route_length = float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    minimum = float("inf")
    hypotheses = [(None, 0.)] + [
        (when, angle) for when in (.6, 1.2)
        for angle in (-np.pi / 2., np.pi / 2.)]
    for ape_speed in (.145, .235):
        for delay in (0., .2):
            for turn_s, angle in hypotheses:
                turned = np.array((
                    np.cos(angle) * velocity[0] - np.sin(angle) * velocity[1],
                    np.sin(angle) * velocity[0] + np.cos(angle) * velocity[1]))
                for elapsed in np.arange(0., duration + .05, .1):
                    future_s = age + elapsed
                    if turn_s is None or future_s <= turn_s:
                        obstacle = center + velocity * future_s
                    else:
                        obstacle = (center + velocity * turn_s +
                                    turned * (future_s - turn_s))
                    ape = path_at_distance(
                        path, max(0., elapsed - delay) * ape_speed)
                    gap = (float(np.linalg.norm(ape - obstacle)) - .17 -
                           track["radius_m"] - pose_error_m - .03 -
                           .02 * future_s)
                    minimum = min(minimum, gap)
    slow_arrival_s = .2 + route_length / .145
    supported = minimum >= .05 and duration - slow_arrival_s >= .4
    return minimum, supported


def audit_early_choice(ep: dict, source: str, branch: str) -> dict:
    summary = ep["memory_return"]
    assert summary["predeparture_future_hold_probe_requested"]
    assert summary["predeparture_future_hold_select_requested"] == (
        branch == "early_route_hold")
    assert summary["best_effort_if_unsupported_requested"] == (
        branch == "first_reachable")
    assert not summary["evaluator_abort_on_margin_breach_requested"]
    scans = [row for row in ep["trace"]
             if row["stage"] == "predeparture_range_scan"]
    probes = [row for row in scans if "actor_early_future_hold_probe" in row]
    assert len(probes) <= 1
    if not probes:
        assert summary["predeparture_escape_goal_xy_m"] is None
        return {"early_decision_time_s": None,
                "known_free_goals": 0, "supported_goals": 0,
                "early_selected_goal_xy_m": None}
    row = probes[0]
    alternatives = row["actor_known_free_alternatives"]
    selected_tracks = [track for track in row["actor_multi_tracks"]
                       if track["id"] in row[
                           "actor_early_future_hold_probe"]["actor_track_ids"]]
    assert selected_tracks
    assert len(alternatives) == row[
        "actor_early_future_hold_probe"]["known_free_candidate_count"]
    assert row["actor_early_future_hold_probe"]["probe_only"] == (
        branch == "first_reachable")
    for candidate in alternatives:
        assert candidate["early_future_hold_by_track"]
        assert candidate["path_cells"] >= 2
        assert candidate["distance_m"] in (.30, .45, .60)
        outcomes = []
        for stored in candidate["early_future_hold_by_track"]:
            track = next(t for t in selected_tracks if t["id"] == stored[
                "track_id"])
            screen = stored["screen"]
            actual, supported = independent_screen(
                screen, track, row["time_s"], trial.POST_STOP_UNTIL_S,
                .05 if source == "range" else .12)
            assert abs(actual - screen["worst_modeled_gap_m"]) < .0002
            assert supported == screen["supported_under_models"]
            outcomes.append(supported)
        assert candidate["early_supported_under_models"] == all(outcomes)
    supported = [candidate for candidate in alternatives if candidate[
        "early_supported_under_models"]]
    assert len(supported) == row[
        "actor_early_future_hold_probe"]["supported_candidate_count"]
    if branch == "early_route_hold":
        if supported:
            best = max(supported, key=lambda candidate: (
                min(item["screen"]["worst_modeled_gap_m"] for item in
                    candidate["early_future_hold_by_track"]),
                -candidate["early_future_hold_by_track"][0][
                    "screen"]["path_length_m"]))
            assert row["actor_early_selected_goal_xy_m"] == best["goal_xy_m"]
            assert row["actor_early_selection_status"] == "modeled_supported"
            assert summary["predeparture_escape_goal_xy_m"] == best[
                "goal_xy_m"]
        else:
            assert row["actor_early_selected_goal_xy_m"] is None
            assert row["actor_early_selection_status"] == "no_supported_pose"
            assert summary["predeparture_escape_goal_xy_m"] is None
            assert any(item["stage"] == "predeparture_owner_help" and
                       item["actor_reason"] == "beam_change_escape"
                       for item in ep["trace"])
    else:
        first = next(item for item in row["actor_escape_candidates"]
                     if item["plan_status"] == "known_goal" and
                     item["has_request"])
        assert summary["predeparture_escape_goal_xy_m"] == first["goal_xy_m"]
    return {"early_decision_time_s": row["time_s"],
            "known_free_goals": len(alternatives),
            "supported_goals": len(supported),
            "early_selected_goal_xy_m": row.get(
                "actor_early_selected_goal_xy_m"),
            "known_help_reason_overwrite": bool(
                branch == "early_route_hold" and not supported)}


def compare_early_pair(first: dict, second: dict) -> dict:
    def packets(ep: dict) -> list[dict]:
        return [row for row in ep["trace"] if row["stage"] ==
                "predeparture_range_scan"]
    a, b = packets(first), packets(second)
    assert len(a) == len(b)
    for left, right in zip(a, b):
        assert left["time_s"] == right["time_s"]
        assert left["range_72_beams_m"] == right["range_72_beams_m"]
        assert left["actor_multi_tracks"] == right["actor_multi_tracks"]
        if "actor_known_free_alternatives" in left:
            assert left["actor_known_free_alternatives"] == right[
                "actor_known_free_alternatives"]
    return {"exact_predeparture_range_packets": len(a),
            "same_candidate_screens": True}


def main() -> None:
    declaration_path = OUT / "predeclared-early-hold-selection.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == owner_audit.sha(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("upper_source_sha256", Path(memory.head_depth_upper_route.__file__)),
                        ("rank_source_sha256", Path(memory.nominal_emergency_rank.__file__)),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
                        ("bearing_source_sha256", bearing.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE),
                        ("joined_source_sha256", Path(joined.__file__)),
                        ("timed_grid_source_sha256", Path(timed_grid.__file__)),
                        ("multi_tracker_source_sha256", Path(multi_tracker.__file__))):
        assert declared[key] == owner_audit.sha(OUT / source.name)
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (seed, direction, source, branch) in zip(
                report["episodes"], trial.PROFILES):
            if direction == "stationary":
                independent_motion = old_motion("stationary")
            else:
                original = old_motion("moving_across_return")
                stage = crossing.STAGE
                end = bearing.ENDS[direction]

                def independent_motion(time_s: float):
                    shifted = time_s - .4
                    if shifted <= crossing.ENTRY_END_S:
                        return original(shifted)
                    portion = min(1., max(0.,
                        (shifted - crossing.ENTRY_END_S) /
                        (crossing.CROSS_END_S - crossing.ENTRY_END_S)))
                    return [stage[i] + portion * (end[i] - stage[i])
                            for i in (0, 1)]

            crossing.prop_motion = lambda _, fn=independent_motion: fn
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{direction}_seed{seed}_{branch}"
            map_result, rounding = prior_audit.audit_map_with_rounding_replay(ep)
            rows.append({"seed": seed, "direction": direction,
                         "source": source, "branch": branch,
                         **owner_audit.audit_episode(ep, direction),
                         **map_result,
                         "map_fresh_count_rounding_replays": rounding,
                         **warning_audit.audit_warnings(ep, 1),
                         **audit_early_choice(ep, source, branch)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 2):
        seed, direction, source, _ = trial.PROFILES[index]
        pairs[f"seed{seed}_{direction}_{source}"] = compare_early_pair(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-early-hold-selection-audit.v1",
        "declaration_sha256": owner_audit.sha(declaration_path),
        "report_sha256": owner_audit.sha(report_path),
        "scope": "Actor-only early route-plus-hold screen, full owner media/geometry and exact predeparture packet pairs; no evaluator-truth action abort",
        "videos_decoded": len(rows),
        "predeparture_pair_comparison": pairs,
        "movement_samples_20ms": sum(row["movement_samples_20ms"]
                                     for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"]
                                 for row in rows),
        "all_integrity_checks_pass": True,
        "known_behavior_defects": [
            "V671 no-supported early branch logged beam_change_escape as its help reason because the legacy reason assignment overwrote the new no-supported reason; corrected in the live callback after this frozen run."],
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
