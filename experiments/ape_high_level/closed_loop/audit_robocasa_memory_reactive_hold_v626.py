"""Audit held range packets and prospective warnings in V625 owner episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
from multi_novel_range_objects import MultiNovelRangeObjects
import run_robocasa_memory_reactive_hold_v625 as trial
import run_robocasa_memory_timed_monitor_v623 as baseline
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
    if not held:
        assert summary["route_status"] == "running"
        return {"held_scan_interval_s": interval_s, "held_range_packets": 0,
                "fresh_dynamic_surfaces_independently_matched": 0,
                "first_actor_prediction_warning_s": None,
                "first_evaluator_5cm_breach_s": None,
                "warning_lead_before_breach_s": None,
                "actor_prediction_warnings": []}
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


def audit_reaction(ep: dict, baseline_ep: dict | None) -> dict:
    summary = ep["memory_return"]
    label = summary["run_label"].split("_")[0]
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    outcomes = [row for row in ep["trace"]
                if row["stage"] == "memory_held_reaction_result"]
    assert summary["held_reactive_escape_requested"]
    assert summary["held_reactive_escape_count"] == len(decisions)
    assert len(decisions) <= 1
    if label == "stationary":
        assert not decisions and not outcomes
        assert summary["item_status"] == "fresh_metric_item_refound"
        return {"matched_pre_action_range_packets": 0,
                "reactive_escapes": 0, "reactive_route_decisions": 0}
    assert baseline_ep is not None
    assert summary["item_status"] == "not_rechecked"
    assert any(row["stage"] == "predeparture_owner_help" for row in ep["trace"])
    cutoff = decisions[0]["time_s"] if decisions else trial.POST_STOP_UNTIL_S

    def before(ep_: dict) -> list[dict]:
        return [row for row in ep_["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                    "memory_route_decision",
                                    "memory_post_stop_range_scan") and
                row["time_s"] <= cutoff and row.get("phase") != "evade_reactive"]

    new_packets, old_packets = before(ep), before(baseline_ep)
    assert len(new_packets) == len(old_packets)
    for new, old in zip(new_packets, old_packets):
        assert new["stage"] == old["stage"] and new["time_s"] == old["time_s"]
        assert new["range_72_beams_m"] == old["range_72_beams_m"]
    reactive_moves = [row for row in ep["trace"]
                      if row["stage"] == "memory_route_decision" and
                      row["phase"] == "evade_reactive"]
    if label == "early":
        assert not decisions and not reactive_moves and not outcomes
    else:
        assert label == "late" and len(decisions) == len(outcomes) == 1
        decision = decisions[0]
        assert decision["time_s"] == 18.3
        scan = next(row for row in ep["trace"]
                    if row["stage"] == "memory_post_stop_range_scan" and
                    row["time_s"] == decision["time_s"])
        track = decision["actor_track"]
        assert track in scan["actor_multi_tracks"]
        assert track["status"] == "confirmed_now"
        assert track["radius_m"] >= .18 and track["velocity_xy_m_s"][0] < -.30
        assert abs(projected_gap(scan, track) -
                   decision["actor_projected_gap_m"]) < .00002
        assert decision["actor_projected_gap_m"] < .05
        velocity = np.asarray(track["velocity_xy_m_s"])
        center = np.asarray(track["center_xy_m"])
        pose = np.asarray(scan["actor_estimated_xy_m"])
        perpendicular = np.array([-velocity[1], velocity[0]])
        if np.dot(perpendicular, pose - center) < 0:
            perpendicular *= -1
        perpendicular /= np.linalg.norm(perpendicular)
        alternatives = decision["actor_candidate_goals"]
        for option, distance in zip(alternatives, (.35, .45, .25)):
            assert np.linalg.norm(np.asarray(option["goal_xy_m"]) -
                                  (pose + distance * perpendicular)) < .00002
        selected = next((option for option in alternatives
                         if option["plan_status"] == "known_goal" and
                         option["has_request"]), None)
        assert selected is not None
        assert np.linalg.norm(np.asarray(selected["goal_xy_m"]) -
                              decision["actor_selected_goal_xy_m"]) < .00002
        assert summary["held_reactive_escape_goal_xy_m"] == selected["goal_xy_m"]
        assert reactive_moves and outcomes[0]["actor_route_status"] == (
            "evaded_wait_for_owner_clearance")
        assert summary["minimum_proxy_gap_m_evaluator_only"] >= .05
    return {"matched_pre_action_range_packets": len(new_packets),
            "reactive_escapes": len(decisions),
            "reactive_route_decisions": len(reactive_moves)}


def main() -> None:
    declaration_path = OUT / "predeclared-reactive-hold.json"
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
    baseline_episodes = json.loads((baseline.OUT / "report.json").read_text())["episodes"]
    prior = np.load(joined.PRIOR)
    rows = []
    original_motion = crossing.prop_motion
    original_horizon = crossing.POST_STOP_UNTIL_S
    try:
        crossing.prop_motion = lambda label: (
            original_motion("stationary") if label == "stationary" else
            lambda time_s: original_motion("moving_across_return")(
                time_s + trial.SHIFTS_S[label]))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (source, label, interval_s) in zip(report["episodes"], trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{label}_{int(interval_s * 1000)}ms_reactive_hold"
            baseline_ep = next((old for old in baseline_episodes
                                if old["memory_return"]["source"] == source and
                                old["memory_return"]["run_label"] ==
                                f"{label}_100ms_timed"), None)
            rows.append({**base_audit.audit_episode(ep, label),
                         **audit_monitor(ep, prior, interval_s),
                         **audit_reaction(ep, baseline_ep)})
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    result = {"schema": "robot-learning-lab.robocasa-memory-reactive-hold-audit.v1",
              "declaration_sha256": base_audit.sha(declaration_path),
              "report_sha256": base_audit.sha(report_path),
              "scope": "One actor-cued second response in the full owner task; provisional cylinder only",
              "matched_no_action_report_sha256": base_audit.sha(baseline.OUT / "report.json"),
              "warning_rule": "Current compact south-aisle track has west velocity <-0.3 m/s and constant-velocity 1.2s projected 17cm proxy gap including 3cm base plus 2cm/s uncertainty <5cm",
              "videos_decoded": len(rows),
              "held_range_packets": sum(row["held_range_packets"] for row in rows),
              "moving_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
              "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
              "matched_pre_action_range_packets": sum(
                  row["matched_pre_action_range_packets"] for row in rows),
              "reactive_escapes": sum(row["reactive_escapes"] for row in rows),
              "all_integrity_checks_pass": True,
              "rows": rows}
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
