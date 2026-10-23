"""Audit V643's explicit unsupported-state emergency action in full episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_future_hold_v642 as screen_audit
import run_robocasa_memory_emergency_hold_v643 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker
import held_pose_screen


OUT = trial.OUT


def audit_response(ep: dict, source: str, label: str, mode: str) -> dict:
    summary = ep["memory_return"]
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    assert len(decisions) <= 1
    assert summary["held_reactive_escape_count"] == len(decisions)
    if not decisions:
        return {"decision_time_s": None, "screened_known_goals": 0,
                "supported_goals": 0, "emergency_route": False,
                "support_status": None}
    decision = decisions[0]
    track = decision["actor_track"]
    scan = next(row for row in ep["trace"]
                if row["stage"] == "memory_post_stop_range_scan" and
                row["time_s"] == decision["time_s"])
    assert track in scan["actor_multi_tracks"]
    assert track["status"] == "confirmed_now"
    assert track["radius_m"] >= .18 and track["velocity_xy_m_s"][0] < -.30
    if mode == "baseline":
        assert not decision["actor_future_hold_screen_requested"]
        assert decision["actor_selected_goal_support_status"] == "current_grid_only"
        return {"decision_time_s": decision["time_s"],
                "screened_known_goals": 0, "supported_goals": 0,
                "emergency_route": False,
                "support_status": "current_grid_only"}
    assert decision["actor_future_hold_screen_requested"]
    assert summary["future_hold_screen_requested"]
    assert summary["best_effort_if_unsupported_requested"]
    assert len(decision["actor_candidate_goals"]) == 24
    checked = []
    for option in decision["actor_candidate_goals"]:
        if "future_hold_screen" in option:
            checked.append((option, screen_audit.independent_screen(
                option, track, decision["time_s"], source)))
    supported = [option for option, result in checked if result["supported"]]
    status = decision["actor_selected_goal_support_status"]
    emergency = decision["actor_emergency_candidates"]
    if supported:
        assert status == "modeled_supported" and not emergency
        selected = max(supported, key=lambda option: (
            option["future_hold_screen"]["worst_modeled_gap_m"],
            -option["future_hold_screen"]["path_length_m"]))
        assert np.linalg.norm(np.asarray(selected["goal_xy_m"]) -
                              decision["actor_selected_goal_xy_m"]) < .00002
    elif emergency:
        first_known = next((option for option in emergency
                            if option["plan_status"] == "known_goal" and
                            option["has_request"]), None)
        if first_known is None:
            assert status == "none"
            assert decision["actor_selected_goal_xy_m"] is None
        else:
            assert status == "unqualified_emergency_known_free"
            assert np.linalg.norm(np.asarray(first_known["goal_xy_m"]) -
                                  decision["actor_selected_goal_xy_m"]) < .00002
            assert summary["item_status"] == "not_rechecked"
    else:
        assert status == "none" and decision["actor_selected_goal_xy_m"] is None
    return {"decision_time_s": decision["time_s"],
            "screened_known_goals": len(checked),
            "supported_goals": len(supported),
            "emergency_route": status == "unqualified_emergency_known_free",
            "support_status": status}


def main() -> None:
    declaration_path = OUT / "predeclared-emergency-hold.json"
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
        for ep, (source, label, seed, mode) in zip(report["episodes"],
                                                   trial.PROFILES):
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{label}_seed{seed}_{mode}_held"
            rows.append({"paired_scan_seed": seed, "mode": mode,
                         **base_audit.audit_episode(ep, label),
                         **audit_response(ep, source, label, mode)})
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    pairs = {}
    for index in (4, 6, 8, 10):
        source, _, seed, _ = trial.PROFILES[index]
        pairs[f"{source}_seed{seed}"] = screen_audit.compare_predecision(
            report["episodes"][index], report["episodes"][index + 1])
    result = {
        "schema": "robot-learning-lab.robocasa-memory-emergency-hold-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Matched current-grid baseline versus explicit unqualified emergency action after finite future screen; provisional body only",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
        "screened_known_goals": sum(row["screened_known_goals"] for row in rows),
        "supported_goals": sum(row["supported_goals"] for row in rows),
        "emergency_routes": sum(row["emergency_route"] for row in rows),
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
