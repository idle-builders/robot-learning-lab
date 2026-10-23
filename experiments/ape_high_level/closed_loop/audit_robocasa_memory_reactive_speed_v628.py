"""Replay V627 videos and geometry; compare frozen action with matched no action."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_reactive_hold_v626 as response_audit
import run_robocasa_memory_reactive_speed_v627 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_episode(ep: dict, speed: float, reaction: bool, prior) -> dict:
    summary = ep["memory_return"]
    assert summary["run_label"] == f"speed_{speed:.1f}_{'response' if reaction else 'no_action'}"
    assert summary["held_reactive_escape_requested"] == reaction
    baseline_motion = crossing.prop_motion("moving_across_return")

    def independent_motion(time_s: float):
        old_time = time_s - .4
        if old_time > crossing.ENTRY_END_S:
            old_time = crossing.ENTRY_END_S + speed * (
                old_time - crossing.ENTRY_END_S)
        return baseline_motion(old_time)

    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    try:
        crossing.prop_motion = lambda _: independent_motion
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        geometry = base_audit.audit_episode(ep, "late")
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    monitoring = response_audit.audit_monitor(ep, prior, .1)
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    assert len(decisions) == summary["held_reactive_escape_count"]
    if not reaction:
        assert not decisions
    for decision in decisions:
        scan = next(row for row in ep["trace"]
                    if row["stage"] == "memory_post_stop_range_scan" and
                    row["time_s"] == decision["time_s"])
        track = decision["actor_track"]
        assert track in scan["actor_multi_tracks"]
        assert track["status"] == "confirmed_now"
        assert track["velocity_xy_m_s"][0] < -.3
        assert track["radius_m"] >= .18
        assert abs(response_audit.projected_gap(scan, track) -
                   decision["actor_projected_gap_m"]) < .00002
        assert decision["actor_projected_gap_m"] < .05
        assert decision["actor_selected_goal_xy_m"] is not None
        assert any(row["stage"] == "memory_route_decision" and
                   row["phase"] == "evade_reactive" for row in ep["trace"])
        assert any(row["stage"] == "memory_held_reaction_result" for row in ep["trace"])
    assert summary["item_status"] == "not_rechecked"
    assert any(row["stage"] == "predeparture_owner_help" for row in ep["trace"])
    return {"speed": speed, "reaction": reaction,
            "reaction_time_s": decisions[0]["time_s"] if decisions else None,
            **geometry, **monitoring}


def compare_pair(no_action: dict, response: dict) -> dict:
    decisions = [row for row in response["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    cutoff = decisions[0]["time_s"] if decisions else trial.POST_STOP_UNTIL_S

    def packets(ep):
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                     "memory_route_decision",
                                     "memory_post_stop_range_scan") and
                row["time_s"] <= cutoff and
                row.get("phase") != "evade_reactive"]

    left, right = packets(no_action), packets(response)
    assert len(left) == len(right)
    unequal = []
    for old, new in zip(left, right):
        assert old["stage"] == new["stage"]
        assert old["time_s"] == new["time_s"]
        if old["range_72_beams_m"] != new["range_72_beams_m"]:
            unequal.append({"stage": old["stage"], "time_s": old["time_s"],
                            "max_beam_difference_m": round(float(np.max(np.abs(
                                np.asarray(old["range_72_beams_m"]) -
                                np.asarray(new["range_72_beams_m"])))), 5)})
    return {"packets_compared": len(left),
            "packets_with_identical_beams": len(left) - len(unequal),
            "first_unequal_packet": unequal[0] if unequal else None,
            "owner_confirmation_time_s": [
                next(row["time_s"] for row in ep["trace"] if
                     row["stage"] == "owner_confirms_taught_item")
                for ep in (no_action, response)]}


def main() -> None:
    declaration_path = OUT / "predeclared-reactive-speed.json"
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
    rows = [audit_episode(ep, speed, reaction, prior)
            for ep, (speed, reaction) in zip(report["episodes"], trial.PROFILES)]
    matched = {str(speed): compare_pair(report["episodes"][index],
                                        report["episodes"][index + 1])
               for index, speed in ((0, .7), (2, 1.5))}
    result = {
        "schema": "robot-learning-lab.robocasa-memory-reactive-speed-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "videos_decoded": len(rows),
        "matched_pre_action_range_packets": matched,
        "moving_samples_20ms": sum(row["movement_samples_20ms"] for row in rows),
        "held_samples_20ms": sum(row["post_stop_samples_20ms"] for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
