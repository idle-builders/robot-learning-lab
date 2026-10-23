"""Audit source-loss packets, owner outcomes and full proxy geometry in V631."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_reactive_hold_v626 as response_audit
import run_robocasa_memory_held_dropout_v631 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def audit_episode(ep: dict, source: str, label: str,
                  dropout: tuple[float, float] | None, prior) -> dict:
    summary = ep["memory_return"]
    assert summary["source"] == source
    assert summary["run_label"] == f"{label}_{'drop' if dropout else 'healthy'}_held"
    assert summary["hold_scan_dropout_window_s_requested"] == (
        None if dropout is None else list(dropout))
    original_motion, original_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    try:
        crossing.prop_motion = lambda condition: (
            original_motion("stationary") if condition == "stationary" else
            lambda time_s: original_motion("moving_across_return")(
                time_s + trial.SHIFTS_S[condition]))
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        geometry = base_audit.audit_episode(ep, label)
    finally:
        crossing.prop_motion = original_motion
        crossing.POST_STOP_UNTIL_S = original_horizon
    monitor = response_audit.audit_monitor(ep, prior, .1)
    missing = [row for row in ep["trace"]
               if row["stage"] == "memory_post_stop_range_missing"]
    healthy = [row for row in ep["trace"]
               if row["stage"] == "memory_post_stop_range_scan"]
    assert len(missing) == summary["hold_missing_scan_count"]
    if dropout is None:
        assert not missing
    else:
        assert missing
        assert all(row["source_health"] == "missing" and
                   dropout[0] <= row["time_s"] < dropout[1] for row in missing)
        assert all(not (dropout[0] <= row["time_s"] < dropout[1])
                   for row in healthy)
        assert all("range_72_beams_m" not in row for row in missing)
        assert all(row["actor_last_track_capture_time_s"] < row["time_s"]
                   for row in missing)
    decisions = [row for row in ep["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    assert len(decisions) == summary["held_reactive_escape_count"]
    assert len(decisions) <= 1
    for decision in decisions:
        scan = next(row for row in healthy
                    if row["time_s"] == decision["time_s"])
        track = decision["actor_track"]
        assert track in scan["actor_multi_tracks"]
        assert track["status"] == "confirmed_now"
        assert track["radius_m"] >= .18
        assert track["velocity_xy_m_s"][0] < -.3
        assert abs(response_audit.projected_gap(scan, track) -
                   decision["actor_projected_gap_m"]) < .00002
        assert decision["actor_projected_gap_m"] < .05
        assert decision["actor_selected_goal_xy_m"] is not None
        assert any(row["stage"] == "memory_route_decision" and
                   row["phase"] == "evade_reactive" for row in ep["trace"])
    if label == "stationary":
        assert summary["item_status"] == "fresh_metric_item_refound"
    else:
        assert summary["item_status"] == "not_rechecked"
        assert any(row["stage"] == "predeparture_owner_help" for row in ep["trace"])
    return {"source": source, "scenario": label,
            "dropout_window_s": dropout,
            "missing_range_packets": len(missing),
            "second_action_time_s": decisions[0]["time_s"] if decisions else None,
            **geometry, **monitor}


def compare_before_loss(healthy: dict, dropped: dict) -> dict:
    cutoff = trial.DROP_WINDOW_S[0]

    def packets(ep):
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                     "memory_route_decision",
                                     "memory_post_stop_range_scan") and
                row["time_s"] < cutoff]

    left, right = packets(healthy), packets(dropped)
    assert len(left) == len(right)
    unequal = []
    for a, b in zip(left, right):
        assert a["stage"] == b["stage"] and a["time_s"] == b["time_s"]
        if a["range_72_beams_m"] != b["range_72_beams_m"]:
            unequal.append(a["time_s"])
    return {"packets_compared": len(left),
            "identical_packets": len(left) - len(unequal),
            "first_unequal_time_s": unequal[0] if unequal else None,
            "owner_confirmation_time_s": [
                next(row["time_s"] for row in ep["trace"]
                     if row["stage"] == "owner_confirms_taught_item")
                for ep in (healthy, dropped)]}


def main() -> None:
    declaration_path = OUT / "predeclared-held-dropout.json"
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
    rows = [audit_episode(ep, source, label, dropout, prior)
            for ep, (source, label, dropout) in zip(
                report["episodes"], trial.PROFILES)]
    pairs = {source: compare_before_loss(report["episodes"][i],
                                         report["episodes"][i + 1])
             for source, i in (("range", 2), ("rgbd", 4))}
    result = {
        "schema": "robot-learning-lab.robocasa-memory-held-dropout-audit.v1",
        "declaration_sha256": base_audit.sha(declaration_path),
        "report_sha256": base_audit.sha(report_path),
        "videos_decoded": len(rows),
        "pre_loss_pair_comparison": pairs,
        "missing_range_packets": sum(row["missing_range_packets"] for row in rows),
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
