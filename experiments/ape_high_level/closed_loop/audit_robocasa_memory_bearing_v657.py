"""Independently replay V656 oblique crossings and response/no-action pairs."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_moving_return_v604 as base_audit
import audit_robocasa_memory_trail_motion_v651 as trail_audit
import audit_robocasa_memory_warning_confirmation_v653 as warning_audit
import dynamic_trail_map
import held_pose_screen
import run_robocasa_memory_bearing_v656 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout4_wait_and_remap_v461 as timed_grid
import multi_novel_range_objects as multi_tracker


OUT = trial.OUT


def compare_predecision(no_action: dict, response: dict) -> dict:
    decisions = [row for row in response["trace"]
                 if row["stage"] == "memory_held_hazard_decision"]
    cutoff = decisions[0]["time_s"] if decisions else trial.POST_STOP_UNTIL_S

    def packets(ep: dict) -> list[dict]:
        return [row for row in ep["trace"]
                if row["stage"] in ("predeparture_range_scan",
                                    "memory_route_decision",
                                    "memory_post_stop_range_scan") and
                row["time_s"] <= cutoff and row.get("phase") != "evade_reactive"]

    first, second = packets(no_action), packets(response)
    exact = sum(a["stage"] == b["stage"] and a["time_s"] == b["time_s"] and
                a["range_72_beams_m"] == b["range_72_beams_m"]
                for a, b in zip(first, second))
    return {"cutoff_s": cutoff, "no_action_packets": len(first),
            "response_packets": len(second), "exact_packets": exact,
            "exact_pair": len(first) == len(second) == exact}


def main() -> None:
    declaration = OUT / "predeclared-bearing.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == base_audit.sha(declaration)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("trail_source_sha256", Path(dynamic_trail_map.__file__)),
                        ("screen_source_sha256", Path(held_pose_screen.__file__)),
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
        for ep, (bearing, source, reaction) in zip(report["episodes"],
                                                    trial.PROFILES):
            original = old_motion("moving_across_return")
            stage = crossing.STAGE
            end = trial.ENDS[bearing]

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
                f"{bearing}_seed{trial.SEED}_"
                f"{'response' if reaction else 'no_action'}")
            assert summary["held_reactive_escape_requested"] == reaction
            rows.append({"bearing": bearing, "source": source,
                         "reaction": reaction,
                         **base_audit.audit_episode(ep, "late"),
                         **trail_audit.audit_map_evidence(ep, True),
                         **warning_audit.audit_warnings(ep, 1)})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 2):
        bearing, source, _ = trial.PROFILES[index]
        pairs[f"{bearing}_{source}"] = compare_predecision(
            report["episodes"][index], report["episodes"][index + 1])
    cross_bearing = {
        "range": compare_predecision(report["episodes"][1],
                                     report["episodes"][5]),
        "rgbd": compare_predecision(report["episodes"][3],
                                    report["episodes"][7]),
    }
    for source, first_index, second_index in (("range", 1, 5),
                                               ("rgbd", 3, 7)):
        first = next(row for row in report["episodes"][first_index]["trace"]
                     if row["stage"] == "memory_held_hazard_decision")
        second = next(row for row in report["episodes"][second_index]["trace"]
                      if row["stage"] == "memory_held_hazard_decision")
        assert cross_bearing[source]["exact_pair"]
        assert first["actor_selected_goal_xy_m"] == (
            second["actor_selected_goal_xy_m"])
        cross_bearing[source]["same_selected_goal"] = True
    result = {
        "schema": "robot-learning-lab.robocasa-memory-bearing-audit.v1",
        "declaration_sha256": base_audit.sha(declaration),
        "report_sha256": base_audit.sha(report_path),
        "scope": "Changed crossing bearings in complete owner task; independently reconstructed prop motion, fixture/body clearance, video frames, range/map warning evidence",
        "videos_decoded": len(rows),
        "predecision_pair_comparison": pairs,
        "cross_bearing_predecision_comparison": cross_bearing,
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
