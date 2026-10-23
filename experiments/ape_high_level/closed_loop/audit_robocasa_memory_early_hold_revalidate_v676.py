"""Independent media, geometry and actor-only revalidation audit for V675."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_early_hold_select_v672 as earlier
import audit_robocasa_memory_head_upper_latency_v669 as map_audit
import audit_robocasa_memory_moving_return_v604 as owner_audit
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_early_hold_revalidate_v675 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT


def audit_revalidation(ep: dict, branch: str) -> dict:
    summary = ep["memory_return"]
    assert summary["predeparture_future_hold_select_requested"]
    assert summary["predeparture_future_hold_revalidate_requested"] == (
        branch == "revalidate_route_hold")
    original = next((row for row in ep["trace"] if
                     "actor_early_future_hold_probe" in row), None)
    rows = [row for row in ep["trace"] if row["stage"] ==
            "memory_early_hold_revalidation"]
    if original is None or original["actor_early_selected_goal_xy_m"] is None:
        assert not rows
        return {"revalidations": 0, "first_lost_support_s": None,
                "first_lost_support_gap_m": None}
    assert branch == "revalidate_route_hold" or not rows
    if branch == "revalidate_route_hold":
        assert rows
    first_lost = None
    for row in rows:
        assert row["actor_goal_xy_m"] == original[
            "actor_early_selected_goal_xy_m"]
        tracks = next(snapshot["tracks"] for snapshot in reversed(ep["multi_history"])
                      if abs(snapshot["capture_time_s"] - row["time_s"]) < .001
                      and any(track["id"] in row["actor_original_track_ids"]
                              for track in snapshot["tracks"]))
        checked = []
        for stored in row["actor_screens"]:
            track = next(track for track in tracks if track["id"] ==
                         stored["track_id"] and track["status"] ==
                         "confirmed_now" and abs(track["last_seen_s"] -
                                                  row["time_s"]) < .001)
            gap, supported = earlier.independent_screen(
                stored["screen"], track, row["time_s"],
                trial.POST_STOP_UNTIL_S, .05)
            assert abs(gap - stored["screen"]["worst_modeled_gap_m"]) < .0002
            assert supported == stored["screen"]["supported_under_models"]
            checked.append(supported)
        assert len(checked) == row["actor_fresh_track_count"]
        independently_supported = (
            len(checked) == len(row["actor_original_track_ids"]) and all(checked))
        assert independently_supported == row["actor_supported_now"]
        if not independently_supported and first_lost is None:
            first_lost = row
    if first_lost is not None:
        assert summary["item_status"] == "not_rechecked"
        assert any(row["stage"] == "predeparture_owner_help" and
                   row.get("actor_route_status") ==
                   "early_hold_model_invalidated" for row in ep["trace"])
        assert not any(row["stage"] == "memory_route_decision" and
                       row["time_s"] > first_lost["time_s"] and
                       row["phase"] == "evade" and
                       row.get("actor_bounded_command_xy_m") is not None
                       for row in ep["trace"])
    return {"revalidations": len(rows),
            "first_lost_support_s": (None if first_lost is None else
                                     first_lost["time_s"]),
            "first_lost_support_gap_m": (None if first_lost is None else
                min((item["screen"]["worst_modeled_gap_m"]
                     for item in first_lost["actor_screens"]), default=None))}


def main() -> None:
    declaration_path = OUT / "predeclared-early-hold-revalidation.json"
    report_path = OUT / "report.json"
    declared = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == owner_audit.sha(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES)
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("screen_source_sha256", Path(memory.held_pose_screen.__file__)),
                        ("bearing_source_sha256", bearing.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE)):
        assert declared[key] == owner_audit.sha(OUT / source.name)
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (seed, direction, source, branch) in zip(
                report["episodes"], trial.PROFILES):
            assert source == "range"
            if direction == "stationary":
                motion = old_motion("stationary")
            else:
                original = old_motion("moving_across_return")
                stage = crossing.STAGE
                end = bearing.ENDS[direction]

                def motion(time_s: float):
                    shifted = time_s - .4
                    if shifted <= crossing.ENTRY_END_S:
                        return original(shifted)
                    fraction = min(1., max(0.,
                        (shifted - crossing.ENTRY_END_S) /
                        (crossing.CROSS_END_S - crossing.ENTRY_END_S)))
                    return [stage[i] + fraction * (end[i] - stage[i])
                            for i in (0, 1)]

            crossing.prop_motion = lambda _, fn=motion: fn
            summary = ep["memory_return"]
            assert summary["source"] == source
            assert summary["run_label"] == f"{direction}_seed{seed}_{branch}"
            map_result, rounding = map_audit.audit_map_with_rounding_replay(ep)
            rows.append({"seed": seed, "direction": direction,
                         "source": source, "branch": branch,
                         **owner_audit.audit_episode(ep, direction),
                         **map_result,
                         "map_fresh_count_rounding_replays": rounding,
                         **audit_revalidation(ep, branch)})
    finally:
        crossing.prop_motion, crossing.POST_STOP_UNTIL_S = old_motion, old_horizon
    pairs = {}
    for index in range(0, len(trial.PROFILES), 2):
        seed, direction, source, _ = trial.PROFILES[index]
        pairs[f"seed{seed}_{direction}_{source}"] = (
            earlier.compare_early_pair(report["episodes"][index],
                                       report["episodes"][index + 1]))
    result = {
        "schema": "robot-learning-lab.robocasa-memory-early-hold-revalidation-audit.v1",
        "declaration_sha256": owner_audit.sha(declaration_path),
        "report_sha256": owner_audit.sha(report_path),
        "scope": "Independent saved-range revalidation arithmetic, full owner videos, 20ms geometry and exact predeparture sensor pairs; evaluator geometry never controls motion",
        "videos_decoded": len(rows),
        "predeparture_pair_comparison": pairs,
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
