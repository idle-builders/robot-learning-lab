"""Independent full-owner and 20 ms geometry audit of V680 meeting choices."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_memory_head_upper_latency_v669 as map_audit
import audit_robocasa_memory_moving_return_v604 as memory_audit
import audit_robocasa_person_standoff_v508 as meeting_audit
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_meeting_standoff_v680 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT


def route_core(ep: dict) -> list[dict]:
    return [{key: value for key, value in row.items() if key != "rgb_sha256"}
            for row in ep["route_episode"]["trace"]]


def audit_meeting(ep: dict, branch: str) -> dict:
    first_watch = next(row for row in ep["trace"] if row["stage"] ==
                       "predeparture_guard")
    pre_memory = {**ep,
                  "final_actual_xy_m": first_watch["evaluator_actual_xy_m"],
                  "hold_until_s": round(first_watch["time_s"] - .02, 5)}
    physical = meeting_audit.audit_episode(pre_memory)
    assert physical["scene_fingerprint_pass"]
    assert physical["video_sha_pass"] and physical["video_frames_pass"]
    assert physical["person_room_5cm_pass"]
    assert physical["can_room_nonoverlap_pass"]
    choice = next((row for row in ep["trace"] if row["stage"] ==
                   "choose_owner_standoff"), None)
    backtrack = [row for row in ep["trace"] if row["stage"] ==
                 "standoff_backtrack"]
    if branch == "standoff_meeting":
        assert choice is not None and choice["chosen"] is not None
        selected = choice["chosen"]
        assert abs(selected["distance_m"] - .3) < 1.e-6
        assert selected["known_free_path"]
        assert selected["reserve_m"] >= .25
        assert all(cell["free"] and not cell["occupied"]
                   for cell in selected["cells"])
        assert len(backtrack) == 60
        assert all(row["skill_status"] == "kinematic_surrogate_unqualified"
                   for row in backtrack)
        assert np.linalg.norm(np.asarray(backtrack[-1]["ape_xy_m"]) -
                              np.asarray(first_watch[
                                  "evaluator_actual_xy_m"])) < 1.e-5
    else:
        assert choice is None and not backtrack
    return {"selected_standoff_m": None if choice is None else
            choice["chosen"]["distance_m"],
            "backtrack_20ms_samples": len(backtrack),
            "pre_memory_geometry_samples": physical["geometry_samples"],
            "pre_memory_minimum_ape_gap_m_capped_at_10cm": physical[
                "nearest"]["ape_any"]["gap_m_capped_at_10cm"],
            "pre_memory_minimum_person_gap_m_capped_at_10cm": physical[
                "nearest"]["ape_person"]["gap_m_capped_at_10cm"],
            "pre_memory_5cm_pass": physical["ape_5cm_margin_pass"]}


def main() -> None:
    declaration_path = OUT / "predeclared-meeting-standoff.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    assert report["declaration_sha256"] == memory.digest(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 24
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("standoff_source_sha256", Path(trial.standoff.__file__)),
                        ("bearing_source_sha256", bearing.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE)):
        assert declaration[key] == memory.digest(OUT / source.name)
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    old_meeting_motion = meeting_audit.motion
    rows = []
    pairs = {}
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for index, (ep, (seed, direction, source, branch)) in enumerate(
                zip(report["episodes"], trial.PROFILES)):
            crossing.prop_motion = old_motion
            script = (old_motion("stationary") if direction == "stationary"
                      else bearing.challenge_motion(direction))
            crossing.prop_motion = lambda _, script=script: script
            meeting_audit.motion = lambda _, script=script: script
            summary = ep["memory_return"]
            assert summary is not None
            assert summary["source"] == source
            assert summary["run_label"] == f"{direction}_seed{seed}_{branch}"
            assert summary["predeparture_future_hold_select_requested"]
            assert summary["predeparture_future_hold_revalidate_requested"]
            assert summary["predeparture_future_hold_replan_probe_requested"]
            assert not summary["evaluator_abort_on_margin_breach_requested"]
            assert ep["outcome"] == "owner_confirmed_metric_find"
            first_watch = next(row["time_s"] for row in ep["trace"]
                               if row["stage"] == "predeparture_range_scan")
            assert abs(first_watch - 15.) < .001
            meeting = audit_meeting(ep, branch)
            map_result, rounding = map_audit.audit_map_with_rounding_replay(ep)
            memory_result = memory_audit.audit_episode(ep, direction)
            rows.append({"seed": seed, "direction": direction,
                         "source": source, "branch": branch,
                         **meeting, **map_result,
                         "map_fresh_count_rounding_replays": rounding,
                         **memory_result})
            if branch == "standoff_meeting":
                previous = report["episodes"][index - 1]
                assert previous["scene_fingerprint"] == ep["scene_fingerprint"]
                assert route_core(previous) == route_core(ep)
                pairs[f"seed{seed}_{direction}_{source}"] = {
                    "same_pre_owner_route_except_render_hash": True,
                    "same_absolute_predeparture_scan_time_s": first_watch,
                    "current_return_pose_xy_m": previous[
                        "memory_return"]["return_view_xy_m"],
                    "standoff_return_pose_xy_m": summary["return_view_xy_m"],
                    "current_gap_m": previous["memory_return"][
                        "minimum_proxy_gap_m_evaluator_only"],
                    "standoff_gap_m": summary[
                        "minimum_proxy_gap_m_evaluator_only"]}
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
        meeting_audit.motion = old_meeting_motion
    result = {
        "schema": "robot-learning-lab.robocasa-memory-meeting-standoff-audit.v1",
        "declaration_sha256": memory.digest(declaration_path),
        "report_sha256": memory.digest(report_path),
        "scope": "Independent owner reach/standoff and memory-route 20ms geometry, all full videos, exact pre-owner route state except occasional render hash; no truth-backed actor stop",
        "videos_decoded": len(rows),
        "paired_meeting_comparison": pairs,
        "pre_memory_geometry_samples": sum(row["pre_memory_geometry_samples"]
                                           for row in rows),
        "memory_movement_samples_20ms": sum(row["movement_samples_20ms"]
                                            for row in rows),
        "memory_held_samples_20ms": sum(row["post_stop_samples_20ms"]
                                        for row in rows),
        "all_integrity_checks_pass": True,
        "rows": rows,
    }
    (OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
