"""Independent replay of paired range-ID continuity intervention."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_joint_station_v687 as pilot_audit
import audit_robocasa_memory_moving_return_v604 as memory_audit
import audit_robocasa_person_standoff_v508 as meeting_audit
import run_robocasa_memory_joint_station_continuity_v691 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT
BASELINE = trial.BASELINE
BASELINE_AUDIT = trial.ROOT / "artifacts/ape-robocasa-memory-joint-station-heldout-v689/audit.json"
PREFLIGHT = trial.ROOT / "artifacts/ape-robocasa-meeting-refuge-preflight-v685/report.json"


def range_packets(ep: dict) -> list[tuple[float, list[float | None]]]:
    return [(row["time_s"], row["range_72_beams_m"]) for row in ep["trace"]
            if row["stage"] == "predeparture_range_scan"]


def main() -> None:
    declaration_path = OUT / "predeclared.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    baseline = json.loads(BASELINE.read_text())
    baseline_audit = json.loads(BASELINE_AUDIT.read_text())
    snapshot = json.loads(PREFLIGHT.read_text())["actor_map"]
    assert baseline_audit["all_integrity_checks_pass"]
    assert baseline_audit["report_sha256"] == memory.digest(BASELINE)
    assert declaration["frozen_shifted_baseline_report_sha256"] == memory.digest(BASELINE)
    assert report["declaration_sha256"] == memory.digest(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 10
    for key, source in (("source_sha256", trial.SOURCE),
                        ("selector_source_sha256", Path(trial.joint_meeting_refuge.__file__)),
                        ("standoff_source_sha256", Path(trial.standoff.__file__)),
                        ("memory_source_sha256", memory.SOURCE),
                        ("tracker_source_sha256", Path(trial.multi_tracker.__file__))):
        assert declaration[key] == memory.digest(OUT / source.name)
    old_motion = crossing.prop_motion
    old_horizon = crossing.POST_STOP_UNTIL_S
    old_meeting_motion = meeting_audit.motion
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = 20.8
        for ep, before, (direction, shift_s, source) in zip(
                report["episodes"], baseline["episodes"], trial.PROFILES):
            crossing.prop_motion = old_motion
            base_script = (old_motion("stationary") if direction == "stationary"
                           else trial.bearing.challenge_motion(direction))
            script = lambda t, base_script=base_script, shift_s=shift_s: base_script(t + shift_s)
            crossing.prop_motion = lambda _, script=script: script
            meeting_audit.motion = lambda _, script=script: script
            assert ep["scene_fingerprint"] == before["scene_fingerprint"]
            assert pilot_audit.route_core(ep) == pilot_audit.route_core(before)
            assert ep["outcome"] == before["outcome"] == "owner_confirmed_metric_find"
            assert ep["memory_return"]["source"] == source
            assert ep["memory_return"]["predeparture_track_continuity_gate_requested"]
            assert not ep["memory_return"]["evaluator_abort_on_margin_breach_requested"]
            current_packets, old_packets = range_packets(ep), range_packets(before)
            paired_packets = min(len(current_packets), len(old_packets))
            assert current_packets[:paired_packets] == old_packets[:paired_packets]
            station = pilot_audit.audit_station(ep, snapshot)
            geometry = memory_audit.audit_episode(ep, direction)
            assert geometry["proxy_minimum_gap_m_capped_at_10cm"] >= .05
            events = [row for row in ep["trace"] if row["stage"] ==
                      "predeparture_range_scan" and (
                          row.get("actor_unresolved_new_track_ids") or
                          row.get("actor_confirmed_track_trigger"))]
            if direction == "stationary":
                assert not events
                assert geometry["item_status"] == "fresh_metric_item_refound"
            else:
                assert events and geometry["item_status"] == "not_rechecked"
                assert all(track["id"] != 1 for event in events for track in
                           event["actor_multi_tracks"] if track["status"] ==
                           "confirmed_now")
            rows.append({"direction": direction, "shift_s": shift_s,
                         "source": source,
                         "exact_paired_watch_range_packets": paired_packets,
                         "continuity_event_time_s": None if not events else
                         events[0]["time_s"],
                         "continuity_event": (None if not events else
                         "unresolved_new_object" if events[0].get(
                             "actor_unresolved_new_track_ids") else
                         "confirmed_new_moving_track"),
                         "baseline_proxy_gap_m": before["memory_return"][
                             "minimum_proxy_gap_m_evaluator_only"],
                         **station, **geometry})
    finally:
        crossing.prop_motion = old_motion
        crossing.POST_STOP_UNTIL_S = old_horizon
        meeting_audit.motion = old_meeting_motion
    result = {
        "schema": "robot-learning-lab.robocasa-memory-joint-station-continuity-audit.v1",
        "declaration_sha256": memory.digest(declaration_path),
        "report_sha256": memory.digest(report_path),
        "baseline_audit_sha256": memory.digest(BASELINE_AUDIT),
        "scope": "Paired actor range packets, track split/uncertainty gate, 20ms owner/memory geometry and complete videos; no selected articulated body",
        "videos_decoded": len(rows),
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
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2))


if __name__ == "__main__":
    main()
