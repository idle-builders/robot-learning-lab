"""Audit the V683 actor-track trigger against frozen V680 standoff episodes."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_early_hold_select_v672 as screen_audit
import audit_robocasa_memory_moving_return_v604 as owner_audit
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory
import run_robocasa_memory_track_hold_standoff_v683 as trial


OUT = trial.OUT
BASELINE_AUDIT = trial.ROOT / "artifacts/ape-robocasa-memory-meeting-standoff-v680/audit.json"


def scans_to_15_4(ep: dict) -> list[dict]:
    return [{"time_s": row["time_s"],
             "range_72_beams_m": row["range_72_beams_m"],
             "actor_multi_tracks": row["actor_multi_tracks"]}
            for row in ep["trace"] if row["stage"] ==
            "predeparture_range_scan" and row["time_s"] <= 15.4]


def audit_actor_screen(ep: dict, direction: str, source: str) -> dict:
    decision = next((row for row in ep["trace"] if
                     "actor_confirmed_track_trigger" in row), None)
    if direction == "stationary":
        assert decision is None
        assert ep["memory_return"]["item_status"] == "fresh_metric_item_refound"
        return {"track_trigger_time_s": None,
                "known_free_goals": 0, "supported_goals": 0,
                "first_lost_support_s": None}
    assert decision is not None and decision["time_s"] == 15.4
    probe = decision["actor_early_future_hold_probe"]
    assert probe["trigger"] == "confirmed_motion_track"
    tracks = [track for track in decision["actor_multi_tracks"] if
              track["id"] in probe["actor_track_ids"]]
    assert len(tracks) == 1
    alternatives = decision["actor_known_free_alternatives"]
    assert len(alternatives) == probe["known_free_candidate_count"]
    for candidate in alternatives:
        assert candidate["path_cells"] >= 2
        results = []
        for item in candidate["early_future_hold_by_track"]:
            track = next(track for track in tracks if track["id"] ==
                         item["track_id"])
            gap, supported = screen_audit.independent_screen(
                item["screen"], track, decision["time_s"],
                trial.POST_STOP_UNTIL_S,
                .05 if source == "range" else .12)
            assert abs(gap - item["screen"]["worst_modeled_gap_m"]) < .0002
            assert supported == item["screen"]["supported_under_models"]
            results.append(supported)
        assert candidate["early_supported_under_models"] == all(results)
    supported_goals = [candidate for candidate in alternatives if candidate[
        "early_supported_under_models"]]
    assert len(supported_goals) == probe["supported_candidate_count"]
    best = max(supported_goals, key=lambda candidate: (
        min(item["screen"]["worst_modeled_gap_m"] for item in candidate[
            "early_future_hold_by_track"]),
        -candidate["early_future_hold_by_track"][0]["screen"][
            "path_length_m"]))
    assert decision["actor_early_selected_goal_xy_m"] == best["goal_xy_m"]
    revalidations = [row for row in ep["trace"] if row["stage"] ==
                     "memory_early_hold_revalidation"]
    assert revalidations
    first_lost = None
    for row in revalidations:
        snapshot = next(snapshot for snapshot in reversed(ep["multi_history"])
                        if abs(snapshot["capture_time_s"] - row["time_s"]) < .001)
        results = []
        for item in row["actor_screens"]:
            track = next(track for track in snapshot["tracks"] if track["id"] ==
                         item["track_id"] and track["status"] ==
                         "confirmed_now" and abs(track["last_seen_s"] -
                                                  row["time_s"]) < .001)
            gap, supported = screen_audit.independent_screen(
                item["screen"], track, row["time_s"],
                trial.POST_STOP_UNTIL_S,
                .05 if source == "range" else .12)
            assert abs(gap - item["screen"]["worst_modeled_gap_m"]) < .0002
            assert supported == item["screen"]["supported_under_models"]
            results.append(supported)
        supported_now = len(results) == len(row["actor_original_track_ids"]) and all(results)
        assert row["actor_supported_now"] == supported_now
        if not supported_now and first_lost is None:
            first_lost = row
    assert first_lost is not None
    assert any(row["stage"] == "predeparture_owner_help" and
               row.get("actor_route_status") == "early_hold_model_invalidated"
               for row in ep["trace"])
    assert ep["memory_return"]["item_status"] == "not_rechecked"
    return {"track_trigger_time_s": decision["time_s"],
            "known_free_goals": len(alternatives),
            "supported_goals": len(supported_goals),
            "first_lost_support_s": first_lost["time_s"]}


def main() -> None:
    declaration_path = OUT / "predeclared-track-hold-standoff.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    baseline = json.loads(trial.BASELINE.read_text())
    baseline_audit = json.loads(BASELINE_AUDIT.read_text())
    assert baseline_audit["all_integrity_checks_pass"]
    assert baseline_audit["report_sha256"] == memory.digest(trial.BASELINE)
    assert declaration["frozen_baseline_report_sha256"] == memory.digest(
        trial.BASELINE)
    assert report["declaration_sha256"] == memory.digest(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 12
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("standoff_source_sha256", Path(trial.standoff.__file__)),
                        ("track_hold_source_sha256", Path(trial.predeparture_track_hold.__file__)),
                        ("bearing_source_sha256", bearing.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE)):
        assert declaration[key] == memory.digest(OUT / source.name)
    prior_by_key = {(
        ep["memory_return"]["run_label"], ep["memory_return"]["source"]): ep
        for ep in baseline["episodes"]}
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (seed, direction, source, branch) in zip(
                report["episodes"], trial.PROFILES):
            assert branch == "track_trigger_standoff"
            crossing.prop_motion = old_motion
            script = (old_motion("stationary") if direction == "stationary"
                      else bearing.challenge_motion(direction))
            crossing.prop_motion = lambda _, script=script: script
            summary = ep["memory_return"]
            assert summary is not None and summary["source"] == source
            assert summary["run_label"] == f"{direction}_seed{seed}_{branch}"
            assert summary["predeparture_confirmed_track_hold_select_requested"]
            assert not summary["evaluator_abort_on_margin_breach_requested"]
            old = prior_by_key[(f"{direction}_seed{seed}_standoff_meeting", source)]
            assert ep["scene_fingerprint"] == old["scene_fingerprint"]
            assert [row for row in ep["trace"] if row["stage"] ==
                    "standoff_backtrack"] == [row for row in old["trace"] if
                                                row["stage"] == "standoff_backtrack"]
            assert scans_to_15_4(ep) == scans_to_15_4(old)
            assert max(row["time_s"] for row in ep["trace"] if row[
                "stage"] == "fresh_owner_view") < 14.
            actor_result = audit_actor_screen(ep, direction, source)
            geometry = owner_audit.audit_episode(ep, direction)
            rows.append({"seed": seed, "direction": direction,
                         "source": source, "branch": branch,
                         "exact_pretrigger_range_packets": len(scans_to_15_4(ep)),
                         "baseline_gap_m": old["memory_return"][
                             "minimum_proxy_gap_m_evaluator_only"],
                         **actor_result, **geometry})
    finally:
        crossing.prop_motion, crossing.POST_STOP_UNTIL_S = old_motion, old_horizon
    result = {
        "schema": "robot-learning-lab.robocasa-memory-track-hold-standoff-audit.v1",
        "declaration_sha256": memory.digest(declaration_path),
        "report_sha256": memory.digest(report_path),
        "baseline_audit_sha256": memory.digest(BASELINE_AUDIT),
        "scope": "Exact V680 pretrigger range and standoff-motion traces, independent early/route-revalidation arithmetic, full videos and 20ms route/held geometry; pre-show geometry inherited from audited identical V680 backtrack",
        "videos_decoded": len(rows),
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
