"""Independent actor-screen and full-owner replay of the V677 replan probe."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_memory_early_hold_select_v672 as screen_audit
import audit_robocasa_memory_head_upper_latency_v669 as map_audit
import audit_robocasa_memory_moving_return_v604 as owner_audit
import run_robocasa_memory_bearing_v656 as bearing
import run_robocasa_memory_early_replan_probe_v677 as trial
import run_robocasa_memory_moving_return_v603 as crossing
import run_robocasa_memory_return_v599 as memory


OUT = trial.OUT


def main() -> None:
    declaration_path = OUT / "predeclared-early-replan-probe.json"
    report_path = OUT / "report.json"
    declaration = json.loads(declaration_path.read_text())
    report = json.loads(report_path.read_text())
    prior = json.loads((trial.ROOT / "artifacts/ape-robocasa-memory-early-hold-revalidate-v675/report.json").read_text())
    assert report["declaration_sha256"] == owner_audit.sha(declaration_path)
    assert len(report["episodes"]) == len(trial.PROFILES) == 4
    for key, source in (("source_sha256", trial.SOURCE),
                        ("memory_source_sha256", memory.SOURCE),
                        ("bearing_source_sha256", bearing.SOURCE),
                        ("motion_source_sha256", crossing.SOURCE)):
        assert declaration[key] == owner_audit.sha(OUT / source.name)
    previous_by_label = {ep["memory_return"]["run_label"]: ep
                         for ep in prior["episodes"]}
    old_motion, old_horizon = crossing.prop_motion, crossing.POST_STOP_UNTIL_S
    rows = []
    try:
        crossing.POST_STOP_UNTIL_S = trial.POST_STOP_UNTIL_S
        for ep, (seed, direction, source, branch) in zip(
                report["episodes"], trial.PROFILES):
            assert source == "range" and branch == "replan_probe"
            original = old_motion("moving_across_return")
            stage = crossing.STAGE
            end = bearing.ENDS[direction]

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
            assert summary["run_label"] == f"{direction}_seed{seed}_replan_probe"
            matched = previous_by_label[f"{direction}_seed{seed}_revalidate_route_hold"]
            def packet_prefix(item: dict) -> list[dict]:
                return [{"time_s": row["time_s"],
                         "scan": row["range_72_beams_m"],
                         "tracks": row["actor_multi_tracks"]}
                        for row in item["trace"] if row["stage"] ==
                        "predeparture_range_scan"]
            assert packet_prefix(ep) == packet_prefix(matched)
            early = next(row for row in ep["trace"] if
                         "actor_early_future_hold_probe" in row)
            matched_early = next(row for row in matched["trace"] if
                                 "actor_early_future_hold_probe" in row)
            assert early["actor_known_free_alternatives"] == matched_early[
                "actor_known_free_alternatives"]
            assert early["actor_early_selected_goal_xy_m"] == matched_early[
                "actor_early_selected_goal_xy_m"]
            probes = [row for row in ep["trace"] if row["stage"] ==
                      "memory_early_hold_replan_probe"]
            assert len(probes) == 1
            probe = probes[0]
            assert probe["time_s"] == 16.2 and probe["probe_only"]
            alternatives = probe["actor_known_free_alternatives"]
            assert len(alternatives) == probe["known_free_candidate_count"]
            independently_supported = 0
            for candidate in alternatives:
                assert candidate["path_cells"] >= 2
                results = []
                for stored in candidate["screens"]:
                    track = next(track for track in probe["actor_fresh_tracks"]
                                 if track["id"] == stored["track_id"])
                    gap, supported = screen_audit.independent_screen(
                        stored["screen"], track, probe["time_s"],
                        trial.POST_STOP_UNTIL_S, .05)
                    assert abs(gap - stored["screen"]["worst_modeled_gap_m"]) < .0002
                    assert supported == stored["screen"]["supported_under_models"]
                    results.append(supported)
                independently_supported += bool(results) and all(results)
                assert candidate["supported_under_models"] == (
                    bool(results) and all(results))
            assert independently_supported == probe["supported_candidate_count"]
            map_result, rounding = map_audit.audit_map_with_rounding_replay(ep)
            rows.append({"seed": seed, "direction": direction,
                         "exact_predeparture_packets": len(packet_prefix(ep)),
                         "invalidation_time_s": probe["time_s"],
                         "known_free_candidates": len(alternatives),
                         "supported_candidates": independently_supported,
                         **owner_audit.audit_episode(ep, direction),
                         **map_result,
                         "map_fresh_count_rounding_replays": rounding})
    finally:
        crossing.prop_motion, crossing.POST_STOP_UNTIL_S = old_motion, old_horizon
    result = {
        "schema": "robot-learning-lab.robocasa-memory-early-replan-probe-audit.v1",
        "declaration_sha256": owner_audit.sha(declaration_path),
        "report_sha256": owner_audit.sha(report_path),
        "scope": "Fresh actor-only low-plane alternative screen, exact V675 predeparture packet match, full owner videos and independent 20ms geometry replay",
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
