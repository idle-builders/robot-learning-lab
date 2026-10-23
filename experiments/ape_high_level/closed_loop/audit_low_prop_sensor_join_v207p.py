"""Independently score V206p low-prop gaps, packets, videos and task claims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_dynamic_suite_v440 as route_audit
from audit_robocasa_dynamic_owner_v472 import audit_owner
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-robocasa-low-prop-sensor-v206p"
REPORT = OUT / "report.json"


def low_evaluator_scene(start_xy):
    model, data, ape_body, ape_geom = layout4_evaluator_scene(start_xy)
    model.body_pos[model.body("route_obstacle").id, 2] = .09
    model.geom("route_obstacle_geom").size[1] = .09
    return model, data, ape_body, ape_geom


def packets(case, expected_beams):
    rows = case["route_episode"]["trace"]
    sources = case["sensor_packets"]
    checks = []
    for index, (row, source) in enumerate(zip(rows, sources)):
        path = ROOT / source["path"]
        with np.load(path, allow_pickle=False) as packet:
            ranges = packet["ranges_m"]
            status = packet["status"]
            sample_index = int(packet["sample_index"])
            decision_time = float(packet["decision_time_s"])
        valid = status == 0
        counts = {"valid_hit": int(np.count_nonzero(valid)),
                  "too_near": int(np.count_nonzero(status == 1)),
                  "no_target": int(np.count_nonzero(status == 2))}
        checks.append({
            "index": index,
            "hash_match": sha(path) == source["sha256"],
            "index_match": sample_index == index,
            "time_match": abs(decision_time - row["time_s"]) <= 1e-5,
            "beam_count_match": len(status) == expected_beams,
            "counts_match": counts == source["status_counts"],
            "actor_returns_match": counts["valid_hit"] == row["range_returns"],
            "invalid_unmeasured": bool(np.all(np.isnan(ranges[~valid]))),
            "valid_finite_and_capped": bool(np.all(
                np.isfinite(ranges[valid]) &
                (ranges[valid] >= .06) & (ranges[valid] <= 5.02))),
        })
    return {
        "count": len(sources),
        "count_match": len(sources) == len(rows),
        "all_checks_pass": all(all(value for key, value in row.items()
                               if key != "index") for row in checks),
        "first_failure": next((row for row in checks
                               if not all(value for key, value in row.items()
                                          if key != "index")), None),
    }


def owner_boundary(case):
    owner = audit_owner(case)
    claimed = case["outcome"] == "owner_confirmed_metric_find"
    if case["route_state"] == "goal_viewpoint_reached":
        expected = claimed and owner["claim_evidence_pass"]
    elif case["route_state"] == "ask_owner_clear_corridor":
        expected = (not claimed and case["outcome"] ==
                    "ask_owner_clear_corridor" and any(
                        row["stage"] == "request_owner_clear_corridor"
                        for row in case["trace"]))
    else:
        expected = not claimed and not any(
            row["stage"] == "ask_owner_to_show" for row in case["trace"])
    owner["actor_outcome_matches_evidence"] = bool(expected)
    return owner


def main():
    report = json.loads(REPORT.read_text())
    original_scene, original_path = route_audit.evaluator_scene, route_audit.prop_at
    route_audit.evaluator_scene = low_evaluator_scene
    route_audit.prop_at = evaluator_prop_at
    entries = []
    try:
        for case in report["episodes"]:
            profile = case["profile"]
            geometry = route_audit.audit_case({
                "scenario": case["scenario"], "package": profile,
                "episode": case["route_episode"]})
            packet_audit = packets(case, 360)
            owner = owner_boundary(case)
            entries.append({
                "scenario": case["scenario"]["id"],
                "profile": profile,
                "actor_route_state": case["route_state"],
                "actor_outcome": case["outcome"],
                "geometry": geometry,
                "packets": packet_audit,
                "owner": owner,
                "tracked_frames": sum(t["status"] == "tracked"
                                      for t in case["track_history"]),
                "hold_actions": sum(d["decision"] ==
                                    "hold_and_reobserve_moving_object"
                                    for d in case["decision_history"]),
            })
    finally:
        route_audit.evaluator_scene = original_scene
        route_audit.prop_at = original_path
    scene_match = all(len({case["scene_fingerprint"]
                           for case in report["episodes"]
                           if case["scenario"]["id"] == scenario}) == 1
                      for scenario in ("stationary", "returning_crossing",
                                       "one_way_crossing"))
    output = {
        "schema": "robot-learning-lab.ape-low-prop-sensor-join-audit.v1",
        "source_sha256": sha(SOURCE),
        "report_sha256": sha(REPORT),
        "experiment_source_match": report["source_sha256"] == sha(
            SOURCE.with_name("run_low_prop_sensor_join_v206p.py")),
        "config_source_match": report["config_sha256"] == sha(
            SOURCE.with_name("robocasa_layout4_returning_prop_v460.json")),
        "prior_source_match": report["prior_sha256"] == sha(
            ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
        "matched_scenes": scene_match,
        "episodes": entries,
    }
    output["evidence_integrity_pass"] = (
        output["experiment_source_match"] and output["config_source_match"]
        and output["prior_source_match"] and scene_match
        and all(e["packets"]["count_match"] and
                e["packets"]["all_checks_pass"] and
                e["geometry"]["video_sha_pass"] and
                e["geometry"]["video_frames_pass"] and
                e["geometry"]["prop_fixture_nonoverlap_pass"] and
                e["owner"]["video_sha_pass"] and
                e["owner"]["video_frame_count_pass"] and
                e["owner"]["actor_outcome_matches_evidence"]
                for e in entries))
    (OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
    for e in entries:
        print(e["scenario"], e["profile"], e["actor_route_state"],
              "gap", round(e["geometry"]["nearest_ape_prop"]["signed_gap_m"], 3),
              "hold", e["hold_actions"], "track", e["tracked_frames"],
              flush=True)
    print("evidence integrity passed", output["evidence_integrity_pass"])
    if not output["evidence_integrity_pass"]:
        raise AssertionError("Low-prop audit integrity failed")


if __name__ == "__main__":
    main()
