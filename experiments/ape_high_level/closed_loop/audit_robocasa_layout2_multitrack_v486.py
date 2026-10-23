"""Audit multi-object identity, decisions, geometry and videos for V485."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_robocasa_layout2_two_props_v483 import audit_case
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-multitrack-v485"
SOURCE = Path(__file__).with_name("run_robocasa_layout2_multitrack_v485.py")
TRACKER = Path(__file__).with_name("multi_novel_range_objects.py")
CONFIG = Path(__file__).with_name("robocasa_layout2_two_props_v482.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
SCENE = Path(__file__).with_name("run_robocasa_layout2_two_props_v482.py")
OLD = ROOT / "artifacts/ape-robocasa-layout2-two-props-v482/report.json"


def main():
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    old = json.loads(OLD.read_text())
    cases = [audit_case(case) for case in report["cases"]]
    present = report["cases"][1:]
    identity = []
    for case in present:
        at_stage = case["multi_history"][-1]
        confirmed = [track for track in at_stage["tracks"] if track["hits"] >= 2]
        truth = [np.asarray(case["scenario"]["primary_xy_m"]),
                 np.asarray(case["scenario"]["secondary_from_xy_m"])]
        errors = [min(float(np.linalg.norm(np.asarray(track["center_xy_m"]) - expected))
                      for track in confirmed) if confirmed else float("inf")
                  for expected in truth]
        decision = case["multi_decisions"][-1] if case["multi_decisions"] else None
        identity.append({"scenario_id": case["scenario"]["id"],
                         "confirmed_ids": [track["id"] for track in confirmed],
                         "truth_match_error_m": errors,
                         "stage_before_motion_pass": bool(
                             decision and decision["decision"] ==
                             "request_owner_show_here_safe_side" and
                             len(decision["confirmed_track_ids"]) == 2 and
                             case["episode"]["end_time_s"] < case["scenario"]["begin_s"]),
                         "both_tracks_within_10cm_pass": max(errors) <= .10})
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(SOURCE),
              "tracker_source_sha_pass": report["multi_tracker_source_sha256"] == sha256(TRACKER),
              "config_source_sha_pass": report["config_sha256"] == sha256(CONFIG),
              "prior_source_sha_pass": report["prior_map_sha256"] == sha256(PRIOR),
              "scene_source_sha_pass": report["two_prop_scene_source_sha256"] == sha256(SCENE),
              "absent_route_pair_pass": report["cases"][0]["episode"]["trace"] ==
              old["cases"][0]["episode"]["trace"],
              "all_geometry_pass": all(case["ape_5cm_margin_pass"] and
                                       case["both_props_5cm_margin_pass"] and
                                       case["prop_paths_nonoverlap_pass"] for case in cases),
              "all_videos_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                     for case in cases),
              "all_two_track_identity_pass": all(item["both_tracks_within_10cm_pass"]
                                                 for item in identity),
              "all_early_stage_pass": all(item["stage_before_motion_pass"]
                                          for item in identity),
              "cases": cases, "identity": identity}
    (OUT / "audit-v486.json").write_text(json.dumps(output, indent=2) + "\n")
    for case, item in zip(cases[1:], identity):
        print(case["scenario_id"], case["outcome"],
              "second_gap", round(case["ape_secondary_gap_m_capped_at_50cm"], 3),
              "track_errors", [round(x, 3) for x in item["truth_match_error_m"]],
              flush=True)
    print({key: value for key, value in output.items() if key.endswith("_pass")}, flush=True)


if __name__ == "__main__":
    main()
