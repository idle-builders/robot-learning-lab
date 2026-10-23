"""Independent 3-D swept-proxy and video audit for V433 scan-height routes."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import sha256
from audit_robocasa_margin_ablation_v432 import score_route


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-scan-height-v433"


def main() -> None:
    trial_path = OUT / "report.json"
    report = json.loads(trial_path.read_text())
    cases = []
    for case in report["cases"]:
        geometry = score_route(case["episode"])
        prop = case["prop_audit"]
        cases.append({"case": case["case"],
                      "scan_heights_m": case["scan_heights_m"],
                      "outcome": case["episode"]["outcome"],
                      "all_body_clearance_m": geometry["closest"]["minimum_fixture_gap_capped_m"],
                      "closest_geom": geometry["closest"]["nearest_geom"],
                      "closest_time_s": geometry["closest"]["time_s"],
                      "all_body_5cm_margin_pass": geometry["all_body_5cm_margin_pass"],
                      "actual_goal_gap_m": geometry["end_actual_goal_gap_m"],
                      "video_sha_pass": prop["video_sha_pass"],
                      "video_frames_pass": prop["video_frames_pass"],
                      "prop_replay_pass": prop["distance_replay_max_error_m"] < .001})
        print(case["case"], round(geometry["closest"]["minimum_fixture_gap_capped_m"], 3),
              geometry["closest"]["nearest_geom"],
              round(geometry["end_actual_goal_gap_m"], 3), flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(trial_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_scan_height_v433.py")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_prop_checks_pass": all(case["video_sha_pass"] and
                                                case["video_frames_pass"] and
                                                case["prop_replay_pass"] for case in cases),
              "cases": cases}
    (OUT / "audit-v434.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
