"""Independent full-horizon audit for away-direction control and V465."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_layout4_wait_and_remap_v462 import (
    decision_contract, evaluator_prop_at,
)
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
AWAY = ROOT / "artifacts/ape-robocasa-layout4-away-motion-v464/report.json"
DIRECTIONAL = ROOT / "artifacts/ape-robocasa-layout4-directional-wait-v465/report.json"
OUT = ROOT / "artifacts/ape-robocasa-layout4-directional-wait-v465"


def main():
    away = json.loads(AWAY.read_text())
    directional = json.loads(DIRECTIONAL.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    common_audit.prop_at = evaluator_prop_at
    rows = []
    for label, report in (("away_control", away), ("directional_wait", directional)):
        for case in report["cases"]:
            result = common_audit.audit_case(case)
            result["variant"] = label
            if label == "directional_wait":
                result["decision_contract"] = decision_contract(case)
            rows.append(result)
            print(label, case["scenario"]["id"], result["outcome"],
                  "room_gap", round(result["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
                  "prop_gap", round(result["nearest_ape_prop"]["signed_gap_m"], 3),
                  "goal_gap", round(result["actual_goal_gap_m"], 2), flush=True)
    config_paths = [Path(__file__).with_name("robocasa_layout4_returning_prop_v460.json"),
                    Path(__file__).with_name("robocasa_layout4_away_motion_v464.json")]
    output = {"source_sha256": sha256(Path(__file__)),
              "away_report_sha256": sha256(AWAY),
              "directional_report_sha256": sha256(DIRECTIONAL),
              "away_source_sha_pass": away["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_away_motion_v464.py")),
              "directional_source_sha_pass": directional["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_directional_wait_v465.py")),
              "config_source_sha_pass": all(
                  directional["config_sha256"][str(path)] == sha256(path)
                  for path in config_paths) and away["config_sha256"] == sha256(config_paths[1]),
              "tracker_source_sha_pass": directional["tracker_source_sha256"] == sha256(
                  Path(__file__).with_name("novel_range_object.py")),
              "prior_map_sha_pass": directional["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "all_video_pass": all(c["video_sha_pass"] and c["video_frames_pass"]
                                    for c in rows),
              "all_prop_paths_nonoverlap": all(c["prop_fixture_nonoverlap_pass"]
                                               for c in rows),
              "all_policy_decisions_pass": all(c["decision_contract"]["pass"]
                                               for c in rows if c["variant"] == "directional_wait"),
              "cases": rows}
    (OUT / "audit-v466.json").write_text(json.dumps(output, indent=2) + "\n")
    print("video", output["all_video_pass"],
          "prop_paths", output["all_prop_paths_nonoverlap"],
          "decisions", output["all_policy_decisions_pass"], flush=True)


if __name__ == "__main__":
    main()
