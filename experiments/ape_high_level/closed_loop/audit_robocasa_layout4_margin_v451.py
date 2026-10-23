"""Independent full-horizon geometry and video audit of V450 margin sweep."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-margin-v450"


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    cases = []
    for case in report["cases"]:
        audit = common_audit.audit_case(case)
        cases.append({"margin_m": case["map_margin_m"], **audit})
        print(case["map_margin_m"], audit["scenario_id"], audit["outcome"],
              round(audit["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              round(audit["actual_goal_gap_m"], 3), flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_margin_v450.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_crossing_v447.json")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                    for case in cases),
              "all_prop_paths_nonoverlap": all(case["prop_fixture_nonoverlap_pass"]
                                               for case in cases),
              "cases": cases}
    (OUT / "audit-v451.json").write_text(json.dumps(output, indent=2) + "\n")
    print("all_video_pass", output["all_video_pass"],
          "all_prop_paths_nonoverlap", output["all_prop_paths_nonoverlap"], flush=True)


if __name__ == "__main__":
    main()
