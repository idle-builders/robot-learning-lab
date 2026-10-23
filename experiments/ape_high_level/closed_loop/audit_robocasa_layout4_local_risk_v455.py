"""Independent complete-trajectory geometry and retained-video audit for V454."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-local-risk-v454"


def main() -> None:
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    results = []
    for case in report["cases"]:
        result = common_audit.audit_case({**case,
            "package": f"local_risk_{int(case['risk_radius_m'] * 100):02d}cm"})
        results.append(result)
        print(result["package"], result["scenario_id"], result["outcome"],
              "gap", round(result["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "phase", result["nearest_ape_any"]["phase"], flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_local_risk_v454.py")),
              "prior_map_sha_pass": report["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_crossing_v447.json")),
              "all_videos_pass": all(row["video_sha_pass"] and row["video_frames_pass"]
                                 for row in results),
              "all_prop_paths_nonoverlap": all(row["prop_fixture_nonoverlap_pass"]
                                               for row in results),
              "cases": results}
    (OUT / "audit-v455.json").write_text(json.dumps(output, indent=2) + "\n")
    print("all_videos_pass", output["all_videos_pass"],
          "all_prop_paths_nonoverlap", output["all_prop_paths_nonoverlap"], flush=True)


if __name__ == "__main__":
    main()
