"""Independent full-horizon geometry/video audit of the slow-motion challenge."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-slow-motion-v457"


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    cases = [common_audit.audit_case(case) for case in report["cases"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_slow_motion_v457.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_slow_motion_v457.json")),
              "scene_pair_pass": len({c["episode"]["scene_fingerprint"]
                                      for c in report["cases"]}) == 1,
              "all_videos_pass": all(c["video_sha_pass"] and c["video_frames_pass"]
                                     for c in cases),
              "all_prop_paths_nonoverlap": all(c["prop_fixture_nonoverlap_pass"]
                                               for c in cases),
              "cases": cases}
    (OUT / "audit-v458.json").write_text(json.dumps(output, indent=2) + "\n")
    for c in cases:
        print(c["scenario_id"], c["outcome"],
              round(c["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              c["nearest_ape_any"]["phase"],
              "goal", round(c["actual_goal_gap_m"], 2), flush=True)


if __name__ == "__main__":
    main()
