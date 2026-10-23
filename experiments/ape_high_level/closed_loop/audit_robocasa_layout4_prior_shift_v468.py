"""Independent full-horizon/hold/video audit of the shifted-prior challenge."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from audit_robocasa_layout4_wait_and_remap_v462 import evaluator_prop_at
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-prior-shift-v467"


def main():
    source = OUT / "report.json"
    report = json.loads(source.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    common_audit.prop_at = evaluator_prop_at
    cases = []
    for case in report["cases"]:
        result = common_audit.audit_case(case)
        hold_rows = [row for row in case["episode"]["trace"]
                     if row["plan_status"] == "hold_and_reobserve_moving_object"]
        result["holds_zero_requested_translation_pass"] = all(
            np.linalg.norm(np.asarray(row["requested_xy_m"]) -
                           np.asarray(row["estimated_xy_m"])) < 1e-9
            for row in hold_rows)
        result["hold_count"] = len(hold_rows)
        cases.append(result)
        print(case["package"], result["scenario_id"], result["outcome"],
              "hold", len(hold_rows),
              "room_gap", round(result["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_gap", round(result["nearest_ape_prop"]["signed_gap_m"], 3), flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(source),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_prior_shift_v467.py")),
              "config_source_sha_pass": report["config_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_layout4_prior_shift_v467.json")),
              "policy_source_sha_pass": report["policy_source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_layout4_directional_wait_v465.py")),
              "prior_map_sha_pass": report["prior_map_sha256"] == sha256(
                  ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"),
              "all_video_pass": all(c["video_sha_pass"] and c["video_frames_pass"]
                                    for c in cases),
              "all_prop_paths_nonoverlap": all(c["prop_fixture_nonoverlap_pass"]
                                               for c in cases),
              "all_holds_zero_translation": all(c["holds_zero_requested_translation_pass"]
                                                for c in cases),
              "cases": cases}
    (OUT / "audit-v468.json").write_text(json.dumps(output, indent=2) + "\n")
    print("video", output["all_video_pass"],
          "prop_paths", output["all_prop_paths_nonoverlap"],
          "zero_holds", output["all_holds_zero_translation"], flush=True)


if __name__ == "__main__":
    main()
