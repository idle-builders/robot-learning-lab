"""Independent full-horizon geometry and artifact audit for V478 transfer."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout2-safe-side-v478"
SOURCE = Path(__file__).with_name("run_robocasa_layout2_safe_side_v478.py")
CONFIG = Path(__file__).with_name("robocasa_layout2_safe_side_v478.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout2-survey-v477/sensor-derived-map.npz"
POLICY = Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    cases = [common_audit.audit_case(case) for case in report["cases"]]
    output = {
        "source_sha256": sha256(Path(__file__)),
        "trial_report_sha256": sha256(report_path),
        "trial_source_sha_pass": report["source_sha256"] == sha256(SOURCE),
        "config_source_sha_pass": report["config_sha256"] == sha256(CONFIG),
        "prior_source_sha_pass": report["prior_map_sha256"] == sha256(PRIOR),
        "frozen_policy_source_sha_pass": report["frozen_policy_source_sha256"] == sha256(POLICY),
        "east_initial_scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                             for case in report["cases"]
                                             if case["scenario"]["id"].startswith("east")}) == 1,
        "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                              for case in cases),
        "all_room_margin_pass": all(case["ape_5cm_margin_pass"] for case in cases),
        "all_prop_nonoverlap_pass": all(case["prop_fixture_nonoverlap_pass"] for case in cases),
        "all_ape_prop_margin_pass": all(case["nearest_ape_prop"]["signed_gap_m"] >= .05
                                    for case in cases),
        "all_goal_pass": all(case["outcome"] == "goal_viewpoint_reached" for case in cases),
        "cases": cases,
    }
    (OUT / "audit-v479.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["scenario_id"], case["outcome"],
              "room_gap", round(case["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_gap", round(case["nearest_ape_prop"]["signed_gap_m"], 3),
              "samples", case["geometry_samples"], flush=True)
    print({key: value for key, value in output.items() if key.endswith("_pass")}, flush=True)


if __name__ == "__main__":
    main()
