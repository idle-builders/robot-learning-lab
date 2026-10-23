"""Independent geometry, full-motion horizon, video and decision audit for V473."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_dynamic_suite_v440 as common_audit
from audit_robocasa_layout4_crossing_v448 import layout4_evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-safe-side-v473"
SOURCE = Path(__file__).with_name("run_robocasa_layout4_safe_side_v473.py")
CONFIG = Path(__file__).with_name("robocasa_layout4_safe_side_v473.json")
PRIOR = ROOT / "artifacts/ape-robocasa-layout4-survey-v452/sensor-derived-map.npz"
TRACKER = Path(__file__).with_name("novel_range_object.py")


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    common_audit.evaluator_scene = layout4_evaluator_scene
    cases = [common_audit.audit_case(case) for case in report["cases"]]
    by_id = {case["scenario_id"]: case for case in cases}
    stage_cases = [case for case in report["cases"]
                   if case["episode"]["outcome"] == "request_owner_show_here_safe_side"]
    stage_decision_pass = all(
        case["safe_side_history"] and case["safe_side_history"][-1]["should_stage"]
        and case["safe_side_history"][-1]["current_staging_reserve_m"] >=
        json.loads(CONFIG.read_text())["minimum_staging_reserve_m"]
        and case["episode"]["trace"][-1]["requested_xy_m"] is None
        and case["episode"]["end_time_s"] < case["scenario"]["begin_s"]
        for case in stage_cases)
    output = {
        "source_sha256": sha256(Path(__file__)),
        "trial_report_sha256": sha256(report_path),
        "trial_source_sha_pass": report["source_sha256"] == sha256(SOURCE),
        "config_source_sha_pass": report["config_sha256"] == sha256(CONFIG),
        "prior_source_sha_pass": report["prior_map_sha256"] == sha256(PRIOR),
        "tracker_source_sha_pass": report["tracker_source_sha256"] == sha256(TRACKER),
        "late_stationary_scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                                for case in report["cases"]
                                                if case["scenario"]["id"] != "absent"}) == 1,
        "stage_decision_pass": stage_decision_pass,
        "absent_goal_pass": report["cases"][0]["scenario"]["id"] == "absent"
        and report["cases"][0]["episode"]["outcome"] == "goal_viewpoint_reached",
        "stationary_and_late_stage_pass": len(stage_cases) == 2
        and {case["scenario"]["id"] for case in stage_cases} ==
        {"stationary", "late_crossing"},
        "all_5cm_room_margin_pass": all(case["ape_5cm_margin_pass"] for case in cases),
        "all_prop_nonoverlap_pass": all(case["prop_fixture_nonoverlap_pass"] for case in cases),
        "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                              for case in cases),
        "late_full_horizon_prop_gap_pass": by_id["late_crossing"]
        ["nearest_ape_prop"]["signed_gap_m"] >= .05,
        "cases": cases,
    }
    (OUT / "audit-v474.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["scenario_id"], case["outcome"],
              "room_gap", round(case["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "prop_gap", round(case["nearest_ape_prop"]["signed_gap_m"], 3),
              "samples", case["geometry_samples"], flush=True)
    print({key: value for key, value in output.items() if key.endswith("_pass")},
          flush=True)


if __name__ == "__main__":
    main()
