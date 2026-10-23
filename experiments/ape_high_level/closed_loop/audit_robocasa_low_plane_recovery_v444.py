"""Audit full-horizon geometry, stationary holds and recovery event order."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_robocasa_dynamic_suite_v440 import audit_case
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-low-plane-recovery-v443"


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    cases = []
    for case in report["cases"]:
        episode = case["episode"]
        geometry = audit_case(case)
        health = case["sensor_health_trace"]
        holds = [row for row in episode["trace"]
                 if row["plan_status"] == "hold_for_fresh_low_scan"]
        hold_zero_motion = all(
            row["requested_xy_m"] is not None and
            np.linalg.norm(np.asarray(row["requested_xy_m"]) -
                           np.asarray(row["estimated_xy_m"])) < 1e-9
            for row in holds)
        health_order = (len(health) == len(episode["trace"]) and
                        all(h["scan_step"] == i and
                            h["plan_status"] == episode["trace"][i]["plan_status"]
                            for i, h in enumerate(health)))
        missing_motion_pass = all(
            h["low_packet_valid"] or
            h["plan_status"] in ("hold_for_fresh_low_scan", "low_range_unhealthy_stop")
            for h in health)
        cases.append({"package": case["package"],
                      "outcome": episode["outcome"],
                      "hold_count": case["hold_count"],
                      "hold_zero_motion_pass": hold_zero_motion,
                      "health_order_pass": health_order,
                      "missing_motion_pass": missing_motion_pass,
                      "geometry": geometry})
        print(case["package"], episode["outcome"],
              "holds", len(holds), "clearance",
              round(geometry["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "goal", round(geometry["actual_goal_gap_m"], 3), flush=True)
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_low_plane_recovery_v443.py")),
              "suite_source_sha_pass": report["suite_sha256"] == sha256(
                  Path(__file__).with_name("robocasa_dynamic_suite_v439.json")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_contract_and_video_pass": all(
                  case["hold_zero_motion_pass"] and case["health_order_pass"] and
                  case["missing_motion_pass"] and case["geometry"]["video_sha_pass"] and
                  case["geometry"]["video_frames_pass"] for case in cases),
              "cases": cases}
    (OUT / "audit-v444.json").write_text(json.dumps(output, indent=2) + "\n")
    print("all_contract_and_video_pass", output["all_contract_and_video_pass"], flush=True)


if __name__ == "__main__":
    main()
