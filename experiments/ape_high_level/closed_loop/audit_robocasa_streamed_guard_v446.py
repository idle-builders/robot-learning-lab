"""Audit rotating-sector guard runs through prop motion after each stop."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_dynamic_suite_v440 import audit_case
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-streamed-guard-v445"
SUITE = Path(__file__).with_name("robocasa_dynamic_suite_v439.json")


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    scenario = next(case for case in json.loads(SUITE.read_text())["cases"]
                    if case["id"] == "east_pinch")
    cases = [audit_case({"scenario": scenario,
                         "package": f"phase{case['phase_sector']}",
                         "episode": case["episode"]})
             for case in report["cases"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_streamed_guard_v445.py")),
              "scene_pair_pass": len({case["episode"]["scene_fingerprint"]
                                      for case in report["cases"]}) == 1,
              "all_video_pass": all(case["video_sha_pass"] and case["video_frames_pass"]
                                    for case in cases),
              "cases": cases}
    (OUT / "audit-v446.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in cases:
        print(row["package"], row["outcome"],
              "full_horizon_gap", round(row["nearest_ape_any"]["gap_m_capped_at_10cm"], 3),
              "phase", row["nearest_ape_any"]["phase"], flush=True)
    print("all_video_pass", output["all_video_pass"], flush=True)


if __name__ == "__main__":
    main()
