"""Score heading-fault route videos against independently rebuilt geometry."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import audit_episode as audit_layout2
from audit_robocasa_layout4_transfer_v413 import audit_episode as audit_layout4
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-heading-fault-v417"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    actor_source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout2_localized_route_v408.py"
    fault_source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_heading_fault_v417.py"
    cases = []
    for case in report["cases"]:
        episode = case["episode"]
        geometry = (audit_layout2(episode, "heading_fault") if case["layout"] == 2
                    else audit_layout4(episode))
        minimum = (geometry["minimum_signed_clearance_m"] if case["layout"] == 2
                   else geometry["clearance"]["distance_m_capped_at_10cm"])
        cases.append({"layout": case["layout"], "mode": case["mode"],
                      "outcome": episode["outcome"], "steps": episode["steps"],
                      "final_pose_error_m": episode["end_pose_error_m"],
                      "maximum_pose_error_m": episode["max_pose_error_m"],
                      "unsafe_requested_steps": episode["unsafe_requested_steps"],
                      "minimum_signed_clearance_m_capped_at_10cm": minimum,
                      "video_sha_pass": (geometry["video_sha_match"] if case["layout"] == 2
                                         else geometry["video_sha_pass"]),
                      "video_frames_pass": (geometry["video_frames_match"] if case["layout"] == 2
                                            else geometry["video_frames_pass"])})
    output = {"source_sha256": sha256(Path(__file__)),
              "fault_source_sha_pass": report["source_sha256"] == sha256(fault_source),
              "frozen_actor_source_sha_pass": report["frozen_actor_source_sha256"] == sha256(actor_source),
              "cases": cases}
    (OUT / "audit-v418.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["layout"], case["mode"], case["outcome"],
              "pose error", round(case["final_pose_error_m"], 3),
              "clearance", case["minimum_signed_clearance_m_capped_at_10cm"],
              "unsafe requests", case["unsafe_requested_steps"], flush=True)


if __name__ == "__main__":
    main()
