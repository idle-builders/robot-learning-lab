"""Audit development pose-health stops against matched healthy routes."""

from __future__ import annotations

import json
from pathlib import Path

from audit_robocasa_layout2_routes_v409 import audit_episode as audit_layout2
from audit_robocasa_layout4_transfer_v413 import audit_episode as audit_layout4
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-pose-health-gate-v419"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_pose_health_gate_v419.py"
    actor = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout2_localized_route_v408.py"
    cases = []
    for case in report["cases"]:
        episode = case["episode"]
        geometry = (audit_layout2(episode, "pose_health_gate") if case["layout"] == 2
                    else audit_layout4(episode))
        minimum = (geometry["minimum_signed_clearance_m"] if case["layout"] == 2
                   else geometry["clearance"]["distance_m_capped_at_10cm"])
        expected = (episode["outcome"] == "goal_viewpoint_reached" and case["health_trigger"] is None
                    if case["mode"] == "healthy" else
                    episode["outcome"] == "localization_uncertain_stop" and
                    case["health_trigger"] is not None)
        cases.append({"layout": case["layout"], "mode": case["mode"],
                      "outcome": episode["outcome"], "steps": episode["steps"],
                      "end_pose_error_m": episode["end_pose_error_m"],
                      "health_trigger": case["health_trigger"],
                      "expected_state_pass": bool(expected),
                      "minimum_signed_clearance_m_capped_at_10cm": minimum,
                      "proxy_5cm_margin_pass": minimum >= .05,
                      "video_sha_pass": (geometry["video_sha_match"] if case["layout"] == 2
                                         else geometry["video_sha_pass"]),
                      "video_frames_pass": (geometry["video_frames_match"] if case["layout"] == 2
                                            else geometry["video_frames_pass"])})
    output = {"source_sha256": sha256(Path(__file__)),
              "gate_source_sha_pass": report["source_sha256"] == sha256(source),
              "frozen_actor_source_sha_pass": report["frozen_actor_source_sha256"] == sha256(actor),
              "cases": cases,
              "all_pass": all(x["expected_state_pass"] and x["proxy_5cm_margin_pass"] and
                              x["video_sha_pass"] and x["video_frames_pass"] for x in cases)}
    (OUT / "audit-v420.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["layout"], case["mode"], case["outcome"],
              round(case["end_pose_error_m"], 3),
              case["health_trigger"]["frame"] if case["health_trigger"] else None,
              "clearance", case["minimum_signed_clearance_m_capped_at_10cm"], flush=True)
    print("all_pass", output["all_pass"], flush=True)


if __name__ == "__main__":
    main()
