"""Audit the dual-height joined owner turn against static prop and fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_joined_owner_route_v411 as joined_audit
from audit_robocasa_layout2_routes_v409 import evaluator_scene, sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-dual-height-owner-v437"


def main() -> None:
    report_path = OUT / "report.json"
    report = json.loads(report_path.read_text())
    # V411's independent scene helper is reused with the added fixed prop.
    # The replacement affects this audit only, never the actor's observations.
    joined_audit.evaluator_scene = lambda _unused: evaluator_scene(
        list(report["static_prop_xy_m"]))
    cases = [joined_audit.audit_case(episode) for episode in report["episodes"]]
    output = {"source_sha256": sha256(Path(__file__)),
              "trial_report_sha256": sha256(report_path),
              "trial_source_sha_pass": report["source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_dual_height_owner_v437.py")),
              "frozen_owner_source_sha_pass": report["frozen_owner_source_sha256"] == sha256(
                  Path(__file__).with_name("run_robocasa_joined_owner_route_v410.py")),
              "scene_pair_pass": len({episode["scene_fingerprint"]
                                      for episode in report["episodes"]}) == 1,
              "cases": cases,
              "all_pass": all(case["proxy_margin_pass"] and
                              case["fresh_claim_evidence_pass"] and
                              case["response_contract_pass"] and
                              case["video_sha_pass"] and
                              case["video_frame_count_pass"] for case in cases)}
    (OUT / "audit-v438.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["response"], case["outcome"],
              round(case["minimum_signed_clearance_m_capped_at_10cm"]["distance_m"], 3),
              case["claim_iou"], flush=True)
    print("all_pass", output["all_pass"], flush=True)


if __name__ == "__main__":
    main()
