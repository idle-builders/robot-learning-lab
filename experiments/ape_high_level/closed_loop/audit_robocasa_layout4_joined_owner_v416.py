"""Audit fresh owner claims and 3D proxy clearance in layout-4 transfer."""

from __future__ import annotations

import json
from pathlib import Path

import audit_robocasa_joined_owner_route_v411 as owner_audit
from audit_robocasa_layout4_transfer_v413 import evaluator_scene
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-layout4-joined-owner-v415"


def main() -> None:
    report = json.loads((OUT / "report.json").read_text())
    owner_audit.evaluator_scene = lambda _obstacle=None: evaluator_scene()
    cases = [owner_audit.audit_case(episode) for episode in report["episodes"]]
    source = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_layout4_joined_owner_v415.py"
    actor = ROOT / "experiments/ape_high_level/closed_loop/run_robocasa_joined_owner_route_v410.py"
    output = {"source_sha256": sha256(Path(__file__)),
              "scene_source_sha_pass": report["source_sha256"] == sha256(source),
              "frozen_joined_actor_source_sha_pass": report["frozen_joined_actor_source_sha256"] == sha256(actor),
              "same_seeded_scene_pass": len({e["scene_fingerprint"] for e in report["episodes"]}) == 1,
              "cases": cases,
              "all_pass": all(case["proxy_margin_pass"] and case["fresh_claim_evidence_pass"] and
                              case["response_contract_pass"] and case["video_sha_pass"] and
                              case["video_frame_count_pass"] for case in cases)}
    (OUT / "audit-v416.json").write_text(json.dumps(output, indent=2) + "\n")
    for case in cases:
        print(case["response"], case["outcome"],
              "clearance >=", case["minimum_signed_clearance_m_capped_at_10cm"]["distance_m"],
              "claim IoU", case["claim_iou"], flush=True)
    print("all_pass", output["all_pass"], flush=True)


if __name__ == "__main__":
    main()
