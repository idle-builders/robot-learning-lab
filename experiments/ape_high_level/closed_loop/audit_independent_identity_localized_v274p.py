"""Audit the full owner replay and the paired visual-pose intervention."""

from __future__ import annotations

import json
from pathlib import Path

import audit_independent_identity_rooms_v269p as owner_audit
from run import ROOT, sha


OUT = ROOT / "artifacts/ape-articulated-depth-route-v273p"
BASELINE = ROOT / "artifacts/ape-articulated-depth-route-v268p"


def main() -> None:
    previous_out = owner_audit.OUT
    try:
        owner_audit.OUT = OUT
        owner_audit.main()
    finally:
        owner_audit.OUT = previous_out
    owner = json.loads((OUT / "audit.json").read_text())
    pairs_path = OUT / "localization-pair-report.json"
    report = json.loads(pairs_path.read_text())
    declaration_path = OUT / "predeclared-localization-pair.json"
    declaration = json.loads(declaration_path.read_text())
    assert declaration["source_sha256"] == sha(OUT / "run_independent_identity_localized_v273p.py")
    assert declaration["baseline_case_source_sha256"] == sha(OUT / "run_independent_identity_rooms_v268p.py")
    assert declaration["runner_source_sha256"] == sha(OUT / "run_blocked_forward_viewpoint_v321.py")
    assert declaration["challenge_source_sha256"] == sha(OUT / "run_varied_owner_route_v258p.py")
    assert report["declaration_sha256"] == sha(declaration_path)
    assert report["baseline_report_sha256"] == sha(BASELINE / "report.json")
    assert report["localized_report_sha256"] == sha(OUT / "report.json")
    assert owner["wrong_success_claims"] == 0
    assert owner["videos_decoded"] == 6
    assert len(report["pairs"]) == 6
    observations = []
    for pair in report["pairs"]:
        trace = json.loads((ROOT / pair["trace_path"]).read_text())
        events = trace["localization_events"]
        assert trace["pose_source"] == "motion_checked_rgbd"
        assert len(events) - 1 == pair["motion_checked_rgbd"]["base_moves"]
        assert len(trace["slips_evaluator_only"]) == pair["motion_checked_rgbd"]["base_moves"]
        assert all(a["capture_time_s"] <= b["capture_time_s"]
                   for a, b in zip(events, events[1:]))
        assert all(event.get("gate") in ("fused", "motion_rejected", "visual_unavailable")
                   for event in events[1:])
        assert pair["motion_checked_rgbd"]["minimum_clearance_m"] >= .05
        if pair["motion_checked_rgbd"]["status"] in owner_audit.SUCCESS:
            assert pair["case"][1] == "present"
            assert any(event["action"] == "view" and
                       event["accepted_box"] is not None and
                       event["evaluator_correct_item"]
                       for event in trace["controller_events"])
        observations.append({
            "seed": pair["case"][0],
            "command_final_pose_error_m": pair["command"]["pose_error_final_m"],
            "localized_final_pose_error_m": pair["motion_checked_rgbd"]["pose_error_final_m"],
            "base_moves_command": pair["command"]["base_moves"],
            "base_moves_localized": pair["motion_checked_rgbd"]["base_moves"],
            "status_command": pair["command"]["status"],
            "status_localized": pair["motion_checked_rgbd"]["status"],
        })
    result = {
        "schema": "robot-learning-lab.independent-identity-localized-audit.v1",
        "paired_report_sha256": sha(pairs_path),
        "owner_audit_sha256": sha(OUT / "audit.json"),
        "wrong_success_claims": owner["wrong_success_claims"],
        "videos_decoded": owner["videos_decoded"],
        "localization_event_integrity_pass": True,
        "observations": observations,
    }
    (OUT / "localization-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
