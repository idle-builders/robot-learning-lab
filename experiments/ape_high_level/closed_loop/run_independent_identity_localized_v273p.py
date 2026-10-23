"""Replay the frozen V268 owner rooms with motion-checked visual odometry.

The V268 command-pose episodes are the retained baseline. Only the pose
source changes here; scene declarations and owner-response scripts are reused.
This remains a kinematic run on the rejected historical Direct Frame body.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live
import run_independent_identity_rooms_v268p as baseline
import run_varied_owner_route_v258p as challenge


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v273p"
BASELINE = ROOT / "artifacts/ape-articulated-depth-route-v268p"


def pose_error(event: dict) -> float:
    estimated = event["actor_estimated_xyyaw"]
    actual = event["evaluator_actual_xyyaw"]
    return math.hypot(estimated[0] - actual[0], estimated[1] - actual[1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_report_path = BASELINE / "report.json"
    old_report = json.loads(old_report_path.read_text())
    if [tuple(row["case"]) for row in old_report["rows"]] != list(baseline.CASES):
        raise ValueError("V268 baseline cases changed")
    manifest = {
        "schema": "robot-learning-lab.independent-identity-localized-declaration.v1",
        "role": "paired continuation of the V268 frozen owner scenes",
        "body": "rejected historical Cove-50 Direct Frame surrogate",
        "only_changed_actor_input": "motion-checked upper RGB-D displacement replaces command pose after each base move",
        "unchanged": "same geometry, hidden slip schedule, head and near-body cameras/ranges, owner script and route gates",
        "cases": baseline.CASES,
        "scene": baseline.SCENE,
        "owner_responses": baseline.OWNER_RESPONSES,
        "pose_source": "motion_checked_rgbd",
        "baseline_report_sha256": sha(old_report_path),
        "source_sha256": sha(SOURCE),
        "baseline_case_source_sha256": sha(Path(baseline.__file__)),
        "runner_source_sha256": sha(Path(live.__file__)),
        "challenge_source_sha256": sha(Path(challenge.__file__)),
    }
    declaration = OUT / "predeclared-localization-pair.json"
    declaration.write_text(json.dumps(manifest, indent=2) + "\n")
    previous_out, previous_pose = baseline.OUT, challenge.POSE_SOURCE
    try:
        baseline.OUT = OUT
        challenge.POSE_SOURCE = "motion_checked_rgbd"
        baseline.main()
    finally:
        baseline.OUT = previous_out
        challenge.POSE_SOURCE = previous_pose
    new_report = json.loads((OUT / "report.json").read_text())
    if [tuple(row["case"]) for row in new_report["rows"]] != list(baseline.CASES):
        raise ValueError("Localized cases changed")
    if json.loads((OUT / "geometry-preflight.json").read_text())["rows"] != \
            json.loads((BASELINE / "geometry-preflight.json").read_text())["rows"]:
        raise ValueError("Geometry preflight differs from baseline")
    new_run = json.loads((OUT / "blocked-forward-viewpoint-audit.json").read_text())
    old_run = json.loads((BASELINE / "blocked-forward-viewpoint-audit.json").read_text())
    pairs = []
    for previous, current, before, after in zip(
            old_report["rows"], new_report["rows"],
            old_run["rows"], new_run["rows"], strict=True):
        old_trace = json.loads((ROOT / before["trace_path"]).read_text())
        new_trace = json.loads((ROOT / after["trace_path"]).read_text())
        old_events = old_trace["localization_events"]
        new_events = new_trace["localization_events"]
        if after["pose_source"] != "motion_checked_rgbd":
            raise ValueError("Localized actor source not recorded")
        if not new_events or not old_events:
            raise ValueError("Missing paired localization packets")
        pairs.append({
            "case": current["case"],
            "command": {"status": previous["status"],
                        "base_moves": previous["base_moves"],
                        "owner_questions": previous["owner_questions"],
                        "minimum_clearance_m": previous["minimum_clearance_m"],
                        "pose_error_final_m": round(pose_error(old_events[-1]), 5),
                        "pose_error_max_m": round(max(map(pose_error, old_events)), 5)},
            "motion_checked_rgbd": {
                "status": current["status"],
                "base_moves": current["base_moves"],
                "owner_questions": current["owner_questions"],
                "minimum_clearance_m": current["minimum_clearance_m"],
                "pose_error_final_m": round(pose_error(new_events[-1]), 5),
                "pose_error_max_m": round(max(map(pose_error, new_events)), 5),
                "registered_updates": sum(e["status"]["status"] == "registered"
                                          for e in new_events[1:]),
                "rejected_updates": sum(e["status"]["status"] != "registered"
                                        for e in new_events[1:]),
                "gate_results": [e["gate"] for e in new_events[1:] if "gate" in e],
            },
            "trace_path": after["trace_path"],
            "video_path": after["video_path"],
        })
    (OUT / "localization-pair-report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.independent-identity-localized-result.v1",
        "declaration_sha256": sha(declaration),
        "baseline_report_sha256": sha(old_report_path),
        "localized_report_sha256": sha(OUT / "report.json"),
        "pairs": pairs,
    }, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
