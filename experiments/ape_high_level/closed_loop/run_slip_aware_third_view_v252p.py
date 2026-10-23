"""A fresh third viewpoint after the second command slips short of a find.

The actor repeats the same head-depth/near-range/shin free-space gate at its
new estimated pose. It never uses the evaluator's actual step distance,
target box or prop trajectory to authorize the next command.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from depth_quality import reject_isolated_depth_spikes
from run import ROOT, sha
import run_blocked_forward_viewpoint_v321 as live
from run_medium_depth_recovery_v246p import CASES, MediumWindowMap, profile_depth
from run_second_viewpoint_v250p import after_first_view as one_extra_view


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-articulated-depth-route-v252p"


def two_extra_views(room, observe, branch: Path, video, robot,
                    filtered_depth, seed: int) -> dict:
    second_branch = branch / "second-stage"
    second_branch.mkdir(parents=True, exist_ok=True)
    second = one_extra_view(
        room, observe, second_branch, video, robot, filtered_depth, seed)
    executed = ([second["audit"]["actor_gate"]]
                if second["audit"]["actor_motion_authorized"] else [])
    audit = {"stages": [second["audit"]],
             "actor_motion_authorized": bool(executed),
             "actor_gate": second["audit"]["actor_gate"]}
    if second["found_event"] is not None or not executed:
        return {"audit": audit, "executed_actor_gates": executed,
                "found_event": second["found_event"]}

    third_branch = branch / "third-stage"
    third_branch.mkdir(parents=True, exist_ok=True)
    previous_depth = live.noisy_depth

    def new_capture_noise(ideal: np.ndarray, mode: str, seed: int,
                          serial: int) -> np.ndarray:
        return previous_depth(ideal, mode, seed, serial + 100)

    try:
        live.noisy_depth = new_capture_noise
        third = one_extra_view(
            room, observe, third_branch, video, robot, filtered_depth, seed)
    finally:
        live.noisy_depth = previous_depth
    audit["stages"].append(third["audit"])
    audit["actor_gate"] = third["audit"]["actor_gate"]
    if third["audit"]["actor_motion_authorized"]:
        executed.append(third["audit"]["actor_gate"])
    audit["actor_motion_authorized"] = bool(executed)
    return {"audit": audit, "executed_actor_gates": executed,
            "found_event": third["found_event"]}


def main() -> None:
    original_cases = live.CASES
    original_depth = live.noisy_depth
    original_map = live.DepthVoxelMap
    try:
        live.CASES = CASES
        live.noisy_depth = profile_depth("medium_30cm_1m", original_depth)
        live.DepthVoxelMap = MediumWindowMap
        live.main(output_root=OUT, source=SOURCE,
                  map_depth_filter=reject_isolated_depth_spikes,
                  experiment_label="V252p repeated actor-certified viewpoint",
                  after_first_view=two_extra_views,
                  max_recovery_steps=3)
    finally:
        live.CASES = original_cases
        live.noisy_depth = original_depth
        live.DepthVoxelMap = original_map
    path = OUT / "blocked-forward-viewpoint-audit.json"
    report = json.loads(path.read_text())
    report["scope_override"] = (
        "Historical articulated 50cm owner task with V246p medium virtual "
        "head depth. After the first certified viewpoint yields no item, "
        "up to two further 8cm forward commands each need new actor-only "
        "head/range/shin free/occupied/unknown evidence. A +0.4/+0.5rad "
        "fresh item view follows each executed step; owner confirmation "
        "still requires a second fresh view. All base motion is kinematic.")
    path.write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.slip-aware-third-view.v1",
        "source_sha256": sha(SOURCE),
        "runner_report_path": str(path.relative_to(ROOT)),
        "runner_report_sha256": sha(path),
        "cases": CASES,
        "rows": [{"case": row["case"], "status": row["status"],
                  "owner_questions": row["owner_questions"],
                  "base_moves": row["base_moves"],
                  "minimum_clearance_m": row[
                      "evaluator_path_min_clearance_m"]}
                 for row in report["rows"]]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
