"""Compare a fixed travel-cost meeting-pose order against the earlier grid order."""

from __future__ import annotations

import argparse
import json
import math

import numpy as np

import run_meeting_pose_replan_3 as baseline
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
import run_cooperative_represent_probe as presentation


OUT = ROOT / "assets/real_indoor_scenes/meeting_pose_cost_order_3_20261015"
PRECOMMIT = ART / "meeting_pose_cost_order_3_precommit.json"
REPORT = ART / "meeting_pose_cost_order_3_report.json"
MAX_CANDIDATES = 8
MAX_ACTIONS = 96
MAX_PER_CANDIDATE = 36
PROPOSAL_POOL = 32


def ranked_proposals(source: dict, required: float) -> list[dict]:
    """Rank old-scan proposals using only the believed station and action limits."""
    start = np.asarray(source["final_actor_xz_m"], float)
    yaw = float(source["final_actor_yaw_deg"])
    proposals = baseline.propose(source, required, PROPOSAL_POOL)

    def cost(candidate: dict) -> tuple[int, int, float]:
        displacement = np.asarray(candidate["actor_goal_xz_m"], float) - start
        bearing = math.degrees(math.atan2(displacement[0], displacement[1]))
        turn = abs((bearing - yaw + 180.) % 360. - 180.)
        distance = float(np.linalg.norm(displacement))
        turn_actions = math.ceil(turn / 15. - 1e-9)
        forward_actions = math.ceil(distance / .04 - 1e-9)
        return (turn_actions + forward_actions, turn_actions, distance)

    return sorted(proposals, key=cost)[:MAX_CANDIDATES]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    cases = baseline.cases()
    required = json.loads(presentation.geometry.OUT.read_text())[
        "required_nearest_two_plane_m"]
    frozen = {
        "schema": "ape.meeting-pose-cost-order-3-precommit.v1",
        "baseline_precommit": relative(baseline.PRECOMMIT),
        "cases": cases,
        "source_station_sha256": {c["scene_key"]: sha(ROOT / c["source"])
                                  for c in cases},
        "required_wait_clearance_m": required,
        "proposal_pool": PROPOSAL_POOL,
        "max_candidates": MAX_CANDIDATES,
        "max_actions": MAX_ACTIONS,
        "max_per_candidate": MAX_PER_CANDIDATE,
        "policy": "Use the unchanged old person-aware two-height range grid and corridor test. Rank all 32 observed proposals by estimated turn actions plus forward actions from the believed station, break ties by turn cost and distance, and try the first eight. Allow at most 36 actions per candidate and 96 total. Every move needs fresh RGB-D/range; only a new local wait scan may permit owner presentation. No scene-specific adjustment or evaluator pose in selection.",
        "limits": "Development-exposed stations; static scripted owner, ideal kinematic disk, simulated range and yaw. This compares action scheduling, not a deployable sensor or body."
    }
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Cost-order precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    if args.freeze_only:
        print(PRECOMMIT)
        return
    rows = []
    for case in cases:
        path = OUT / case["scene_key"] / "trace.json"
        if path.exists():
            x = json.loads(path.read_text())
            row = {"scene_key": case["scene_key"], "trace": relative(path),
                   "outcome": x["presentation_outcome"],
                   "selected_candidate": x["selected_candidate"],
                   "video": x["video"]}
        else:
            row = baseline.run_one(case, required,
                proposal_provider=ranked_proposals, out=OUT,
                max_actions=MAX_ACTIONS,
                max_per_candidate=MAX_PER_CANDIDATE,
                schema="ape.meeting-pose-cost-order-3.v1")
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.meeting-pose-cost-order-3-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["scene_key"], row["outcome"], flush=True)


if __name__ == "__main__":
    main()
