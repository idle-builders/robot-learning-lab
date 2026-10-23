"""Publish the five source-frozen V59 branch checks from saved native reports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v59_combined_screen as v59,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = v59.freeze()
    wanted = {"large-30", "large-36", "edge-01", "edge-12", "positive-10"}
    cases = [case for case in plan["cases"] if case["id"] in wanted]
    if len(cases) != len(wanted):
        raise ValueError("Source-frozen pilot identities missing")
    for case in cases:
        base = v59.DEST / f"seed-{case['seed']}-{case['id']}"
        if not base.with_suffix(".json").is_file() or not base.with_suffix(".npz").is_file():
            raise ValueError(f"Pilot native evidence incomplete: {case['id']}")
    rows = [v59.run_case(case, plan) for case in cases]
    pilot = {
        "schema": "robot-learning-lab.cove50-v59-combined-staged-pilot.v1",
        "role": "source_frozen_branch_pilot_not_full_screen_or_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(v59.PLAN), "pilot_runner_sha256": sha(Path(__file__)),
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "complete": True, "cases": rows,
    }
    out = v59.OUT / "staged-pilot"
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.json").write_text(json.dumps(pilot, indent=2) + "\n")
    print(json.dumps({"case_count": pilot["case_count"],
                      "pass_count": pilot["pass_count"]}, indent=2))


if __name__ == "__main__":
    main()
