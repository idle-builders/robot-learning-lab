"""Gate the two contrasting native scenes before widening a conditioned route."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"


def main() -> None:
    plan = json.loads((ARTIFACT / "v8/conditioned-route-plan.json").read_text())
    actor_sha = hashlib.sha256((ARTIFACT / "v7/actor.pt").read_bytes()).hexdigest()
    if actor_sha != plan["source_actor_sha256"]:
        raise ValueError("Saved actor differs from the route plan")
    out = ARTIFACT / "v8/conditioned-route/diagnostic"
    rows = []
    for seed in (9923, 9926):
        case = next(row for row in plan["development_replay"] if row["seed"] == seed)
        result_path = out / f"seed-{seed}.json"
        trace_path = out / f"seed-{seed}.npz"
        report = json.loads(result_path.read_text())
        if (report["seed"] != seed or report["reach_checkpoint_sha256"] != actor_sha or
                report["reach_controller"] != "learned-residual" or
                report["terminal_roll_mode"] != "entry-error-x" or
                report["terminal_roll_offset_rad"] != -.08 or
                report["reach_duration_s"] != 9.0 or
                report["pitch_offset_rad"] != .105 or
                report["roll_clearance_rad"] != -.25 or
                report["one_sim_without_skill_reset"] is not True or
                report["duration_s"] != 16.5):
            raise ValueError(f"Wrong route or entry for {result_path}")
        faults = gate(report)
        reach = report["phases"]["reach"]
        rows.append({"case": f"seed-{seed}", **case, "pass": not faults, "faults": faults,
                     "contact_s": reach["continuous_finger_contact_s"],
                     "both_feet_fraction": reach["both_feet_fraction"],
                     "rated_leg_fraction": reach["rated_leg_fraction"],
                     "self_body_contact_fraction": reach["self_body_contact_fraction"],
                     "max_abs_unbounded_action": report["max_abs_unbounded_action"],
                     "route_entry_error_x_m": report["route_entry_error_x_m"],
                     "applied_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
                     "result": str(result_path.relative_to(ROOT)),
                     "trace": str(trace_path.relative_to(ROOT))})
    matrix = {"schema": "robot-learning-lab.cove50-handoff-v8-conditioned-diagnostic.v1",
              "all_pass": all(row["pass"] for row in rows), "cases": rows}
    (out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
