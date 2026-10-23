"""Isolate the V41 versus V57 arm on V83's older 10534 regression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v83_thin_route_pilot as v83,
    run_v83_thin_route_retention as retention,
)


OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v83"
DEST = OUT / "prior10534-v41-arm-diagnostic"
ACTOR = OUT.parent / "v41/actor.pt"
PLAN = DEST / "plan.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    pilot = json.loads((OUT / "prior75-pilot-matrix.json").read_text())
    if (not pilot["complete"] or
            not any(row["seed"] == 10534 and not row["pass"] for row in pilot["cases"])):
        raise ValueError("Need complete pilot and observed prior10534 regression first")
    parent = OUT / "prior75-retention-plan.json"
    older = json.loads(parent.read_text())
    case = next(row for row in older["cases"] if row["seed"] == 10534)
    case = {**case, "id": "prior-10534-v41-arm"}
    sources = {**older["source_sha256"], "actor": sha(ACTOR)}
    plan = {
        "schema": "robot-learning-lab.cove50-v83-prior10534-arm-ablation.v1",
        "role": "post_pilot_opened_single_factor_diagnostic_not_retention",
        "body_role": older["body_role"],
        "prior75_plan_sha256": sha(parent),
        "prior75_pilot_matrix_sha256": sha(OUT / "prior75-pilot-matrix.json"),
        "evaluator_sha256": sha(v83.EVALUATOR),
        "v41_actor_sha256": sha(ACTOR),
        "v83_actor_sha256": older["source_sha256"]["actor"],
        "source_sha256": sources,
        "case_count": 1, "cases": [case],
        "changed_factor": "Use the saved V41 arm weights instead of V57 under the same V83 evaluator, image selector, body and motor settings on the exact prior10534 start. No training or automatic selection is inferred.",
    }
    DEST.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen prior10534 actor ablation differs")
    PLAN.write_bytes(encoded)
    v83.v57.ACTOR = ACTOR
    v83.DEST = DEST
    retention.DEST = DEST
    row = retention.run_retention_case(case, plan)
    row["baseline_contact_s"] = row.pop("v73_contact_s")
    matrix = {
        "schema": "robot-learning-lab.cove50-v83-prior10534-arm-ablation-matrix.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "case_count": 1, "pass_count": int(row["pass"]), "cases": [row],
    }
    (DEST / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({key: row[key] for key in ("id", "pass", "contact_s", "faults")},
                     indent=2))


if __name__ == "__main__":
    main()
