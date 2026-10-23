"""Complete the invalid V72 voltage grid as an explicitly post-result diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v70_center_route_screen as v70,
)


BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OLD_PLAN = BASE / "v72/center-view-holdout-run-plan.json"
OLD_DIAGNOSTIC = BASE / "v72/invalid-input-diagnostic.json"
OUT = BASE / "v76"
PLAN = OUT / "center-voltage-domain-repair-plan.json"
DEST = OUT / "native-corrected-sixth"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    old_plan = json.loads(OLD_PLAN.read_text())
    old = json.loads(OLD_DIAGNOSTIC.read_text())
    if (old["gate_status"] != "invalid_input_and_missing_required_branch_exposure" or
            old["valid_case_count"] != 5 or old["valid_case_pass_count"] != 5 or
            old["center_rule_activation_count"] != 0 or old["invalid_rail_v"] != 9.5):
        raise ValueError("Expected saved V72 invalid-voltage diagnostic")
    cases = [dict(case) for case in old_plan["cases"]]
    if len(cases) != 6 or cases[-1]["id"] != old["invalid_case_id"]:
        raise ValueError("V72 final case identity changed")
    cases[-1]["rail_v"] = 10.0
    cases[-1]["id"] = "center-neighbor-06-10v"
    cases[-1]["development_role"] = "post_result_voltage_domain_repair"
    if any(not 10 <= case["rail_v"] <= 12 for case in cases):
        raise ValueError("Repair still contains an unsupported rail voltage")
    reused = [{"id": row["id"],
               "result_sha256": sha(ROOT / row["result"]),
               "trace_sha256": sha(ROOT / row["trace"]),
               "log_sha256": sha(ROOT / row["log"])}
              for row in old["valid_cases"]]
    plan = {
        "schema": "robot-learning-lab.cove50-v76-center-domain-repair-plan.v1",
        "role": "post_result_valid_domain_completion_not_predeclared_holdout",
        "body_role": old_plan["body_role"],
        "v70_plan_sha256": sha(v70.PLAN),
        "v70_matrix_sha256": sha(v70.DEST / "matrix.json"),
        "v72_plan_sha256": sha(OLD_PLAN),
        "v72_invalid_diagnostic_sha256": sha(OLD_DIAGNOSTIC),
        "source_sha256": old_plan["source_sha256"],
        "runner_sha256": sha(Path(__file__)),
        "reused_valid_evidence": reused,
        "case_count": 6, "cases": cases,
        "gate": "All six in-domain native visits must pass and at least three must activate the center rule. The first five outcomes were already observed before this repair; this is a post-result diagnostic and cannot satisfy V72's predeclared holdout gate.",
        "change": "Only the invalid sixth V72 voltage changes from 9.5 to 10.0 V. Its seed, toy, delay, gaze and route are held fixed. The five valid V72 outcomes are reused byte-for-byte.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V76 domain-repair plan differs from current source")
    PLAN.write_bytes(encoded)
    return plan


def run(plan: dict) -> dict:
    old = json.loads(OLD_DIAGNOSTIC.read_text())
    for saved, evidence in zip(old["valid_cases"], plan["reused_valid_evidence"], strict=True):
        if (saved["id"] != evidence["id"] or
                any(sha(ROOT / saved[key]) != evidence[f"{key}_sha256"]
                    for key in ("result", "trace", "log"))):
            raise ValueError("Reused V72 evidence changed")
    DEST.mkdir(parents=True, exist_ok=True)
    v70.DEST = DEST
    new = v70.run_case(plan["cases"][-1], plan)
    new["baseline_v59_center_contact_s"] = plan["cases"][-1]["baseline_v59_center_contact_s"]
    new["center_view_family"] = "post_result_voltage_domain_repair"
    rows = old["valid_cases"] + [new]
    center_uses = sum(row["automatic_front_center_bias_applied"] for row in rows)
    matrix = {
        "schema": "robot-learning-lab.cove50-v76-center-domain-repair-diagnostic.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "complete": True,
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "front_center_bias_use_count": center_uses,
        "diagnostic_gate_met": all(row["pass"] for row in rows) and center_uses >= 3,
        "v72_predeclared_holdout_gate_met": False,
        "cases": rows,
    }
    (OUT / "domain-repair-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-only", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if args.freeze_only:
        print(f"frozen {PLAN} sha256={sha(PLAN)}")
        return
    matrix = run(plan)
    print(json.dumps({key: matrix[key] for key in
                      ("case_count", "pass_count", "front_center_bias_use_count",
                       "diagnostic_gate_met", "v72_predeclared_holdout_gate_met")}, indent=2))


if __name__ == "__main__":
    main()
