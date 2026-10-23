"""Unscored V167 CUDA closure check on one opened training donor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/reach-bc-v167"
V158 = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v158/new-seed-contact-plan.json"
WRAPPER = Path(__file__).with_name("evaluate_reach_bc_v167_cli.py")
EVALUATOR = Path(__file__).with_name("evaluate_reach_bc_v167.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=("sentinel", "sentinel-r2", "sentinel-r3"), default="sentinel")
    args = parser.parse_args()
    method = json.loads((BASE / "sentinel-method.json").read_text())
    if (method["seed"] != 24602 or
            method["launch_sha256"] != sha(BASE / "launch-r2.json")):
        raise ValueError("Unscored sentinel method changed")
    v158 = json.loads(V158.read_text())
    case = next(c for c in v158["cases"] if c["seed"] == 24602 and c["split"] == "development")
    case = {**case, "id": "v167-unscored-training-donor-24602"}
    dest = BASE / args.run_id
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError("Refusing to replace sentinel original")
    dest.mkdir(parents=True, exist_ok=True)
    original = native.EVALUATOR
    try:
        native.EVALUATOR = WRAPPER
        result = native.run_case(case, {"source_sha256": {
            **v158["native_source_sha256"], "evaluator": sha(EVALUATOR),
        }}, dest)
    finally:
        native.EVALUATOR = original
    report = json.loads((ROOT / result["result"]).read_text())
    if (not report["student_route_active"] or
            report["student_actor_sha256"] != sha(BASE / "checkpoint.pt")):
        raise ValueError("Student actor did not execute on opened donor")
    (dest / "sentinel-result.json").write_text(json.dumps({
        "schema": "robot-learning-lab.reach-v167-unscored-sentinel-result.v1",
        "seed": 24602, "method_sha256": sha(BASE / "sentinel-method.json"),
        "pass": result["pass"], "contact_s": result["contact_s"],
        "margin_1p2_s_met": result["margin_1p2_s_met"], "faults": result["faults"],
        "result": result["result"], "report_sha256": sha(ROOT / result["result"]),
        "trace": result["trace"], "trace_sha256": sha(ROOT / result["trace"]),
    }, indent=2) + "\n")
    print(json.dumps({"pass": result["pass"], "contact_s": result["contact_s"],
                      "faults": result["faults"]}))


if __name__ == "__main__":
    main()
