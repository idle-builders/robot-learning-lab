"""Freeze four disjoint fresh V4 teacher/student pairs after opened CUDA sentinels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/compact_morphology/cove50_low_level_handoffs"
PLAN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-l4-screen-method.json"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
INDICES = (0, 5, 8, 11)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("bank", "fit_report", "student_checkpoint", "parity_report", "student_sentinel", "out"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError("Frozen score method already exists")
    plan = json.loads(PLAN.read_text())
    bank = json.loads(args.bank.read_text())
    fit = json.loads(args.fit_report.read_text())
    parity = json.loads(args.parity_report.read_text())
    sentinel = json.loads(args.student_sentinel.read_text())
    if (bank["source_coverage"] != bank["planned_source_families"] == 16
            or bank["accepted_whole_episodes"] < 6
            or fit["bank_sha256"] != sha(args.bank)
            or fit["checkpoint_sha256"] != sha(args.student_checkpoint)
            or not parity["pass"] or parity["device"] != "cuda:0"
            or sentinel["device"] != "cuda:0" or sentinel["steps"] != 1
            or sentinel["seed"] not in {item["seed"] for item in bank["entries"]}
            or sentinel["student_checkpoint_sha256"] != sha(args.student_checkpoint)
            or sentinel["max_target_mapping_gap"] > 1e-5):
        raise RuntimeError("Full source bank, fit or opened native sentinels are not ready")
    result = {
        "schema": "robot-learning-lab.broad-handoff-v4-l4-paired-score-method.v1",
        "classification": "frozen four-pair fresh CUDA development score; no mechanical transfer",
        "screen_plan_sha256": sha(PLAN),
        "paired_runner_sha256": sha(BASE / "run_v4_paired_score.py"),
        "source_evaluator_sha256": sha(BASE / "evaluate_walk_squat_arms.py"),
        "candidate_evaluator_sha256": sha(BASE / "evaluate_v4_shared_bc.py"),
        "action_map_sha256": sha(BASE / "v4_action_map.py"),
        "bank_builder_sha256": sha(BASE / "build_v4_command_bank.py"),
        "fit_source_sha256": sha(BASE / "train_v4_shared_bc.py"),
        "body": reference(BODY),
        "actors": plan["actors"],
        "bank": reference(args.bank),
        "fit_report": reference(args.fit_report),
        "student_checkpoint": reference(args.student_checkpoint),
        "parity_report": reference(args.parity_report),
        "student_sentinel": reference(args.student_sentinel),
        "cases": [plan["reserved_fresh_evaluation"][i] for i in INDICES],
        "matched_entry_tolerance": 1e-4,
        "minimum_pair_start_reserve_s": 500,
        "minimum_visit_start_reserve_s": 240,
        "time_limit_s": 3600,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"method_sha256": sha(args.out), "cases": result["cases"]}, indent=2))


if __name__ == "__main__":
    main()
