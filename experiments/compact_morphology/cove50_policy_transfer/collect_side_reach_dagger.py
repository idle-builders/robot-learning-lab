"""Collect feedback labels on states visited by a mixed Cove reach actor.

These are development data, not qualification replays. The frozen feedback
teacher labels every visited native MJLab state; a deterministic Bernoulli
choice executes the teacher or the saved direct clone. Reserved 901x cases
are never included here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
NOMINAL = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
HEAVY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
GEOMETRY = ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"
CLONE = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-clone-v1/actor.pt"
FEEDBACK = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-v1/controller.pt"
OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-feedback-dagger-v1/collection"

# case, split, heavy body, rail, delay, start XY offsets, toy XY offsets,
# toy step time and XY. Seeds are development/validation and disjoint from 901x.
CASES = (
    ("nominal-a", "train", False, 12, 0, 0, 0, 0, 0, -1, 0, 0, 4301),
    ("toy-left", "train", False, 12, 0, 0, 0, -.004, .003, -1, 0, 0, 4302),
    ("toy-right", "train", False, 12, 0, 0, 0, .004, -.003, -1, 0, 0, 4303),
    ("start-left", "train", False, 12, 0, -.004, .003, 0, 0, -1, 0, 0, 4304),
    ("start-right", "train", False, 12, 0, .004, -.003, 0, 0, -1, 0, 0, 4305),
    ("step-forward", "train", False, 12, 0, 0, 0, 0, 0, 2.4, .008, -.006, 4306),
    ("step-reverse", "train", False, 12, 0, 0, 0, 0, 0, 2.1, -.006, .005, 4307),
    ("heavy11-delay", "train", True, 11, 10, 0, 0, 0, 0, 2.4, .006, -.004, 4308),
    ("val-toy", "validation", False, 12, 0, 0, 0, -.003, -.004, -1, 0, 0, 4401),
    ("val-step", "validation", False, 11, 10, 0, 0, 0, 0, 2.3, -.005, .007, 4402),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-prob", type=float, default=0.8)
    parser.add_argument("cases", nargs="*")
    args = parser.parse_args()
    selected = set(args.cases) if args.cases else {case[0] for case in CASES}
    known = {case[0] for case in CASES}
    if selected - known:
        raise ValueError(f"Unknown cases: {sorted(selected - known)}")
    if not 0 < args.teacher_prob < 1:
        raise ValueError("Mixed teacher probability must be strictly between zero and one")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in CASES:
        (name, split, heavy, rail, delay, sx, sy, tx, ty,
         step_time, step_x, step_y, seed) = case
        if name not in selected:
            continue
        body = HEAVY if heavy else NOMINAL
        result = OUT / f"{name}.json"
        trace = OUT / f"{name}.npz"
        command = [sys.executable, "-m",
                   "experiments.compact_morphology.cove50_policy_transfer.probe_side_reach_native",
                   "--geometry", str(GEOMETRY), "--body-xml", str(body),
                   "--out", str(result), "--dataset-out", str(trace),
                   "--policy-checkpoint", str(CLONE),
                   "--label-feedback-checkpoint", str(FEEDBACK),
                   "--teacher-execution-prob", str(args.teacher_prob),
                   "--seed", str(seed), "--rail-v", str(rail), "--delay-ms", str(delay),
                   "--start-x-offset-m", str(.07 + sx), "--start-y-offset-m", str(sy),
                   "--toy-x-offset-m", str(tx), "--toy-y-offset-m", str(ty),
                   "--toy-step-time-s", str(step_time),
                   "--toy-step-x-m", str(step_x), "--toy-step-y-m", str(step_y)]
        with (OUT / f"{name}.log").open("w") as log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report = json.loads(result.read_text())
        rows.append({
            "case": name, "split": split, "body": "heavy_lowgrip" if heavy else "nominal",
            "rail_v": rail, "delay_ms": delay, "seed": seed,
            "start_xy_delta_m": [sx, sy], "toy_xy_delta_m": [tx, ty],
            "toy_step": [step_time, step_x, step_y],
            "survived": report["survived"], "duration_s": report["duration_s"],
            "contact_s": report["longest_continuous_finger_toy_contact_s"],
            "both_feet_fraction": report["both_feet_grounded_fraction"],
            "rated_leg_fraction": report["rated_leg_fraction"],
            "hand_ground_force_n": report["max_hand_ground_force_n"],
            "teacher_executed_fraction": report["teacher_executed_fraction"],
            "trace": str(trace.relative_to(ROOT)), "trace_sha256": sha(trace),
            "result": str(result.relative_to(ROOT)), "result_sha256": sha(result),
        })
        print(json.dumps(rows[-1]), flush=True)
    matrix = {
        "schema": "robot-learning-lab.cove50-reach-feedback-dagger-collection.v1",
        "method": "native_mixed_feedback_teacher_and_direct_bc_rollout",
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "nominal_body_sha256": sha(NOMINAL), "heavy_body_sha256": sha(HEAVY),
        "clone_sha256": sha(CLONE), "feedback_teacher_sha256": sha(FEEDBACK),
        "teacher_execution_probability": args.teacher_prob,
        "reserved_evaluation_seeds": [9011, 9012, 9013, 9014, 9015],
        "note": "Mixed rollouts collect corrective labels on visited states. Teacher intervention means survival/contact here is not an autonomous actor success.",
        "cases": rows,
    }
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")


if __name__ == "__main__":
    main()
