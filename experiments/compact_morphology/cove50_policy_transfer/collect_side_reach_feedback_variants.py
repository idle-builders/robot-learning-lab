"""Collect selected-body feedback reach teachers over a declared development plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.collect_side_reach_native import (
    GEOMETRY, HEAVY, NOMINAL, ROOT,
)


PROBE = Path(__file__).with_name("probe_side_reach_native.py")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.out = args.out.resolve()
    plan = json.loads(args.plan.read_text())
    if sha(ROOT / "robot/ape-current-design.json") != plan["selected_pointer_sha256"]:
        raise ValueError("Selected body pointer differs from declared plan")
    if sha(NOMINAL) != plan["selected_nominal_body_sha256"]:
        raise ValueError("Nominal body differs from declared plan")
    if sha(args.checkpoint) != plan["teacher_checkpoint_sha256"]:
        raise ValueError("Feedback teacher differs from declared plan")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        name = case["case"]
        if case["split"] not in ("train", "validation") or case["seed"] >= 9000:
            raise ValueError(f"Invalid development split or seed: {name}")
        body = HEAVY if case["body"] == "heavy_lowgrip" else NOMINAL
        result = args.out / f"{name}.json"
        trace = args.out / f"{name}.npz"
        log = args.out / f"{name}.log"
        sx, sy = case.get("start_xy_delta_m", [0, 0])
        tx, ty = case.get("toy_xy_delta_m", [0, 0])
        dx, dy = case.get("toy_step_xy_m", [0, 0])
        command = [sys.executable, str(PROBE), "--geometry", str(GEOMETRY),
                   "--body-xml", str(body), "--out", str(result),
                   "--dataset-out", str(trace),
                   "--policy-checkpoint", str(args.checkpoint),
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--start-x-offset-m", str(.07 + sx),
                   "--start-y-offset-m", str(sy),
                   "--toy-x-offset-m", str(tx), "--toy-y-offset-m", str(ty),
                   "--toy-step-time-s", str(case.get("toy_step_time_s", -1)),
                   "--toy-step-x-m", str(dx), "--toy-step-y-m", str(dy)]
        with log.open("w") as stream:
            subprocess.run(command, check=True, cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT)
        report = json.loads(result.read_text())
        passed = (report["survived"] and
                  report["longest_continuous_finger_toy_contact_s"] >= 1 and
                  report["both_feet_grounded_fraction"] >= .99 and
                  report["rated_leg_fraction"] <= .1 and
                  report["max_hand_ground_force_n"] < 1)
        row = {**case, "pass": passed,
               "contact_s": report["longest_continuous_finger_toy_contact_s"],
               "survived": report["survived"],
               "both_feet_fraction": report["both_feet_grounded_fraction"],
               "rated_leg_fraction": report["rated_leg_fraction"],
               "max_hand_ground_force_n": report["max_hand_ground_force_n"],
               "result": str(result.relative_to(ROOT)),
               "trace": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace)}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-feedback-variant-collection.v1",
                  "plan": str(args.plan.relative_to(ROOT)),
                  "plan_sha256": sha(args.plan),
                  "teacher_checkpoint": str(args.checkpoint.relative_to(ROOT)),
                  "teacher_checkpoint_sha256": sha(args.checkpoint),
                  "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
                  "nominal_body_sha256": sha(NOMINAL),
                  "heavy_body_sha256": sha(HEAVY),
                  "probe_sha256": sha(PROBE),
                  "cases": rows}
        (args.out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
        print(json.dumps({"case": name, "pass": passed, "contact_s": row["contact_s"]}), flush=True)


if __name__ == "__main__":
    main()
