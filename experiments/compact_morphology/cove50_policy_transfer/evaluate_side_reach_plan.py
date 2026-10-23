"""Fresh native saved-checkpoint reach replay for predeclared scene lists."""

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
    parser.add_argument("--cases-key", choices=("saved_actor_development_gate", "new_reserved_cases",
                                                 "on_policy_cases", "development_cases", "reserved_cases"),
                        required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-checkpoint", type=Path,
                        help="Save frozen feedback-teacher labels on the actor's visited states")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.plan = args.plan.resolve()
    args.checkpoint = args.checkpoint.resolve()
    if args.label_checkpoint:
        args.label_checkpoint = args.label_checkpoint.resolve()
    args.out = args.out.resolve()
    plan = json.loads(args.plan.read_text())
    if sha(ROOT / "robot/ape-current-design.json") != plan["selected_pointer_sha256"]:
        raise ValueError("Selected pointer differs from declared plan")
    if sha(NOMINAL) != plan["selected_nominal_body_sha256"]:
        raise ValueError("Nominal body differs from declared plan")
    if sha(args.checkpoint) != plan["frozen_checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from declared plan")
    if args.label_checkpoint and sha(args.label_checkpoint) != plan["label_feedback_checkpoint_sha256"]:
        raise ValueError("Label teacher differs from declared plan")
    if args.cases_key == "on_policy_cases" and not args.label_checkpoint:
        raise ValueError("On-policy collection requires a frozen label checkpoint")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan[args.cases_key]:
        name = case["case"]
        if args.cases_key in ("new_reserved_cases", "reserved_cases") and case["seed"] < 9000:
            raise ValueError(f"Reserved case has a development seed: {name}")
        if args.cases_key == "on_policy_cases" and case["seed"] >= 9000:
            raise ValueError(f"On-policy development case has a reserved seed: {name}")
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
                   "--hand", case.get("hand", plan.get("hand", "right")),
                   "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                   "--delay-ms", str(case["delay_ms"]),
                   "--start-x-offset-m", str(.07 + sx),
                   "--start-y-offset-m", str(sy),
                   "--toy-x-offset-m", str(tx), "--toy-y-offset-m", str(ty),
                   "--toy-step-time-s", str(case.get("toy_step_time_s", -1)),
                   "--toy-step-x-m", str(dx), "--toy-step-y-m", str(dy)]
        for joint, value in plan.get("joint_target_offsets_rad", {}).items():
            command += ["--joint-target-offset", joint, str(value)]
        if args.label_checkpoint:
            command += ["--label-feedback-checkpoint", str(args.label_checkpoint)]
        with log.open("w") as stream:
            subprocess.run(command, check=True, cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT)
        report = json.loads(result.read_text())
        contact_s = report["longest_continuous_finger_toy_contact_s"]
        both_feet = report["both_feet_grounded_fraction"]
        rated_leg = report["rated_leg_fraction"]
        hand_force = report["max_hand_ground_force_n"]
        passed = (report["survived"] and contact_s >= 1 and both_feet >= .99
                  and rated_leg <= .1 and hand_force < 1)
        stable_for_labels = (report["survived"] and both_feet >= .99 and
                             rated_leg <= .1 and hand_force < 1)
        row = {**case, "pass": passed, "contact_s": contact_s,
               "stable_for_labels": stable_for_labels,
               "survived": report["survived"],
               "both_feet_fraction": both_feet,
               "rated_leg_fraction": rated_leg,
               "max_hand_ground_force_n": hand_force,
               "result": str(result.relative_to(ROOT)),
               "trace": str(trace.relative_to(ROOT)),
               "trace_sha256": sha(trace)}
        rows.append(row)
        matrix = {"schema": "robot-learning-lab.cove50-reach-saved-actor-plan-evaluation.v1",
                  "cases_key": args.cases_key,
                  "plan": str(args.plan.relative_to(ROOT)), "plan_sha256": sha(args.plan),
                  "checkpoint": str(args.checkpoint.relative_to(ROOT)),
                  "checkpoint_sha256": sha(args.checkpoint),
                  "label_checkpoint": (str(args.label_checkpoint.relative_to(ROOT))
                                       if args.label_checkpoint else None),
                  "label_checkpoint_sha256": sha(args.label_checkpoint) if args.label_checkpoint else None,
                  "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
                  "nominal_body_sha256": sha(NOMINAL), "heavy_body_sha256": sha(HEAVY),
                  "probe_sha256": sha(PROBE), "cases": rows}
        (args.out / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
        print(json.dumps({"case": name, "pass": passed, "contact_s": contact_s,
                          "survived": report["survived"],
                          "rated_leg_fraction": rated_leg}), flush=True)


if __name__ == "__main__":
    main()
