"""Fresh-process native contact matrix for the saved Cove side-reach BC actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from experiments.compact_morphology.cove50_policy_transfer.collect_side_reach_native import (
    CASES, GEOMETRY, HEAVY, NOMINAL, REFERENCE, ROOT,
)


PROBE = Path(__file__).with_name("probe_side_reach_native.py")
DEFAULT_OUT = ROOT / "artifacts/ape-cove50-side-reach-20261009/reach-bc-v1/evaluation"
DEFAULT_CHECKPOINT = DEFAULT_OUT.parent / "actor.pt"
EVAL_CASES = ("held-toy-diagonal-a", "held-toy-diagonal-b",
              "held-start-diagonal", "held-heavy11-shift",
              "held-heavy10-boundary")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--teacher", action="store_true", help="Native authored reference negative control")
    parser.add_argument("cases", nargs="*", choices=EVAL_CASES)
    args = parser.parse_args()
    selected = args.cases or EVAL_CASES
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in selected:
        split, heavy, rail, delay, sx, sy, tx, ty, seed = CASES[name]
        assert split == "held"
        body = HEAVY if heavy else NOMINAL
        result = out / f"{name}.json"
        trace = out / f"{name}.npz"
        command = [sys.executable, str(PROBE), "--geometry", str(GEOMETRY),
                   "--body-xml", str(body), "--out", str(result),
                   "--dataset-out", str(trace),
                   "--seed", str(seed), "--rail-v", str(rail),
                   "--delay-ms", str(delay),
                   "--start-x-offset-m", str(.07 + sx),
                   "--start-y-offset-m", str(sy),
                   "--toy-x-offset-m", str(tx),
                   "--toy-y-offset-m", str(ty)]
        if args.teacher:
            command += ["--xy-correction-reference", str(REFERENCE)]
        else:
            command += ["--policy-checkpoint", str(args.checkpoint)]
        with (out / f"{name}.log").open("w") as log:
            subprocess.run(command, check=True, cwd=ROOT, stdout=log,
                           stderr=subprocess.STDOUT)
        report = json.loads(result.read_text())
        passed = (report["survived"] and
                  report["longest_continuous_finger_toy_contact_s"] >= 1.0 and
                  report["both_feet_grounded_fraction"] >= .99 and
                  report["rated_leg_fraction"] <= .1 and
                  report["max_hand_ground_force_n"] < 1.0)
        row = {"case": name, "body": "heavy_lowgrip" if heavy else "nominal",
               "rail_v": rail, "delay_ms": delay,
               "start_xy_delta_m": [sx, sy], "toy_xy_delta_m": [tx, ty],
               "seed": seed, "pass": passed,
               "contact_s": report["longest_continuous_finger_toy_contact_s"],
               "survived": report["survived"],
               "both_feet_fraction": report["both_feet_grounded_fraction"],
               "rated_leg_fraction": report["rated_leg_fraction"],
               "max_hand_ground_force_n": report["max_hand_ground_force_n"],
               "result": str(result.relative_to(ROOT)),
               "trace": str(trace.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = {"schema": "robot-learning-lab.cove50-reach-bc-evaluation.v1",
               "method": ("authored_native_teacher" if args.teacher else
                          "fresh_process_closed_loop_native_MJLab"),
               "checkpoint": None if args.teacher else str(args.checkpoint.relative_to(ROOT)),
               "checkpoint_sha256": None if args.teacher else sha(args.checkpoint),
               "nominal_body_sha256": sha(NOMINAL),
               "heavy_body_sha256": sha(HEAVY),
               "probe_sha256": sha(PROBE),
               "success_count": sum(row["pass"] for row in rows),
               "case_count": len(rows), "cases": rows,
               "limitations": ["Prepositioned attained upright start; no locomotion handoff",
                               "MJLab ee_to_cube is simulator truth, not perception",
                               "Catalog motor/contact properties are provisional"]}
    (out / "matrix.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
