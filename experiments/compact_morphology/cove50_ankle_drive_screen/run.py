"""Paired native replay for a motor-law-only Cove ankle sensitivity.

The selected CAD-derived body and policy action interface stay identical. The
ST-3250 coupon excludes its extra mass, packaging and electrical implications.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-ankle-drive-screen-20261010"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
STRESS_BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
FORWARD = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
TURN = ROOT / "artifacts/ape-cove50-turn-recovery-20261008/turneffort25-model950.pt"

CASES = {
    "forward": (FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", 5501, 0.35, 0.0),
    "turn-left": (TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", 5502, 0.0, 0.4),
    "turn-right": (TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", 5502, 0.0, -0.4),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(name: str, model: str, stress: bool) -> None:
    checkpoint, task, seed, vx, omega = CASES[name]
    body = STRESS_BODY if stress else BODY
    case_id = f"{name}-{model}" + ("-stress" if stress else "")
    command = [sys.executable, str(EVALUATOR), str(checkpoint),
               "--task", task, "--out", str(OUT / f"{case_id}.json"),
               "--diagnostic-trace", str(OUT / f"{case_id}.npz"),
               "--video", str(OUT / f"{case_id}.mp4"),
               "--worlds", "4", "--seed", str(seed),
               "--body-xml", str(body), "--physics-model", "frame50",
               "--ankle-motor-model", model,
               "--yaw-only-head", "--arms-down", "--physics-dt", "0.002",
               "--velocity-x", str(vx), "--velocity-y", "0.0",
               "--angular-z", str(omega)]
    if name == "forward":
        command += ["--ankle-target-margin-rad", "0.02"]
    if stress:
        command += ["--motor-rail-v", "10.0", "--command-delay-ms", "10.0",
                    "--contact-slide-friction", "0.55"]
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "case": case_id, "motor_law": model, "stress": stress,
        "identity_boundary": "Only the two ankle catalog motor laws change; body mass, case geometry and all contacts remain selected. Candidate is not build-qualified.",
        "command": command,
        "sources": {"pointer": sha(ROOT / "robot/ape-current-design.json"),
                    "catalog": sha(ROOT / "robot/actuator-catalog.yaml"),
                    "body": sha(body), "checkpoint": sha(checkpoint),
                    "evaluator": sha(EVALUATOR),
                    "motor_model": sha(ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py")},
        "definition": {"actor": name, "seed": seed, "vx": vx, "omega": omega,
                       "ankle_clip": 0.02 if name == "forward" else None},
    }
    (OUT / f"{case_id}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (OUT / f"{case_id}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(case_id, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("selected", "st3250"), required=True)
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("cases", nargs="*", choices=tuple(CASES))
    args = parser.parse_args()
    for case in args.cases or CASES:
        run(case, args.model, args.stress)
