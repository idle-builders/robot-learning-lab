"""Fresh, matched native Cove effort traces on a temporary GPU worker.

This invokes the existing MJLab evaluator, retaining its JSON outcomes and
synchronized NPZ diagnostics. Copy outputs back locally before deleting the pod.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
OUT = ROOT / "artifacts/ape-cove50-effort-envelope"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
STRESS_BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
FORWARD = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
TURN = ROOT / "artifacts/ape-cove50-turn-recovery-20261008/turneffort25-model950.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CASES = {
    "passive-zero": {"actor": "forward", "seed": 5501, "vx": 0.0, "omega": 0.0,
                     "zero": True},
    "forward-nominal": {"actor": "forward", "seed": 5501, "vx": 0.35, "omega": 0.0,
                        "ankle_clip": 0.02, "video": True},
    "forward-slow": {"actor": "forward", "seed": 5501, "vx": 0.20, "omega": 0.0,
                     "ankle_clip": 0.02},
    "forward-stop-early": {"actor": "forward", "seed": 5501, "vx": 0.35, "omega": 0.0,
                           "ankle_clip": 0.02, "stop": 2.5},
    "forward-stop-late": {"actor": "forward", "seed": 5501, "vx": 0.35, "omega": 0.0,
                          "ankle_clip": 0.02, "stop": 3.0},
    "forward-stress": {"actor": "forward", "seed": 5501, "vx": 0.35, "omega": 0.0,
                       "ankle_clip": 0.02, "stress": True, "video": True},
    "turn-left-nominal": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": 0.4,
                          "video": True},
    "turn-right-nominal": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": -0.4,
                           "video": True},
    "turn-left-stress": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": 0.4,
                         "stress": True, "video": True},
    "turn-right-stress": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": -0.4,
                          "stress": True},
    "turn-left-ankle-clip": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": 0.4,
                              "ankle_clip": 0.02},
    "turn-right-ankle-clip": {"actor": "turn", "seed": 5502, "vx": 0.0, "omega": -0.4,
                               "ankle_clip": 0.02},
}


def run_case(case_name: str) -> None:
    case = CASES[case_name]
    actor = case["actor"]
    body = STRESS_BODY if case.get("stress") else BODY
    cmd = [sys.executable, str(EVALUATOR), str(FORWARD if actor == "forward" else TURN),
           "--task", "Mjlab-Velocity-Flat-Ape-Elbows" if actor == "forward"
           else "Mjlab-Velocity-Flat-Ape-Elbows-Planar",
           "--out", str(OUT / f"{case_name}.json"),
           "--diagnostic-trace", str(OUT / f"{case_name}.npz"),
           "--worlds", "4", "--seed", str(case["seed"]),
           "--body-xml", str(body), "--physics-model", "frame50",
           "--yaw-only-head", "--arms-down", "--physics-dt", "0.002",
           "--velocity-x", str(case["vx"]), "--velocity-y", "0.0",
           "--angular-z", str(case["omega"])]
    if case.get("ankle_clip"):
        cmd += ["--ankle-target-margin-rad", str(case["ankle_clip"])]
    if case.get("zero"):
        cmd += ["--zero-action"]
    if case.get("stop"):
        cmd += ["--stop-at-s", str(case["stop"])]
    if case.get("stress"):
        cmd += ["--motor-rail-v", "10.0", "--command-delay-ms", "10.0",
                "--contact-slide-friction", "0.55"]
    if case.get("video"):
        cmd += ["--video", str(OUT / f"{case_name}.mp4")]
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {"case": case_name, "definition": case, "command": cmd,
            "sources": {"pointer": sha(ROOT / "robot/ape-current-design.json"),
                        "catalog": sha(ROOT / "robot/actuator-catalog.yaml"),
                        "body": sha(body), "checkpoint": sha(FORWARD if actor == "forward" else TURN),
                        "evaluator": sha(EVALUATOR)}}
    (OUT / f"{case_name}-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (OUT / f"{case_name}.log").open("w") as log:
        subprocess.run(cmd, check=True, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    print(case_name, "complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", choices=list(CASES),
                        help="Specific cases to run; omit for all")
    args = parser.parse_args()
    for case in args.cases or CASES:
        run_case(case)


if __name__ == "__main__":
    main()
