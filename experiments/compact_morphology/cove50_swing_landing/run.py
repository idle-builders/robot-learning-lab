"""Paired native forward screen for inward swing-foot landing targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-swing-landing-20261010"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
STRESS_BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"
CHECKPOINT = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
OFFSETS = {"selected": 0.0, "inward05": 0.05, "inward10": 0.10}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(case: str, stress: bool) -> None:
    body = STRESS_BODY if stress else BODY
    case_id = case + ("-stress" if stress else "")
    command = [sys.executable, str(EVALUATOR), str(CHECKPOINT),
               "--task", "Mjlab-Velocity-Flat-Ape-Elbows",
               "--out", str(OUT / f"{case_id}.json"),
               "--diagnostic-trace", str(OUT / f"{case_id}.npz"),
               "--video", str(OUT / f"{case_id}.mp4"),
               "--worlds", "4", "--seed", "5501",
               "--body-xml", str(body), "--physics-model", "frame50",
               "--yaw-only-head", "--arms-down", "--physics-dt", "0.002",
               "--velocity-x", "0.35", "--velocity-y", "0.0", "--angular-z", "0.0",
               "--ankle-target-margin-rad", "0.02"]
    if OFFSETS[case]:
        command += ["--swing-hip-roll-inward-rad", str(OFFSETS[case])]
    if stress:
        command += ["--motor-rail-v", "10.0", "--command-delay-ms", "10.0",
                    "--contact-slide-friction", "0.55"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{case_id}-plan.json").write_text(json.dumps({
        "case": case_id, "offset_rad": OFFSETS[case], "stress": stress,
        "contact_timing": "Observed one-foot contact; 40 ms first-order target entry/exit filter",
        "command": command,
        "sources": {"pointer": sha(ROOT / "robot/ape-current-design.json"),
                    "catalog": sha(ROOT / "robot/actuator-catalog.yaml"),
                    "body": sha(body), "checkpoint": sha(CHECKPOINT),
                    "evaluator": sha(EVALUATOR),
                    "motor_model": sha(ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py")},
    }, indent=2) + "\n")
    with (OUT / f"{case_id}.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
    print(case_id, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+", choices=tuple(OFFSETS))
    parser.add_argument("--stress", action="store_true")
    args = parser.parse_args()
    for name in args.cases:
        run(name, args.stress)
