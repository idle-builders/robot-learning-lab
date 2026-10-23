"""Screen one authored right-ankle reach target factor on an opened stress miss."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
OUT = ARTIFACT / "v8/conditioned-route/ankle-target-screen"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    plan = json.loads((ARTIFACT / "v8/ankle-target-screen-plan.json").read_text())
    source = json.loads((ARTIFACT / "v8/conditioned-route/stress/seed-9964.json").read_text())
    if source["seed"] != plan["development_seed"] or source["phases"]["reach"]["rated_leg_fraction"] != .5667:
        raise ValueError("Opened stress source changed")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in plan["screen_offsets_rad"]:
        label = f"seed-{plan['development_seed']}-ankle-{round(offset * 1000):03d}mrad"
        result_path, trace_path, log_path = [OUT / f"{label}.{suffix}"
                                             for suffix in ("json", "npz", "log")]
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-heavy-lowgrip.xml"),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ARTIFACT / "v7/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(plan["development_seed"]),
                   "--rail-v", "10", "--delay-ms", "10",
                   "--toy-offset-x-m", ".004", "--toy-offset-y-m", "-.004",
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9.0", "--record-leg-efforts",
                   "--right-ankle-target-offset-rad", str(offset),
                   "--out", str(result_path), "--trace", str(trace_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log_path.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        report = json.loads(result_path.read_text())
        faults = gate(report)
        ankle_index = report["leg_joint_order"].index("right_ankle_pitch")
        reach = [frame for frame in report["timeseries"] if frame["phase"] == "reach"]
        row = {"case": label, "offset_rad": offset, "pass": not faults, "faults": faults,
               "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
               "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
               "rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
               "right_ankle_peak_effort_nm": max(frame["leg_effort_nm"][ankle_index] for frame in reach),
               "right_ankle_rated_nm": report["leg_rated_limit_nm"][ankle_index],
               "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-handoff-v8-ankle-target-screen-matrix.v1",
        "status": "opened-development-diagnostic", "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
