"""Run the predeclared native boundary matrix after nominal qualification."""

from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    plan = json.loads((ARTIFACT / "v8/conditioned-stress-plan.json").read_text())
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(ARTIFACT / "v7/actor.pt") != plan["actor_sha256"]:
        raise ValueError("Saved actor changed after stress plan")
    bodies = ROOT / "artifacts/ape-cove50-transfer-20261008"
    if (sha(bodies / "cove50-direct-training.xml") != plan["nominal_body_sha256"] or
            sha(bodies / "cove50-direct-heavy-lowgrip.xml") != plan["heavy_lowgrip_body_sha256"]):
        raise ValueError("Body source changed after stress plan")
    nominal = ARTIFACT / "v8/conditioned-route"
    for split in ("reference", "development", "reserved"):
        matrix = json.loads((nominal / split / "matrix.json").read_text())
        if not matrix["all_pass"]:
            raise ValueError(f"{split} gate failed; preserve stress seeds")
    out = nominal / "stress"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in plan["cases"]:
        label = f"seed-{case['seed']}"
        body = ("cove50-direct-training.xml" if case["body"] == "nominal" else
                "cove50-direct-heavy-lowgrip.xml")
        result_path, trace_path, log_path = [out / f"{label}.{suffix}"
                                             for suffix in ("json", "npz", "log")]
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008" / body),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ARTIFACT / "v7/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                   "--toy-offset-x-m", str(case["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(case["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9.0",
                   "--out", str(result_path), "--trace", str(trace_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with log_path.open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        report = json.loads(result_path.read_text())
        faults = gate(report)
        row = {"case": label, **case, "pass": not faults, "faults": faults,
               "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
               "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
               "rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
               "self_body_contact_fraction": report["phases"]["reach"]["self_body_contact_fraction"],
               "max_abs_unbounded_action": report["max_abs_unbounded_action"],
               "route_entry_error_x_m": report["route_entry_error_x_m"],
               "applied_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
               "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (out / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-squat-handoff-v8-conditioned-stress-matrix.v1",
        "all_pass": all(row["pass"] for row in rows), "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
