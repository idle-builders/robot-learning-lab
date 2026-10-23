"""Replay the opened heavy/low-grip failure with per-leg-joint force telemetry."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_policy_transfer.run_squat_learned_reach_matrix import gate

ARTIFACT = ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010"
OUT = ARTIFACT / "v8/conditioned-route/stress-effort-diagnostic"
EVALUATOR = Path(__file__).with_name("evaluate_squat_reach_handoff.py")


def main() -> None:
    plan = json.loads((ARTIFACT / "v8/stress-effort-diagnostic-plan.json").read_text())
    source = json.loads((ARTIFACT / "v8/conditioned-route/stress/seed-9964.json").read_text())
    if (source["seed"] != plan["source_seed"] or source["rail_v"] != 10 or
            source["phases"]["reach"]["rated_leg_fraction"] <= .1):
        raise ValueError("Opened stress failure differs from diagnostic source")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for rail_v in plan["rail_v"]:
        label = f"seed-{plan['source_seed']}-{rail_v}v"
        result_path, trace_path, log_path = [OUT / f"{label}.{suffix}"
                                             for suffix in ("json", "npz", "log")]
        command = [sys.executable, str(EVALUATOR),
                   "--body-xml", str(ROOT / "artifacts/ape-cove50-transfer-20261008" / plan["body"]),
                   "--squat-checkpoint", str(ROOT / "artifacts/ape-cove50-squat-recovery-20261009/squat-teacher-bc-blendmask.pt"),
                   "--reach-checkpoint", str(ARTIFACT / "v7/actor.pt"),
                   "--reach-controller", "learned-residual",
                   "--reach-reference", str(ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry/development-selected.npz"),
                   "--geometry", str(ROOT / "artifacts/ape-cove50-side-reach-20261009/geometry-center.json"),
                   "--entry", "continuous", "--seed", str(plan["source_seed"]),
                   "--rail-v", str(rail_v), "--delay-ms", str(plan["delay_ms"]),
                   "--toy-offset-x-m", str(plan["toy_offset_x_m"]),
                   "--toy-offset-y-m", str(plan["toy_offset_y_m"]),
                   "--pitch-offset-rad", ".105", "--roll-clearance-rad", "-.25",
                   "--terminal-roll-offset-rad", "-.08", "--terminal-roll-mode", "entry-error-x",
                   "--reach-duration-s", "9.0", "--record-leg-efforts",
                   "--out", str(result_path), "--trace", str(trace_path)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        if not result_path.exists():
            with log_path.open("w") as stream:
                subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                               stderr=subprocess.STDOUT, check=True)
        report = json.loads(result_path.read_text())
        source_qpos_delta = None
        if rail_v == 10:
            original = np.load(ARTIFACT / "v8/conditioned-route/stress/seed-9964.npz")
            replay = np.load(trace_path)
            source_qpos_delta = float(np.max(np.abs(original["qpos"] - replay["qpos"])))
            source_action_delta = float(np.max(np.abs(original["action"] - replay["action"])))
            if (report["boundary_state_sha256"]["posture"] != source["boundary_state_sha256"]["posture"] or
                    report["first_action_sha256"] != source["first_action_sha256"] or
                    report["phases"]["reach"]["rated_leg_fraction"] !=
                    source["phases"]["reach"]["rated_leg_fraction"] or
                    report["phases"]["reach"]["continuous_finger_contact_s"] !=
                    source["phases"]["reach"]["continuous_finger_contact_s"] or
                    source_qpos_delta > 5e-5 or source_action_delta > 5e-5):
                raise ValueError("Instrumented 10 V replay changed the measured failure")
        reach = [frame for frame in report["timeseries"] if frame["phase"] == "reach"]
        joints = []
        for index, name in enumerate(report["leg_joint_order"]):
            limit = report["leg_rated_limit_nm"][index]
            rated_frames = [frame for frame in reach if frame["leg_effort_nm"][index] >= limit * .995]
            joints.append({"joint": name, "rated_limit_nm": limit,
                           "rated_fraction": round(len(rated_frames) / len(reach), 4),
                           "first_rated_s": rated_frames[0]["time_s"] if rated_frames else None,
                           "peak_effort_nm": max(frame["leg_effort_nm"][index] for frame in reach)})
        row = {"case": label, "rail_v": rail_v, "pass": not gate(report),
               "faults": gate(report),
               "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
               "both_feet_fraction": report["phases"]["reach"]["both_feet_fraction"],
               "rated_leg_fraction": report["phases"]["reach"]["rated_leg_fraction"],
               "source_max_qpos_delta": source_qpos_delta,
               "joints": joints, "result": str(result_path.relative_to(ROOT)),
               "trace": str(trace_path.relative_to(ROOT))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    (OUT / "matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-handoff-v8-stress-effort-diagnostic-matrix.v1",
        "status": "opened-development-diagnostic", "cases": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
