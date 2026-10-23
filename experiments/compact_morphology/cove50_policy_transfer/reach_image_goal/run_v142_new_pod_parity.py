"""Compare an opened saved CUDA start with the V142 replacement GPU host."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v95_native_case as baseline,
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v139/reserved-contact-score-plan.json"
MATRIX = BASE / "v139/native-reserved-contact/matrix.json"
OUT = BASE / "v142/new-pod-cuda-parity"
SUMMARY = OUT / "parity-summary.json"
SEED = 24119


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["plan_sha256"] != sha(PLAN) or
            matrix["score_gate_met"]):
        raise ValueError("Exact completed failed score required")
    case = next(case for case in plan["cases"] if case["seed"] == SEED)
    row = next(row for row in matrix["cases"] if row["seed"] == SEED)
    saved_path = ROOT / row["baseline_result"]
    saved_trace = ROOT / row["baseline_trace"]
    if not saved_path.is_file() or not saved_trace.is_file():
        raise ValueError("Saved CUDA reference report/trace required")
    OUT.mkdir(parents=True, exist_ok=True)
    fresh = baseline.run_case({**case, "id": "newpod-parity-24119"},
                              {"source_sha256": plan["source_sha256"]}, OUT)
    saved = json.loads(saved_path.read_text())
    replay = json.loads((ROOT / fresh["result"]).read_text())
    source_match = all(saved[key] == replay[key] for key in (
        "evaluator_sha256", "body_sha256", "selected_pointer_sha256",
        "reach_checkpoint_sha256", "selection_mode", "adaptive_variant"))
    entry_a = score_runner.observations.observed_features(saved)
    entry_b = score_runner.observations.observed_features(replay)
    entry_faults = score_runner.entry_parity(entry_a, entry_b)
    with np.load(saved_trace) as a, np.load(ROOT / fresh["trace"]) as b:
        qa, qb = a["qpos"], b["qpos"]
        if qa.shape != qb.shape:
            raise ValueError("CUDA parity trajectories have different shapes")
        dt = float(a["step_dt_s"])
        index = round(saved["reach_start_s"] / dt)
        entry_qpos_max_abs = float(np.max(np.abs(qa[index] - qb[index])))
        trajectory_qpos_max_abs = float(np.max(np.abs(qa - qb)))
    contact_delta = round(fresh["contact_s"] - row["baseline_contact_s"], 3)
    passed = bool(source_match and not entry_faults and
                  fresh["pass"] == row["baseline_native_pass"] and
                  abs(contact_delta) <= .04 and
                  entry_qpos_max_abs <= .0001 and
                  trajectory_qpos_max_abs <= .001)
    result = {
        "schema": "robot-learning-lab.cove50-v142-new-pod-cuda-parity.v1",
        "role": "opened_replacement_cuda_runtime_check_before_V142",
        "body_role": plan["body_role"],
        "pod_id": "l5ts5gmcflpefh",
        "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
        "gpu": "NVIDIA GeForce RTX 4090",
        "source_sha256": {"runner": sha(Path(__file__)),
                          "score_plan": sha(PLAN), "score_matrix": sha(MATRIX),
                          "saved_report": sha(saved_path),
                          "saved_trace": sha(saved_trace),
                          "replacement_report": sha(ROOT / fresh["result"]),
                          "replacement_trace": sha(ROOT / fresh["trace"])},
        "source_match": source_match,
        "entry_parity_faults": entry_faults,
        "contact_saved_s": row["baseline_contact_s"],
        "contact_replacement_s": fresh["contact_s"],
        "contact_delta_s": contact_delta,
        "full_gate_saved": row["baseline_native_pass"],
        "full_gate_replacement": fresh["pass"],
        "entry_qpos_max_abs": entry_qpos_max_abs,
        "trajectory_qpos_max_abs": trajectory_qpos_max_abs,
        "parity_gate_met": passed,
        "saved_result": row["baseline_result"],
        "replacement_result": fresh["result"],
        "replacement_trace": fresh["trace"],
        "interpretation": ("Opened CUDA runtime parity before V142; even a "
                           "pass cannot repair V139 or qualify the body."),
    }
    SUMMARY.write_text(json.dumps(result, indent=2) + "\n")
    print(f"V142 new-pod parity {'met' if passed else 'FAILED'}: "
          f"contact {row['baseline_contact_s']:.2f}→{fresh['contact_s']:.2f}s, "
          f"entry qpos gap {entry_qpos_max_abs:.7f}, "
          f"trajectory gap {trajectory_qpos_max_abs:.7f}", flush=True)


if __name__ == "__main__":
    main()
