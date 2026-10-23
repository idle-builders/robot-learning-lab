"""Paired CUDA feasibility diagnostic on the six already-opened V25 starts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-reference-cuda-v26"
V25 = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-ppo-pilot-v25"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py"
CHECKPOINT = V25 / "checkpoints/model_25.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed: int, route: str, timeout_s: int) -> dict:
    stem = ART / f"seed{seed}-{route}"
    command = [
        sys.executable, str(EVALUATOR), "--out", str(stem.with_suffix(".json")),
        "--native-trajectory-out", str(stem.with_suffix(".npz")),
        "--seed", str(seed), "--device", "cuda:0",
        "--cove50-body-xml", str(BODY), "--mode", "cycle",
        "--start-squat", "--recover-from-start", "--seated-rise-stage",
        "--randomized-physics",
    ]
    if route == "model25":
        command.insert(2, str(CHECKPOINT))
    elif route == "scripted":
        command.append("--scripted-target-probe")
    else:
        raise ValueError(route)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    with stem.with_suffix(".log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout_s,
                                   check=False)
    if completed.returncode:
        raise RuntimeError(f"{seed} {route}: native evaluator exit {completed.returncode}")
    return json.loads(stem.with_suffix(".json").read_text())


def compare(seed: int, learned: dict, scripted: dict) -> dict:
    original = json.loads((V25 / f"checkpoint-25-seed{seed}.json").read_text())
    for report in (learned, scripted):
        if (report["device"] != "cuda:0" or report["seed"] != seed
                or report["body_sha256"] != sha(BODY)
                or not report["seated_rise_stage"]
                or not report["randomized_physics"]):
            raise ValueError(f"{seed}: wrong CUDA task/body/seed")
        if not all(isinstance(x, int) and 0 <= x <= 4
                   for x in report["initial_motors"]["delay_lag_steps"]):
            raise ValueError(f"{seed}: actual per-motor lag draw missing")
    fields = ("initial_state_sha256", "initial_qpos", "initial_qvel",
              "initial_predicted_sole_gap_m")
    motors = ("target_names", "rated_effort_nm", "effort_factor",
              "stiffness_factor", "damping_factor")
    for field in fields:
        if learned[field] != scripted[field] or learned[field] != original[field]:
            raise ValueError(f"{seed}: matched start differs in {field}")
    for field in ("torso_mass_kg", "sole_slide_friction"):
        if (learned["initial_physics"][field] != scripted["initial_physics"][field]
                or learned["initial_physics"][field] != original["initial_physics"][field]):
            raise ValueError(f"{seed}: original sampled physics differs in {field}")
    for field in ("robot_body_names", "robot_body_mass_kg",
                  "robot_body_inertia_kg_m2", "robot_body_ipos_m",
                  "robot_body_iquat_wxyz"):
        if (not learned["initial_physics"][field]
                or learned["initial_physics"][field] != scripted["initial_physics"][field]):
            raise ValueError(f"{seed}: full sampled robot body physics differs in {field}")
    for field in motors:
        if (learned["initial_motors"][field] != scripted["initial_motors"][field]
                or learned["initial_motors"][field] != original["initial_motors"][field]):
            raise ValueError(f"{seed}: matched motor draw differs in {field}")
    if (learned["initial_motors"]["delay_lag_steps"] !=
            scripted["initial_motors"]["delay_lag_steps"]):
        raise ValueError(f"{seed}: per-motor lag draw differs between new routes")
    return {
        "seed": seed,
        "initial_state_sha256": learned["initial_state_sha256"],
        "matched_original_v25_recorded_start_physics_motors": True,
        "matched_new_routes_all_robot_mass_inertial_fields": True,
        "matched_new_route_per_motor_lag_draw": True,
        "new_lag_steps": learned["initial_motors"]["delay_lag_steps"],
        "model25_report_sha256": sha(ART / f"seed{seed}-model25.json"),
        "scripted_report_sha256": sha(ART / f"seed{seed}-scripted.json"),
    }


def main() -> None:
    plan_path = ART / "plan.json"
    plan = json.loads(plan_path.read_text())
    source = plan["source_sha256"]
    for label, path in (
        ("body", BODY), ("evaluator", EVALUATOR),
        ("trainer", ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py"),
        ("task", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py"),
        ("frame50", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py"),
        ("runner", Path(__file__)), ("checkpoint", CHECKPOINT),
    ):
        if sha(path) != source[label]:
            raise ValueError(f"V26 frozen {label} source changed")
    import mjlab_microduck.tasks.mdp as upstream_mdp
    if sha(Path(upstream_mdp.__file__)) != source["upstream_mdp"]:
        raise ValueError("V26 installed upstream MDP differs from V25")
    if not sys.version.startswith(plan["runtime_versions"]["python"]):
        raise ValueError("V26 Python runtime differs from freeze")
    for package, expected in plan["runtime_versions"].items():
        if package != "python" and importlib.metadata.version(package) != expected:
            raise ValueError(f"V26 {package} runtime differs from freeze")
    if sha(V25 / "plan.json") != plan["v25_plan_sha256"]:
        raise ValueError("V25 pilot plan changed")
    seeds = plan["opened_development_seeds"]
    if seeds != [2231, 2232, 2233, 2234, 2235, 2236]:
        raise ValueError("V26 may use only the six already-opened development starts")
    ART.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    rows = []
    for seed in seeds:
        remaining = plan["max_worker_wall_s"] - (time.monotonic() - started)
        if remaining < 1:
            raise TimeoutError("V26 total worker wall cap reached")
        learned = run(seed, "model25", min(plan["max_route_wall_s"], int(remaining)))
        remaining = plan["max_worker_wall_s"] - (time.monotonic() - started)
        if remaining < 1:
            raise TimeoutError("V26 total worker wall cap reached")
        scripted = run(seed, "scripted", min(plan["max_route_wall_s"], int(remaining)))
        row = compare(seed, learned, scripted)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (ART / "source-pair-ledger.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-seated-cuda-reference-pairs.v26",
        "plan_sha256": sha(plan_path), "rows": rows,
        "claim_limit": "Opened development diagnostic only; not a frozen score or learned policy qualification.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
