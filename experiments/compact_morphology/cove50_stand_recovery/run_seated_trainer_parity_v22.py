"""Run the frozen V22 native seated reference and parked controls."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-trainer-parity-v22"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed: int, route: str) -> tuple[int, str, int]:
    stem = ART / f"seed{seed}-{route}"
    if stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file():
        return seed, route, 0
    command = [
        sys.executable, str(EVALUATOR),
        "--out", str(stem.with_suffix(".json")),
        "--native-trajectory-out", str(stem.with_suffix(".npz")),
        "--seed", str(seed), "--device", "cpu",
        "--cove50-body-xml", str(BODY), "--mode", "cycle",
        "--start-squat", "--recover-from-start", "--seated-rise-stage",
        "--randomized-physics", "--scripted-target-probe",
    ]
    if route == "parked":
        command += ["--scripted-min-blend", "0.5"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    with stem.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return seed, route, result.returncode


def main() -> None:
    plan = json.loads((ART / "plan.json").read_text())
    for label, path in (
        ("body_sha256", BODY),
        ("trainer_source_sha256", ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py"),
        ("evaluator_source_sha256", EVALUATOR),
        ("task_source_sha256", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py"),
        ("motor_source_sha256", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py"),
    ):
        if sha(path) != plan[label]:
            raise ValueError(f"Frozen V22 source changed: {label}")
    jobs = [(seed, route) for seed in plan["seeds"] for route in plan["routes"]]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for future in as_completed(futures):
            seed, route, code = future.result()
            print(seed, route, code, flush=True)
            if code:
                raise RuntimeError(f"V22 native run failed: {seed} {route}")


if __name__ == "__main__":
    main()
