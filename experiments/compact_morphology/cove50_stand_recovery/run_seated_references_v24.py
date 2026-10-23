"""Run frozen V24 seated environment references on fresh CPU start families."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-stand-recovery/seated-rise-reset-repair-v24"
OUT = ART / "fresh-references"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(seed: int, route: str) -> tuple[int, str, int]:
    stem = OUT / f"seed{seed}-{route}"
    if stem.with_suffix(".json").is_file() and stem.with_suffix(".npz").is_file():
        return seed, route, 0
    command = [
        sys.executable, str(EVALUATOR), "--out", str(stem.with_suffix(".json")),
        "--native-trajectory-out", str(stem.with_suffix(".npz")),
        "--seed", str(seed), "--device", "cpu",
        "--cove50-body-xml", str(BODY), "--mode", "cycle",
        "--start-squat", "--recover-from-start", "--seated-rise-stage",
        "--randomized-physics",
    ]
    if route == "scripted":
        command.append("--scripted-target-probe")
    elif route == "parked":
        command += ["--scripted-target-probe", "--scripted-min-blend", "0.5"]
    elif route == "zero":
        command.append("--zero-actions")
    else:
        raise ValueError(route)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    with stem.with_suffix(".log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    return seed, route, result.returncode


def main() -> None:
    plan = json.loads((ART / "plan.json").read_text())
    for name, path in (
        ("body_sha256", BODY),
        ("train_posture_arm.py", ROOT / "experiments/compact_morphology/cove50_policy_transfer/train_posture_arm.py"),
        ("evaluate_posture.py", EVALUATOR),
        ("task.py", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/task.py"),
        ("frame50.py", ROOT / "experiments/compact_morphology/mjlab_ape/src/ape_walk/physics/frame50.py"),
    ):
        expected = plan[name] if name == "body_sha256" else plan["source_sha256"][name]
        if sha(path) != expected:
            raise ValueError(f"Frozen V24 source changed: {name}")
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(seed, route) for seed in plan["fresh_environment_references"]["seeds"]
            for route in ("scripted", "parked")]
    jobs.append((plan["fresh_environment_references"]["separate_zero_seed"], "zero"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run, *job) for job in jobs]
        for future in as_completed(futures):
            seed, route, code = future.result()
            print(seed, route, code, flush=True)
            if code:
                raise RuntimeError(f"V24 native run failed: {seed} {route}")


if __name__ == "__main__":
    main()
