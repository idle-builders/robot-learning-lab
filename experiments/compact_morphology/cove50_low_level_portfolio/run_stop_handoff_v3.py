"""Run frozen paired whole-start stop-handoff visits on native CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-stop-handoff-v3"
METHOD = BASE / "method.json"
EVALUATOR = Path(__file__).with_name("evaluate_stop_handoff_v3.py")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    for name, relative in method["sources"].items():
        if digest(ROOT / relative) != method["source_sha256"][name]:
            raise ValueError(f"Frozen source changed: {name}")
    if not 0 <= args.start < len(method["cases"]) or not 1 <= args.limit <= len(method["cases"]):
        parser.error("invalid case slice")
    assets = method["assets"]
    out = BASE / "native"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": "experiments/compact_morphology/mjlab_ape/src"})
    for case in method["cases"][args.start:args.start + args.limit]:
        for variant, duration in (("baseline", None), ("early", method["stop_actor_s"])):
            stem = out / f"seed-{case['seed']}-{variant}"
            paths = {suffix: stem.with_suffix(suffix) for suffix in (".json", ".npz", ".mp4", ".log")}
            if any(path.exists() for path in paths.values()):
                raise FileExistsError(f"Original visit already attempted: {stem}")
            command = [sys.executable, str(EVALUATOR), "--body", assets["body"],
                       "--gait-actor", assets["gait"], "--stop-actor", assets["stop"],
                       "--squat-actor", assets["squat"], "--arm-actor", assets["arms"],
                       "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                       "--delay-ms", str(case["delay_ms"]), "--walk-velocity", str(case["speed_m_s"]),
                       "--walk-s", str(case["walk_s"]), "--stop-s", str(case["stop_s"]),
                       "--squat-start-s", str(case["squat_start_s"]), "--squat-s", str(case["squat_s"]),
                       "--arm-start-s", str(case["arm_start_s"]), "--arm-s", str(case["arm_s"]),
                       "--sole-friction", str(case["sole_friction"]), "--device", "cuda:0",
                       "--native-evaluation", "--out", str(paths[".json"]),
                       "--trace", str(paths[".npz"]), "--video", str(paths[".mp4"])]
            if duration is not None:
                command += ["--stop-actor-s", str(duration)]
            with paths[".log"].open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT, check=False)
            print(json.dumps({"seed": case["seed"], "variant": variant,
                              "exit_code": result.returncode, "report_exists": paths[".json"].exists()}),
                  flush=True)
            if result.returncode:
                raise RuntimeError(f"Native visit failed; inspect {paths['.log']}")
        baseline = json.loads((out / f"seed-{case['seed']}-baseline.json").read_text())
        early = json.loads((out / f"seed-{case['seed']}-early.json").read_text())
        gap = max(abs(a - b) for a, b in zip(baseline["initial_qpos"], early["initial_qpos"]))
        if gap > 1e-4:
            raise ValueError(f"Matched start differs for seed {case['seed']}: {gap}")


if __name__ == "__main__":
    main()
