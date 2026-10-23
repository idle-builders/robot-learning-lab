"""Run one frozen varied whole-start portfolio screen on native CUDA.

This orchestrates original single-world visits for physical scoring. It does
not train a policy or turn camera/render throughput into a learner benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-low-level-training-20261015/modular-portfolio-breadth-v2"
METHOD = BASE / "method.json"
SCRIPT = Path(__file__).with_name("evaluate_portfolio_breadth_v2.py")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=16)
    args = ap.parse_args()
    method = json.loads(METHOD.read_text())
    sources = {name: ROOT / path for name, path in method["sources"].items()}
    if any(digest(path) != method["source_sha256"][name]
           for name, path in sources.items()):
        raise ValueError("Frozen portfolio breadth source changed")
    if not 0 <= args.start < len(method["cases"]) or not 1 <= args.limit <= len(method["cases"]):
        ap.error("invalid case slice")
    cases = method["cases"][args.start:args.start + args.limit]
    out = BASE / "native"
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({"MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "PYTHONPATH": "experiments/compact_morphology/mjlab_ape/src"})
    assets = method["assets"]
    for case in cases:
        stem = out / f"seed-{case['seed']}-original"
        paths = {suffix: stem.with_suffix(suffix) for suffix in (".json", ".npz", ".mp4", ".log")}
        if any(path.exists() for path in paths.values()):
            raise FileExistsError(f"Original native case already attempted: {stem}")
        cmd = [sys.executable, str(SCRIPT), "--body", assets["body"],
               "--gait-actor", assets["gait"], "--stop-actor", assets["stop"],
               "--squat-actor", assets["squat"], "--arm-actor", assets["arms"],
               "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
               "--delay-ms", str(case["delay_ms"]), "--walk-velocity", str(case["speed_m_s"]),
               "--walk-s", str(case["walk_s"]), "--stop-s", str(case["stop_s"]),
               "--squat-start-s", str(case["squat_start_s"]), "--squat-s", str(case["squat_s"]),
               "--arm-start-s", str(case["arm_start_s"]), "--arm-s", str(case["arm_s"]),
               "--sole-friction", str(case["sole_friction"]),
               "--device", "cuda:0", "--native-evaluation",
               "--out", str(paths[".json"]), "--trace", str(paths[".npz"]),
               "--video", str(paths[".mp4"])]
        with paths[".log"].open("w") as log:
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  check=False)
        print(json.dumps({"seed": case["seed"], "exit_code": done.returncode,
                          "report_exists": paths[".json"].exists()}), flush=True)
        if done.returncode:
            raise RuntimeError(f"Native original failed; inspect {paths['.log']}")


if __name__ == "__main__":
    main()
