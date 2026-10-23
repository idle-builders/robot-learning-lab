"""Run one source-pinned A40 evaluation sweep of retained Direct Frame actors.

This executes full native episodes, never fits a checkpoint, and never replaces
an original. Each source report, numeric trace, video and log is kept together.
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
METHOD = ROOT / "artifacts/ape-low-level-training-20261016/basic-sweep-posture-locomotion/method.json"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
LOC_EVAL = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
POST_EVAL = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate_posture.py"
SEATED_EVAL = Path(__file__).with_name("evaluate_seated_bc_sweep.py")
BC_LOC = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt"
BC_SQUAT = ROOT / "artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt"
BC_SEATED = ROOT / "artifacts/ape-cove50-stand-recovery/seated-bc-v35/model_bc_seed.pt"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(case: dict, out: Path) -> list[str]:
    name = case["name"]
    report, trace, video = (out / f"{name}.{suffix}" for suffix in ("json", "npz", "mp4"))
    if case["kind"] == "stand":
        return [sys.executable, str(POST_EVAL), "--zero-actions", "--mode", "stand",
                "--cove50-body-xml", str(BODY), "--device", "cuda:0",
                "--seed", str(case["seed"]), "--duration-s", "8", "--out", str(report),
                "--native-trajectory-out", str(trace), "--video", str(video)]
    if case["kind"] == "squat":
        return [sys.executable, str(POST_EVAL), str(BC_SQUAT), "--student-bc",
                "--task", "Mjlab-Posture-Flat-Ape-Elbows-HeadingObserved",
                "--mode", "cycle", "--squat-active-mask", "--action-abs-bound", "1",
                "--expose-squat-blend", "--mask-action-history",
                "--cove50-body-xml", str(BODY), "--device", "cuda:0",
                "--seed", str(case["seed"]), "--duration-s", "10",
                "--motor-rail-v", str(case["motor_rail_v"]),
                "--command-delay-ms", str(case["command_delay_ms"]),
                "--action-smoothing-alpha", str(case["action_smoothing_alpha"]),
                "--out", str(report), "--native-trajectory-out", str(trace),
                "--video", str(video)]
    if case["kind"] == "seated":
        return [sys.executable, str(SEATED_EVAL), str(BC_SEATED),
                "--intervention-route", "bc", "--start-squat", "--recover-from-start",
                "--seated-rise-stage", "--randomized-physics", "--mode", "cycle",
                "--cove50-body-xml", str(BODY), "--device", "cuda:0",
                "--seed", str(case["seed"]), "--duration-s", "10",
                "--out", str(report), "--native-trajectory-out", str(trace),
                "--video", str(video)]
    cmd = [sys.executable, str(LOC_EVAL), str(BC_LOC),
           "--task", case["task"], "--body-xml", str(BODY),
           "--physics-model", "frame50", "--initial-root-z", "0.231",
           "--yaw-only-head", "--arms-down", "--worlds", "1",
           "--seed", str(case["seed"]),
           "--velocity-x", str(case["command"][0]),
           "--velocity-y", str(case["command"][1]),
           "--angular-z", str(case["command"][2]),
           "--motor-rail-v", str(case["motor_rail_v"]),
           "--command-delay-ms", str(case["command_delay_ms"]),
           "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
           "--duration-s", "4", "--out", str(report),
           "--diagnostic-trace", str(trace), "--video", str(video)]
    if case["kind"] == "stop":
        cmd += ["--stop-at-s", "2"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--case", action="append", help="Run only selected original names")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.basic-sweep.posture-locomotion.method.v1":
        raise RuntimeError("method schema differs")
    for key, item in method["sources"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"source differs: {key}: {path}")
    cases = [case for case in method["cases"] if not args.case or case["name"] in args.case]
    if args.case and len(cases) != len(set(args.case)):
        raise ValueError("requested case is not unique in frozen method")
    for case in cases:
        cmd = command(case, args.out)
        if args.dry_run:
            print(json.dumps({"case": case["name"], "command": cmd}))
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        stem = args.out / case["name"]
        files = [stem.with_suffix(ext) for ext in (".json", ".npz", ".mp4", ".log")]
        if any(path.exists() for path in files):
            raise RuntimeError(f"original already exists: {case['name']}")
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
        env["MUJOCO_GL"] = "egl"
        env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
        with files[-1].open("x") as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, check=False, timeout=300)
        if result.returncode:
            raise RuntimeError(f"native original failed: {case['name']} exit {result.returncode}")
        if not all(path.is_file() and path.stat().st_size for path in files):
            raise RuntimeError(f"native original incomplete: {case['name']}")
        report = json.loads(files[0].read_text())
        if report["body_sha256"] != method["sources"]["body"]["sha256"]:
            raise RuntimeError(f"body identity differs: {case['name']}")
        if report["seed"] != case["seed"]:
            raise RuntimeError(f"seed identity differs: {case['name']}")
        print(json.dumps({"case": case["name"], "duration_s": report.get("duration_s"),
                          "files": {path.suffix: sha(path) for path in files}}), flush=True)


if __name__ == "__main__":
    main()
