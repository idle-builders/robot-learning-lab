"""Run the separately frozen both-arm and no-reset handoff addendum once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261016/basic-sweep-posture-locomotion/addendum-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.basic-sweep.arm-handoff-addendum.v1":
        raise RuntimeError("addendum method differs")
    main_method = METHOD.with_name("method.json")
    if sha(main_method) != method["main_method_sha256"]:
        raise RuntimeError("main frozen method differs")
    for name, item in method["sources"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"source differs: {name}")
    src = method["sources"]
    body = ROOT / src["body"]["path"]
    arm = ROOT / src["arm_bc"]["path"]
    gait = ROOT / src["gait_bc"]["path"]
    squat = ROOT / src["squat_bc"]["path"]
    stop = ROOT / src["stop_actor"]["path"]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'experiments/compact_morphology/mjlab_ape/src'}:{ROOT}"
    env["MUJOCO_GL"] = "egl"
    env["NVIDIA_DRIVER_CAPABILITIES"] = "compute,utility,graphics"
    for case in method["cases"]:
        stem = args.out / case["name"]
        report, trace, video, log = (stem.with_suffix(s) for s in (".json", ".npz", ".mp4", ".log"))
        if case["name"].startswith("both-arms"):
            cmd = [sys.executable, str(ROOT / src["arm_evaluator"]["path"]), str(arm),
                   "--student-bc", "--cove50-body-xml", str(body),
                   "--shoulder-target-margin-rad", "0.05", "--device", "cuda:0",
                   "--seed", str(case["seed"]), "--motor-rail-v", str(case["motor_rail_v"]),
                   "--command-delay-ms", str(case["command_delay_ms"]),
                   "--action-smoothing-alpha", str(case["action_smoothing_alpha"]),
                   "--out", str(report), "--native-trajectory-out", str(trace),
                   "--video", str(video)]
        else:
            cmd = [sys.executable, str(ROOT / src["handoff_evaluator"]["path"]),
                   "--body", str(body), "--squat-actor", str(squat),
                   "--arm-actor", str(arm), "--gait-actor", str(gait),
                   "--stop-actor", str(stop), "--device", "cuda:0",
                   "--native-evaluation", "--seed", str(case["seed"]),
                   "--rail-v", str(case["motor_rail_v"]),
                   "--delay-ms", str(case["command_delay_ms"]),
                   "--walk-velocity", str(case["walk_velocity_m_s"]),
                   "--max-steps", "1200", "--out", str(report), "--trace", str(trace),
                   "--video", str(video)]
        if args.dry_run:
            print(json.dumps({"case": case["name"], "command": cmd}))
            continue
        args.out.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (report, trace, video, log)):
            raise RuntimeError(f"addendum original already exists: {case['name']}")
        with log.open("x") as stream:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=360, check=False)
        if result.returncode:
            raise RuntimeError(f"addendum original failed: {case['name']} exit {result.returncode}")
        for path in (report, trace, video):
            if not path.is_file() or not path.stat().st_size:
                raise RuntimeError(f"addendum output missing: {path}")
        native = json.loads(report.read_text())
        if native["body_sha256"] != src["body"]["sha256"] or native["seed"] != case["seed"]:
            raise RuntimeError(f"addendum identity differs: {case['name']}")
        print(json.dumps({"case": case["name"], "report_sha256": sha(report),
                          "trace_sha256": sha(trace), "video_sha256": sha(video)}), flush=True)


if __name__ == "__main__":
    main()
