"""Collect whole native forward and signed-turn motion examples in one GPU batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-low-level-training-20261015/shared-forward-turn/method.json"
OUT = Path("/workspace/robot-learning-lab/ape-low-level-training-20261015/shared-forward-turn/teachers")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case: dict, method: dict, folder: Path) -> dict:
    name = case["name"]
    source = method["sources"]
    checkpoint = ROOT / source[case["teacher"]]["path"]
    body = ROOT / source["body"]["path"]
    evaluator = ROOT / source["evaluator"]["path"]
    report, trace, imitation, video, log = (folder / f"{name}.{suffix}" for suffix in
                                            ("json", "npz", "imitation.npz", "mp4", "log"))
    if any(path.exists() for path in (report, trace, imitation, video, log)):
        raise RuntimeError(f"Original teacher visit already exists: {name}")
    command = [sys.executable, str(evaluator), str(checkpoint),
               "--task", case["task"], "--body-xml", str(body),
               "--physics-model", "frame50", "--initial-root-z", "0.231",
               "--yaw-only-head", "--arms-down", "--worlds", str(case["worlds"]),
               "--seed", str(case["seed"]),
               "--velocity-x", str(case["command"][0]),
               "--velocity-y", str(case["command"][1]),
               "--angular-z", str(case["command"][2]),
               "--motor-rail-v", str(case["motor_rail_v"]),
               "--command-delay-ms", str(case["command_delay_ms"]),
               "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
               "--duration-s", str(case["duration_s"]),
               "--out", str(report), "--diagnostic-trace", str(trace),
               "--imitation-trace", str(imitation)]
    if case.get("video", False):
        command += ["--video", str(video)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    with log.open("x") as stream:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, timeout=480, check=False)
    if proc.returncode:
        raise RuntimeError(f"Native teacher visit failed: {name}, exit={proc.returncode}, log={log}")
    saved = json.loads(report.read_text())
    if (saved["body_sha256"] != source["body"]["sha256"]
            or saved["checkpoint"] != str(checkpoint)
            or saved["seed"] != case["seed"]
            or saved["task"] != case["task"]
            or len(saved["worlds"]) != case["worlds"]):
        raise RuntimeError(f"Native teacher identity mismatch: {name}")
    with np.load(imitation, allow_pickle=False) as values:
        expected_steps = round(case["duration_s"] * 50)
        if (values["actor_observation"].shape != (expected_steps, case["worlds"], 72)
                or values["policy_action"].shape != (expected_steps, case["worlds"], 21)
                or values["command"].shape != (expected_steps, case["worlds"], 3)
                or not np.isfinite(values["actor_observation"]).all()
                or not np.isfinite(values["policy_action"]).all()):
            raise RuntimeError(f"Incomplete or nonfinite imitation trace: {name}")
        if not np.allclose(values["actor_observation"][..., -3:], values["command"], atol=1e-5):
            raise RuntimeError(f"Actor command observation mismatch: {name}")
        if not np.allclose(values["command"], case["command"], atol=1e-5):
            raise RuntimeError(f"Wrong teacher command: {name}")
        first_action = hashlib.sha256(values["policy_action"][0].tobytes()).hexdigest()
        initial = hashlib.sha256(values["qpos"][0].tobytes()
                                  + values["qvel"][0].tobytes()
                                  + values["command"][0].tobytes()).hexdigest()
        if (first_action != saved["first_action_sha256"]
                or initial != saved["initial_state_sha256"]):
            raise RuntimeError(f"Pre-action native parity mismatch: {name}")
    return {"name": name, "case": case, "report": str(report.relative_to(OUT)),
            "trace": str(trace.relative_to(OUT)),
            "imitation": str(imitation.relative_to(OUT)),
            "video": str(video.relative_to(OUT)) if case.get("video", False) else None,
            "files_sha256": {p.name: sha(p) for p in (report, trace, imitation)
                             if p.is_file()},
            "video_sha256": sha(video) if case.get("video", False) else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sentinel", "train", "dev"), required=True)
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    for item in method["sources"].values():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Frozen source mismatch: {path}")
    folder = OUT / args.phase
    if folder.exists():
        raise RuntimeError(f"One-shot phase already exists: {folder}")
    folder.mkdir(parents=True)
    cases = ([method["sentinel"]] if args.phase == "sentinel"
             else [c for c in method["cases"] if c["split"] == args.phase])
    rows = []
    for case in cases:
        row = run_case(case, method, folder)
        rows.append(row)
        (folder / "partial.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(case["name"], "source-valid", flush=True)
    result = {"schema": "robot-learning-lab.shared-forward-turn.teacher-collection.v1",
              "phase": args.phase, "method_sha256": sha(METHOD),
              "visits": len(rows), "worlds": sum(r["case"]["worlds"] for r in rows),
              "optimizer_updates": 0, "rows": rows,
              "claim_limit": "Native teacher motion data only; physical acceptance is decided by the dataset builder."}
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"phase": args.phase, "visits": len(rows),
                      "worlds": result["worlds"]}), flush=True)


if __name__ == "__main__":
    main()
