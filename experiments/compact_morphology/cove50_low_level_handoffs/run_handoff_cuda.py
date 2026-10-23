"""Execute the frozen short handoff CUDA development packet once per new seed."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-cove50-low-level-handoffs/squat-arms-cuda-method.json"
OUT = ROOT / "artifacts/ape-cove50-low-level-handoffs/squat-arms-cuda-v1"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_case(report_path, trace_path, video_path, case, method):
    report = json.loads(report_path.read_text())
    if (report["body_sha256"] != method["source"]["body"]["sha256"]
            or report["device"] != "cuda:0" or report["seed"] != case["seed"]
            or report["rail_v"] != case["rail_v"] or report["delay_ms"] != case["delay_ms"]
            or report["steps"] != 1000 or report["ended_early"]):
        raise ValueError(f"source, clock or run length differs: {report_path}")
    with np.load(trace_path, allow_pickle=False) as trace:
        if (len(trace["pre_qpos"]) != 1000 or len(trace["actor_obs"]) != 1000
                or trace["actor_obs"].shape[1] != 72
                or trace["applied_action"].shape[1] != 21
                or abs(float(trace["dt_s"]) - 0.02) > 1e-8):
            raise ValueError(f"trace clock or shape differs: {trace_path}")
        if np.max(abs(trace["post_qpos"][499] - trace["pre_qpos"][500])) > 1e-9:
            raise ValueError(f"physics state jumped at actor switch: {trace_path}")
    reader = imageio.get_reader(video_path)
    count = 0
    nonblank = 0
    for frame in reader:
        count += 1
        nonblank += int(np.asarray(frame).std() > 0)
    reader.close()
    if count != 500 or nonblank != 500:
        raise ValueError(f"video frames {count}/{nonblank} differ: {video_path}")
    return {
        "case": case, "passed": report["full_continuity_gate_pass"],
        "physical_reasons": [key for key in (
            "squat_depth_return_pass", "arm_rise_return_pass",
            "no_hand_or_body_contact", "support_windows_pass", "rated_duty_windows_pass",
        ) if not report[key]],
        "report": str(report_path.relative_to(ROOT)), "report_sha256": digest(report_path),
        "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": digest(trace_path),
        "video": str(video_path.relative_to(ROOT)), "video_sha256": digest(video_path),
        "video_frames": count, "nonblank_frames": nonblank,
    }


def main():
    method = json.loads(METHOD.read_text())
    for entry in method["source"].values():
        path = ROOT / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"frozen input differs: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for case in method["cases"]:
        label = f"seed{case['seed']}-rail{case['rail_v']}-delay{case['delay_ms']}"
        report = OUT / f"{label}.json"
        trace = OUT / f"{label}.npz"
        video = OUT / f"{label}.mp4"
        log = OUT / f"{label}.log"
        if report.exists() or trace.exists() or video.exists():
            raise ValueError(f"refusing to overwrite original visit: {label}")
        command = [sys.executable, str(ROOT / method["source"]["evaluator"]["path"]),
                   "--body", str(ROOT / method["source"]["body"]["path"]),
                   "--squat-actor", str(ROOT / method["source"]["squat_actor"]["path"]),
                   "--arm-actor", str(ROOT / method["source"]["arm_actor"]["path"]),
                   "--out", str(report), "--trace", str(trace), "--video", str(video),
                   "--device", "cuda:0", "--seed", str(case["seed"]),
                   "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"])]
        env = dict(os.environ, MUJOCO_GL="egl",
                   NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics",
                   PYTHONPATH=f"{ROOT}/experiments/compact_morphology/mjlab_ape/src:{ROOT}")
        with log.open("w") as stream:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, check=False)
        if run.returncode:
            raise RuntimeError(f"{label} exited {run.returncode}; see {log}")
        results.append(verify_case(report, trace, video, case, method))
        (OUT / "index.json").write_text(json.dumps({
            "schema": "robot-learning-lab.squat-arms-cuda-index.v1",
            "method_sha256": digest(METHOD), "completed": len(results),
            "planned": len(method["cases"]), "results": results,
        }, indent=2) + "\n")
        print(json.dumps({"completed": len(results), "case": label,
                          "passed": results[-1]["passed"]}), flush=True)


if __name__ == "__main__":
    main()
