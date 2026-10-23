"""Run or verify the frozen four-family walk→stop→posture CUDA packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed_runtime(method: dict) -> dict:
    import torch

    versions = {name: metadata.version(name) for name in method["runtime_versions"]}
    if versions != method["runtime_versions"] or not torch.cuda.is_available():
        raise RuntimeError(f"Installed CUDA runtime differs: {versions}")
    bam = importlib.util.find_spec("bam")
    if bam is None or bam.origin is None:
        raise RuntimeError("BAM runtime is missing")
    bam_mjlab = Path(bam.origin).parent / "mjlab.py"
    if sha(bam_mjlab) != method["bam_mjlab_sha256"]:
        raise RuntimeError("BAM source differs from pinned commit")
    inspection = json.loads(subprocess.check_output(
        [sys.executable, "-m", "pip", "inspect", "--local"], text=True))
    packages = sorted([item["metadata"]["name"].lower(), item["metadata"]["version"],
                       item.get("metadata_location"), item.get("direct_url")]
                      for item in inspection["installed"])
    return {"versions": versions, "bam_mjlab_sha256": sha(bam_mjlab),
            "bam_origin": str(bam.origin), "installed_packages": packages}


def case_paths(out: Path, case: dict) -> tuple[Path, Path, Path, Path]:
    stem = f"seed-{case['seed']}-rail-{case['rail_v']}-delay-{case['delay_ms']}"
    return tuple(out / f"{stem}.{suffix}" for suffix in ("json", "npz", "mp4", "log"))


def verify(case: dict, out: Path, method: dict, *, sentinel: bool) -> dict:
    report_path, trace_path, video_path, log_path = case_paths(out, case)
    for path in (report_path, trace_path, video_path, log_path):
        if not path.is_file():
            raise RuntimeError(f"Missing original source: {path}")
    report = json.loads(report_path.read_text())
    expected = method["sentinel"]["steps"] if sentinel else method["clock"]["steps"]
    if (report["body_sha256"] != method["body"]["sha256"]
            or report["device"] != "cuda:0" or report["seed"] != case["seed"]
            or report["rail_v"] != case["rail_v"] or report["delay_ms"] != case["delay_ms"]
            or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]
            or report["steps"] != expected or report["ended_early"]):
        raise RuntimeError(f"Original source identity or clock differs: {report_path}")
    for name, source in method["actors"].items():
        if report["checkpoint_sha256"][name] != source["sha256"]:
            raise RuntimeError(f"Checkpoint differs in original: {name}")
    with np.load(trace_path, allow_pickle=False) as values:
        if (len(values["pre_qpos"]) != expected
                or len(values["post_qpos"]) != expected
                or values["actor_obs"].shape != (expected, 72)
                or values["applied_action"].shape != (expected, 21)
                or abs(float(values["dt_s"]) - 0.02) > 1e-8
                or not np.isfinite(values["post_qpos"]).all()):
            raise RuntimeError(f"Trace is incomplete: {trace_path}")
        if not sentinel:
            for step in (100, 200, 800):
                if float(np.max(abs(values["post_qpos"][step - 1]
                                    - values["pre_qpos"][step]))) > 1e-9:
                    raise RuntimeError(f"Physics jumped at step {step}: {trace_path}")
    for row in report["timeseries"]:
        if row["phase"] in ("walk", "stop"):
            expected_command = (case["walk_velocity_m_s"] if row["phase"] == "walk" else 0.0)
            if abs(row["gait_actor_command"] - expected_command) > 1e-7:
                raise RuntimeError(f"Gait actor received wrong command: {report_path}")
    reader = imageio.get_reader(video_path)
    frames = nonblank = 0
    try:
        for frame in reader:
            frames += 1
            nonblank += int(np.asarray(frame).std() > 0)
    finally:
        reader.close()
    expected_frames = 1 if sentinel else 600
    if frames != expected_frames or nonblank != expected_frames:
        raise RuntimeError(f"Video frame/blank fault: {video_path}: {frames}/{nonblank}")
    failures = [] if sentinel else [name for name in (
        "squat_depth_return_pass", "arm_rise_return_pass", "no_hand_or_body_contact",
        "support_windows_pass", "rated_duty_windows_pass",
    ) if not report[name]]
    if not sentinel and report["walk_distance_m"] < method["physical_gate"]["minimum_heading_walk_m"]:
        failures.append("walk_progress")
    if not sentinel and report["stop_tail_xy_m"] > method["physical_gate"]["maximum_stop_tail_xy_m"]:
        failures.append("stop_tail_drift")
    if not sentinel and report["switch_qpos_gap"] > 1e-9:
        failures.append("state_discontinuity")
    if not sentinel and bool(report["full_continuity_gate_pass"]) != (len(failures) == 0):
        raise RuntimeError("Evaluator gate and source-derived failures disagree")
    return {
        "seed": case["seed"], "rail_v": case["rail_v"],
        "delay_ms": case["delay_ms"], "walk_velocity_m_s": case["walk_velocity_m_s"],
        "unscored": sentinel, "pass": None if sentinel else report["full_continuity_gate_pass"],
        "failure_reasons": failures, "walk_distance_m": report["walk_distance_m"],
        "stop_tail_xy_m": report["stop_tail_xy_m"],
        "squat_depth_m": (report["windows"]["stand_before"]["pelvis_z"]
                          - report["windows"]["squat_hold"]["pelvis_z"]
                          if not sentinel else None),
        "minimum_hand_rise_m": (min(np.subtract(
            report["windows"]["arm_raised"]["hand_z"],
            report["windows"]["arm_before"]["hand_z"])) if not sentinel else None),
        "moving_rated_leg_fraction": (report["windows"]["walk_end"]["leg_duty_fraction"]
                                      if not sentinel else None),
        "report": str(report_path), "report_sha256": sha(report_path),
        "trace": str(trace_path), "trace_sha256": sha(trace_path),
        "video": str(video_path), "video_sha256": sha(video_path),
        "video_frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sentinel", "development"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    method = json.loads(METHOD.read_text())
    if method["schema"] != "robot-learning-lab.walk-squat-arms-cuda-method.v1":
        raise RuntimeError("Method schema differs")
    for source in (method["body"], method["evaluator"], method["runner"], method["runtime"],
                   *method["sources"].values(), *method["actors"].values()):
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Frozen source differs: {source['path']}")
    if not args.verify_only:
        runtime_path = args.out.resolve() / "runtime-attestation.json"
        current = installed_runtime(method)
        if args.phase == "sentinel":
            if runtime_path.exists():
                raise RuntimeError("Original runtime attestation already exists")
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_text(json.dumps(current, indent=2) + "\n")
        elif not runtime_path.is_file() or current != json.loads(runtime_path.read_text()):
            raise RuntimeError("Runtime changed since the one-step sentinel")
    cases = ([method["sentinel"]] if args.phase == "sentinel"
             else method["development"])
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "development":
        verify(method["sentinel"], out, method, sentinel=True)
    rows = []
    for case in cases:
        report, trace, video, log = case_paths(out, case)
        if not args.verify_only:
            if any(path.exists() for path in (report, trace, video, log)):
                raise RuntimeError(f"Original visit exists: {report}")
            command = [sys.executable, str(ROOT / method["evaluator"]["path"]),
                       "--body", str(ROOT / method["body"]["path"]),
                       "--gait-actor", str(ROOT / method["actors"]["gait"]["path"]),
                       "--stop-actor", str(ROOT / method["actors"]["stop"]["path"]),
                       "--squat-actor", str(ROOT / method["actors"]["squat"]["path"]),
                       "--arm-actor", str(ROOT / method["actors"]["arms"]["path"]),
                       "--device", "cuda:0", "--native-evaluation",
                       "--seed", str(case["seed"]), "--rail-v", str(case["rail_v"]),
                       "--delay-ms", str(case["delay_ms"]),
                       "--walk-velocity", str(case["walk_velocity_m_s"]),
                       "--max-steps", str(method["sentinel"]["steps"] if args.phase == "sentinel"
                                          else method["clock"]["steps"]),
                       "--out", str(report), "--trace", str(trace), "--video", str(video)]
            env = dict(os.environ, MUJOCO_GL="egl",
                       NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics",
                       PYTHONPATH=f"{ROOT}/experiments/compact_morphology/mjlab_ape/src:{ROOT}")
            with log.open("x") as stream:
                run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, check=False)
            if run.returncode:
                raise RuntimeError(f"Original visit exited {run.returncode}: {report}")
        row = verify(case, out, method, sentinel=args.phase == "sentinel")
        rows.append(row)
        (out / "partial.json").write_text(json.dumps({"method_sha256": sha(METHOD),
                                                       "phase": args.phase,
                                                       "rows": rows}, indent=2) + "\n")
        print(json.dumps({"seed": case["seed"], "pass": row["pass"],
                          "failure_reasons": row["failure_reasons"]}), flush=True)
    result = {"schema": "robot-learning-lab.walk-squat-arms-cuda-score.v1",
              "claim_limit": method["claim_limit"], "method_sha256": sha(METHOD),
              "phase": args.phase, "visits": len(rows), "rows": rows,
              "new_optimizer_updates": 0}
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
