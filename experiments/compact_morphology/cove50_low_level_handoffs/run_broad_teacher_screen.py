"""Bounded fresh no-reset teacher screen for a broad command-conditioned learner.

Reuses the source-checked V3 exact native evaluator. Each original is retained;
this script collects full teacher trajectories and does not fit a new policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
V3_METHOD = ROOT / "artifacts/ape-cove50-low-level-handoffs/walk-squat-arms-cuda-method.json"
PLAN = ROOT / "artifacts/ape-low-level-training-20261015/broad-handoff-v4-a5000-screen-method.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(method: dict, plan: dict) -> None:
    if sha(V3_METHOD) != plan["source_v3_method_sha256"]:
        raise RuntimeError("V3 method changed")
    if sha(Path(__file__)) != plan["new_runner_sha256"]:
        raise RuntimeError("Frozen teacher runner changed")
    for source in (method["body"], method["evaluator"], method["runner"], method["runtime"],
                   *method["sources"].values(), *method["actors"].values()):
        if sha(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Exact native source changed: {source['path']}")
    if plan["exact_body_sha256"] != method["body"]["sha256"]:
        raise RuntimeError("Plan body differs")


def hardware_attestation(plan: dict) -> dict:
    import platform
    import torch

    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    smi = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    expected_name = plan["hardware"]["provider_gpu_id"]
    smi_name = smi[0].split(",", 1)[0].strip() if len(smi) == 1 else None
    if (name.strip() != expected_name
            or capability != plan["hardware"]["compute_capability"]
            or total_gib < plan["hardware"]["minimum_vram_gib"]
            or smi_name != expected_name):
        raise RuntimeError(f"GPU differs from frozen A5000 method: {name}, {capability}, {total_gib}, {smi}")
    return {"torch_device_name": name, "compute_capability": capability,
            "torch_total_gib": total_gib, "nvidia_smi_name_driver_memory": smi[0],
            "host_platform": platform.platform()}


def stem(case: dict) -> str:
    return f"seed-{case['seed']}-rail-{case['rail_v']}-delay-{case['delay_ms']}"


def verify_case(case: dict, out: Path, method: dict, *, sentinel: bool) -> dict:
    report_path = out / (stem(case) + ".json")
    trace_path = out / (stem(case) + ".npz")
    log_path = out / (stem(case) + ".log")
    timing_path = out / (stem(case) + ".timing.json")
    if not all(p.is_file() for p in (report_path, trace_path, log_path)):
        raise RuntimeError(f"Incomplete original: {stem(case)}")
    report = json.loads(report_path.read_text())
    steps = int(report["steps"])
    if (sentinel and (steps != 1 or report["ended_early"])) or (
            not sentinel and not 1 <= steps <= 1200):
        raise RuntimeError(f"Original step count invalid: {stem(case)}")
    if (report["body_sha256"] != method["body"]["sha256"]
            or report["device"] != "cuda:0"
            or report["seed"] != case["seed"]
            or report["rail_v"] != case["rail_v"]
            or report["delay_ms"] != case["delay_ms"]
            or report["walk_velocity_m_s"] != case["walk_velocity_m_s"]):
        raise RuntimeError(f"Original report identity/length mismatch: {stem(case)}")
    if any(report["checkpoint_sha256"][name] != source["sha256"]
           for name, source in method["actors"].items()):
        raise RuntimeError(f"Actor SHA mismatch: {stem(case)}")
    with np.load(trace_path, allow_pickle=False) as trace:
        pre, post = trace["pre_qpos"], trace["post_qpos"]
        obs, action, target = trace["actor_obs"], trace["applied_action"], trace["processed_target"]
        if (len(pre) != steps or len(post) != steps or obs.shape != (steps, 72)
                or action.shape != (steps, 21) or target.shape != (steps, 21)
                or not all(np.isfinite(v).all() for v in (pre, post, obs, action, target))):
            raise RuntimeError(f"Original trace invalid: {stem(case)}")
        max_gap = float(np.max(np.abs(post[:-1] - pre[1:]))) if steps > 1 else None
        if max_gap is not None and max_gap > 1e-9:
            raise RuntimeError(f"Physics state discontinuity: {stem(case)}")
    passed = None if sentinel else bool(report["full_continuity_gate_pass"])
    if passed and (steps != 1200 or report["ended_early"]):
        raise RuntimeError(f"Full pass contradicts early termination: {stem(case)}")
    failures = [] if sentinel or passed else [name for name in (
        "squat_depth_return_pass", "arm_rise_return_pass", "no_hand_or_body_contact",
        "support_windows_pass", "rated_duty_windows_pass") if not report[name]]
    if not sentinel and report["walk_distance_m"] < 0.06:
        failures.append("walk_progress")
    if not sentinel:
        if report["ended_early"] or steps < 1200:
            failures.append("early_termination")
        if report["stop_tail_xy_m"] is None:
            failures.append("stop_window_incomplete")
        elif report["stop_tail_xy_m"] > 0.03:
            failures.append("stop_tail_drift")
    if not sentinel and passed != (len(failures) == 0):
        raise RuntimeError(f"Saved gate and derived failure disagree: {stem(case)}")
    timing = json.loads(timing_path.read_text()) if timing_path.is_file() else None
    if timing is not None and timing["elapsed_s"] <= 0:
        raise RuntimeError(f"Invalid recorded runtime: {stem(case)}")
    return {"case": case, "pass": passed, "failure_reasons": failures,
            "report": str(report_path), "report_sha256": sha(report_path),
            "trace": str(trace_path), "trace_sha256": sha(trace_path),
            "log": str(log_path), "log_sha256": sha(log_path),
            "max_physics_gap": max_gap,
            "steps": steps, "ended_early": bool(report["ended_early"]),
            "elapsed_s_including_constructor": timing["elapsed_s"] if timing else None,
            "useful_transitions_per_s": steps / timing["elapsed_s"] if timing else None,
            "walk_distance_m": report["walk_distance_m"],
            "stop_tail_xy_m": report["stop_tail_xy_m"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sentinel", "screen"), required=True)
    parser.add_argument("--plan", type=Path, required=True,
                        help="Frozen hardware-specific method JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--pod-start-unix", type=float,
                        help="Provider recorded start time of the paid pod")
    parser.add_argument("--deadline-unix", type=float,
                        help="Paid-run stop time, at most one hour after provider pod start")
    args = parser.parse_args()
    if not args.verify_only and (
            args.pod_start_unix is None or args.deadline_unix is None
            or args.pod_start_unix > time.time()
            or not args.pod_start_unix < args.deadline_unix <= args.pod_start_unix + 3600
            or args.deadline_unix <= time.time()):
        parser.error("Paid run needs a provider pod start and a future deadline within one hour")
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text())
    method = json.loads(V3_METHOD.read_text())
    verify_source(method, plan)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = [plan["sentinel"]] if args.phase == "sentinel" else plan["screen"]
    runtime_path = out / "runtime-attestation.json"
    if not args.verify_only:
        from experiments.compact_morphology.cove50_low_level_handoffs.run_walk_squat_arms_cuda import installed_runtime
        runtime = installed_runtime(method)
        runtime["hardware"] = hardware_attestation(plan)
        if args.phase == "sentinel" and not runtime_path.exists():
            runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
        elif not runtime_path.exists() or json.loads(runtime_path.read_text()) != runtime:
            raise RuntimeError("Runtime differs from unscored sentinel")
    if args.phase == "screen":
        verify_case(plan["sentinel"], out, method, sentinel=True)
    rows = []
    for case in cases:
        if not args.verify_only and time.time() >= args.deadline_unix:
            raise RuntimeError("Pilot wall deadline reached before next original")
        report = out / (stem(case) + ".json")
        trace = out / (stem(case) + ".npz")
        log = out / (stem(case) + ".log")
        if not args.verify_only:
            existing = [path.exists() for path in (report, trace, log)]
            if any(existing) and not all(existing):
                raise RuntimeError(f"Incomplete original preserved; no replacement: {stem(case)}")
            if not all(existing):
                started = time.monotonic()
                command = [sys.executable, str(ROOT / method["evaluator"]["path"]),
                       "--body", str(ROOT / method["body"]["path"]),
                       "--gait-actor", str(ROOT / method["actors"]["gait"]["path"]),
                       "--stop-actor", str(ROOT / method["actors"]["stop"]["path"]),
                       "--squat-actor", str(ROOT / method["actors"]["squat"]["path"]),
                       "--arm-actor", str(ROOT / method["actors"]["arms"]["path"]),
                       "--device", "cuda:0", "--native-evaluation", "--seed", str(case["seed"]),
                       "--rail-v", str(case["rail_v"]), "--delay-ms", str(case["delay_ms"]),
                       "--walk-velocity", str(case["walk_velocity_m_s"]),
                       "--max-steps", str(1 if args.phase == "sentinel" else 1200),
                       "--out", str(report), "--trace", str(trace)]
                env = dict(os.environ, MUJOCO_GL="egl",
                       NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics",
                       PYTHONPATH=f"{ROOT}/experiments/compact_morphology/mjlab_ape/src:{ROOT}")
                with log.open("x") as stream:
                    run = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, check=False,
                                         timeout=max(1, args.deadline_unix - time.time()))
                if run.returncode:
                    raise RuntimeError(f"Original exited {run.returncode}: {stem(case)}")
                (out / (stem(case) + ".timing.json")).write_text(json.dumps({
                    "elapsed_s": time.monotonic() - started,
                    "scope": "subprocess constructor plus native episode; excludes pod bootstrap",
                }, indent=2) + "\n")
        row = verify_case(case, out, method, sentinel=args.phase == "sentinel")
        rows.append(row)
        (out / f"{args.phase}-partial.json").write_text(json.dumps({
            "plan_sha256": sha(plan_path), "completed": len(rows), "rows": rows,
        }, indent=2) + "\n")
        print(json.dumps({"seed": case["seed"], "pass": row["pass"],
                          "failures": row["failure_reasons"]}), flush=True)
    (out / f"{args.phase}-report.json").write_text(json.dumps({
        "schema": "robot-learning-lab.broad-handoff-v4-teacher-screen.v1",
        "claim_limit": "Fresh teacher data only; zero new learner updates or learned skill score",
        "plan_sha256": sha(plan_path), "phase": args.phase, "visits": len(rows),
        "accepted_whole_episodes": sum(row["pass"] is True for row in rows),
        "rows": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
