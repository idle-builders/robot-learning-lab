"""Attest the V31d/V32b runtime, then smoke the exact locomotion evaluator.

`contract` is local and read-only. The other phases require the frozen CUDA
worker. Two disjoint one-step sentinels precede all opened probe seeds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch"
PLAN = ART / "plan.json"
GUARD = ART / "fresh-pod-attestation.json"
CLOSURE = ART / "installed-closure.json"
RUNTIME = ART / "runtime-attestation.json"
READY = ART / "runtime-ready.json"
ATTESTOR = ROOT / "experiments/compact_morphology/cove50_stand_recovery/preflight_seated_runtime_v31d.py"
EXPECTED_CLOSURE_SHA = "7c2f52255fd27e42d2d6153b61d05c9eb7b05116a8134b65541f53451cefdfef"
SENTINELS = (
    ("forward", 46998, "forward_checkpoint", "Mjlab-Velocity-Flat-Ape-Elbows", 0.35, 0.0, True),
    ("turn", 46999, "turn_checkpoint", "Mjlab-Velocity-Flat-Ape-Elbows-Planar", 0.0, 0.4, False),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def plan_contract() -> dict:
    plan = json.loads(PLAN.read_text())
    if (plan["schema"] != "robot-learning-lab.cove50-locomotion-probe.v1.plan"
            or plan["status"] != "frozen_before_paid_opened_diagnostic"
            or plan["compiled_control_body"] != "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
            or plan["opened_native_visits_max"] != 18
            or plan["optimizer_updates_max"] != 0
            or plan["sentinel_seeds"] != [46998, 46999]
            or plan["required_installed_closure_content_sha256"] != EXPECTED_CLOSURE_SHA):
        raise RuntimeError("Frozen locomotion plan contract changed")
    for name, rel in plan["source_paths"].items():
        path = (ROOT / rel).resolve()
        if (not path.is_relative_to(ROOT.resolve()) or not path.is_file()
                or sha(path) != plan["source_sha256"][name]):
            raise RuntimeError(f"Frozen material source changed: {name}")
    if plan["source_paths"]["preflight"] != str(Path(__file__).resolve().relative_to(ROOT)):
        raise RuntimeError("Wrong preflight source path")
    return plan


def guard_binding(plan: dict, *, fresh_s: int | None) -> dict:
    attestation = json.loads(GUARD.read_text())
    if (attestation["schema"] != "robot-learning-lab.locomotion-probe-fresh-pod-attestation.v1"
            or attestation["plan_sha256"] != sha(PLAN)
            or attestation["image"] != "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
            or attestation["gpu_id"] != "NVIDIA GeForce RTX 4090"
            or attestation["listed_rate_usd_per_hour"] != 0.89
            or attestation["gpu_count"] != 1
            or attestation["provider_desired_status"] != "RUNNING"
            or attestation["provider_runtime_status"] != "running"):
        raise RuntimeError("Guard/provider attestation does not match frozen host")
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(attestation["hard_deadline_utc"]):
        raise RuntimeError("Local guard deadline has passed")
    if fresh_s is not None and (now - datetime.fromisoformat(
            attestation["attested_at_utc"])).total_seconds() > fresh_s:
        raise RuntimeError("Refresh the live local guard attestation before this CUDA phase")
    return attestation


def installed_closure(plan: dict) -> dict:
    spec = importlib.util.spec_from_file_location("v31d_runtime_attestor", ATTESTOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    closure = module.installed_closure(plan)
    if closure["content_sha256"] != EXPECTED_CLOSURE_SHA:
        raise RuntimeError("V31d/V32b installed closure differs")
    return closure


def attest_runtime(plan: dict) -> None:
    guard_binding(plan, fresh_s=None)
    closure = installed_closure(plan)
    write(CLOSURE, closure)
    write(RUNTIME, {
        "schema": "robot-learning-lab.cove50-locomotion-runtime.v1",
        "status": "installed_closure_attested_no_locomotion_cuda_task",
        "plan_sha256": sha(PLAN),
        "guard_attestation_sha256": sha(GUARD),
        "closure_sha256": closure["content_sha256"],
        "source_sha256": plan["source_sha256"],
    })
    print(RUNTIME)


def sentinels(plan: dict) -> None:
    if READY.exists():
        raise RuntimeError("Exact sentinels already completed; no replacement")
    guard_binding(plan, fresh_s=240)
    runtime = json.loads(RUNTIME.read_text())
    closure = installed_closure(plan)
    if (runtime["plan_sha256"] != sha(PLAN)
            or runtime["closure_sha256"] != EXPECTED_CLOSURE_SHA
            or closure != json.loads(CLOSURE.read_text())):
        raise RuntimeError("Installed runtime changed after attestation")
    evaluator = ROOT / plan["source_paths"]["evaluator"]
    body = ROOT / plan["source_paths"]["body"]
    results = []
    for name, seed, checkpoint_key, task, vx, yaw, ankle_clip in SENTINELS:
        binding = guard_binding(plan, fresh_s=240)
        if (datetime.fromisoformat(binding["hard_deadline_utc"])
                - datetime.now(timezone.utc)).total_seconds() < 210:
            raise RuntimeError("Insufficient guard time for sentinel and evidence stop reserve")
        report = ART / f"sentinel-{name}.json"
        log_path = ART / f"sentinel-{name}.log"
        if report.exists() or log_path.exists():
            raise RuntimeError(f"Exact {name} sentinel output already exists; no replacement")
        command = [sys.executable, str(evaluator),
                   str(ROOT / plan["source_paths"][checkpoint_key]),
                   "--task", task, "--body-xml", str(body),
                   "--physics-model", "frame50", "--yaw-only-head", "--arms-down",
                   "--worlds", "1", "--seed", str(seed),
                   "--velocity-x", str(vx), "--velocity-y", "0",
                   "--angular-z", str(yaw), "--motor-rail-v", "12",
                   "--command-delay-ms", "0", "--duration-s", "0.02",
                   "--physics-dt", "0.002", "--out", str(report)]
        if ankle_clip:
            command += ["--ankle-target-margin-rad", "0.02"]
        with log_path.open("x") as log:
            proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT, check=False, timeout=90)
        if proc.returncode:
            raise RuntimeError(f"Exact {name} CUDA sentinel failed; see saved log")
        native = json.loads(report.read_text())
        action = native["first_action_world0"]
        if (native["seed"] != seed or native["body_sha256"] != plan["source_sha256"]["body"]
                or native["checkpoint"] != str(ROOT / plan["source_paths"][checkpoint_key])
                or native["task"] != task
                or native["velocity_command_m_s"] != [vx, 0.0]
                or native["angular_command_rad_s"] != yaw
                or native["physics_dt_s"] != 0.002 or native["stop_at_s"] is not None
                or len(native["worlds"]) != 1 or native["worlds"][0]["duration_s"] != 0.02
                or native["candidate_motor_rail_v"] != 12
                or native["candidate_command_delay_ms"] != 0
                or native["ankle_target_margin_rad"] != (0.02 if ankle_clip else None)
                or len(action) != 21 or not all(math.isfinite(x) for x in action)
                or not native["initial_state_sha256"] or not native["first_action_sha256"]):
            raise RuntimeError(f"Exact {name} CUDA sentinel interface differs")
        results.append({"name": name, "seed": seed, "report_sha256": sha(report),
                        "initial_state_sha256": native["initial_state_sha256"],
                        "first_action_sha256": native["first_action_sha256"]})
    write(READY, {"schema": "robot-learning-lab.cove50-locomotion-runtime-ready.v1",
                  "status": "two_exact_cuda_sentinels_passed",
                  "claim_limit": "Two unscored one-step entries; no signed-motion score or optimizer update.",
                  "plan_sha256": sha(PLAN), "closure_sha256": EXPECTED_CLOSURE_SHA,
                  "guard_attestation_sha256": sha(GUARD), "sentinels": results})
    print(READY)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("contract", "attest-runtime", "sentinels"))
    phase = parser.parse_args().phase
    plan = plan_contract()
    if phase == "contract":
        print(json.dumps({"plan_sha256": sha(PLAN), "source_count": len(plan["source_paths"])}))
    elif phase == "attest-runtime":
        attest_runtime(plan)
    else:
        sentinels(plan)


if __name__ == "__main__":
    main()
