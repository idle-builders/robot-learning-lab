"""Frozen native replay plan for opened Direct Frame locomotion starts.

Default prints the exact commands. --run executes the same 18 one-world cases
in the current CUDA Python environment; no worker provisioning is performed here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
FORWARD = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
TURN = ROOT / "artifacts/ape-cove50-turn-recovery-20261008/turneffort25-model950.pt"
OUT = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/fresh-native-probe"
LAUNCH = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/cuda-launch"
EXPECTED = {
    BODY: "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c",
    EVALUATOR: "bc7a35eba4e3ce378f5809152ec9927b13c6f88a5ddf7aa7fb0d8b09e1fea8fc",
    FORWARD: "1d01d490e3754bf9cb14e19588d784cb7bd1c9d6e82ae0b216c3ce06590e27a2",
    TURN: "e56cb7f4da0099e6e31333dc4957b943b375dc5e98af1902b13a77be4651d301",
}
# Two distinct RNG starts per command. These are opened development probes,
# not the sealed independent score sets held by other tracks.
CASES = (
    ("forward", FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", (0.35, 0.0, 0.0), None, (47001, 47002)),
    ("backward", FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", (-0.35, 0.0, 0.0), None, (47003, 47004)),
    ("left-side", FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", (0.0, 0.2, 0.0), None, (47005, 47006)),
    ("right-side", FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", (0.0, -0.2, 0.0), None, (47007, 47008)),
    ("left-turn", TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", (0.0, 0.0, 0.4), None, (47009, 47010)),
    ("right-turn", TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", (0.0, 0.0, -0.4), None, (47011, 47012)),
    ("forward-stop", FORWARD, "Mjlab-Velocity-Flat-Ape-Elbows", (0.35, 0.0, 0.0), 4.0, (47013, 47014)),
    ("left-turn-stop", TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", (0.0, 0.0, 0.4), 4.0, (47015, 47016)),
    ("right-turn-stop", TURN, "Mjlab-Velocity-Flat-Ape-Elbows-Planar", (0.0, 0.0, -0.4), 4.0, (47017, 47018)),
)
LEG_NAMES = (
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle_pitch",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle_pitch",
)
TRACE_SHAPES = {
    "active": (1,), "root_pose": (1, 7), "root_velocity": (1, 3),
    "joint_pos": (1, 10), "joint_vel": (1, 10),
    "actuator_force": (1, 10), "raw_action": (1, 10),
    "unclipped_position_target": (1, 10), "sole_found": (1, 2),
    "sole_contact_force": (1, 2, 3), "sole_speed": (1, 2),
    "hand_force": (1,), "body_contact": (1,),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_finite_json(value: object, path: str = "report") -> None:
    """Reject nonfinite metrics anywhere in the native JSON, including nested stops."""
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite native metric: {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite_json(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            require_finite_json(child, f"{path}[{index}]")
        return
    raise TypeError(f"Unsupported native JSON value: {path}")


def validate_native_case(native: dict, trace: Path, *, seed: int,
                         checkpoint: Path, task: str, vx: float, vy: float,
                         yaw: float, stop: float | None) -> dict:
    """Accept only a finite, step-aligned one-world report and complete trace."""
    import numpy as np  # Only the guarded CUDA worker needs NumPy.

    require_finite_json(native)
    first_action = native["first_action_world0"]
    worlds = native["worlds"]
    if (native["seed"] != seed or native["body_sha256"] != EXPECTED[BODY]
            or native["body_xml"] != str(BODY)
            or native["checkpoint"] != str(checkpoint)
            or native["task"] != task or native["physics_model"] != "frame50"
            or native["physics_dt_s"] != 0.002
            or native["episode_length_s"] != 8
            or native["stop_at_s"] != stop or len(worlds) != 1
            or native["velocity_command_m_s"] != [vx, vy]
            or native["angular_command_rad_s"] != yaw
            or native["candidate_motor_rail_v"] != 12
            or native["candidate_command_delay_ms"] != 0
            or native["ankle_target_margin_rad"] != (0.02 if checkpoint == FORWARD else None)
            or native["diagnostic_trace"] != str(trace)
            or native["yaw_only_head"] is not True or native["arms_down"] is not True
            or native["zero_action"] is not False
            or not isinstance(first_action, list) or len(first_action) != 21
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                       and math.isfinite(x) for x in first_action)
            or not re.fullmatch(r"[0-9a-f]{64}", native["initial_state_sha256"])
            or not re.fullmatch(r"[0-9a-f]{64}", native["first_action_sha256"])):
        raise ValueError("Native source, command, action or one-world interface differs")
    world = worlds[0]
    if (not isinstance(world, dict) or world["world"] != 0
            or not isinstance(world["duration_s"], (int, float))
            or isinstance(world["duration_s"], bool)
            or not 0 < world["duration_s"] <= 8
            or (stop is None and "stop" in world)
            or (stop is not None and not isinstance(world.get("stop"), dict))
            or not isinstance(native["one_meter_with_alternating_steps"], int)
            or native["one_meter_with_alternating_steps"] not in (0, 1)):
        raise ValueError("Native world metrics or stop record differ")
    if not trace.is_file():
        raise ValueError("Native diagnostic trace missing")
    with np.load(trace, allow_pickle=False) as archive:
        expected_keys = set(TRACE_SHAPES) | {
            "time_s", "leg_names", "rated_effort_limits_nm", "step_dt_s"}
        if set(archive.files) != expected_keys:
            raise ValueError("Native diagnostic trace fields differ")
        arrays = {key: archive[key] for key in archive.files}
    dt = arrays["step_dt_s"]
    times = arrays["time_s"]
    if (dt.shape != () or dt.dtype.kind not in "fiu"
            or not math.isclose(float(dt), 0.02, rel_tol=0, abs_tol=1e-9)
            or times.ndim != 1 or times.dtype.kind not in "fiu"):
        raise ValueError("Native diagnostic trace clock differs")
    steps = len(times)
    if (not 1 <= steps <= 400
            or not math.isclose(float(world["duration_s"]), steps * float(dt),
                                rel_tol=0, abs_tol=0.0005)
            or not np.allclose(times, np.arange(1, steps + 1) * float(dt),
                               rtol=0, atol=1e-8)):
        raise ValueError("Native report/trace step alignment differs")
    for key, shape in TRACE_SHAPES.items():
        array = arrays[key]
        if array.shape != (steps, *shape) or array.dtype.kind not in "biuf":
            raise ValueError(f"Native diagnostic trace shape/type differs: {key}")
    if (arrays["leg_names"].shape != (10,)
            or arrays["leg_names"].dtype.kind not in "US"
            or arrays["leg_names"].tolist() != list(LEG_NAMES)
            or arrays["rated_effort_limits_nm"].shape != (10,)
            or arrays["rated_effort_limits_nm"].dtype.kind not in "fiu"
            or native["leg_rated_effort_limits_nm"] is None
            or len(native["leg_rated_effort_limits_nm"]) != 10):
        raise ValueError("Native diagnostic leg joint metadata differs")
    for key, array in arrays.items():
        if key != "leg_names" and (array.dtype.kind not in "biuf" or not np.isfinite(array).all()):
            raise ValueError(f"Nonfinite or unsupported native diagnostic trace: {key}")
    if (not np.allclose(arrays["rated_effort_limits_nm"],
                        native["leg_rated_effort_limits_nm"], rtol=0, atol=1e-6)
            or not np.all(arrays["active"][:, 0])
            or not np.allclose(arrays["raw_action"][0, 0], first_action[:10],
                               rtol=0, atol=1e-6)):
        raise ValueError("Native report/trace action, support or motor law differs")
    return {"native_steps": steps, "trace_step_dt_s": float(dt),
            "all_numeric_report_and_trace_finite": True}


def guarded_runtime() -> datetime:
    plan_path = LAUNCH / "plan.json"
    plan = json.loads(plan_path.read_text())
    guard = json.loads((LAUNCH / "fresh-pod-attestation.json").read_text())
    ready = json.loads((LAUNCH / "runtime-ready.json").read_text())
    if (plan["schema"] != "robot-learning-lab.cove50-locomotion-probe.v1.plan"
            or plan["status"] != "frozen_before_paid_opened_diagnostic"
            or plan["opened_native_visits_max"] != 18
            or plan["optimizer_updates_max"] != 0
            or plan["source_sha256"]["probe"] != sha(Path(__file__))
            or ready["status"] != "two_exact_cuda_sentinels_passed"
            or ready["plan_sha256"] != sha(plan_path)
            or ready["closure_sha256"] != plan["required_installed_closure_content_sha256"]
            or len(ready["sentinels"]) != 2
            or guard["plan_sha256"] != sha(plan_path)
            or guard["image"] != "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
            or guard["gpu_id"] != "NVIDIA GeForce RTX 4090"
            or guard["listed_rate_usd_per_hour"] != 0.89
            or guard["provider_desired_status"] != "RUNNING"
            or guard["provider_runtime_status"] != "running"):
        raise RuntimeError("Frozen source, exact CUDA sentinels or guarded host missing")
    for sentinel, expected_seed in zip(ready["sentinels"], (46998, 46999)):
        report = LAUNCH / f"sentinel-{sentinel['name']}.json"
        if sentinel["seed"] != expected_seed or sha(report) != sentinel["report_sha256"]:
            raise RuntimeError("Exact one-step sentinel report changed")
    for name, rel in plan["source_paths"].items():
        source = (ROOT / rel).resolve()
        if (not source.is_relative_to(ROOT.resolve()) or not source.is_file()
                or sha(source) != plan["source_sha256"][name]):
            raise RuntimeError(f"Frozen source changed: {name}")
    now = datetime.now(timezone.utc)
    attested = datetime.fromisoformat(guard["attested_at_utc"])
    deadline = datetime.fromisoformat(guard["hard_deadline_utc"])
    if (now - attested).total_seconds() > 180 or now >= deadline:
        raise RuntimeError("Refresh live guard attestation or stop: budget deadline reached")
    return deadline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Execute locally after source check")
    args = parser.parse_args()
    for path, expected in EXPECTED.items():
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"Frozen source changed: {path}: {actual} != {expected}")
    deadline = guarded_runtime() if args.run else None
    if args.run:
        OUT.mkdir(parents=True, exist_ok=True)
        lock = OUT / "execution.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(f"plan_sha256={sha(LAUNCH / 'plan.json')}\n")
    rows = []
    def save_progress() -> None:
        result = {"schema": "robot-learning-lab.cove50-signed-stop-probe.v1",
                  "mode": "guarded_cuda_native" if args.run else "frozen_dry_run",
                  "source_sha256": {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
                  "claims": "Opened diagnostic only; no learned update or sealed score",
                  "rows": rows}
        if args.run:
            target = OUT / "manifest.json"
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(result, indent=2) + "\n")
            temp.replace(target)
        else:
            print(json.dumps(result, indent=2))

    if args.run:
        save_progress()
    for name, checkpoint, task, (vx, vy, yaw), stop, seeds in CASES:
        for seed in seeds:
            stem = f"{name}-seed{seed}"
            report = OUT / f"{stem}.json"
            trace = OUT / f"{stem}.npz"
            command = [sys.executable, str(EVALUATOR), str(checkpoint),
                       "--task", task, "--body-xml", str(BODY),
                       "--physics-model", "frame50", "--yaw-only-head",
                       "--arms-down", "--worlds", "1", "--seed", str(seed),
                       "--velocity-x", str(vx), "--velocity-y", str(vy),
                       "--angular-z", str(yaw), "--motor-rail-v", "12",
                       "--command-delay-ms", "0", "--duration-s", "8",
                       "--physics-dt", "0.002",
                       "--out", str(report), "--diagnostic-trace", str(trace)]
            if checkpoint == FORWARD:
                command += ["--ankle-target-margin-rad", "0.02"]
            if stop is not None:
                command += ["--stop-at-s", str(stop)]
            row = {"case": name, "seed": seed, "command": command,
                   "report": str(report), "trace": str(trace)}
            if args.run:
                remaining = (deadline - datetime.now(timezone.utc)).total_seconds() - 120
                if remaining <= 0:
                    row["fault"] = "guard_deadline_with_120s_sync_stop_reserve"
                    rows.append(row)
                    save_progress()
                    break
                if report.exists() or trace.exists():
                    row["fault"] = "no_replacement_existing_output"
                    rows.append(row)
                    save_progress()
                    break
                with (OUT / f"{stem}.log").open("w") as log:
                    try:
                        proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                              stderr=subprocess.STDOUT, check=False,
                                              timeout=min(90, remaining))
                    except subprocess.TimeoutExpired:
                        row["fault"] = "native_case_timeout"
                        rows.append(row)
                        save_progress()
                        break
                row["exit_code"] = proc.returncode
                if proc.returncode:
                    row["fault"] = "native_case_failed"
                    rows.append(row)
                    save_progress()
                    break
                try:
                    native = json.loads(report.read_text())
                    row.update(validate_native_case(native, trace, seed=seed,
                                                    checkpoint=checkpoint, task=task,
                                                    vx=vx, vy=vy, yaw=yaw, stop=stop))
                    row["native_initial_state_sha256"] = native["initial_state_sha256"]
                    row["native_first_action_sha256"] = native["first_action_sha256"]
                    row["report_sha256"] = sha(report)
                    row["trace_sha256"] = sha(trace)
                except Exception as exc:
                    row["fault"] = f"native_output_parity_{type(exc).__name__}"
                    rows.append(row)
                    save_progress()
                    break
            rows.append(row)
            if args.run:
                save_progress()
        if args.run and rows[-1].get("fault"):
            break
    save_progress()


if __name__ == "__main__":
    main()
