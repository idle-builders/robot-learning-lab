"""Fresh, source-bound whole-trajectory support and short-step measurement sweep.

This runner evaluates saved actors only. It never starts training or retries an
incomplete original. Run on the shared CUDA worker, then sync the whole output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.cove50_support_recovery.summarize_release_v19 import sole_corners

TRAINER = ROOT / "experiments/compact_morphology/cove50_support_recovery/train_release_residual_v19.py"
WALK_EVALUATOR = ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
WALK_ACTOR = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
RELEASE_ACTOR = ROOT / "artifacts/ape-low-level-training-20261015/support-release-v19/model_3.pt"
POSE = ROOT / "artifacts/ape-cove50-support-recovery-20261009"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_original(cmd: list[str], report: Path, trace: Path, video: Path) -> None:
    log = report.with_suffix(".log")
    present = [path.exists() for path in (report, trace, video, log)]
    if any(present):
        if not all(present):
            raise RuntimeError(f"Incomplete original {report}; inspect it, do not rerun silently")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "experiments/compact_morphology/mjlab_ape/src") + ":" + str(ROOT)
    env["MUJOCO_GL"] = "egl"
    with log.open("x") as stream:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=300, check=False)
    if result.returncode or not all(path.exists() for path in (report, trace, video)):
        raise RuntimeError(f"Original failed or incomplete: {report}; see {log}")


def support_originals(method: dict, out: Path) -> None:
    for case in method["support_cases"]:
        for controller in ("reference", "checkpoint"):
            stem = f"support-seed{case['seed']}-{controller}"
            report, trace, video = (out / f"{stem}.{suffix}"
                                    for suffix in ("json", "npz", "mp4"))
            cmd = [sys.executable, str(TRAINER), "--body-xml", str(BODY),
                   "--right-pose-json", str(POSE / "right-low-swing-endpoint.json"),
                   "--left-pose-json", str(POSE / "left-low-swing-endpoint.json"),
                   "--lesson", "release_transition", "--reference-residual",
                   "--preflight-only", "--preflight-reward-controller",
                   "reference" if controller == "reference" else "ppo",
                   "--preflight-side", case["side"], "--seed", str(case["seed"]),
                   "--eval-device", "cuda:0", "--run-name", "support-step-sweep-20261016",
                   "--preflight-out", str(report), "--eval-video", str(video)]
            if case["domain_variation"]:
                cmd += ["--domain-variation", "--eval-fixed-lag-steps", str(case["fixed_lag_steps"])]
            if controller == "checkpoint":
                cmd += ["--eval-checkpoint", str(RELEASE_ACTOR)]
            run_original(cmd, report, trace, video)


def walk_originals(method: dict, out: Path) -> None:
    for case in method["step_cases"]:
        stem = f"step-seed{case['seed']}"
        report, trace, video = (out / f"{stem}.{suffix}"
                                for suffix in ("json", "npz", "mp4"))
        cmd = [sys.executable, str(WALK_EVALUATOR), str(WALK_ACTOR),
               "--task", "Mjlab-Velocity-Flat-Ape-Elbows", "--body-xml", str(BODY),
               "--physics-model", "frame50", "--initial-root-z", "0.231",
               "--yaw-only-head", "--arms-down", "--worlds", "1",
               "--seed", str(case["seed"]), "--velocity-x", "0.25",
               "--velocity-y", "0", "--angular-z", "0",
               "--motor-rail-v", str(case["motor_rail_v"]),
               "--command-delay-ms", str(case["command_delay_ms"]),
               "--ankle-target-margin-rad", "0.02", "--physics-dt", "0.002",
               "--stop-at-s", str(case["stop_at_s"]), "--duration-s", "3.0",
               "--out", str(report), "--diagnostic-trace", str(trace),
               "--video", str(video)]
        run_original(cmd, report, trace, video)


def support_measurements(case: dict, controller: str, out: Path,
                         model: mujoco.MjModel, data: mujoco.MjData) -> dict:
    stem = f"support-seed{case['seed']}-{controller}"
    report_path = out / f"{stem}.json"
    trace_path = out / f"{stem}.npz"
    video_path = out / f"{stem}.mp4"
    report = json.loads(report_path.read_text())
    ep = report["reward_episode_preflight"]
    if report["body_sha256"] != sha(BODY) or ep["seed"] != case["seed"] or ep["side"] != case["side"]:
        raise RuntimeError(f"Support source identity mismatch: {stem}")
    if ep["eval_device"] != "cuda:0" or ep["controller"] != ("reference" if controller == "reference" else "ppo"):
        raise RuntimeError(f"Support runtime/controller mismatch: {stem}")
    expected_actor = None if controller == "reference" else sha(RELEASE_ACTOR)
    if ep["checkpoint_sha256"] != expected_actor:
        raise RuntimeError(f"Support checkpoint mismatch: {stem}")
    with np.load(trace_path, allow_pickle=False) as z:
        initial, initial_vel, qpos = z["initial_qpos"], z["initial_qvel"], z["qpos"]
    if len(qpos) != len(ep["rows"]) or len(qpos) != ep["steps"]:
        raise RuntimeError(f"Support trace/report length mismatch: {stem}")
    if hashlib.sha256(initial.tobytes() + initial_vel.tobytes()).hexdigest() != ep["initial_state_sha256"]:
        raise RuntimeError(f"Support initial physical state mismatch: {stem}")
    side = case["side"]
    other = "left" if side == "right" else "right"
    start_patch = sole_corners(model, data, initial, side)
    max_slip = 0.0
    max_full_corner_z = -1e9
    airborne_s = 0.0
    max_airborne_s = current_airborne_s = 0.0
    for row, q in zip(ep["rows"], qpos, strict=True):
        stance = sole_corners(model, data, q, side)
        swing = sole_corners(model, data, q, other)
        max_slip = max(max_slip, float(np.linalg.norm(stance[:, :2] - start_patch[:, :2], axis=1).max()))
        corner_z = float(swing[:, 2].min())
        max_full_corner_z = max(max_full_corner_z, corner_z)
        in_release_window = 2.0 <= row["time_s"] < 5.0
        airborne = in_release_window and row["feet_grounded"] == ([False, True] if side == "right" else [True, False])
        if airborne:
            airborne_s += .02
            current_airborne_s += .02
            max_airborne_s = max(max_airborne_s, current_airborne_s)
        else:
            current_airborne_s = 0.0
    rows = ep["rows"]
    load_window = [r["stance_load_fraction"] for r in rows if 1.8 <= r["time_s"] <= 2.2]
    return {
        "seed": case["seed"], "side": side, "controller": controller,
        "condition": "source-reset-varied" if case["domain_variation"] else "nominal",
        "fixed_lag_steps": case["fixed_lag_steps"],
        "duration_s": float(rows[-1]["time_s"]),
        "max_stance_load_fraction_1p8_to_2p2_s": max(load_window) if load_window else None,
        "max_full_swing_sole_corner_z_m": max_full_corner_z,
        "airborne_time_2_to_5_s": round(airborne_s, 3),
        "longest_continuous_airborne_s": round(max_airborne_s, 3),
        "max_stance_patch_motion_m": max_slip,
        "max_tilt_deg": max(r["tilt_deg"] for r in rows),
        "terminal_pelvis_z_m": rows[-1]["pelvis_z_m"],
        "terminal_tilt_deg": rows[-1]["tilt_deg"],
        "rated_leg_time_s": round(.02 * sum(r["rated_leg"] for r in rows), 3),
        "rated_arm_time_s": round(.02 * sum(r["rated_arm"] for r in rows), 3),
        "nonsole_or_self_contact_time_s": round(.02 * sum(r["nonsole_contact"] or r["self_contact"] for r in rows), 3),
        "max_hand_force_n": max(r["hand_force_n"] for r in rows),
        "initial_state_sha256": ep["initial_state_sha256"],
        "initial_domain_sha256": ep["initial_domain_sha256"],
        "report_sha256": sha(report_path), "trace_sha256": sha(trace_path),
        "video_sha256": sha(video_path),
    }


def step_measurements(case: dict, out: Path) -> dict:
    stem = f"step-seed{case['seed']}"
    report_path = out / f"{stem}.json"
    trace_path = out / f"{stem}.npz"
    video_path = out / f"{stem}.mp4"
    report = json.loads(report_path.read_text())
    if (report["body_sha256"] != sha(BODY) or report["seed"] != case["seed"]
            or report["candidate_motor_rail_v"] != case["motor_rail_v"]
            or report["candidate_command_delay_ms"] != case["command_delay_ms"]
            or report["stop_at_s"] != case["stop_at_s"] or len(report["worlds"]) != 1):
        raise RuntimeError(f"Step source identity mismatch: {stem}")
    with np.load(trace_path, allow_pickle=False) as z:
        t = z["time_s"]
        active = z["active"][:, 0].astype(bool)
        roots = z["root_pose"][:, 0, :3]
        found = z["sole_found"][:, 0].astype(bool)
        efforts = np.abs(z["actuator_force"][:, 0])
        limits = z["rated_effort_limits_nm"]
        hands = z["hand_force"][:, 0]
        body = z["body_contact"][:, 0].astype(bool)
    if len(t) != len(active) or not active.any():
        raise RuntimeError(f"Step empty trace: {stem}")
    at_stop = int(np.argmin(np.abs(t - case["stop_at_s"])))
    valid = np.flatnonzero(active)
    last = int(valid[-1])
    touchdown_events = sorted((round(float(t[i]), 3), side)
                              for side in range(2)
                              for i in np.where(np.diff(found[:, side].astype(int)) == 1)[0] + 1
                              if active[i] and 0.16 <= t[i] <= case["stop_at_s"])
    post = (t >= 2.0) & active
    return {
        "seed": case["seed"], "motor_rail_v": case["motor_rail_v"],
        "command_delay_ms": case["command_delay_ms"], "stop_at_s": case["stop_at_s"],
        "duration_s": float(t[last]),
        "forward_at_stop_m": float(roots[at_stop, 0] - roots[0, 0]),
        "net_forward_m": float(roots[last, 0] - roots[0, 0]),
        "travel_after_stop_m": float(np.linalg.norm(roots[last, :2] - roots[at_stop, :2])),
        "touchdown_events_before_stop": touchdown_events,
        "late_both_soles_fraction": float(found[post].all(axis=1).mean()) if post.any() else None,
        "rated_leg_time_moving_s": round(.02 * np.sum(np.any(efforts[(t <= case["stop_at_s"]) & active] >= .995 * limits, axis=1)), 3),
        "max_hand_force_n": float(hands[active].max()),
        "body_contact_time_s": round(.02 * int(body[active].sum()), 3),
        "report_sha256": sha(report_path), "trace_sha256": sha(trace_path),
        "video_sha256": sha(video_path),
    }


def summarize(method: dict, out: Path, method_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(BODY))
    data = mujoco.MjData(model)
    support = []
    for case in method["support_cases"]:
        pair = [support_measurements(case, controller, out, model, data)
                for controller in ("reference", "checkpoint")]
        if pair[0]["initial_state_sha256"] != pair[1]["initial_state_sha256"] or pair[0]["initial_domain_sha256"] != pair[1]["initial_domain_sha256"]:
            raise RuntimeError(f"Support pair {case['seed']} did not share its source state/domain")
        support.extend(pair)
    steps = [step_measurements(case, out) for case in method["step_cases"]]
    report = {
        "schema": "robot-learning-lab.support-step-sweep-20261016.measurements.v1",
        "method_sha256": sha(method_path), "body_sha256": sha(BODY),
        "release_actor_sha256": sha(RELEASE_ACTOR), "walk_actor_sha256": sha(WALK_ACTOR),
        "native_original_episodes": len(support) + len(steps),
        "independent_start_families": len(method["support_cases"]) + len(method["step_cases"]),
        "world_slots_max": 1, "new_optimizer_updates": 0, "new_policy_checkpoints": 0,
        "support": support, "short_step": steps,
        "scope": "Fresh saved-controller Direct Frame CUDA trajectories only; no redesigned-body or physical qualification.",
    }
    (out / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "support", "step", "summarize"), default="all")
    args = parser.parse_args()
    method = json.loads(args.method.read_text())
    if method["schema"] != "robot-learning-lab.support-step-sweep-20261016.method.v1":
        raise RuntimeError("Unexpected method")
    for relative, expected in method["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Pinned source changed: {relative}")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("all", "support"):
        support_originals(method, args.out)
    if args.stage in ("all", "step"):
        walk_originals(method, args.out)
    if args.stage in ("all", "summarize"):
        summarize(method, args.out, args.method)


if __name__ == "__main__":
    main()
