"""Summarize native effort traces by support phase and joint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts/ape-cove50-ankle-drive-screen-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def summarize(name: str) -> dict:
    trace_path = ARTIFACTS / f"{name}.npz"
    report_path = ARTIFACTS / f"{name}.json"
    plan_path = ARTIFACTS / f"{name}-plan.json"
    data = np.load(trace_path)
    report = json.loads(report_path.read_text())
    plan = json.loads(plan_path.read_text())
    active = data["active"]
    found = data["sole_found"] > 0
    phase = {"double": found[:, :, 0] & found[:, :, 1],
             "left_only": found[:, :, 0] & ~found[:, :, 1],
             "right_only": found[:, :, 1] & ~found[:, :, 0],
             "flight": ~found[:, :, 0] & ~found[:, :, 1]}
    torque = np.abs(data["actuator_force"])
    speed = np.abs(data["joint_vel"])
    rated = data["rated_effort_limits_nm"]
    saturated = torque >= 0.995 * rated[None, None, :]
    names = data["leg_names"].tolist()
    dt = float(data["step_dt_s"])
    by_phase = {}
    for phase_name, mask in phase.items():
        sample = mask & active
        by_phase[phase_name] = {
            "frame_fraction": float(sample.sum() / max(1, active.sum())),
            "mean_abs_sole_vertical_force_n_left_right": [mean(np.abs(
                data["sole_contact_force"][:, :, side, 2][sample]))
                for side in (0, 1)],
            "fraction_with_any_rated_leg": mean(saturated.any(axis=2)[sample]),
            "rated_fraction_by_joint": {joint: mean(saturated[:, :, index][sample])
                                        for index, joint in enumerate(names)},
        }
    joint_summary = {}
    for index, joint in enumerate(names):
        mask = active
        clipped = saturated[:, :, index] & mask
        joint_summary[joint] = {
            "rated_fraction": mean(saturated[:, :, index][mask]),
            "mean_abs_torque_nm": mean(torque[:, :, index][mask]),
            "peak_abs_torque_nm": float(torque[:, :, index][mask].max()),
            "mean_abs_speed_rad_s": mean(speed[:, :, index][mask]),
            "mean_abs_speed_while_rated_rad_s": mean(speed[:, :, index][clipped]),
            "abs_mechanical_work_j": float((torque[:, :, index] * speed[:, :, index] * mask).sum() * dt),
        }
    worlds = report["worlds"]
    move = active & (data["time_s"][:, None] <= plan["definition"].get("stop", 8.0))
    after_stop = active & ~move
    outcome = {
        "worlds": len(worlds),
        "survived_8s": sum(world["duration_s"] >= 8.0 for world in worlds),
        "one_meter_alternating": report["one_meter_with_alternating_steps"],
        "mean_forward_m": float(np.mean([world["forward_m"] for world in worlds])),
        "mean_lateral_m": float(np.mean([world["lateral_m"] for world in worlds])),
        "mean_net_yaw_rad": float(np.mean([world["net_yaw_rad"] for world in worlds])),
        "mean_any_rated_leg_fraction": mean(saturated.any(axis=2)[active]),
        "mean_any_rated_leg_fraction_while_commanded": mean(saturated.any(axis=2)[move]),
        "mean_any_rated_leg_fraction_after_stop": mean(saturated.any(axis=2)[after_stop]),
        "total_sole_slip_m_left_right": [float((data["sole_speed"][:, :, side] *
                                                 found[:, :, side] * active).sum() * dt / len(worlds))
                                         for side in (0, 1)],
        "max_hand_force_n": max(world["hand_force_n"] for world in worlds),
        "max_body_contact_fraction": max(world["fraction_frames_with_body_contact"] or 0
                                         for world in worlds),
    }
    return {"case": name, "definition": plan["definition"], "sources": plan["sources"],
            "trace": {"path": str(trace_path.relative_to(ROOT)), "sha256": sha(trace_path),
                      "frames": int(active.shape[0]), "step_s": dt},
            "replay": {"path": str(report_path.relative_to(ROOT)), "sha256": sha(report_path)},
            "outcome": outcome, "support_phase": by_phase, "joints": joint_summary}


def main() -> None:
    cases = sorted(path.stem for path in ARTIFACTS.glob("*.npz")
                   if (ARTIFACTS / f"{path.stem}.json").exists())
    matrix = {"schema": "robot-learning-lab.cove50-effort-phase-matrix.v1",
              "status": "native_diagnostic_replays",
              "cases": [summarize(case) for case in cases]}
    (ARTIFACTS / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for family in ("forward", "turn"):
        rows = []
        for case in matrix["cases"]:
            if not (case["case"].startswith(family) or family == "forward" and case["case"] == "passive-zero"):
                continue
            outcome = case["outcome"]
            rows.append({
                "case": case["case"],
                "report": case["replay"]["path"],
                "survived": outcome["survived_8s"],
                "one_meter_alternating": outcome["one_meter_alternating"],
                "mean_forward_m": round(outcome["mean_forward_m"], 3),
                "mean_lateral_m": round(outcome["mean_lateral_m"], 3),
                "mean_yaw_rad": round(outcome["mean_net_yaw_rad"], 3),
                "mean_moving_rated_leg_fraction": outcome["mean_any_rated_leg_fraction_while_commanded"],
                "mean_rated_leg_fraction": outcome["mean_any_rated_leg_fraction"],
                "left_hip_roll_rated_fraction": case["joints"]["left_hip_roll"]["rated_fraction"],
                "right_hip_roll_rated_fraction": case["joints"]["right_hip_roll"]["rated_fraction"],
                "left_ankle_rated_fraction": case["joints"]["left_ankle_pitch"]["rated_fraction"],
                "right_ankle_rated_fraction": case["joints"]["right_ankle_pitch"]["rated_fraction"],
                "left_only_support_fraction": case["support_phase"]["left_only"]["frame_fraction"],
                "right_only_support_fraction": case["support_phase"]["right_only"]["frame_fraction"],
                "sole_slip_m_left_right": outcome["total_sole_slip_m_left_right"],
                "max_hand_force_n": outcome["max_hand_force_n"],
            })
        summary = {"schema": "robot-learning-lab.cove50-effort-summary-matrix.v1",
                   "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
                   "selected_body_sha256": sha(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"),
                   "rated_definition": "Fraction of native 50 Hz frames with any leg actuator at or above 99.5% of that case's catalog-rated effort. Moving excludes the interval after an explicit stop command.",
                   "cases": rows}
        (ARTIFACTS / f"{family}-summary-matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    for case in matrix["cases"]:
        result = case["outcome"]
        top = sorted(case["joints"], key=lambda joint: case["joints"][joint]["rated_fraction"],
                     reverse=True)[:3]
        print(case["case"], result["survived_8s"], result["mean_forward_m"],
              result["mean_net_yaw_rad"], result["mean_any_rated_leg_fraction"], top)


if __name__ == "__main__":
    main()
