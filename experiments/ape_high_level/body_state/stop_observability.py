"""Retrospective 50 Hz body-sensor stop gates on retained Cove-50 native traces.

This is an observability experiment on a rejected historical body. It neither
runs a new controller nor treats simulator root velocity as an actor sensor.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-effort-envelope"
OUTPUT = ROOT / "artifacts/ape-stop-observability-v1"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
CHECKPOINT = ROOT / "artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt"
POINTER = ROOT / "robot/ape-current-design.json"
CASES = ("forward-stop-early", "forward-stop-late")
WINDOW = 20  # Twenty consecutive 50 Hz packets, ending at the current tick.
ACTOR_LIMITS = {"gyro_yaw_rad_s": 0.03,
                "mean_abs_leg_encoder_velocity_rad_s": 0.05,
                "minimum_sole_force_n": 2.0}
EVALUATOR_LIMITS = {"planar_speed_m_s": 0.01, "yaw_speed_rad_s": 0.01}
CHANNELS = ("imu", "encoders", "sole_force")
SCENARIOS = {"imu_only": ("imu",), "encoders_only": ("encoders",),
             "sole_force_only": ("sole_force",), "fused": CHANNELS,
             "fused_imu_lost": CHANNELS}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def first_held(condition: np.ndarray, first_index: int) -> int | None:
    for end in range(first_index + WINDOW - 1, len(condition)):
        if bool(np.all(condition[end - WINDOW + 1:end + 1])):
            return end
    return None


def load_case(case: str) -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    plan = json.loads((SOURCE / f"{case}-plan.json").read_text())
    report = json.loads((SOURCE / f"{case}.json").read_text())
    data = np.load(SOURCE / f"{case}.npz")
    sources = plan["sources"]
    if sources["body"] != sha(BODY) or sources["checkpoint"] != sha(CHECKPOINT):
        raise ValueError(f"Historical source body or checkpoint changed: {case}")
    if report["body_sha256"] != sources["body"] or report["seed"] != plan["definition"]["seed"]:
        raise ValueError(f"Historical native report disagrees with plan: {case}")
    if (data["root_pose"].shape != (400, 4, 7) or
        data["joint_pos"].shape != (400, 4, 10) or
        data["sole_contact_force"].shape != (400, 4, 2, 3) or
        not np.isclose(float(data["step_dt_s"]), 0.02) or
        not np.all(data["active"])):
        raise ValueError(f"Unexpected native trace layout or inactive actor: {case}")
    if not np.isclose(float(data["time_s"][-1]), 8.0):
        raise ValueError(f"Truncated native trace: {case}")
    return plan, report, data


def evaluate_world(case: str, world: int, plan: dict, data: np.lib.npyio.NpzFile) -> tuple[dict, list[dict]]:
    time = data["time_s"].astype(float)
    stop_s = float(plan["definition"]["stop"])
    start = int(np.flatnonzero(np.isclose(time, stop_s))[0])
    dt = float(data["step_dt_s"])
    yaw = yaw_from_quaternion(data["root_pose"][:, world, 3:7].astype(float))
    # A lossless, zero-latency IMU-like yaw sample reconstructed from two saved
    # orientations. This is a favorable sensor bound, not a noisy hardware IMU.
    gyro = np.abs(np.diff(yaw, prepend=yaw[0])) / dt
    joints = data["joint_pos"][:, world].astype(float)
    # Velocity estimated from encoder positions, rather than evaluator joint_vel.
    encoder_velocity = np.mean(np.abs(np.diff(joints, axis=0, prepend=joints[:1])), axis=1) / dt
    sole_force = np.linalg.norm(data["sole_contact_force"][:, world].astype(float), axis=2)
    sole_supported = np.all(sole_force >= ACTOR_LIMITS["minimum_sole_force_n"], axis=1)
    actor_good = {
        "imu": gyro <= ACTOR_LIMITS["gyro_yaw_rad_s"],
        "encoders": encoder_velocity <= ACTOR_LIMITS["mean_abs_leg_encoder_velocity_rad_s"],
        "sole_force": sole_supported,
    }
    # Root translation and yaw are evaluator-only. The truth event requires
    # twenty quiet ticks and no later excursion above the stated limits.
    planar_speed = np.linalg.norm(data["root_velocity"][:, world, :2], axis=1)
    planar_position = data["root_pose"][:, world, :2].astype(float)
    truth_quiet = ((planar_speed <= EVALUATOR_LIMITS["planar_speed_m_s"]) &
                   (gyro <= EVALUATOR_LIMITS["yaw_speed_rad_s"]))
    truth_end = first_held(truth_quiet, start)
    if truth_end is None or not bool(np.all(truth_quiet[truth_end + 1:])):
        raise ValueError(f"Evaluator never remains stationary: {case} world {world}")
    # IMU health fails permanently 0.5 s after the stop command. In this
    # declared dropout, the planner must not infer health from a quiet value.
    health_imu = time < stop_s + 0.5
    gate_condition = {}
    gate_first = {}
    for name, required in SCENARIOS.items():
        eligible = np.logical_and.reduce([actor_good[channel] for channel in required])
        if name == "fused_imu_lost":
            eligible &= health_imu
        gate_condition[name] = eligible
        gate_first[name] = first_held(eligible, start)
    trace_rows = []
    for i in range(len(time)):
        row = {"case": case, "world": world, "sample_at_s": round(float(time[i]), 3),
               "stop_command": int(i >= start), "imu_health": int(health_imu[i]),
               "imu_yaw_rate_rad_s": round(float(gyro[i]), 6),
               "leg_encoder_mean_abs_rad_s": round(float(encoder_velocity[i]), 6),
               "left_sole_force_n": round(float(sole_force[i, 0]), 4),
               "right_sole_force_n": round(float(sole_force[i, 1]), 4),
               "evaluator_planar_speed_m_s": round(float(planar_speed[i]), 6),
               "evaluator_yaw_rate_rad_s": round(float(gyro[i]), 6),
               "evaluator_stationary": int(i >= truth_end)}
        for name, first in gate_first.items():
            row[f"{name}_accepted"] = int(first is not None and i >= first)
        trace_rows.append(row)
    gates = {}
    for name, first in gate_first.items():
        gates[name] = {
            "first_accept_at_s": None if first is None else round(float(time[first]), 3),
            "first_gate_authorized_frame_at_s": None if first is None or first + 1 >= len(time)
            else round(float(time[first + 1]), 3),
            "premature_before_evaluator_stationary": bool(first is not None and first < truth_end),
            "lead_or_lag_vs_truth_s": None if first is None
            else round(float(time[first] - time[truth_end]), 3),
            "evaluator_path_until_stationary_m": None if first is None or first >= truth_end else
            round(float(np.linalg.norm(np.diff(planar_position[first:truth_end + 1], axis=0), axis=1).sum()), 5),
            "nonresponse": first is None,
        }
    row = {"case": case, "world": world, "stop_command_s": stop_s,
           "evaluator_stationary_at_s": round(float(time[truth_end]), 3),
           "first_truly_stationary_frame_at_s": round(float(time[truth_end + 1]), 3),
           "evaluator_root_planar_speed_at_fused_accept_m_s": None if gate_first["fused"] is None
           else round(float(planar_speed[gate_first["fused"]]), 5),
           "gates": gates}
    return row, trace_rows


def main() -> None:
    pointer = json.loads(POINTER.read_text())
    if pointer.get("status") != "rejected_no_selected_mechanical_design" or pointer.get("selected_design_id") is not None:
        raise ValueError("Design selection changed; revisit historical evidence label")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, all_packets, sources = [], [], {}
    for case in CASES:
        plan, report, data = load_case(case)
        sources[case] = {"plan_sha256": sha(SOURCE / f"{case}-plan.json"),
                         "report_sha256": sha(SOURCE / f"{case}.json"),
                         "native_trace_sha256": sha(SOURCE / f"{case}.npz"),
                         "historical_pointer_sha256": plan["sources"]["pointer"],
                         "seed": report["seed"], "stop_command_s": report["stop_at_s"]}
        for world in range(4):
            row, packets = evaluate_world(case, world, plan, data)
            rows.append(row)
            all_packets.extend(packets)
    with (OUTPUT / "packets-and-decisions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_packets[0].keys())
        writer.writeheader()
        writer.writerows(all_packets)
    summary = {}
    for name in SCENARIOS:
        observed = [r["gates"][name] for r in rows]
        offsets = [g["lead_or_lag_vs_truth_s"] for g in observed if not g["nonresponse"]]
        summary[name] = {"accepted": len(offsets), "premature": sum(g["premature_before_evaluator_stationary"] for g in observed),
                         "nonresponse": sum(g["nonresponse"] for g in observed),
                         "lead_or_lag_range_s": None if not offsets else [min(offsets), max(offsets)]}
    result = {
        "schema": "robot-learning-lab.ape-stop-observability.v1",
        "classification": "retrospective native trace sensor ablation; historical rejected 50 cm Direct Frame",
        "current_design_pointer": {"status": pointer["status"], "selected_design_id": pointer["selected_design_id"],
                                   "sha256": sha(POINTER)},
        "native_body_sha256": sha(BODY), "native_checkpoint_sha256": sha(CHECKPOINT),
        "analysis_sha256": sha(Path(__file__)), "sources": sources,
        "packet_period_s": 0.02, "consecutive_packet_window": WINDOW,
        "actor_channels": {"imu": "idealized yaw rate from adjacent root orientations, no root translation",
                           "encoders": "mean absolute leg-joint velocity from adjacent saved positions",
                           "sole_force": "simulated left and right sole contact-force norms"},
        "actor_thresholds": ACTOR_LIMITS, "evaluator_only_thresholds": EVALUATOR_LIMITS,
        "dropout": "IMU health false from stop command + 0.5 s through episode end; fused gate requires fresh healthy packets for entire 400 ms",
        "summary": summary, "rows": rows,
        "limits": [
            "All eight worlds use one seed and the same historical rejected body and checkpoint; this is not held-out tuning or broad robustness.",
            "IMU yaw is orientation-derived and lossless; encoder and sole-force packets are idealized native-sim values with no physical calibration or packet latency.",
            "Root planar speed is evaluator-only; IMU yaw, joints and support cannot alone prove zero translation or absence of foot slip.",
            "The video replays recorded poses. It is not a new closed-loop controller rollout or an owner-triggered native stop.",
        ],
    }
    (OUTPUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
