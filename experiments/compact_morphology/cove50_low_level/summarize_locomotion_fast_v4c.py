"""Source-check and score the eight paired V4c development starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/locomotion-fast-v4c-a40"
EVIDENCE = OUT / "candidate"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(world: dict) -> tuple[bool, list[str]]:
    reasons = []
    if world["duration_s"] < 7.98 or not world["ended_by_timeout"]:
        reasons.append("fell_or_ended_early")
    if world["forward_m"] < .7:
        reasons.append("insufficient_forward_travel")
    if world["alternating_touchdowns"] < 6:
        reasons.append("insufficient_alternating_touchdowns")
    if world["hand_force_n"] > 1:
        reasons.append("forbidden_hand_contact")
    if world["stop"]["travel_after_command_m"] > .12:
        reasons.append("stop_tail_over_120mm")
    if world["stop"]["late_double_support_fraction"] < .8:
        reasons.append("insufficient_late_double_support")
    return not reasons, reasons


def main() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    method = json.loads((OUT / "method.json").read_text())
    packet = json.loads((OUT / "packet.json").read_text())
    if (result["method_sha256"] != sha(OUT / "method.json")
            or packet["source_method_sha256"] != result["method_sha256"]
            or result["optimizer_updates"] != 24 or result["native_visits"] != 16):
        raise RuntimeError("V4c method/optimizer identity differs")
    for name, expected in result["files"].items():
        if sha(EVIDENCE / name) != expected:
            raise RuntimeError(f"V4c source hash differs: {name}")
    rows = []
    for case in method["development_cases"]:
        pair = {"seed": case["seed"], "speed_m_s": case["velocity_x_m_s"],
                "rail_v": case["motor_rail_v"], "delay_ms": case["command_delay_ms"],
                "slice": "nominal" if case["motor_rail_v"] == 12 and case["command_delay_ms"] == 0 else "stress"}
        starts = []
        for arm in ("baseline", "candidate"):
            label = f"seed{case['seed']}-{arm}"
            report = json.loads((EVIDENCE / f"{label}.json").read_text())
            if (report["seed"] != case["seed"]
                    or report["body_sha256"] != method["source"]["body"]["sha256"]
                    or report["task"] != "Mjlab-Velocity-Flat-Ape-Elbows"
                    or report["physics_model"] != "frame50"
                    or report["episode_length_s"] != case["duration_s"]
                    or report["initial_root_z_m"] != case["initial_root_z_m"]
                    or report["speed_command_m_s"] != case["velocity_x_m_s"]
                    or report["velocity_command_m_s"] != [case["velocity_x_m_s"], 0.0]
                    or report["physics_dt_s"] != .002
                    or report["ankle_target_margin_rad"] != .02
                    or not report["yaw_only_head"] or not report["arms_down"]
                    or report["zero_action"]
                    or not report["checkpoint"].endswith(
                        "ankleonly100-model844.pt" if arm == "baseline" else "candidate-final.pt")
                    or report["candidate_motor_rail_v"] != case["motor_rail_v"]
                    or report["candidate_command_delay_ms"] != case["command_delay_ms"]
                    or report["stop_at_s"] != case["stop_at_s"]
                    or len(report["worlds"]) != 1):
                raise RuntimeError(f"V4c report identity differs: {label}")
            with np.load(EVIDENCE / f"{label}.npz", allow_pickle=False) as trace:
                if (len(trace["time_s"]) < 50
                        or not np.isfinite(trace["raw_action"]).all()
                        or not np.isfinite(trace["actuator_force"]).all()):
                    raise RuntimeError(f"V4c trace invalid: {label}")
            world = report["worlds"][0]
            passed, reasons = gate(world)
            pair[arm] = {"passed": passed, "failure_reasons": reasons,
                         "duration_s": world["duration_s"],
                         "forward_m": world["forward_m"],
                         "alternating_touchdowns": world["alternating_touchdowns"],
                         "hand_force_n": world["hand_force_n"],
                         "tail_m": world["stop"]["travel_after_command_m"],
                         "late_double_support": world["stop"]["late_double_support_fraction"],
                         "moving_rated_leg_fraction": world["fraction_moving_frames_with_saturated_leg"],
                         "report_sha256": sha(EVIDENCE / f"{label}.json"),
                         "trace_sha256": sha(EVIDENCE / f"{label}.npz")}
            starts.append(report["initial_state_sha256"])
        if starts[0] != starts[1]:
            raise RuntimeError(f"V4c paired start physical state differs: {case['seed']}")
        pair["matched_initial_state_sha256"] = starts[0]
        pair["rescue"] = not pair["baseline"]["passed"] and pair["candidate"]["passed"]
        pair["regression"] = pair["baseline"]["passed"] and not pair["candidate"]["passed"]
        pair["catastrophic_regression"] = (
            (pair["baseline"]["duration_s"] >= 7.98
             and pair["candidate"]["duration_s"] < 7.98)
            or (pair["baseline"]["hand_force_n"] <= 1
                and pair["candidate"]["hand_force_n"] > 1))
        rows.append(pair)
    mean = lambda arm: sum(row[arm]["moving_rated_leg_fraction"] for row in rows) / len(rows)
    report = {"schema": "robot-learning-lab.locomotion-fast-v4c.development-score.v1",
              "scorer_sha256": sha(Path(__file__)),
              "method_sha256": sha(OUT / "method.json"),
              "candidate_result_sha256": sha(EVIDENCE / "result.json"),
              "gpu": packet["gpu"], "matched_cases": len(rows), "native_visits": 2 * len(rows),
              "baseline_passes": sum(row["baseline"]["passed"] for row in rows),
              "candidate_passes": sum(row["candidate"]["passed"] for row in rows),
              "rescues": sum(row["rescue"] for row in rows),
              "regressions": sum(row["regression"] for row in rows),
              "catastrophic_regressions": sum(row["catastrophic_regression"] for row in rows),
              "mean_moving_rated_leg_fraction_baseline": mean("baseline"),
              "mean_moving_rated_leg_fraction_candidate": mean("candidate"),
              "candidate_mean_rated_leg_duty_target_at_or_below_0_8": mean("candidate") <= .8,
              "development_target_met": (sum(row["candidate"]["passed"] for row in rows) >= 6
                                         and not any(row["catastrophic_regression"] for row in rows)
                                         and mean("baseline") - mean("candidate") >= .05),
              "rows": rows,
              "claim_limit": "Eight A40 development pairs only; no independent score, reward shortcut clearance, motor qualification or body transfer."}
    path = OUT / "development-score.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
