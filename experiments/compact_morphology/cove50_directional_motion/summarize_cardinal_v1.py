"""Score the frozen eight paired cardinal development starts from native evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v1"
EVIDENCE = OUT / "candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(world: dict, family: str) -> list[str]:
    reasons = []
    if world["duration_s"] < 7.98 or not world["ended_by_timeout"]:
        reasons.append("fell_or_ended_early")
    progress_min = .7 if family == "forward" else .25 if family == "backward" else .15
    if world["command_progress_m"] < progress_min:
        reasons.append("insufficient_signed_progress")
    if world["alternating_touchdowns"] < (6 if family == "forward" else 2):
        reasons.append("insufficient_alternating_touchdowns")
    if world["hand_force_n"] > 1:
        reasons.append("forbidden_hand_contact")
    stop = world.get("stop")
    if stop is None:
        reasons.append("stop_not_observed")
    else:
        if stop["travel_after_command_m"] > .12:
            reasons.append("stop_tail_over_120mm")
        if stop["late_double_support_fraction"] < .8:
            reasons.append("insufficient_late_double_support")
    return reasons


def main() -> None:
    method = json.loads((OUT / "method.json").read_text())
    result = json.loads((EVIDENCE / "result.json").read_text())
    if (result["method_sha256"] != sha(OUT / "method.json")
            or result["optimizer_updates"] != method["learner"]["additional_updates"]
            or result["native_visits"] != 2 * len(method["development_cases"])):
        raise RuntimeError("Candidate method or update identity differs")
    for name, expected in result["files"].items():
        if sha(EVIDENCE / name) != expected:
            raise RuntimeError(f"Native evidence hash differs: {name}")
    rows = []
    for case in method["development_cases"]:
        row = {k: case[k] for k in ("seed", "family", "velocity_x_m_s", "velocity_y_m_s",
                                    "motor_rail_v", "command_delay_ms")}
        initial = []
        for arm in ("baseline", "candidate"):
            label = f"seed{case['seed']}-{arm}"
            path = EVIDENCE / f"{label}.json"
            report = json.loads(path.read_text())
            if (report["seed"] != case["seed"]
                    or report["body_sha256"] != method["source"]["body"]["sha256"]
                    or report["velocity_command_m_s"] != [case["velocity_x_m_s"], case["velocity_y_m_s"]]
                    or report["candidate_motor_rail_v"] != case["motor_rail_v"]
                    or report["candidate_command_delay_ms"] != case["command_delay_ms"]
                    or report["stop_at_s"] != case["stop_at_s"]
                    or report["episode_length_s"] != case["duration_s"]
                    or len(report["worlds"]) != 1):
                raise RuntimeError(f"Native report identity differs: {label}")
            with np.load(EVIDENCE / f"{label}.npz", allow_pickle=False) as trace:
                if (len(trace["time_s"]) < 50
                        or not np.isfinite(trace["raw_action"]).all()
                        or not np.isfinite(trace["actuator_force"]).all()):
                    raise RuntimeError(f"Native trace invalid: {label}")
            world = report["worlds"][0]
            failures = gate(world, case["family"])
            row[arm] = {
                "passed": not failures, "failures": failures,
                "duration_s": world["duration_s"],
                "signed_progress_m": world["command_progress_m"],
                "alternating_touchdowns": world["alternating_touchdowns"],
                "hand_force_n": world["hand_force_n"],
                "stop_tail_m": world["stop"]["travel_after_command_m"] if world.get("stop") else None,
                "late_double_support": world["stop"]["late_double_support_fraction"] if world.get("stop") else None,
                "moving_rated_leg_duty": world["fraction_moving_frames_with_saturated_leg"],
                "reported_contact_slip_m": sum(world["contact_slip_m_left_right"]),
                "report_sha256": sha(path), "trace_sha256": sha(EVIDENCE / f"{label}.npz"),
            }
            initial.append(report["initial_state_sha256"])
        if initial[0] != initial[1]:
            raise RuntimeError(f"Matched physical start differs for seed {case['seed']}")
        row["initial_state_sha256"] = initial[0]
        row["rescue"] = not row["baseline"]["passed"] and row["candidate"]["passed"]
        row["regression"] = row["baseline"]["passed"] and not row["candidate"]["passed"]
        row["catastrophic_forward_regression"] = (case["family"] == "forward"
            and row["baseline"]["duration_s"] >= 7.98
            and (row["candidate"]["duration_s"] < 7.98 or row["candidate"]["hand_force_n"] > 1))
        rows.append(row)
    new = [r for r in rows if r["family"] != "forward"]
    report = {
        "schema": "robot-learning-lab.cardinal-v1.development-score.v1",
        "method_sha256": sha(OUT / "method.json"), "result_sha256": sha(EVIDENCE / "result.json"),
        "scorer_sha256": sha(Path(__file__)),
        "matched_families": len(rows), "native_visits": 2 * len(rows),
        "baseline_passes": sum(r["baseline"]["passed"] for r in rows),
        "candidate_passes": sum(r["candidate"]["passed"] for r in rows),
        "new_direction_rescues": sum(r["rescue"] for r in new),
        "new_direction_regressions": sum(r["regression"] for r in new),
        "catastrophic_forward_regressions": sum(r["catastrophic_forward_regression"] for r in rows),
        "pilot_signal_met": (sum(r["rescue"] for r in new) >= 2
                             and not any(r["catastrophic_forward_regression"] for r in rows)),
        "rows": rows,
        "claim_limit": "Eight opened development families on Direct Frame; no independent score, motor qualification or body transfer.",
    }
    (OUT / "development-score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
