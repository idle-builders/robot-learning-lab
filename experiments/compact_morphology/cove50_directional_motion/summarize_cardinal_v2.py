"""Score eight fresh three-arm cardinal movement starts from native evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v2"
EVIDENCE = OUT / "candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate(world: dict, family: str) -> list[str]:
    reasons = []
    if world["duration_s"] < 3.98 or not world["ended_by_timeout"]:
        reasons.append("fell_or_ended_early")
    progress_min = .7 if family == "forward" else .25 if family == "backward" else .15
    if world["command_progress_m"] < progress_min:
        reasons.append("insufficient_signed_progress")
    if world["alternating_touchdowns"] < (6 if family == "forward" else 2):
        reasons.append("insufficient_alternating_touchdowns")
    if world["hand_force_n"] > 1:
        reasons.append("forbidden_hand_contact")
    return reasons


def main() -> None:
    method = json.loads((OUT / "method.json").read_text())
    result = json.loads((EVIDENCE / "result.json").read_text())
    if (result["method_sha256"] != sha(OUT / "method.json")
            or result["optimizer_updates"] != method["learner"]["additional_updates"]
            or result["native_visits"] != 3 * len(method["development_cases"])):
        raise RuntimeError("Candidate method or update identity differs")
    for name, expected in result["files"].items():
        if sha(EVIDENCE / name) != expected:
            raise RuntimeError(f"Native evidence hash differs: {name}")
    rows = []
    for case in method["development_cases"]:
        row = {k: case[k] for k in ("seed", "family", "velocity_x_m_s", "velocity_y_m_s",
                                    "motor_rail_v", "command_delay_ms")}
        initial = []
        for arm in ("baseline", "v1", "candidate"):
            label = f"seed{case['seed']}-{arm}"
            path = EVIDENCE / f"{label}.json"
            report = json.loads(path.read_text())
            if (report["seed"] != case["seed"]
                    or report["body_sha256"] != method["source"]["body"]["sha256"]
                    or report["velocity_command_m_s"] != [case["velocity_x_m_s"], case["velocity_y_m_s"]]
                    or report["candidate_motor_rail_v"] != case["motor_rail_v"]
                    or report["candidate_command_delay_ms"] != case["command_delay_ms"]
                    or report["stop_at_s"] is not None
                    or report["episode_length_s"] != case["duration_s"]
                    or len(report["worlds"]) != 1):
                raise RuntimeError(f"Native report identity differs: {label}")
            with np.load(EVIDENCE / f"{label}.npz", allow_pickle=False) as trace:
                if (len(trace["time_s"]) < 50
                        or not np.isfinite(trace["raw_action"]).all()
                        or not np.isfinite(trace["actuator_force"]).all()):
                    raise RuntimeError(f"Native trace invalid: {label}")
                contact_force = np.linalg.norm(trace["sole_contact_force"][:, 0], axis=-1)
                sole_speed = trace["sole_speed"][:, 0]
                loaded = (contact_force > 1.0) & (trace["sole_found"][:, 0] > 0.5)
                loaded_speed_mean = float(sole_speed[loaded].mean()) if loaded.any() else None
                loaded_fraction = float(loaded.mean())
                loaded_motion = (sole_speed * loaded).sum(axis=0) * float(trace["step_dt_s"])
                loaded_samples = loaded.sum(axis=0)
            world = report["worlds"][0]
            failures = gate(world, case["family"])
            row[arm] = {
                "passed": not failures, "failures": failures,
                "duration_s": world["duration_s"],
                "signed_progress_m": world["command_progress_m"],
                "alternating_touchdowns": world["alternating_touchdowns"],
                "hand_force_n": world["hand_force_n"],
                "moving_rated_leg_duty": world["fraction_moving_frames_with_saturated_leg"],
                "loaded_sole_speed_mean_m_s": loaded_speed_mean,
                "loaded_sole_sample_fraction": loaded_fraction,
                "loaded_sole_motion_m_left_right": [float(x) for x in loaded_motion],
                "loaded_sole_samples_left_right": [int(x) for x in loaded_samples],
                "loaded_sole_path_source": "Full force/speed/sole-found time path is in the source-matched NPZ trace",
                "touchdown_sequence": world["touchdown_sequence"],
                "report_sha256": sha(path), "trace_sha256": sha(EVIDENCE / f"{label}.npz"),
            }
            initial.append(report["initial_state_sha256"])
        if len(set(initial)) != 1:
            raise RuntimeError(f"Matched physical start differs for seed {case['seed']}")
        row["initial_state_sha256"] = initial[0]
        row["rescue"] = not row["baseline"]["passed"] and row["candidate"]["passed"]
        row["regression"] = row["baseline"]["passed"] and not row["candidate"]["passed"]
        row["forward_preserved"] = case["family"] != "forward" or row["candidate"]["passed"]
        row["v2_over_v1"] = row["candidate"]["passed"] and not row["v1"]["passed"]
        rows.append(row)
    new = [r for r in rows if r["family"] != "forward"]
    report = {
        "schema": "robot-learning-lab.cardinal-v2.development-score.v1",
        "method_sha256": sha(OUT / "method.json"), "result_sha256": sha(EVIDENCE / "result.json"),
        "scorer_sha256": sha(Path(__file__)),
        "matched_families": len(rows), "native_visits": 3 * len(rows),
        "baseline_passes": sum(r["baseline"]["passed"] for r in rows),
        "v1_passes": sum(r["v1"]["passed"] for r in rows),
        "candidate_passes": sum(r["candidate"]["passed"] for r in rows),
        "new_direction_rescues": sum(r["rescue"] for r in new),
        "new_direction_regressions": sum(r["regression"] for r in new),
        "v2_over_v1_new_direction_passes": sum(r["v2_over_v1"] for r in new),
        "forward_preserved": all(r["forward_preserved"] for r in rows),
        "pilot_signal_met": (sum(r["rescue"] for r in new) >= 2
                             and sum(r["v2_over_v1"] for r in new) >= 2
                             and all(r["forward_preserved"] for r in rows)),
        "rows": rows,
        "claim_limit": "Eight fresh 4 s movement families on Direct Frame; no stop, 8 s endurance, independent score, motor qualification or body transfer.",
    }
    (OUT / "development-score.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
