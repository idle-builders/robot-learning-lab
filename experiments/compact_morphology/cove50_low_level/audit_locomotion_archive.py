"""Audit retained Direct Frame locomotion matrices against their native reports.

This is an offline evidence check, not a new simulation or a policy score.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
SOURCES = (
    "artifacts/ape-cove50-recovery-20261008/ankleonly100-combined-matrix.json",
    "artifacts/ape-cove50-turn-recovery-20261008/turneffort25-combined-matrix.json",
)
OUT = ROOT / "artifacts/ape-cove50-locomotion-audit-20261014/archive-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4)


def audit() -> dict:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    assert pointer["id"] == "cove-50-direct-frame"
    assert pointer["selected_design_id"] is None
    assert pointer["status"] == "rejected_no_selected_mechanical_design"
    body_hash = sha(BODY)
    assert body_hash == "9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c"
    rows = []
    starts = []
    for source in SOURCES:
        matrix = json.loads((ROOT / source).read_text())
        checkpoint = ROOT / matrix["checkpoint"]
        assert sha(checkpoint) == matrix["checkpoint_sha256"]
        for case in matrix["cases"]:
            report_path = ROOT / case["report"]
            report = json.loads(report_path.read_text())
            assert case.get("process_exit_code", case.get("exit_code")) == 0
            assert report["seed"] == case["seed"]
            assert report["body_sha256"] == case["body_sha256"]
            local_body = BODY.parent / Path(report["body_xml"]).name
            assert report["body_sha256"] == sha(local_body)
            assert len(report["worlds"]) == case["worlds"]
            assert report["stop_at_s"] == case["stop_at_s"]
            assert report["candidate_motor_rail_v"] == case["motor_rail_v"]
            assert report["candidate_command_delay_ms"] == case["command_delay_ms"]
            if "velocity_x_m_s" in case:
                assert report["velocity_command_m_s"] == [case["velocity_x_m_s"], 0.0]
                assert report["angular_command_rad_s"] == case["yaw_command_rad_s"]
            else:
                assert report["velocity_command_m_s"] + [report["angular_command_rad_s"]] == case["velocity_command"]
            starts.append((report["seed"], report["initial_state_sha256"]))
            worlds = report["worlds"]
            vx, vy = report["velocity_command_m_s"]
            yaw = report["angular_command_rad_s"]
            if abs(vx) + abs(vy) > 0:
                progress = [w["command_progress_m"] for w in worlds]
                progress_unit = "m"
            else:
                progress = [w["command_yaw_progress_rad"] for w in worlds]
                progress_unit = "rad"
            moving_duty = [w.get("fraction_moving_frames_with_saturated_leg") for w in worlds]
            # The older turn report did not store pre-stop duty. Full-episode duty
            # is only equivalent when there is no stop transition.
            if report["stop_at_s"] is None and any(v is None for v in moving_duty):
                moving_duty = [w["fraction_frames_with_saturated_leg"] for w in worlds]
            row = {
                "source_matrix": source,
                "native_report": case["report"],
                "case": case["case"],
                "seed": case["seed"],
                "worlds": len(worlds),
                "command": [vx, vy, yaw],
                "stop_at_s": report["stop_at_s"],
                "survived": sum(w["ended_by_timeout"] for w in worlds),
                "signed_progress_mean": mean(progress),
                "signed_progress_min": min(progress),
                "progress_unit": progress_unit,
                "alternating_touchdowns_min": min(w["alternating_touchdowns"] for w in worlds),
                "slip_sum_m_mean": mean([sum(w["contact_slip_m_left_right"]) for w in worlds]),
                "slip_sum_m_max": max(round(sum(w["contact_slip_m_left_right"]), 4) for w in worlds),
                "rated_leg_duty_mean": mean([w["fraction_frames_with_saturated_leg"] for w in worlds]),
                "moving_rated_leg_duty_mean": (mean(moving_duty) if all(v is not None for v in moving_duty) else None),
                "stop_travel_m_max": (max(w["stop"]["travel_after_command_m"] for w in worlds)
                                      if report["stop_at_s"] is not None else None),
                "stop_yaw_rad_max_abs": (max(abs(w["stop"]["yaw_after_command_rad"]) for w in worlds)
                                         if report["stop_at_s"] is not None else None),
                "late_speed_m_s_max": (max(w["stop"]["late_mean_planar_speed_m_s"] for w in worlds)
                                       if report["stop_at_s"] is not None else None),
                "late_yaw_speed_rad_s_max": (max(w["stop"]["late_mean_abs_yaw_speed_rad_s"] for w in worlds)
                                             if report["stop_at_s"] is not None else None),
                "late_double_support_min": (min(w["stop"]["late_double_support_fraction"] for w in worlds)
                                            if report["stop_at_s"] is not None else None),
            }
            rows.append(row)
    assert len(starts) == len(set(starts))
    return {
        "schema": "robot-learning-lab.cove50-locomotion-archive-audit.v1",
        "evidence_type": "offline reanalysis of archived native reports; no fresh rollout",
        "training_body_sha256": body_hash,
        "design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "mechanical_selection": "none; Direct Frame is an experimental control body",
        "evaluator_sha256": sha(ROOT / "experiments/compact_morphology/mjlab_ape/evaluate.py"),
        "distinct_native_seed_and_initial_state_groups": len(starts),
        "cases": rows,
        "metric_limits": [
            "Archived turn-stop reports omit pre-stop rated-leg duty; full-episode duty dilutes moving demand.",
            "Cumulative sole-center slip includes unloaded contact and the stop tail; it is not loaded-floor sliding.",
            "Four vector worlds within one report share one RNG seed; a fresh score needs separate seed groups.",
        ],
    }


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit(), indent=2) + "\n")
    print(OUT)
