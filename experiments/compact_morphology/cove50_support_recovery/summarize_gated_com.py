"""Audit and summarize the pinned native load-gated COM experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-gated-com-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    for key in ("selected_design_pointer", "selected_body", "actuator_catalog", "probe_source"):
        pin = plan[key]
        assert sha(ROOT / pin["path"]) == pin["sha256"], key
    for side, pin in plan["baseline_traces"].items():
        assert sha(ROOT / pin["path"]) == pin["sha256"], side
    for line in (OUT / "trials/remote-sha256.txt").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(ROOT / name) == digest, name

    rows = []
    for candidate in plan["cases"]:
        for side in plan["nominal"]["stances"]:
            name = f"{candidate['id']}-{side}-nominal"
            report_path = OUT / "trials" / f"{name}.json"
            report = json.loads(report_path.read_text())
            assert report["design_pointer_sha256"] == plan["selected_design_pointer"]["sha256"]
            assert report["body_sha256"] == plan["selected_body"]["sha256"]
            assert report["stance"] == side
            assert report["stance_com_velocity_horizon_s"] == candidate["velocity_horizon_s"]
            assert report["rail_v"] == 12 and report["delay_ms"] == 0
            assert report["slide_friction"] == 1 and report["seed"] == 7144
            assert report["damping_entry_load_fraction"] == 0.95
            assert report["damping_entry_swing_z_m"] == 0.020
            assert report["damping_ramp_s"] == 0.15
            trace_path = report_path.with_suffix(".npz")
            video_path = report_path.with_suffix(".mp4")
            assert trace_path.is_file() and video_path.is_file()
            if candidate["id"] == "baseline-reload":
                prior = np.load(ROOT / plan["baseline_traces"][side]["path"])
                current = np.load(trace_path)
                for field in ("qpos", "qvel", "joint_target"):
                    assert np.array_equal(prior[field], current[field]), (side, field)
            sign = 1 if side == "right" else -1

            def sample(time_s: float) -> dict:
                row = min(report["timeseries"], key=lambda item: abs(item["time_s"] - time_s))
                return {
                    "time_s": row["time_s"],
                    "com_toward_stance_mm": round(-sign * row["com_minus_stance_world_xy_m"][1] * 1000, 2),
                    "com_velocity_toward_stance_mm_s": round(-sign * row["filtered_com_lateral_velocity_m_s"] * 1000, 2),
                    "stance_load_fraction": row["stance_foot_load_fraction"],
                    "stance_slip_mm": round(row["stance_slip_m"] * 1000, 2),
                    "swing_sole_z_mm": round(row["swing_sole_z_m"] * 1000, 2),
                    "effective_horizon_s": row["effective_com_velocity_horizon_s"],
                    "stance_roll_feedback_rad": row["stance_roll_feedback_rad"],
                    "feet_grounded": row["feet_grounded"],
                }

            first_rated = next((r for r in report["timeseries"] if r["rated_leg_joints"]), None)
            rows.append({
                "case": name, "controller": candidate["id"], "stance": side,
                "velocity_horizon_s": candidate["velocity_horizon_s"],
                "damping_entry_time_s": report["damping_entry_time_s"],
                "duration_s": report["duration_s"],
                "completed_schedule": report["completed_schedule"],
                "exclusive_support_s": report["longest_single_support_s"],
                "rated_leg_fraction": report["rated_leg_fraction"],
                "max_stance_slip_mm": round(report["max_stance_slip_m"] * 1000, 2),
                "first_rated_time_s": first_rated["time_s"] if first_rated else None,
                "first_rated_joints": first_rated["rated_leg_joints"] if first_rated else [],
                "nonsole_floor_frames": report["nonsole_floor_contact_frames"],
                "self_contact_frames": report["self_contact_frames"],
                "hand_contact_frames": report["hand_contact_frames"],
                "full_gate_pass": report["lift_skill_screen_pass"],
                "at_4_1_s": sample(4.1), "at_4_6_s": sample(4.6), "at_5_0_s": sample(5.0),
                "report": str(report_path.relative_to(ROOT)),
                "trace": str(trace_path.relative_to(ROOT)),
                "video": str(video_path.relative_to(ROOT)),
            })
    assert len(rows) == 2 * len(plan["cases"])
    matrix = {"schema": "robot-learning-lab.cove50-gated-com-support-matrix.v1", "cases": rows}
    (OUT / "summary-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    counts = {case["id"]: sum(r["full_gate_pass"] for r in rows if r["controller"] == case["id"])
              for case in plan["cases"]}
    print(json.dumps({"nominal_passes": counts, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
