"""Verify and summarize pinned native swing-corner release cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-corner-release-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "plan.json").read_text())
    for key in ("selected_design_pointer", "selected_body", "actuator_catalog",
                "probe_source", "static_screen"):
        pin = plan[key]
        assert sha(ROOT / pin["path"]) == pin["sha256"], key
    for side, pin in plan["baseline_traces"].items():
        assert sha(ROOT / pin["path"]) == pin["sha256"], side
    for line in (OUT / "trials/remote-sha256.txt").read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert sha(ROOT / path) == digest, path

    model = mujoco.MjModel.from_xml_path(str(ROOT / plan["selected_body"]["path"]))
    data = mujoco.MjData(model)
    rows = []
    for case in plan["cases"]:
        for side in plan["nominal"]["stances"]:
            stem = f"{case['id']}-{side}-nominal"
            report_path = OUT / "trials" / f"{stem}.json"
            trace_path = report_path.with_suffix(".npz")
            video_path = report_path.with_suffix(".mp4")
            assert trace_path.is_file() and video_path.is_file()
            report = json.loads(report_path.read_text())
            assert report["design_pointer_sha256"] == plan["selected_design_pointer"]["sha256"]
            assert report["body_sha256"] == plan["selected_body"]["sha256"]
            assert report["stance"] == side
            assert report["swing_ankle_release_target_rad"] == case["swing_ankle_release_target_rad"]
            assert report["stance_com_velocity_horizon_s"] == 0.15
            assert report["damping_entry_load_fraction"] == 0.82
            assert report["rail_v"] == 12 and report["slide_friction"] == 1
            assert report["delay_ms"] == 0 and report["seed"] == 7144
            swing = "left" if side == "right" else "right"
            with np.load(trace_path) as saved:
                qpos = saved["qpos"].copy()
                dt = float(saved["dt_s"])
                if case["id"] == "baseline-reload":
                    with np.load(ROOT / plan["baseline_traces"][side]["path"]) as prior:
                        for field in ("qpos", "qvel", "joint_target"):
                            assert np.array_equal(saved[field], prior[field]), (side, field)
            geom = model.geom(f"{swing}_sole").id
            joint = model.joint(f"{swing}_ankle_pitch")

            def phase_sample(time_s: float) -> dict:
                index = min(round(time_s / dt) - 1, len(qpos) - 1)
                native = report["timeseries"][index]
                data.qpos[:] = qpos[index]
                mujoco.mj_forward(model, data)
                rotation = data.geom_xmat[geom].reshape(3, 3)
                half = model.geom_size[geom]
                lowest = min(float((data.geom_xpos[geom] + rotation @ np.array(
                    [x * half[0], y * half[1], -half[2]]))[2]) * 1000
                    for x in (-1, 1) for y in (-1, 1))
                return {
                    "time_s": native["time_s"],
                    "actual_swing_ankle_pitch_rad": round(float(data.qpos[joint.qposadr[0]]), 5),
                    "lowest_swing_corner_z_mm": round(lowest, 3),
                    "swing_sole_center_z_mm": round(native["swing_sole_z_m"] * 1000, 2),
                    "stance_load_fraction": native["stance_foot_load_fraction"],
                    "stance_slip_mm": round(native["stance_slip_m"] * 1000, 2),
                    "com_toward_stance_mm": round(
                        (-1 if side == "right" else 1) * native["com_minus_stance_world_xy_m"][1] * 1000, 2),
                    "feet_grounded": native["feet_grounded"],
                }

            first_rated = next((item for item in report["timeseries"] if item["rated_leg_joints"]), None)
            rows.append({
                "case": stem, "controller": case["id"], "stance": side,
                "late_ankle_target_rad": case["swing_ankle_release_target_rad"],
                "damping_entry_time_s": report["damping_entry_time_s"],
                "duration_s": report["duration_s"],
                "exclusive_support_s": report["longest_single_support_s"],
                "rated_leg_fraction": report["rated_leg_fraction"],
                "max_stance_slip_mm": round(report["max_stance_slip_m"] * 1000, 2),
                "first_rated_time_s": first_rated["time_s"] if first_rated else None,
                "first_rated_joints": first_rated["rated_leg_joints"] if first_rated else [],
                "full_gate_pass": report["lift_skill_screen_pass"],
                "at_4_2_s": phase_sample(4.2),
                "at_4_6_s": phase_sample(4.6),
                "at_5_0_s": phase_sample(5.0),
                "report": str(report_path.relative_to(ROOT)),
                "trace": str(trace_path.relative_to(ROOT)),
                "video": str(video_path.relative_to(ROOT)),
            })
    assert len(rows) == 6
    (OUT / "summary-matrix.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-corner-release-matrix.v1", "cases": rows,
    }, indent=2) + "\n")
    print(json.dumps({"passes": {case["id"]: sum(row["full_gate_pass"] for row in rows
                                                 if row["controller"] == case["id"])
                                 for case in plan["cases"]}, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
