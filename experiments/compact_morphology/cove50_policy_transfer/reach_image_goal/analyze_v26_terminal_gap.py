"""Retain the opened fingertip-cube geometry that motivated V26."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v26/terminal-gap-basis.json"
SOURCES = (
    ("V21 scored 10604", BASE / "v21/fresh-reserved/seed-10604.json"),
    ("V24 corrected 10604", BASE / "v24/opened-truth-vector/seed-10604-x3_yminus3.json"),
    ("V21 pass 10601", BASE / "v21/fresh-reserved/seed-10601.json"),
    ("V21 pass 10602", BASE / "v21/fresh-reserved/seed-10602.json"),
)
ARM = ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch",
       "right_elbow_pitch")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    first = json.loads(SOURCES[0][1].read_text())
    source = VirtualRgbGoal(selected_body_sha256=first["body_sha256"], max_age_s=9)
    tip = source.model.geom("right_finger_tip").id
    cube = source.model.geom("offline_visual_cube_geom").id

    def distance(qpos: np.ndarray) -> float:
        source.data.qpos[:] = qpos
        source.data.qvel[:] = 0
        mujoco.mj_forward(source.model, source.data)
        return float(mujoco.mj_geomDistance(source.model, source.data,
                                            tip, cube, 1.0, None) * 1000)

    rows = []
    try:
        for label, report_path in SOURCES:
            report = json.loads(report_path.read_text())
            trace_path = report_path.with_suffix(".npz")
            if (report["body_sha256"] != first["body_sha256"] or
                    report["body_role"] != "rejected_historical_direct_frame_sandbox"):
                raise ValueError("Mixed body or provenance")
            with np.load(trace_path) as trace:
                qpos = trace["qpos"].copy()
                dt = float(trace["step_dt_s"])
            if qpos.shape != (825, source.model.nq) or dt != .02:
                raise ValueError(f"Incomplete 50 Hz native trace: {label}")
            gaps = np.asarray([distance(qpos[i]) for i in range(375, 825)])
            minimum = int(np.argmin(gaps))
            row = {
                "label": label,
                "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                "signed_gap_at_12s_mm": round(float(gaps[225]), 3),
                "signed_gap_at_15s_mm": round(float(gaps[375]), 3),
                "minimum_signed_gap_mm": round(float(gaps[minimum]), 3),
                "minimum_gap_time_s": round(7.5 + minimum * .02, 3),
                "terminal_signed_gap_mm": round(float(gaps[-1]), 3),
                "interpenetrating_frames": int(np.count_nonzero(gaps < 0)),
                "report": str(report_path.relative_to(ROOT)),
                "report_sha256": sha(report_path),
                "trace": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
            }
            if label == "V21 scored 10604":
                terminal = qpos[-1].copy()
                gradients = {}
                for name in ARM:
                    joint = source.model.joint(name)
                    address = int(joint.qposadr[0])
                    epsilon = .002
                    plus, minus = terminal.copy(), terminal.copy()
                    plus[address] += epsilon
                    minus[address] -= epsilon
                    gradients[name] = round((distance(plus) - distance(minus)) /
                                            (2 * epsilon), 2)
                row["terminal_gap_gradient_mm_per_rad"] = gradients
            rows.append(row)
    finally:
        source.close()
    output = {
        "schema": "robot-learning-lab.cove50-v26-opened-terminal-gap-basis.v1",
        "role": "postscore_opened_kinematic_diagnostic_not_native_replay_or_inference",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "body_sha256": first["body_sha256"],
        "source_sha256": sha(Path(__file__)),
        "method": "Reconstruct saved native qpos in the identical body plus visual cube model; MuJoCo signed geom distance between right_finger_tip and offline_visual_cube_geom at each 50 Hz reach frame. Finite-difference the opened final qpos at ±0.002 rad per arm joint. Negative distance indicates geometric overlap only, not sustained physical contact.",
        "cases": rows,
        "qualification": "This post-score kinematic diagnostic motivates a small terminal route screen. Only a new native motor/contact replay can establish contact; it does not alter V21 scoring or qualify a future body.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
