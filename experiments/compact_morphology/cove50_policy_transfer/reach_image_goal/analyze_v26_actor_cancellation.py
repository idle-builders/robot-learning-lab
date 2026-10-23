"""Explain why authored terminal-roll changes failed in opened V26 replays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.virtual_rgb_goal import VirtualRgbGoal


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v26/actor-cancellation.json"
PATHS = (
    BASE / "v21/fresh-reserved/seed-10604.json",
    BASE / "v26/opened-terminal-roll/seed-10604-roll_minus010.json",
    BASE / "v26/opened-terminal-roll/seed-10604-roll_minus020.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    baseline = json.loads(PATHS[0].read_text())
    source = VirtualRgbGoal(selected_body_sha256=baseline["body_sha256"], max_age_s=9)
    roll = int(source.model.joint("right_shoulder_roll").qposadr[0])
    tip = source.model.geom("right_finger_tip").id
    cube = source.model.geom("offline_visual_cube_geom").id
    rows = []
    try:
        for path in PATHS:
            report = json.loads(path.read_text())
            trace_path = path.with_suffix(".npz")
            if report["body_sha256"] != baseline["body_sha256"]:
                raise ValueError("Mixed body source")
            with np.load(trace_path) as trace:
                qpos = trace["qpos"]
                if qpos.shape != (825, source.model.nq):
                    raise ValueError(f"Incomplete native trace: {path}")
                terminal = qpos[-1].copy()
            source.data.qpos[:] = terminal
            source.data.qvel[:] = 0
            mujoco.mj_forward(source.model, source.data)
            gap = float(mujoco.mj_geomDistance(source.model, source.data,
                                               tip, cube, 1.0, None) * 1000)
            rows.append({
                "label": "V21 scored" if path == PATHS[0] else path.stem,
                "authored_terminal_roll_offset_rad": report["applied_terminal_roll_offset_rad"],
                "attained_terminal_shoulder_roll_rad": round(float(terminal[roll]), 6),
                "learned_roll_residual_raw_terminal": report["timeseries"][-1]["learned_arm_residual_raw"][1],
                "terminal_signed_fingertip_cube_gap_mm": round(gap, 3),
                "contact_s": report["phases"]["reach"]["continuous_finger_contact_s"],
                "report": str(path.relative_to(ROOT)),
                "report_sha256": sha(path),
                "trace": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
            })
    finally:
        source.close()
    output = {
        "schema": "robot-learning-lab.cove50-v26-opened-actor-cancellation.v1",
        "role": "postscore_opened_control_interface_diagnostic_not_inference_or_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": sha(Path(__file__)),
        "method": "Compare saved actor residual, attained shoulder-roll qpos and reconstructed terminal signed geom gap on paired native V21/V26 traces.",
        "cases": rows,
        "inference": "The learned residual grew with the negative authored roll offset; attained roll rose slightly and the fingertip gap widened. This is evidence for actor compensation in these opened native runs, not a general guarantee about the controller.",
    }
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
