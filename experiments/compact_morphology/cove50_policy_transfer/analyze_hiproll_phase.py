"""Explain why the wider hip-roll pose prior failed the native motor gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-recovery-20261009"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads((OUT / "hiproll-plan.json").read_text())
    if plan["sources"]["body"]["sha256"] != sha(BODY):
        raise ValueError("Selected body changed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    sole_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
                for n in ("left_sole", "right_sole")]
    data = mujoco.MjData(model)
    rows = []
    for actor in ("hiproll-source", "hiproll100"):
        trace_path = OUT / f"{actor}-seed9144-trace.npz"
        trace = np.load(trace_path)
        active = trace["active"]
        found = trace["sole_found"] > 0
        limits = trace["rated_effort_limits_nm"]
        for side in (0, 1):
            exclusive = active & found[:, :, side] & ~found[:, :, 1 - side]
            offsets = []
            hip_angles = []
            hip_rated = []
            ankle_rated = []
            for tick, world in zip(*np.where(exclusive)):
                data.qpos[:7] = trace["root_pose"][tick, world]
                data.qpos[7:17] = trace["joint_pos"][tick, world]
                mujoco.mj_forward(model, data)
                rotation = np.empty(9)
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                lateral = rotation.reshape(3, 3)[:, 1]
                offsets.append(abs(float((data.subtree_com[0] -
                                          data.geom_xpos[sole_ids[side]]) @ lateral)))
                hip_angles.append(abs(float(trace["joint_pos"][tick, world, side * 5 + 1])))
                effort = abs(trace["actuator_force"][tick, world])
                hip_rated.append(bool(effort[side * 5 + 1] >= .995 * limits[side * 5 + 1]))
                ankle_rated.append(bool(effort[side * 5 + 4] >= .995 * limits[side * 5 + 4]))
            rows.append({
                "actor": actor, "stance": ("left", "right")[side],
                "exclusive_frames": len(offsets),
                "sole_half_width_cm": round(100 * float(model.geom_size[sole_ids[side], 1]), 3),
                "com_lateral_offset_cm": round(100 * float(np.mean(offsets)), 3),
                "hip_roll_abs_rad": round(float(np.mean(hip_angles)), 4),
                "hip_roll_rated_fraction": round(float(np.mean(hip_rated)), 5),
                "ankle_pitch_rated_fraction": round(float(np.mean(ankle_rated)), 5),
                "trace": str(trace_path.relative_to(ROOT)),
                "trace_sha256": sha(trace_path),
            })
    full = np.load(OUT / "hiproll-source-forward-seed144.npz")["qpos"]
    errors = []
    for state in full:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        actual = data.subtree_com[0, :2].copy()
        data.qpos[17:] = model.qpos0[17:]
        mujoco.mj_forward(model, data)
        errors.append(float(np.linalg.norm(data.subtree_com[0, :2] - actual)))
    matrix = {
        "schema": "robot-learning-lab.cove50-hiproll-phase-matrix.v1",
        "note": "Four source-success matched worlds on reserved seed 9144, 50 Hz native traces. Whole-body COM reconstructed from saved root/leg state with default upper joints; separate same-seed full-qpos replay bounds upper-joint planar COM error. Simulated geometry and catalog-rated effort only.",
        "cases": rows,
    }
    report = {
        "schema": "robot-learning-lab.cove50-hiproll-phase-report.v1",
        "selected_body_sha256": sha(BODY),
        "selected_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
        "source_full_qpos": str((OUT / "hiproll-source-forward-seed144.npz").relative_to(ROOT)),
        "source_full_qpos_sha256": sha(OUT / "hiproll-source-forward-seed144.npz"),
        "max_upper_joint_default_com_error_m": round(max(errors), 6),
        "matrix": str((OUT / "hiproll-phase-matrix.json").relative_to(ROOT)),
        "cases": rows,
        "interpretation": "Relaxing the walking hip-roll pose prior increased mean absolute hip-roll angle only slightly and did not shift the COM toward the stance sole. The right hip-roll rated fraction dropped, but ankle rated duty and whole-leg moving duty remained high. This one-factor continuation did not solve the support/load-transfer mechanism.",
        "limits": ["Four matched worlds on one reserved seed diagnose the mechanism; paired two-seed native outcomes determine rejection.",
                   "Upper joints are defaulted only for trace reconstruction; the separate full-qpos replay bounds the approximation, but is not an identical trajectory.",
                   "Motor, contact and geometry are provisional simulation evidence, not physical duty."],
    }
    (OUT / "hiproll-phase-matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    (OUT / "hiproll-phase-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"max_upper_joint_default_com_error_m": max(errors),
                      "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
