"""Reconstruct stance COM from paired native traces on the selected body."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-ankle-drive-screen-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha(BODY) != json.loads((OUT / "plan.json").read_text())["identity"][
        str(BODY.relative_to(ROOT))]["sha256"]:
        raise ValueError("Selected body changed")
    model = mujoco.MjModel.from_xml_path(str(BODY))
    data = mujoco.MjData(model)
    sole_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_sole")
                for side in ("left", "right")]
    rows = []
    for skill in ("forward", "turn-left", "turn-right"):
        for motor in ("selected", "st3250"):
            path = OUT / f"{skill}-{motor}.npz"
            trace = np.load(path)
            active = trace["active"]
            found = trace["sole_found"] > 0
            limits = trace["rated_effort_limits_nm"]
            for side in (0, 1):
                mask = active & found[:, :, side] & ~found[:, :, 1 - side]
                offsets = []
                hip_rated = []
                ankle_rated = []
                effort = []
                for tick, world in zip(*np.where(mask)):
                    data.qpos[:] = model.qpos0
                    data.qpos[:7] = trace["root_pose"][tick, world]
                    data.qpos[7:17] = trace["joint_pos"][tick, world]
                    mujoco.mj_forward(model, data)
                    rotation = np.empty(9)
                    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                    lateral = rotation.reshape(3, 3)[:, 1]
                    offset = float((data.subtree_com[0] - data.geom_xpos[sole_ids[side]]) @ lateral)
                    offsets.append(offset)
                    forces = np.abs(trace["actuator_force"][tick, world])
                    hip_rated.append(forces[side * 5 + 1] >= .995 * limits[side * 5 + 1])
                    ankle_rated.append(forces[side * 5 + 4] >= .995 * limits[side * 5 + 4])
                    effort.append(float(forces[side * 5 + 1]))
                halfwidth = float(model.geom_size[sole_ids[side], 1])
                abs_offsets = np.abs(offsets)
                rows.append({
                    "case": skill, "motor_law": motor,
                    "stance": ("left", "right")[side],
                    "exclusive_frames": len(offsets),
                    "exclusive_fraction": round(float(mask.sum() / active.sum()), 6),
                    "sole_half_width_cm": round(100 * halfwidth, 3),
                    "mean_abs_com_offset_cm": round(100 * float(np.mean(abs_offsets)), 3),
                    "mean_beyond_sole_cm": round(100 * float(np.mean(np.maximum(abs_offsets - halfwidth, 0))), 3),
                    "hip_roll_rated_fraction_exclusive": round(float(np.mean(hip_rated)), 6),
                    "ankle_rated_fraction_exclusive": round(float(np.mean(ankle_rated)), 6),
                    "mean_abs_hip_roll_torque_nm_exclusive": round(float(np.mean(effort)), 4),
                    "trace": str(path.relative_to(ROOT)), "trace_sha256": sha(path),
                })
    result = {
        "schema": "robot-learning-lab.cove50-ankle-law-com-phase.v1",
        "selected_body_sha256": sha(BODY),
        "method": "Whole-body subtree COM from saved root and ten leg joints at 50 Hz; upper joints held at XML defaults. Lateral offset is relative to the active sole in root heading coordinates. This is a trace reconstruction on provisional physics, not a measured physical COM or exact full-upper-joint replay.",
        "cases": rows,
    }
    (OUT / "com-phase-matrix.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row["case"], row["motor_law"], row["stance"],
              row["mean_abs_com_offset_cm"], row["hip_roll_rated_fraction_exclusive"])


if __name__ == "__main__":
    main()
