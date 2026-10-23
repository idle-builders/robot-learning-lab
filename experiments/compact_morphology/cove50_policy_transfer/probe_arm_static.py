"""Measure fixed-root shoulder gravity torque on the selected Cove body."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("body_xml", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml.resolve()))
    data = mujoco.MjData(model)
    shoulder_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                     f"{side}_shoulder_pitch")
                    for side in ("left", "right")]
    if any(j < 0 for j in shoulder_ids):
        raise ValueError("Selected body is missing shoulder pitch axes")
    rows = []
    for angle in (0.0, -0.5, -1.0, -1.5):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.2305
        for joint_id in shoulder_ids:
            data.qpos[model.jnt_qposadr[joint_id]] = angle
        mujoco.mj_forward(model, data)
        rows.append({
            "shoulder_pitch_rad": angle,
            "fixed_root_gravity_torque_nm": [round(float(
                data.qfrc_bias[model.jnt_dofadr[joint_id]]), 4)
                for joint_id in shoulder_ids],
        })
    result = {
        "body_xml": str(args.body_xml),
        "method": "MuJoCo qfrc_bias with root fixed at standing height and zero velocity; this is a static gravity diagnostic, not a full contact or controller replay.",
        "catalog_shoulder_rated_effort_nm_at_12v": 0.980665,
        "poses": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
