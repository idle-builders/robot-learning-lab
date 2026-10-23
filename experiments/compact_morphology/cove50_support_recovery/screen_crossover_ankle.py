"""Screen swing-sole corner clearance on the attained 0.33-rad native path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--body-xml", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    data = mujoco.MjData(model)
    with np.load(args.trace) as saved:
        qpos = saved["qpos"].copy()
        dt = float(saved["dt_s"])
    ankle = model.joint("left_ankle_pitch")
    swing_sole = model.geom("left_sole").id
    stance_site = model.site("right_sole_center").id
    pelvis = model.body("pelvis").id
    rows = []
    for time in (4.4, 5.0, 5.5, 6.0):
        base = qpos[round(time / dt) - 1]
        for ankle_target in (0.0, 0.1, 0.2):
            data.qpos[:] = base
            data.qpos[ankle.qposadr[0]] = ankle_target
            mujoco.mj_forward(model, data)
            rot = data.geom_xmat[swing_sole].reshape(3, 3)
            gap = data.geom_xpos[swing_sole, 2] - np.abs(rot[2, :]) @ model.geom_size[swing_sole]
            pairs = [sorted((model.geom(c.geom1).name, model.geom(c.geom2).name))
                     for c in data.contact[:data.ncon]
                     if "floor" not in (model.geom(c.geom1).name, model.geom(c.geom2).name)]
            rows.append({
                "time_s": time,
                "left_swing_ankle_pitch_rad": ankle_target,
                "lowest_swing_sole_corner_z_mm": round(float(gap * 1000), 3),
                "com_minus_stance_sole_y_mm": round(float(1000 *
                    (data.subtree_com[pelvis, 1] - data.site_xpos[stance_site, 1])), 3),
                "self_contact_pairs": pairs,
            })
    result = {
        "schema": "robot-learning-lab.cove50-crossover-ankle-static.v1",
        "method": "Exact saved 0.33-rad right-stance native qpos at four times, fixed root and all joints except swing ankle; MuJoCo forward geometry only. Positive corner height is not a native motor or support pass.",
        "body_sha256": sha(args.body_xml),
        "trace_sha256": sha(args.trace),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
