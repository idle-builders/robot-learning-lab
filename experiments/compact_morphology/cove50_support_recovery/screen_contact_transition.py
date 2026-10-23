"""Screen small swing-roll changes at exact saved Cove-50 motor states.

Forward geometry only: this cannot establish contact force or motor feasibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--right-trace", type=Path, required=True)
    parser.add_argument("--left-trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    data = mujoco.MjData(model)
    pelvis = model.body("pelvis").id
    rows = []
    traces = {"right": args.right_trace, "left": args.left_trace}
    for stance, trace in traces.items():
        sign = 1.0 if stance == "right" else -1.0
        swing = "left" if stance == "right" else "right"
        roll = model.joint(f"{swing}_hip_roll")
        sole = model.geom(f"{swing}_sole").id
        stance_site = model.site(f"{stance}_sole_center").id
        with np.load(trace) as saved:
            qpos = saved["qpos"].copy()
            dt = float(saved["dt_s"])
        for time in (4.2, 4.4, 5.0, 5.4):
            base = qpos[round(time / dt) - 1]
            for extra in (0.0, 0.04, 0.08):
                data.qpos[:] = base
                data.qpos[roll.qposadr[0]] += sign * extra
                mujoco.mj_forward(model, data)
                rotation = data.geom_xmat[sole].reshape(3, 3)
                corner_z = (data.geom_xpos[sole, 2] -
                            np.abs(rotation[2, :]) @ model.geom_size[sole])
                com = data.subtree_com[pelvis] - data.site_xpos[stance_site]
                self_pairs = [sorted((model.geom(c.geom1).name,
                                      model.geom(c.geom2).name))
                              for c in data.contact[:data.ncon]
                              if "floor" not in (model.geom(c.geom1).name,
                                                 model.geom(c.geom2).name)]
                rows.append({
                    "stance": stance, "time_s": time,
                    "swing_roll_extra_rad": extra,
                    "swing_corner_z_mm": round(float(corner_z * 1000), 3),
                    "com_minus_stance_world_x_mm": round(float(com[0] * 1000), 3),
                    "com_minus_stance_world_y_mm": round(float(com[1] * 1000), 3),
                    "self_contact_pairs": self_pairs,
                })
    result = {
        "schema": "robot-learning-lab.cove50-contact-transition-static.v1",
        "method": "Exact saved 0.33-rad nominal motor qpos on paired stances; add only outward swing hip roll, hold root and all other joints fixed; MuJoCo forward geometry, not a dynamic support pass.",
        "body_sha256": digest(args.body_xml),
        "right_trace_sha256": digest(args.right_trace),
        "left_trace_sha256": digest(args.left_trace),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
