"""Screen legal single-sole Cove poses before training a support policy.

This is a kinematic screen. It does not qualify balance or motor duty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


LEG_JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch")
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--source-pose", type=Path, required=True)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--swing-height", type=float, default=0.05)
    parser.add_argument("--compact-swing", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    source = json.loads(args.source_pose.read_text())
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected the selected 21-axis free-root Cove body")
    swing = "left" if args.stance == "right" else "right"
    ids = {n: model.joint(n).qposadr[0] for n in LEG_JOINTS}
    stance_site = model.site(f"{args.stance}_sole_center").id
    swing_site = model.site(f"{swing}_sole_center").id
    pelvis = model.body("pelvis").id
    qpos = np.zeros(model.nq)
    qpos[3] = 1.0
    for name, value in source["joints_rad"].items():
        qpos[model.joint(name).qposadr[0]] = value
    x0 = np.array([
        source["root_xyz_m"][2],
        *source["root_roll_pitch_yaw_rad"][:2],
        *(source["joints_rad"][name] for name in LEG_JOINTS),
    ], dtype=float)
    if args.compact_swing:
        x0[3 + LEG_JOINTS.index(f"{swing}_hip_yaw")] = 0.0
        x0[3 + LEG_JOINTS.index(f"{swing}_hip_roll")] = 0.0
        x0[3 + LEG_JOINTS.index(f"{swing}_hip_pitch")] = -0.4
        x0[3 + LEG_JOINTS.index(f"{swing}_knee")] = 0.7
        x0[3 + LEG_JOINTS.index(f"{swing}_ankle_pitch")] = 0.0
    bounds_lo = [0.11, -0.35, -0.4]
    bounds_hi = [0.34, 0.35, 0.4]
    for name in LEG_JOINTS:
        joint = model.joint(name)
        if name.endswith("hip_yaw"):
            bounds_lo.append(-0.4)
            bounds_hi.append(0.4)
        elif name == f"{swing}_hip_roll":
            bounds_lo.append(-0.5)
            bounds_hi.append(0.5)
        else:
            bounds_lo.append(float(joint.range[0]) + 0.005)
            bounds_hi.append(float(joint.range[1]) - 0.005)
    lower = np.asarray(bounds_lo)
    upper = np.asarray(bounds_hi)
    x0 = np.clip(x0, lower + 1e-6, upper - 1e-6)
    data = mujoco.MjData(model)

    def place(x: np.ndarray) -> None:
        data.qpos[:] = qpos
        data.qpos[2] = x[0]
        xyzw = Rotation.from_euler("xyz", (x[1], x[2], 0.0)).as_quat()
        data.qpos[3:7] = (xyzw[3], *xyzw[:3])
        for name, value in zip(LEG_JOINTS, x[3:]):
            data.qpos[ids[name]] = value
        mujoco.mj_forward(model, data)

    def residual(x: np.ndarray) -> np.ndarray:
        place(x)
        stance_pos = data.site_xpos[stance_site]
        swing_pos = data.site_xpos[swing_site]
        stance_rot = data.site_xmat[stance_site].reshape(3, 3)
        com = data.subtree_com[pelvis]
        com_dx = com[0] - stance_pos[0]
        com_dy = com[1] - stance_pos[1]
        max_self_penetration = max(
            (max(0.0, -c.dist) for c in data.contact[:data.ncon]
             if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]),
            default=0.0,
        )
        return np.array([
            35.0 * (stance_pos[2] - 0.003),
            6.0 * stance_rot[2, 0],
            6.0 * stance_rot[2, 1],
            3.0 * stance_rot[1, 0],
            20.0 * max(0.0, abs(com_dx) - 0.025),
            20.0 * max(0.0, abs(com_dy) - 0.024),
            15.0 * max(0.0, args.swing_height - swing_pos[2]),
            8.0 * max(0.0, abs(swing_pos[0] - stance_pos[0]) - 0.06),
            20.0 * max(0.0, 0.085 - (swing_pos[1] - stance_pos[1]) * (1 if swing == "left" else -1)),
            50.0 * max_self_penetration,
            0.12 * (x[1] - x0[1]),
            0.12 * (x[2] - x0[2]),
            *(0.12 * (x[3:] - x0[3:])),
        ])

    best = least_squares(residual, x0, bounds=(lower, upper), max_nfev=2500,
                         xtol=1e-10, ftol=1e-10, gtol=1e-10)
    place(best.x)
    stance_pos = data.site_xpos[stance_site].copy()
    swing_pos = data.site_xpos[swing_site].copy()
    stance_rot = data.site_xmat[stance_site].reshape(3, 3).copy()
    com = data.subtree_com[pelvis].copy()
    contacts = []
    for c in data.contact[:data.ncon]:
        contacts.append({
            "geom1": model.geom(c.geom1).name,
            "geom2": model.geom(c.geom2).name,
            "penetration_m": round(float(max(0.0, -c.dist)), 6),
        })
    result = {
        "schema": "robot-learning-lab.cove50-support-static-pose.v1",
        "body_xml": str(args.body_xml),
        "body_sha256": hashlib.sha256(args.body_xml.read_bytes()).hexdigest(),
        "source_pose": str(args.source_pose),
        "stance": args.stance,
        "method": "Bounded static least-squares: minimize stance-sole tilt and COM offset while raising the opposite sole within legal leg joints. Residuals may remain; no motor or transition proof.",
        "optimizer_success": bool(best.success),
        "optimizer_cost": float(best.cost),
        "screen_pass": bool(
            not contacts
            and abs(com[0] - stance_pos[0]) < 0.025
            and abs(com[1] - stance_pos[1]) < 0.024
            and swing_pos[2] > args.swing_height
            and abs(stance_rot[2, 0]) < 0.02
            and abs(stance_rot[2, 1]) < 0.02
        ),
        "root_xyz_m": [0.0, 0.0, round(float(best.x[0]), 6)],
        "root_roll_pitch_yaw_rad": [round(float(best.x[1]), 6), round(float(best.x[2]), 6), 0.0],
        "joints_rad": {n: round(float(data.qpos[adr]), 6) for n, adr in ids.items()},
        "com_xyz_m": np.round(com, 6).tolist(),
        "stance_sole_xyz_m": np.round(stance_pos, 6).tolist(),
        "swing_sole_xyz_m": np.round(swing_pos, 6).tolist(),
        "com_minus_stance_xy_m": np.round(com[:2] - stance_pos[:2], 6).tolist(),
        "stance_sole_normal_xyz": np.round(stance_rot[:, 2], 6).tolist(),
        "contacts": contacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
