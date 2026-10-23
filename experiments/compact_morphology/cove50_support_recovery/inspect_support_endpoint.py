"""Report exact sole geometry and gravity demands of a support reset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoint", type=Path)
    parser.add_argument("--stance", choices=("left", "right"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.endpoint.read_text())
    body = Path(spec["model"])
    model = mujoco.MjModel.from_xml_path(str(body))
    data = mujoco.MjData(model)
    x, y, z, roll, pitch = spec["root_xyz_roll_pitch"]
    data.qpos[:3] = (x, y, z)
    xyzw = Rotation.from_euler("xy", (roll, pitch)).as_quat()
    data.qpos[3:7] = (xyzw[3], *xyzw[:3])
    for name, value in spec["joints"].items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    stance_site = model.site(f"{args.stance}_sole_center").id
    swing = "left" if args.stance == "right" else "right"
    swing_geom = model.geom(f"{swing}_sole").id
    stance_geom = model.geom(f"{args.stance}_sole").id
    com = data.subtree_com[model.body("pelvis").id]
    stance_center = data.geom_xpos[stance_geom]
    stance_rotation = data.geom_xmat[stance_geom].reshape(3, 3)
    local_com = stance_rotation.T @ (com - stance_center)
    swing_rotation = data.geom_xmat[swing_geom].reshape(3, 3)
    swing_floor_gap = (data.geom_xpos[swing_geom, 2] -
                       np.abs(swing_rotation[2, :]) @ model.geom_size[swing_geom])
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jp, jr, stance_site)
    jac = np.vstack((jp, jr))
    wrench = np.linalg.lstsq(jac[:, :6].T, data.qfrc_bias[:6], rcond=None)[0]
    effort = data.qfrc_bias[6:] - jac[:, 6:].T @ wrench
    demands = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        name = model.joint(jid).name
        qadr = model.jnt_qposadr[jid]
        dof = model.jnt_dofadr[jid] - 6
        demands.append({
            "joint": name,
            "angle_rad": round(float(data.qpos[qadr]), 6),
            "range_rad": np.round(model.jnt_range[jid], 6).tolist(),
            "gravity_effort_nm": round(float(effort[dof]), 4),
            "rated_effort_nm": round(float(model.actuator_forcerange[i, 1]), 4),
        })
    contacts = [{"geom1": model.geom(c.geom1).name,
                 "geom2": model.geom(c.geom2).name,
                 "penetration_m": round(float(max(0, -c.dist)), 6)}
                for c in data.contact[:data.ncon]]
    result = {
        "schema": "robot-learning-lab.cove50-support-endpoint-geometry.v1",
        "method": "Static exact-body geometry and one-sole gravity-wrench linearization; not a dynamic hold or reachable transition",
        "body_sha256": hashlib.sha256(body.read_bytes()).hexdigest(),
        "endpoint": str(args.endpoint),
        "stance": args.stance,
        "pelvis_z_m": round(float(data.qpos[2]), 6),
        "com_xyz_m": np.round(com, 6).tolist(),
        "com_in_stance_sole_xy_m": np.round(local_com[:2], 6).tolist(),
        "stance_sole_half_widths_xy_m": np.round(model.geom_size[stance_geom, :2], 6).tolist(),
        "stance_sole_normal_xyz": np.round(stance_rotation[:, 2], 6).tolist(),
        "swing_sole_lowest_point_z_m": round(float(swing_floor_gap), 6),
        "swing_sole_site_z_m": round(float(data.site_xpos[model.site(f"{swing}_sole_center").id, 2]), 6),
        "one_sole_wrench": np.round(wrench, 4).tolist(),
        "joint_demands": demands,
        "contacts": contacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "joint_demands"}, indent=2))


if __name__ == "__main__":
    main()
