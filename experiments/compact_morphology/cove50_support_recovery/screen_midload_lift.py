"""Screen a less displaced swing-roll lift from attained Cove native shift states."""

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.body_xml))
    if model.nu != 21 or model.nq != 28:
        raise ValueError("Expected the selected 21-axis Cove body")
    data = mujoco.MjData(model)
    rows = []
    pitch_rows = []
    sources = {}
    for stance in ("right", "left"):
        swing = "left" if stance == "right" else "right"
        trace = args.trace_dir / f"{stance}-shift0.30.npz"
        sources[stance] = {"path": str(trace), "sha256": sha(trace)}
        with np.load(trace) as saved:
            attained = saved["qpos"][-1].astype(float)
        arm_joint = model.joint(f"{swing}_shoulder_roll")
        roll_joint = model.joint(f"{swing}_hip_roll")
        pitch_joint = model.joint(f"{swing}_hip_pitch")
        stance_site = model.site(f"{stance}_sole_center").id
        swing_site = model.site(f"{swing}_sole_center").id
        pelvis = model.body("pelvis").id
        for extra in (0.0, 0.1, 0.15, 0.2, 0.25):
            data.qpos[:] = attained
            data.qpos[arm_joint.qposadr[0]] = 0.2 if swing == "left" else -0.2
            data.qpos[roll_joint.qposadr[0]] += extra if stance == "right" else -extra
            mujoco.mj_forward(model, data)
            contacts = [
                {"geoms": [model.geom(c.geom1).name, model.geom(c.geom2).name],
                 "penetration_m": round(float(max(0.0, -c.dist)), 6)}
                for c in data.contact[:data.ncon]
                if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]
            ]
            com = data.subtree_com[pelvis] - data.site_xpos[stance_site]
            rows.append({
                "stance": stance,
                "swing_hip_roll_extra_rad": extra,
                "swing_arm_clearance_roll_rad": float(data.qpos[arm_joint.qposadr[0]]),
                "swing_sole_center_z_m": round(float(data.site_xpos[swing_site, 2]), 6),
                "horizontal_com_minus_stance_sole_xy_m": np.round(com[:2], 6).tolist(),
                "self_contacts": contacts,
            })
        for pitch in (0.0, -0.2, -0.4, -0.6):
            data.qpos[:] = attained
            data.qpos[arm_joint.qposadr[0]] = 0.2 if swing == "left" else -0.2
            data.qpos[pitch_joint.qposadr[0]] = pitch
            mujoco.mj_forward(model, data)
            contacts = [
                {"geoms": [model.geom(c.geom1).name, model.geom(c.geom2).name],
                 "penetration_m": round(float(max(0.0, -c.dist)), 6)}
                for c in data.contact[:data.ncon]
                if model.geom_bodyid[c.geom1] and model.geom_bodyid[c.geom2]
            ]
            com = data.subtree_com[pelvis] - data.site_xpos[stance_site]
            pitch_rows.append({
                "stance": stance,
                "swing_hip_pitch_rad": pitch,
                "swing_arm_clearance_roll_rad": float(data.qpos[arm_joint.qposadr[0]]),
                "swing_sole_center_z_m": round(float(data.site_xpos[swing_site, 2]), 6),
                "horizontal_com_minus_stance_sole_xy_m": np.round(com[:2], 6).tolist(),
                "self_contacts": contacts,
            })
    report = {
        "schema": "robot-learning-lab.cove50-midload-static-lift-screen.v1",
        "method": "At each saved 0.30-rad native two-foot shift endpoint, hold root and other joints fixed, clear the swing arm, and separately sweep swing hip roll or hip pitch in MuJoCo forward kinematics. No motor or dynamic support inference.",
        "body_sha256": sha(args.body_xml),
        "sources": sources,
        "rows": rows,
        "pitch_rows": pitch_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
