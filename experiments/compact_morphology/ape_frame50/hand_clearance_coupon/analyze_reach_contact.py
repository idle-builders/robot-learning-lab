"""Compare saved native reach contact geometry without re-running control."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-hand-candidate-mjlab-20261010/reach-entry"
BODIES = {
    "selected": ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml",
    "candidate": ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml",
}


def model_with_cube(body: Path) -> mujoco.MjModel:
    xml = body.read_text().replace(
        "</worldbody>",
        '<body name="view_toy" pos="0.075 -0.08 0.01"><freejoint name="view_toy_free"/>'
        '<geom name="view_toy_geom" type="box" size="0.01 0.01 0.01"/></body></worldbody>',
    )
    return mujoco.MjModel.from_xml_string(xml)


def main() -> None:
    result = {}
    for body, xml in BODIES.items():
        with np.load(OUT / f"development-{body}.npz") as trace:
            qpos = trace["qpos"]
            contact = trace["finger_toy_contact"]
            dt = float(trace["step_dt_s"])
        model = model_with_cube(xml)
        data = mujoco.MjData(model)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_hand_center")
        finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger_tip")
        cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "view_toy_geom")
        rows = []
        for frame in range(len(qpos)):
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            rows.append({
                "time_s": round((frame + 1) * dt, 3),
                "finger_touch": bool(contact[frame]),
                "hand_site_to_cube_xyz_m": np.round(data.site_xpos[site_id] - data.geom_xpos[cube_id], 4).tolist(),
                "finger_geom_to_cube_xyz_m": np.round(data.geom_xpos[finger_id] - data.geom_xpos[cube_id], 4).tolist(),
            })
        touched = [row["time_s"] for row in rows if row["finger_touch"]]
        data.qpos[:] = qpos[min(len(qpos) - 1, round(5 / dt) - 1)]
        mujoco.mj_forward(model, data)
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacGeom(model, data, jacp, jacr, finger_id)
        arm_jacobian = {}
        for name in ("right_shoulder_yaw", "right_shoulder_roll", "right_shoulder_pitch", "right_elbow_pitch"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            arm_jacobian[name] = np.round(jacp[:, model.jnt_dofadr[joint_id]], 4).tolist()
        result[body] = {
            "contact_first_last_s": [min(touched), max(touched)] if touched else None,
            "finger_xyz_jacobian_at_5s_m_per_rad": arm_jacobian,
            "samples": [min(rows, key=lambda r: abs(r["time_s"] - time))
                        for time in (3, 4, 5, 6, 7, 8)],
        }
    (OUT / "contact-geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
