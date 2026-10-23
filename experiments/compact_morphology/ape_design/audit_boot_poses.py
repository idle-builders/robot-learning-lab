"""Place the CAD boot/contact meshes in retained stand, step and squat poses."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-full-boot-control-20261002"
PRIOR = ROOT / "artifacts/ape-physics-pair-20260930"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(OUT / "models/boot_cad_mass.xml"))
    data = mujoco.MjData(model)
    geometry = {
        name: trimesh.load_mesh(OUT / "models" / file).vertices
        for name, file in (("shell", "boot_shell.stl"),
                           ("outsole", "boot_outsole.stl"),
                           ("contact", "boot_contact_hull.stl"))
    }
    squat = np.load(PRIOR / "squat-baseline.npz")["qpos"]
    step = np.load(PRIOR / "step-left-baseline.npz")["qpos"]
    assert model.nq == squat.shape[1] == step.shape[1] == 29
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
    step_foot_z = []
    for qpos in step:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        step_foot_z.append(float(data.xpos[left_id, 2]))
    poses = {
        "stand": (step[0], "step-left-baseline.npz", 0),
        "left_step_peak": (step[int(np.argmax(step_foot_z))],
                           "step-left-baseline.npz", int(np.argmax(step_foot_z))),
        "squat_low": (squat[int(np.argmin(squat[:, 2]))],
                      "squat-baseline.npz", int(np.argmin(squat[:, 2]))),
    }
    result = {"method": "Retained 22-joint qpos placed in the new MJCF; exact CAD STL vertices and collision hull transformed with each foot body. Floor is z=0. These are pose screens, not dynamically replayed boot skills.",
              "source": "artifacts/ape-physics-pair-20260930", "poses": {}}
    for name, (qpos, source, frame) in poses.items():
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        item = {"source": source, "frame": frame, "pelvis_z_m": float(data.qpos[2]),
                "feet": {}}
        for side in ("left", "right"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
            origin = data.xpos[body_id]
            rotation = data.xmat[body_id].reshape(3, 3)
            surfaces = {}
            for part, vertices in geometry.items():
                transformed = vertices @ rotation.T + origin
                surfaces[part] = {"min_z_m": float(transformed[:, 2].min()),
                                  "vertices_below_floor": int((transformed[:, 2] < -0.001).sum())}
            item["feet"][side] = surfaces
        result["poses"][name] = item
    (OUT / "pose-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["poses"], indent=2))


if __name__ == "__main__":
    main()
