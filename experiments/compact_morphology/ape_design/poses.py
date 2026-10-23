"""Kinematic pose states for the assembled ape viewer.

This exposes the existing MJCF joint tree and source-case clearance, not a
claim that the new printed covers clear every coupled trajectory.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from build import MODEL_PATH, OUT, ROOT, battery_pose, poses
from ape_cad.build import clashes


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    neutral = np.array(model.key_qpos[0])
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                  for i in range(1, model.nbody)]
    states = []

    def record(label: str, group: str, qpos: np.ndarray) -> None:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        visible = {}
        for name in body_names:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            visible[name] = {
                "position_m": np.round(data.xpos[bid], 6).tolist(),
                "quaternion_xyzw": np.round(Rotation.from_matrix(
                    data.xmat[bid].reshape(3, 3)).as_quat(), 7).tolist(),
            }
        source_cases = poses(model, data) + [battery_pose(model, data)]
        states.append({
            "label": label, "group": group, "bodies": visible,
            "motor_envelope_clashes_1mm": clashes(source_cases, 1),
        })

    record("Neutral", "neutral", neutral)
    for group, path in (
        ("walk", ROOT / "artifacts/ape-cad-replay-20260930/walk-v8.npz"),
        ("squat", ROOT / "artifacts/ape-cad-replay-20260930/squat-v8.npz"),
        ("step", ROOT / "artifacts/ape-cad-replay-20260930/step-left-v8.npz"),
        ("floor_acquire_place", ROOT / "artifacts/ape-physics-pair-20260930/acquire-place-candidate.npz"),
    ):
        if not path.exists():
            continue
        with np.load(path) as trace:
            frames = trace["qpos"]
        for i in np.unique(np.linspace(0, len(frames) - 1, 13).astype(int)):
            record(f"{group} · frame {i}", group, frames[i])

    # An isolated-axis control reveals the simulated travel without pretending
    # that it establishes shell-to-shell clearance under coupled motion.
    for jid in range(1, model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        lo, hi = model.jnt_range[jid]
        for i, angle in enumerate(np.linspace(lo, hi, 13)):
            qpos = neutral.copy()
            qpos[model.jnt_qposadr[jid]] = angle
            record(f"{name} · {np.degrees(angle):.0f}°",
                   f"joint:{name}", qpos)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "poses.json").write_text(json.dumps({
        "source_model": str(MODEL_PATH.relative_to(ROOT)),
        "warning": "Printed-shell collision and structural strength are not certified by these kinematic poses.",
        "states": states,
    }, separators=(",", ":")))
    print(f"{len(states)} pose states → {OUT / 'poses.json'}")


if __name__ == "__main__":
    main()
