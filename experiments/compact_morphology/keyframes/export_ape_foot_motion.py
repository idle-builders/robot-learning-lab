"""Export a physics-replayed Robot Keyframe Kit motion for MJLab tracking.

Use only a replay that has passed its physical contact/return-to-stand review.
The upstream MJLab MotionCommand format uses joint and body world trajectories.
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


MODEL = Path(__file__).with_name("ape_foot_upright_scene.xml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    frames = np.load(args.replay)["qpos"]
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    dt = 0.02
    body_pos, body_quat = [], []
    for q in frames:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[1:].copy())
        body_quat.append(data.xquat[1:].copy())
    body_pos = np.asarray(body_pos)
    body_quat = np.asarray(body_quat)
    quat_xyzw = body_quat[..., [1, 2, 3, 0]]
    rotation = Rotation.from_quat(quat_xyzw.reshape(-1, 4)).as_matrix().reshape(*quat_xyzw.shape[:2], 3, 3)
    relative = np.einsum("tbij,tbjk->tbik", rotation[1:], rotation[:-1].transpose(0, 1, 3, 2))
    angular_step = Rotation.from_matrix(relative.reshape(-1, 3, 3)).as_rotvec().reshape(len(frames) - 1, model.nbody - 1, 3)
    body_ang_vel = np.concatenate([angular_step[:1], angular_step], axis=0) / dt
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        fps=np.array([round(1 / dt)]),
        joint_pos=frames[:, 7:].astype(np.float32),
        joint_vel=np.gradient(frames[:, 7:], dt, axis=0).astype(np.float32),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=np.gradient(body_pos, dt, axis=0).astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
    )
    print(args.out, len(frames), "frames", model.nbody - 1, "bodies")


if __name__ == "__main__":
    main()
