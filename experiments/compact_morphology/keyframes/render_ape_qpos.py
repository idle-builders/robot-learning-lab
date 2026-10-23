"""Render saved native MJLab ape poses locally without a remote GL context."""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np


SCENE = Path(__file__).with_name("ape_foot_upright_scene.xml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motion", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=SCENE,
                        help="MJCF body used to render the saved qpos (default: upright scene)")
    parser.add_argument("--follow-root", action="store_true",
                        help="Follow root x/y for walking trajectories")
    args = parser.parse_args()
    with np.load(args.motion) as trajectory:
        poses = trajectory["qpos"]
        dt = float(trajectory["dt"])
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if poses.shape[1] != model.nq:
        raise ValueError(f"motion has {poses.shape[1]} qpos values; model needs {model.nq}")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.15]
    camera.distance = 0.62
    camera.azimuth = 135
    camera.elevation = -18
    stride = max(1, round(1 / (25 * dt)))
    frames = []
    with mujoco.Renderer(model, 480, 640) as renderer:
        for pose in poses[::stride]:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            if args.follow_root:
                camera.lookat[:2] = pose[:2]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=round(1 / (dt * stride)), macro_block_size=16)
    print(args.out)


if __name__ == "__main__":
    main()
