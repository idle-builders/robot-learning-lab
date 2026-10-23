"""Render a saved closed-loop MJLab qpos trace locally without a remote GPU viewer."""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "artifacts/ape-frame50-bare/ape-frame50-nominal.xml"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    trace = np.load(args.trace)
    poses = trace["qpos"]
    dt = float(trace["dt"])
    if poses.shape[1] != model.nq:
        raise ValueError(f"Trace has {poses.shape[1]} qpos, body expects {model.nq}")
    # MJLab distributes worlds on a terrain grid; show world zero around its
    # own reset origin rather than looking at the global (0, 0) grid origin.
    origin_xy = poses[0, :2].copy()
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 0.23]
    camera.distance = 1.05
    camera.azimuth = 90
    camera.elevation = -15
    step = max(1, round(1 / (25 * dt)))
    frames = []
    with mujoco.Renderer(model, width=640, height=480) as renderer:
        for pose in poses[::step]:
            data.qpos[:] = pose
            data.qpos[:2] -= origin_xy
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], 0.23]
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    args.video.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(args.video, frames, fps=round(1 / (dt * step)), macro_block_size=16)
    print(args.video, len(frames), "frames")


if __name__ == "__main__":
    main()
