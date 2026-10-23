"""Render a reloaded-policy state trace on the Mac with native MuJoCo."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", choices=("low_shoulder", "ape_short_torso", "waist_pitch"))
    parser.add_argument("task", choices=("locomotion", "reach"))
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model_dir = "reach_models" if args.task == "reach" else "playground_models"
    xml = Path(__file__).with_name(model_dir) / f"{args.shape}.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    trace = np.load(args.trace)
    stop = np.flatnonzero(trace["done"] > 0)
    end = int(stop[0] + 1) if len(stop) else len(trace["qpos"])
    stride = 1 if args.task == "reach" else 2
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.09, 0.0, 0.21]
    camera.distance = 0.83
    camera.azimuth = 135
    camera.elevation = -18
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        with imageio.get_writer(args.output, fps=25, codec="libx264", quality=7) as writer:
            for q, mocap in zip(trace["qpos"][:end:stride], trace["mocap_pos"][:end:stride]):
                data.qpos[:] = q
                if len(mocap):
                    data.mocap_pos[:] = mocap
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render().copy())
    print(args.output, args.output.stat().st_size)


if __name__ == "__main__":
    main()
