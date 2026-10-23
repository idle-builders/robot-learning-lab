"""Render the retained learned walk with only the articulated Cove shell shown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from experiments.compact_morphology.ape_frame50 import build_cove_shell as shell
from experiments.compact_morphology.ape_frame50 import render_cad_replay as replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=shell.DEFAULT_OUT)
    args = parser.parse_args()
    folder = args.out.resolve()
    model = mujoco.MjModel.from_xml_path(str(folder / "cove50-shell-visual.xml"))
    data = mujoco.MjData(model)
    trace = np.load(replay.TRACE)
    poses, dt = trace["qpos"], float(trace["dt"])
    if poses.shape[1] != model.nq:
        raise ValueError("saved trace joint topology does not match visual shell")
    option = mujoco.MjvOption()
    option.geomgroup[1] = 0  # selected internal frame
    option.geomgroup[4] = 0  # old physics contact/mass proxies
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, .25]
    camera.distance = .87
    camera.azimuth = 135
    camera.elevation = -12
    frames = []
    step = max(1, round(1 / (25 * dt)))
    origin = poses[0, :2].copy()
    with mujoco.Renderer(model, width=960, height=720) as renderer:
        for pose in poses[::step]:
            data.qpos[:] = pose
            data.qpos[:2] -= origin
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [data.qpos[0], data.qpos[1], .25]
            renderer.update_scene(data, camera=camera, scene_option=option)
            frames.append(renderer.render().copy())
    imageio.imwrite(folder / "shell-walk-first-frame.png", frames[0])
    movie = folder / "shell-walk.mp4"
    imageio.mimwrite(movie, frames, fps=round(1 / (dt * step)), macro_block_size=16)
    (folder / "video-provenance.json").write_text(json.dumps({
        "saved_walk_video": str(movie.relative_to(shell.ROOT)),
        "saved_walk_video_sha256": shell.digest(movie),
        "saved_walk_trace_sha256": shell.digest(replay.TRACE),
        "saved_walk_frames": len(frames),
        "qualification": "articulated render of previously saved qpos, not a new learned-policy rollout",
    }, indent=2) + "\n")
    print(f"Rendered {len(frames)} retained shell poses: {movie}")


if __name__ == "__main__":
    main()
