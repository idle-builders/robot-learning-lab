"""Render a saved native reach trace without changing its physics evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--body-xml", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--azimuth", type=float, default=135)
    parser.add_argument("--elevation", type=float, default=-20)
    parser.add_argument("--lookat", type=float, nargs=3, default=(0.10, 0.0, 0.17),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--distance", type=float, default=0.65)
    parser.add_argument("--toy-marker", action="store_true",
                        help="Draw a display-only amber post above the toy so its position remains visible")
    args = parser.parse_args()
    # The native scene appends a free-joint 2 cm cube after the robot. Add a
    # display-only copy so its qpos ordering matches the saved native trace.
    marker = ('<geom name="view_toy_marker" type="capsule" fromto="0 0 0.012 0 0 0.08" '
              'size="0.002" rgba="1 0.8 0.08 0.9" contype="0" conaffinity="0"/>'
              if args.toy_marker else '')
    xml = args.body_xml.read_text().replace(
        "</worldbody>",
        '<body name="view_toy" pos="0.075 -0.08 0.01">'
        '<freejoint name="view_toy_free"/>'
        '<geom name="view_toy_geom" type="box" size="0.01 0.01 0.01" '
        'rgba="1 0.42 0.08 1"/>' + marker + '</body></worldbody>',
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    with np.load(args.trace) as trace:
        qpos = trace["qpos"]
        dt = float(trace["step_dt_s"])
    if qpos.shape[1] != model.nq:
        raise ValueError(f"Native trace has {qpos.shape[1]} qpos; display model has {model.nq}")
    toy_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "view_toy")
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] == toy_body:
            # The Cove MJCF assigns group 4 by default. MuJoCo's default
            # viewer hides that group, including this display-only cube.
            model.geom_group[geom_id] = 0
        if model.geom_bodyid[geom_id] != 0 and model.geom_group[geom_id] == 4 and model.geom_rgba[geom_id, 3] == 0:
            model.geom_rgba[geom_id, 3] = 1
            model.geom_group[geom_id] = 0
    camera = mujoco.MjvCamera()
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    args.video.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    fps = 20
    frame_times = np.arange(0, len(qpos) * dt, 1 / fps)
    with imageio.get_writer(args.video, fps=20, codec="libx264", quality=8,
                            macro_block_size=16) as writer:
        for time_s in frame_times:
            frame = min(len(qpos) - 1, round(time_s / dt))
            data.qpos[:] = qpos[frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    renderer.close()
    print(f"rendered {args.video} from {len(qpos)} native qpos at {dt:g}s")


if __name__ == "__main__":
    main()
