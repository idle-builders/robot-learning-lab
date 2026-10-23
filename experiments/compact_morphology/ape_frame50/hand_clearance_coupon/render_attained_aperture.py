"""Render one *static IK* three-angle jaw triplet; physics is not advanced."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50.hand_clearance_coupon.audit_attained_aperture import with_cube


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-case", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT / "artifacts/ape-cove50-hand-clearance-coupon/attained-aperture-ik-v2"
    fits = json.loads((folder / "fit-matrix.json").read_text())["cases"]
    fit = next(row for row in fits if row["case"] == args.fit_case)
    if not fit["clean_static_triplet_placements"]:
        raise ValueError("The selected static fit has no clean closing triplet")
    with np.load(folder / "fitted-states.npz") as saved:
        index = list(saved["fit_case"]).index(args.fit_case)
        qpos = saved["qpos"][index].copy()
    model = with_cube(ROOT / "artifacts/ape-cove50-hand-clearance-coupon/physics-v2/candidate-training.xml")
    data = mujoco.MjData(model)
    cube_id = model.geom("audit_cube_geom").id
    model.geom_rgba[cube_id] = (1, .36, .09, 1)
    for geom_id in range(model.ngeom):
        if model.geom_bodyid[geom_id] != 0 and model.geom_group[geom_id] == 4:
            model.geom_group[geom_id] = 0
            if model.geom_rgba[geom_id, 3] == 0:
                model.geom_rgba[geom_id, 3] = .5
    cube_adr = model.joint("audit_cube_free").qposadr[0]
    jaw_adr = model.joint("right_gripper").qposadr[0]
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (qpos[cube_adr], qpos[cube_adr + 1], .065)
    camera.distance = .32
    camera.azimuth = 145
    camera.elevation = -18
    args.out.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=480, width=640)
    with imageio.get_writer(args.out, fps=20, codec="libx264", quality=8,
                            macro_block_size=16) as writer:
        for jaw in np.r_[np.zeros(10), np.linspace(0, .25, 10),
                         np.linspace(.25, .5, 10), np.full(10, .5)]:
            data.qpos[:] = qpos
            data.qpos[jaw_adr] = jaw
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frame = Image.fromarray(renderer.render())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0, 0, 640, 30), fill=(18, 31, 39))
            draw.text((9, 8), f"STATIC IK ONLY - no motor or physics step  |  jaw {jaw:.2f} rad",
                      fill=(245, 248, 249))
            writer.append_data(np.asarray(frame))
    renderer.close()
    print(f"rendered static IK triplet {args.fit_case} -> {args.out}")


if __name__ == "__main__":
    main()
