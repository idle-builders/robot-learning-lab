"""Render retained packaging meshes with shells hidden, without rerunning geometry."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str((source / "mechanism.xml").absolute()))
    d = mujoco.MjData(m)
    hidden = []
    for name in ("torso_front", "torso_rear"):
        mesh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, name)
        if mesh >= 0:
            m.geom_rgba[m.geom_dataid == mesh, 3] = 0
            hidden.append(name)
    mesh = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "carrier")
    if mesh >= 0:
        m.geom_rgba[m.geom_dataid == mesh, 3] = 0.2
    # The saved XML already places everything in world space. Center on board mesh vertices.
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MESH, "logic_supply_populated")
    gids = np.where(m.geom_dataid == mid)[0]
    mujoco.mj_forward(m, d)
    center = d.geom_xpos[gids[0]].copy()
    c = mujoco.MjvCamera()
    c.lookat[:] = center
    c.distance = 0.27
    c.elevation = -18
    with mujoco.Renderer(m, height=800, width=960) as renderer:
        for name, azimuth in [("rear-cutaway", 135), ("front-cutaway", 45)]:
            c.azimuth = azimuth
            renderer.update_scene(d, camera=c)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            Image.fromarray(renderer.render()).save(output / (name + ".png"))
    refs = [
        source / "mechanism.xml",
        source / "report.json",
        Path(__file__),
        *source.glob("*.stl"),
    ]
    report = {
        "scope": "visual review only; hidden shells and translucent carrier do not change fit evidence",
        "source": str(source),
        "hidden": hidden,
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in refs
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Logic supply packaging review</title><style>body{font:18px system-ui;background:#e6ebe8;padding:24px}img{max-width:48%}</style><h1>Logic supply: allocated position</h1><p>Green: populated buck board. Amber: battery and connector space allocations. Shells hidden; controller carrier translucent. Mounting structure and harness remain open.</p><img src="rear-cutaway.png"><img src="front-cutaway.png">'
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.source, a.output)
