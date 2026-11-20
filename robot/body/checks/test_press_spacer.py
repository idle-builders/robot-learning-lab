"""Spacer-block press model on synthetic parts with known answers (run: .venv/bin/python -B robot/body/checks/test_press_spacer.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import manifold3d as M
import numpy as np
import trimesh

from checks import press_lib as PC


def box(x0, x1, y0, y1, z0, z1):
    return M.Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def mesh_of(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.array(mm.vert_properties[:, :3]), np.array(mm.tri_verts), process=True)


def read(m):
    mesh = mesh_of(m)
    # exit plane z = 0, part above (z 0..5), counter side towards -z; sleeve OD 4.5
    return PC.counter_face(mesh, PC._rays(mesh), np.zeros(3), np.array([0.0, 0.0, -1.0]), 2.25)


plate = box(-30, 30, -30, 30, 0, 5)
CASES = {
    "plate": (plate, dict(flat=(99, 100.1), wall=(0, 0.1), free=(39.9, 40.0))),
    "wall": (plate + box(3, 30, -30, 30, -20, 0.0), dict(wall=(16.5, 20.5), flat=(79.5, 83.5), free=(39.9, 40.0))),
    "pocket32": (plate + box(-30, 30, -30, 30, -35, -32), dict(flat=(99, 100.1), free=(31.9, 32.1))),
    "recess": (plate - M.Manifold.cylinder(2, 5.2, 5.2, 64).translate((0, 0, 0)), dict(recess=(99, 100.1), flat=(0, 0.1), free=(0, 0))),
    "open": (box(-30, 30, -30, 30, 0, 5) - M.Manifold.cylinder(12, 5.2, 5.2, 64).translate((0, 0, -1)), dict(open=(99, 100.1))),
}
if __name__ == "__main__":
    for name, (m, want) in CASES.items():
        f = read(m)
        print(name, {k: f[k] for k in ("flat_pct", "wall_pct", "recess_pct", "open_pct", "free_depth_mm", "axis_free_depth_mm")})
        for k, (lo, hi) in want.items():
            v = f["free_depth_mm" if k == "free" else k + "_pct"]
            assert lo <= v <= hi, (name, k, v, lo, hi)
    print("ok")
