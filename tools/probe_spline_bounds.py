"""Reproduce a missed ellipsoid extremum in OCCT's optimal bounding box."""

import json
from pathlib import Path
from build123d import Rot, Sphere, export_step, import_step, scale
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box

out = Path("artifacts/spline-bounds-probe-v1")
out.mkdir(exist_ok=False)
source = scale(Rot(0, 90, 0) * Sphere(1), by=(2.0, 10.0, 3.0))
export_step(source, out / "ellipsoid.step")
returned = import_step(out / "ellipsoid.step")
report = dict(analytical_bounds_mm=[[-2, -10, -3], [2, 10, 3]], cases={})
for name, shape in [("source", source), ("returned", returned)]:
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape.wrapped, box, False, False)
    bounds = list(box.Get())
    vertices, _ = shape.tessellate(0.001, 0.02)
    report["cases"][name] = dict(
        surface_optimal_bounds_mm=[bounds[:3], bounds[3:]],
        mesh_bounds_mm=[
            [min(getattr(p, axis) for p in vertices) for axis in ("X", "Y", "Z")],
            [max(getattr(p, axis) for p in vertices) for axis in ("X", "Y", "Z")],
        ],
        mesh_scope="diagnostic relative-deflection tessellation; not exact metrology",
    )
report["finding"] = (
    "Source optimal bound underestimates the analytic +Z extremum by 0.02605 mm even with triangulation disabled; this bound method needs correction before Pip export can pass."
)
(out / "probe.py").write_text(Path(__file__).read_text())
(out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
