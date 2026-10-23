"""Optimistic circular-actuator envelopes in the exact 35 cm sketch thigh.

Only transverse knee-axis placement is tested. Each unit is represented by a
perfect cylinder of its published diameter and axial length, which can *under*
count housings, connectors and mounts. A failed cylinder fit rejects that
location; a passing one would still not prove an assembly.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd

from lightweight_ape_thigh_envelope import thigh_interior, volume


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lightweight-ape-20261004/round-thigh-screen.json"
UNITS = (
    ("RobStride05", 46.0, 44.0, 191, 1.8, 110,
     "https://robstride.com/products/robStride05"),
    ("EduLite05", 46.0, 44.0, 242, 1.8, 80,
     "https://www.robstride.com/products/eduLite05"),
    ("CubeMars_AK40-10", 53.0, 37.0, 185, 1.3, 99.90,
     "https://www.cubemars.com/product/ak40-10-robotic-actuator.html"),
    ("CubeMars_AK40-10_V3", 53.0, 40.2, 190, 1.3, 135.90,
     "https://www.cubemars.com/de/product/ak40-10-v3-0-kv170-robotic-actuator.html"),
)


def main() -> None:
    sections = {wall: thigh_interior(wall) for wall in (1.0, 2.0)}
    rows = []
    for name, diameter, length, grams, rated_nm, usd, source in UNITS:
        for x in (37.0, 39.0, 41.0):
            for z in (120.0, 124.0, 128.0):
                center = (x, 3.0, z)
                cylinder = bd.Solid.make_cylinder(
                    diameter/2, length,
                    bd.Plane(origin=(x-length/2, 3.0, z), z_dir=(1,0,0)))
                rows.append({"unit": name, "center_mm": center,
                             "ideal_cylinder_outside_1mm_thigh_mm3": round(volume(cylinder-sections[1.0]), 3),
                             "ideal_cylinder_outside_2mm_thigh_mm3": round(volume(cylinder-sections[2.0]), 3)})
    report = {"shell": "experiments/sketch_ape_shell/build.py exact 350 mm thigh loft",
              "status": "optimistic body envelope only; does not include output/support/mount/connector",
              "axis": "X / transverse knee axis",
              "units": [{"name": n, "diameter_mm": d, "axial_length_mm": l,
                         "mass_g": m, "rated_nm": t, "retail_usd": p, "source": s,
                         "best_1mm": min((r for r in rows if r["unit"] == n),
                                         key=lambda r:r["ideal_cylinder_outside_1mm_thigh_mm3"])}
                        for n,d,l,m,t,p,s in UNITS],
              "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for unit in report["units"]:
        print(unit["name"], unit["best_1mm"])


if __name__ == "__main__":
    main()
