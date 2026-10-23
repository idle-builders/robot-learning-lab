"""Bounded bare-case envelope screen for a compact upper-thigh knee drive.

No motor or cable is selected by this screen. It deliberately omits output
shaft, connector, mounts, support, tensioner and skin articulation. The case
axis mapping is a hypothesis because the manufacturer drawing datum is not
yet imported. This is a reject-or-continue packaging calculation only.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import build123d as bd


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/lightweight-ape-20261004/thigh-case-envelope.json"
STATIONS = [(87, 42, 3, 16, 18), (98, 43, 3, 20, 22),
            (124, 39, 3, 24, 25), (145, 35, 4, 22, 23),
            (157, 33, 4, 16, 18), (163, 31, 4, 8, 10)]
CASE_SIZE = (36.5, 15.0, 34.0)  # FT-3332 official envelope, provisional orientation.


def thigh_interior(wall: float) -> bd.Shape:
    return bd.Solid.make_loft([
        bd.Wire.make_ellipse(rx-wall, ry-wall, bd.Plane(origin=(x, y, z)))
        for z, x, y, rx, ry in STATIONS
    ])


def volume(shape: bd.Shape) -> float:
    return sum(s.volume for s in shape.solids())


def screen(wall: float) -> dict:
    inside = thigh_interior(wall)
    rows = []
    for x, y, z in itertools.product((38, 39, 40, 41, 42), (1, 3, 5), (120, 122, 124, 126, 128)):
        case = bd.Pos(x, y, z) * bd.Box(*CASE_SIZE)
        clearance = bd.Pos(x, y, z) * bd.Box(*(d+2 for d in CASE_SIZE))
        rows.append({"center_mm": [x,y,z],
                     "bare_case_outside_interior_mm3": round(volume(case-inside), 3),
                     "one_mm_service_box_outside_interior_mm3": round(volume(clearance-inside), 3)})
    best = min(rows, key=lambda item:(item["one_mm_service_box_outside_interior_mm3"],
                                       item["bare_case_outside_interior_mm3"]))
    return {"wall_mm": wall, "searched_centers": len(rows), "best": best,
            "zero_bare_case_locations": sum(row["bare_case_outside_interior_mm3"] < .01 for row in rows),
            "zero_service_envelope_locations": sum(row["one_mm_service_box_outside_interior_mm3"] < .01 for row in rows),
            "rows": rows}


def main() -> None:
    report = {
        "source_shell": "experiments/sketch_ape_shell/build.py, 350 mm nominal thigh loft",
        "source_case": "https://www.feetechrc.com/538886",
        "status": "bare-case reject screen only; orientation/output datum not verified",
        "case_size_xyz_mm": CASE_SIZE,
        "service_envelope_adds_mm_per_side": 1,
        "variants": [screen(1), screen(2)],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for item in report["variants"]:
        print(json.dumps({k: item[k] for k in ("wall_mm", "best", "zero_bare_case_locations",
                                                  "zero_service_envelope_locations")}, indent=2))


if __name__ == "__main__":
    main()
