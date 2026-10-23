"""Screen source shell collisions if the right-knee pitch pivot moves in Y/Z.

This intentionally moves only the pivot: neutral shin/thigh appearance is unchanged.
It does not create a shaft, support, drive, skin closure, or knee mechanism.
"""

from __future__ import annotations

import json
from pathlib import Path

import build123d as bd

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-20261004/knee-axis-screen.json"


def volume(shape: bd.Shape | None) -> float:
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def overlap(a: bd.Shape, b: bd.Shape) -> float:
    # OpenCascade's compound/compound boolean can report an empty intersection
    # even when individual child solids intersect. Test every solid pair.
    return sum(volume(left & right) for left in a.solids() for right in b.solids())


def main() -> None:
    source = {child.label: child for child in bd.import_step(SOURCE).children}
    thigh = bd.Compound(children=[source[name] for name in ("R_thigh_front", "R_thigh_back")])
    shin = bd.Compound(children=[source[name] for name in ("R_shin_front", "R_shin_back")])
    rows = []
    centers = [(y, z) for y in (-5, 0, 3, 6, 10)
               for z in (82, 86, 90, 94, 98, 102)]
    centers += [(y, z) for y in (10, 14, 18) for z in (106, 110, 114)]
    centers += [(3, 99)]  # Both bounded coupons used this axis.
    for y, z in centers:
            axis = (42, y, z)
            hits = {}
            for angle in (45, 60, 90, 115, 120):
                moved = bd.Pos(*axis) * bd.Rot(angle, 0, 0) * bd.Pos(-axis[0], -axis[1], -axis[2]) * shin
                hits[str(angle)] = round(overlap(thigh, moved), 3)
            rows.append({"axis_mm": list(axis), "thigh_shin_overlap_mm3": hits,
                         "worst_overlap_mm3": max(hits.values())})
    rows.sort(key=lambda row: (row["worst_overlap_mm3"],
                               row["thigh_shin_overlap_mm3"]["115"]))
    reference_gaps = {}
    for y, z in ((3, 99), (10, 110), (14, 106)):
        axis = (42, y, z)
        moved = bd.Pos(*axis) * bd.Rot(120, 0, 0) * bd.Pos(-axis[0], -axis[1], -axis[2]) * shin
        reference_gaps[f"{y},{z}"] = round(min(left.distance_to(right)
            for left in thigh.solids() for right in moved.solids()), 3)
    report = {"source": str(SOURCE.relative_to(ROOT)),
              "status": "source-skin collision screen only; no supported or enclosed joint",
              "neutral_pose_unchanged": True,
              "reference_120_deg_gap_mm": reference_gaps, "rows": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows[:10]:
        print(row)
    print("Reference gaps at 120°:", reference_gaps)


if __name__ == "__main__":
    main()
