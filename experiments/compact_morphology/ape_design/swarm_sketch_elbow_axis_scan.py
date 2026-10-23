"""Scan an outboard elbow-axis offset on the exact 350 mm design shell."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import build123d as bd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/sketch_ape_shell"))
from build import Study  # noqa: E402

OUT = ROOT / "artifacts/swarm-sketch-elbow-axis-20261004"


def volume(s):
    return sum(p.volume for p in s) if isinstance(s, bd.ShapeList) else (s.volume if s else 0.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    study = Study(height=350)
    study.build()
    p = {s["name"]: s["shape"] for s in study.parts}
    fixed = [p[n] for n in ("R_upper_arm_front", "R_upper_arm_back")]
    moving = [p[n] for n in ("R_forearm_front", "R_forearm_back")]
    rows = []
    for x in (96, 100, 104, 108):
        axis = (x, 2, 194.5)
        row = {"axis_xyz_mm": axis, "poses": []}
        for deg in (0, 45, 80, 115):
            moved = [bd.Pos(*axis) * bd.Rot(0, -deg, 0) *
                     bd.Pos(*(-v for v in axis)) * s for s in moving]
            row["poses"].append({"deg": deg, "overlap_mm3": round(sum(
                volume(a.intersect(b)) for a in moved for b in fixed), 2)})
        rows.append(row)
        print(row, flush=True)
    (OUT / "report.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
