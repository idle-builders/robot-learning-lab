"""Name the fixed part causing each failed knee pose before changing skin."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import (
    integration_shapes,
    turn,
)

OUT = ROOT / "artifacts/knee-gaiter-20261005"


def overlap(a, b):
    bb_a, bb_b = a.bounding_box(), b.bounding_box()
    if any(getattr(bb_a.min, axis) > getattr(bb_b.max, axis)
           or getattr(bb_b.min, axis) > getattr(bb_a.max, axis)
           for axis in "XYZ"):
        return 0.0
    common = a & b
    return 0.0 if common is None else sum(s.volume for s in common.solids())


def main():
    case = integration_shapes(500)
    fixed = {**case["fixed_knee_skin"], **case["fixed_knee_hardware"]}
    moving = {**case["moving_shin_skin"],
              "yoke": case["moving_knee_hardware"]["yoke"],
              "shin_cross_pad": case["moving_knee_hardware"]["shin_cross_pad"]}
    rows = []
    for flex in (0, 2, 15, 30, 60, 90, 113, 115):
        for name, shape in moving.items():
            posed = turn(shape, case["knee_axis_mm"], flex)
            for other, solid in fixed.items():
                common = overlap(posed, solid)
                if common > 0.05:
                    rows.append({"flex_deg": flex, "moving": name, "fixed": other,
                                 "common_mm3": round(common, 3)})
    rows.sort(key=lambda r: (r["flex_deg"], -r["common_mm3"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "named-contacts.json").write_text(json.dumps(rows, indent=2))
    for flex in (0, 2, 15, 30, 60, 90, 113, 115):
        print(flex, [(r["moving"], r["fixed"], r["common_mm3"])
                     for r in rows if r["flex_deg"] == flex][:8])


if __name__ == "__main__":
    main()
