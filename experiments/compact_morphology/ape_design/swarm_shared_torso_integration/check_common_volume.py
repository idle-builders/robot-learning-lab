"""Cross-check independently designed ape joint coupons in one source frame.

This is a direct CAD execution gate, not a test suite. A local coupon pass does
not imply the mirrored shoulder, hip, battery and service all fit together.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import build123d as b

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from experiments.compact_morphology.ape_design.swarm_hip_yaw_rethink.hip_yaw_rethink import (
    assembly_solids,
)
from experiments.compact_morphology.ape_design.swarm_scaled_shoulder.scaled_shoulder_coupon import (
    mechanism,
)
from experiments.compact_morphology.ape_design.swarm_rated_leg_skin_v2.skin_v2_case import (
    integration_shapes,
)


OUT = ROOT / "artifacts/shared-torso-integration"


def broad_pair(a, c, gap=0.5):
    x, y = a.bounding_box(), c.bounding_box()
    return all(
        getattr(x.min, axis) <= getattr(y.max, axis) + gap
        and getattr(y.min, axis) <= getattr(x.max, axis) + gap
        for axis in "XYZ"
    )


def volume(shape):
    return 0.0 if shape is None else sum(piece.volume for piece in shape.solids())


def robust_common(a, c):
    """Conservatively retain both OCCT Boolean operand-order results.

    Some near-tangent source shell pairs return zero in one intersect order and
    a finite solid in the other. The larger result is a collision warning,
    while the order disagreement is retained for inspection.
    """
    ab = volume(a.intersect(c))
    ba = volume(c.intersect(a))
    return max(ab, ba), ab, ba


def pair_report(group_a, group_b):
    hits, near, asymmetric = [], [], []
    for name_a, shape_a in group_a:
        for name_b, shape_b in group_b:
            if not broad_pair(shape_a, shape_b):
                continue
            common, ab, ba = robust_common(shape_a, shape_b)
            distance = shape_a.distance_to(shape_b)
            row = {"a": name_a, "b": name_b, "common_mm3": round(common, 4),
                   "surface_gap_mm": round(distance, 4)}
            if abs(ab - ba) > 0.01:
                asymmetric.append({**row, "a_intersect_b_mm3": round(ab, 4),
                                   "b_intersect_a_mm3": round(ba, 4)})
            if common > 0.01:
                hits.append(row)
            elif distance < 0.5:
                near.append(row)
    return {"intersections": hits, "less_than_0p5mm_gap": near,
            "operand_order_disagreements": asymmetric}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=500, choices=(350, 500))
    parser.add_argument("--offset-knee", action="store_true",
                        help="screen the +10 mm rear, +12 mm up 50 cm knee coupon")
    args = parser.parse_args()
    height = args.height
    if args.offset_knee and height != 500:
        parser.error("the offset-knee coupon exists only at 500 mm")
    right_shoulder, _ = mechanism(height)
    shoulders = []
    for part in right_shoulder:
        shoulders.append(("R " + part["name"], part["shape"]))
        shoulders.append(("L " + part["name"], part["shape"].mirror(b.Plane.YZ)))

    hip = assembly_solids(height=height, layout="remote",
                          pack_z_source=200, service_z_source=225)
    power = [(name, shape) for name, shape in hip["fixed"]
             if "battery" in name or "service" in name]
    hip_hardware = [(name, shape) for name, shape in hip["fixed"] + hip["moving"]
                    if "battery" not in name and "service" not in name]
    knees = []
    if height == 500:
        knee = integration_shapes(height)
        for frame in ("fixed_knee_hardware", "moving_knee_hardware"):
            for name, shape in knee[frame].items():
                if args.offset_knee and name != "shin_cross_pad":
                    shape = b.Pos(0, 10, 12) * shape
                knees.append(("R " + frame + " " + name, shape))
                knees.append(("L " + frame + " " + name, shape.mirror(b.Plane.YZ)))
        if args.offset_knee:
            bridge = b.Pos(61, 11, 116) * b.Box(6, 12, 12)
            knees += [("R moving bridge", bridge),
                      ("L moving bridge", bridge.mirror(b.Plane.YZ))]
    report = {
        "height_mm": height,
        "source_frame": "sketch_ape_shell: X lateral, Y rear, Z up",
        "scope": "cross-coupon neutral only; each coupon's own shell and moving-pose gates remain separate",
        "shoulder_vs_hip": pair_report(shoulders, hip_hardware),
        "shoulder_vs_power": pair_report(shoulders, power),
        "hip_vs_power": pair_report(hip_hardware, power),
        "knee_vs_hip_yaw": pair_report(knees, hip_hardware),
        "knee_vs_shoulder": pair_report(knees, shoulders),
        "knee_vs_power": pair_report(knees, power),
        "battery_vs_service": pair_report(power[:1], power[1:]),
        "shoulder_parts": len(shoulders),
        "hip_parts": len(hip_hardware),
        "knee_parts": len(knees),
        "offset_knee": args.offset_knee,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "-offset-knee" if args.offset_knee else ""
    (OUT / f"report-{height}{suffix}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: len(value["intersections"])
                      for key, value in report.items()
                      if isinstance(value, dict) and "intersections" in value}, indent=2))


if __name__ == "__main__":
    main()
