"""Small ST-3250 torso/cable screen against the exported 350 mm ape STEP.

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_blank_knee_lowcost.py
Boxes are Feetech-source-size case proxies; cable exits and spools are proposed
envelopes, not vendor CAD, manufacturing clearance, or a motion-certified route.
"""

from __future__ import annotations

import json
from pathlib import Path

from build123d import Plane, Solid, Vector, import_step


ROOT = Path(__file__).resolve().parents[3]
STEP = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUT = Path(__file__).with_suffix(".json")
CASE_XYZ_MM = (35.0, 24.72, 45.22)  # official 45.22 x 24.72 x 35, rotated
RATED_NM = 16 * 9.80665 / 100  # kgf cm to Nm; stall torque is excluded
DRUM_RADIUS_MM = 8.0
SHEATH_RADIUS_MM = 1.5  # assumed conduit outer radius for packaging only


def overlap(a, b):
    hit = a & b
    return 0.0 if hit is None else hit.volume


def bounds(shape):
    bb = shape.bounding_box()
    return [[round(v, 3) for v in bb.min], [round(v, 3) for v in bb.max]]


def capsule_segment(a, b, radius):
    direction = Vector(b) - Vector(a)
    return Solid.make_cylinder(
        radius, direction.length, Plane(origin=a, z_dir=direction)
    )


def main():
    assembly = import_step(STEP)
    parts = {item.label: item for item in assembly.children}
    assert len(assembly.solids()) == 67
    shell_names = [
        name for name in parts
        if any(token in name for token in
               ("torso_", "pelvis_", "shoulder_trim", "upper_arm_", "thigh_"))
    ]

    sides = []
    cases = []
    spools = []
    for sign, side in ((-1, "L"), (1, "R")):
        # Each output face is toward the centreline. A and C are reoriented to
        # place the 45.22 mm long dimension vertically and the 35 mm dimension
        # along the shaft axis X. Shaft position, tabs, horns are not known.
        x0 = -44.5 if sign < 0 else 9.5
        case = Solid.make_box(*CASE_XYZ_MM, Plane(origin=(x0, -9.36, 195)))
        spool_x0 = -9.5 if sign < 0 else 3.5
        spool = Solid.make_cylinder(
            DRUM_RADIUS_MM, 6,
            Plane(origin=(spool_x0, 3, 224), z_dir=(1, 0, 0)),
        )
        cases.append(case)
        spools.append(spool)

        sheaths = []
        for y in (-7.5, 13.5):
            path = [
                (sign * 6.5, y, 224),
                (sign * 6.5, y, 190),
                (sign * 31, y, 174),
                (sign * 31, y, 167),
            ]
            segments = [
                capsule_segment(a, b, SHEATH_RADIUS_MM)
                for a, b in zip(path, path[1:])
            ]
            sheaths.append({
                "centerline_mm": path,
                "shell_overlap_mm3": round(sum(
                    overlap(segment, parts[name])
                    for segment in segments for name in shell_names
                ), 3),
            })

        sides.append({
            "side": side,
            "case_bounds_mm": bounds(case),
            "case_shell_overlap_mm3": round(sum(
                overlap(case, parts[name]) for name in shell_names
            ), 3),
            "case_min_distance_to_torso_shell_mm": round(min(
                case.distance_to(parts[name])
                for name in ("torso_front", "torso_back")
            ), 3),
            "case_min_distance_to_pelvis_shell_mm": round(min(
                case.distance_to(parts[name])
                for name in ("pelvis_front", "pelvis_back")
            ), 3),
            "case_distance_to_own_shoulder_trim_mm": round(
                case.distance_to(parts[side + "_shoulder_trim"]), 3
            ),
            "spool_bounds_mm": bounds(spool),
            "spool_shell_overlap_mm3": round(sum(
                overlap(spool, parts[name]) for name in shell_names
            ), 3),
            "sheaths": sheaths,
        })

    result = {
        "source_step": str(STEP.relative_to(ROOT)),
        "source_step_solids": len(assembly.solids()),
        "case_source": "Feetech ST-3250-C001, 45.22 x 24.72 x 35 mm",
        "screened_shell_parts": shell_names,
        "placement": sides,
        "case_to_case_overlap_mm3": round(overlap(cases[0], cases[1]), 3),
        "spool_to_spool_overlap_mm3": round(overlap(spools[0], spools[1]), 3),
        "rating_and_arithmetic": {
            "rated_output_nm_at_12v": round(RATED_NM, 3),
            "one_to_one_knee_nm_by_assumed_bowden_efficiency": {
                str(eta): round(RATED_NM * eta, 3)
                for eta in (0.5, 0.7, 0.9)
            },
            "required_efficiency_for_0p745_nm": round(0.745 / RATED_NM, 3),
            "required_efficiency_for_1p569_nm": round(1.569 / RATED_NM, 3),
            "servo_angle_needed_for_120deg_knee_at_1to1_deg": 120,
            "cable_travel_per_side_mm": round(
                DRUM_RADIUS_MM * 120 * 3.141592653589793 / 180, 3
            ),
            "feetech_no_load_time_for_120deg_s": 0.266,
            "required_loaded_knee_rpm_for_120deg_in_0p5s": 40,
        },
        "limits": [
            "Only fixed case boxes, fixed illustrative spools and cable-exit segments were screened.",
            "Official case size does not resolve output shaft location, mounting ears, horn, connectors or service clearance.",
            "The original sketch STEP has cosmetic pelvis and shoulder trim, not installed hip/shoulder actuators or a moving hip envelope.",
            "Cable paths end inside the pelvis cavity at z=167 mm; routing through moving hip joints and down the thigh is unscreened.",
            "Feetech no-load speed does not specify loaded speed; rated torque is not a published duty-cycle or thermal guarantee.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
