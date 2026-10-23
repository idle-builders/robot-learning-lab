"""Blank-slate knee packaging screen against the exported 350 mm ape STEP.

Run: .venv/bin/python experiments/compact_morphology/ape_design/swarm_blank_knee_screen.py
This is a B-rep interference/scale screen, not a mechanism design or a load test.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from build123d import Axis, Plane, Solid, import_step


ROOT = Path(__file__).resolve().parents[3]
STEP = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUTPUT = Path(__file__).with_name("swarm_blank_knee_screen.json")
PIVOT = (42, 3, 86)  # the exported right knee cuff's nominal centre, mm
DRUM_RADIUS = 8  # hypothetical cable pitch radius, mm
TENSION_CASES_NM = (0.745, 1.569, 2.0)


def volume_of_intersection(a, b):
    hit = a & b
    return 0.0 if hit is None else hit.volume


def bounds(shape):
    bb = shape.bounding_box()
    return [[round(v, 3) for v in bb.min], [round(v, 3) for v in bb.max]]


def main():
    assembly = import_step(STEP)
    parts = {shape.label: shape for shape in assembly.children}
    required = (
        "R_thigh_front", "R_thigh_back", "R_knee_cuff",
        "R_shin_front", "R_shin_back", "torso_front", "torso_back",
    )
    assert all(name in parts for name in required), "named STEP parts missing"
    assert len(assembly.solids()) == 67, "unexpected source assembly"

    axis = Axis(PIVOT, (1, 0, 0))
    motion = []
    for angle in (0, 30, 60, 90, 120):
        shins = {
            name: parts[name].rotate(axis, angle)
            for name in ("R_shin_front", "R_shin_back")
        }
        row = {"flex_deg": angle}
        for fixed in ("R_thigh_front", "R_thigh_back", "R_knee_cuff"):
            row[fixed + "_overlap_mm3"] = round(sum(
                volume_of_intersection(parts[fixed], shins[moving])
                for moving in shins
            ), 3)
        if angle == 120:
            hit = parts["R_thigh_back"] & shins["R_shin_back"]
            row["back_thigh_back_shin_overlap_bounds_mm"] = bounds(hit)
        motion.append(row)

    # A single tendon drum at the knee, with its axle along X. It overlaps the
    # existing cosmetic cuff; the volume quantifies a required cuff redesign.
    drum = Solid.make_cylinder(
        DRUM_RADIUS, 6, Plane(origin=(39, 3, 86), z_dir=(1, 0, 0))
    )
    drum_interference = {
        name: round(volume_of_intersection(drum, parts[name]), 3)
        for name in ("R_knee_cuff", "R_thigh_front", "R_thigh_back",
                     "R_shin_front", "R_shin_back")
    }

    # Generous geometric placeholder for one 22-mm class motor+gear assembly
    # per knee; no claim that a configured product is 85 mm long or complete.
    torso_envelopes = []
    for x in (-22, 22):
        motor = Solid.make_cylinder(11, 85, Plane(origin=(x, 3, 177)))
        torso_envelopes.append({
            "axis_x_mm": x,
            "bounds_mm": bounds(motor),
            "torso_shell_overlap_mm3": round(sum(
                volume_of_intersection(motor, parts[name])
                for name in ("torso_front", "torso_back")
            ), 3),
        })

    result = {
        "source_step": str(STEP.relative_to(ROOT)),
        "source_step_part_count": len(assembly.solids()),
        "right_knee_pivot_mm": PIVOT,
        "flexion_convention": "+X rotation moves the shin posteriorly",
        "motion": motion,
        "drum": {
            "pitch_radius_mm": DRUM_RADIUS,
            "body_bounds_mm": bounds(drum),
            "fixed_shell_overlap_mm3": drum_interference,
            "one_side_cable_travel_0_to_120_mm": round(
                DRUM_RADIUS * math.radians(120), 3
            ),
            "differential_tension_at_knee_N": {
                str(torque): round(torque * 1000 / DRUM_RADIUS, 3)
                for torque in TENSION_CASES_NM
            },
            "cable_speed_for_120deg_in_0p5s_mm_s": round(
                DRUM_RADIUS * math.radians(120) / 0.5, 3
            ),
            "drum_speed_for_120deg_in_0p5s_rpm": 40,
        },
        "torso_envelope_screen": torso_envelopes,
        "limits": [
            "B-rep overlap is between unmodified decorative shell solids, not a structural clearance model.",
            "Zero overlap for torso cylinders does not prove full cavity containment, cable routing, mount clearance, or heat rejection.",
            "No bearings, pulley flanges, cable groove, actuator, brake, or hip-motion route are modeled.",
        ],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
