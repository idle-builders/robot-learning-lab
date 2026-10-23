"""500 mm sketch-shell, internal ST3250 and Gates 123-3MGT-15 knee study.

The drive is a 20T:20T, 1:1 parallel-axis timing belt. This coupon tests a
deliberate narrow rear channel, not subtraction at a set of moving poses.
Coordinate convention: X lateral, Y rearward, Z upward, millimetres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

from swarm_belt_knee_coupon import (ROOT, box, cyl_x, compound, add, overlap,
                                     gap, bbox, volume)

SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-belt-knee-500-v2-20261004"
KNEE = (60.0, 18.0, 155.0)
SPAN = 31.5
DRIVE = (60.0, 2.0, KNEE[2] + math.sqrt(SPAN**2 - 16**2))
MOTOR_CENTER = (45.0, DRIVE[1], DRIVE[2] + 12.2)
MOTOR_SIZE = (35.0, 24.72, 45.22)
BELT_X0, BELT_X1 = 63.0, 78.0
PITCH_RADIUS = 60 / (2 * math.pi)
ANGLES = np.arange(0.0, 120.0001, 2.5)
DISTANCE_CHECK_ANGLES = (0, 5, 15, 30, 45, 60, 75, 90, 105, 115, 120)
VIEW_ANGLES = (0, 30, 60, 90, 120)


def turn(shape, deg):
    return bd.Pos(*KNEE) * bd.Rot(float(deg), 0, 0) * \
           bd.Pos(*(-v for v in KNEE)) * shape


def angled_capsule(radius, x0, x1):
    dy, dz = DRIVE[1]-KNEE[1], DRIVE[2]-KNEE[2]
    angle = math.degrees(math.atan2(-dy, dz))
    bar = bd.Pos((x0+x1)/2, (KNEE[1]+DRIVE[1])/2,
                 (KNEE[2]+DRIVE[2])/2) * bd.Rot(angle, 0, 0) * \
          bd.Box(x1-x0, 2*radius, SPAN)
    return cyl_x(radius, x0, x1, KNEE[1], KNEE[2]).fuse(
        cyl_x(radius, x0, x1, DRIVE[1], DRIVE[2]), bar)


def rod(x, y0, z0, y1, z1, radius):
    direction = bd.Vector(0, y1-y0, z1-z0)
    return bd.Solid.make_cylinder(radius, direction.length,
                                  bd.Plane(origin=(x, y0, z0), z_dir=direction))


def posterior_channel():
    # One designed, linearly tapered 13 mm wide opening. It follows the rear
    # throat rather than interpolating or subtracting any of the pose samples.
    outline = bd.Wire.make_polygon([
        (47.5, 18.0, 121.0), (47.5, 55.0, 121.0),
        (47.5, 55.0, 177.0), (47.5, 30.0, 177.0),
    ], close=True)
    return bd.Solid.extrude(bd.Face(outline), (13.0, 0, 0))


def build():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    OUT.mkdir(parents=True, exist_ok=True)
    p = {part.label: part for part in bd.import_step(SOURCE).children}
    thigh_orig = [p[f"R_thigh_{side}"] for side in ("front", "back")]
    cuff_orig = p["R_knee_cuff"]
    shin_orig = [p[f"R_shin_{side}"] for side in ("front", "back")]
    foot_orig = compound([p[f"R_{label}"] for label in
                          ("foot_dorsal", "toe_1", "toe_2", "toe_3", "sole")])
    original_fixed = compound([*thigh_orig, cuff_orig])
    original_moving = compound([*shin_orig, foot_orig])
    # The cosmetic source cuff passes within 0.23 mm of the source calf at
    # standing. A 2.07 mm planar top trim opens a working seam in this coupon.
    shin_top_cut = box((60, 5, 58.25), (200, 200, 116.5))  # top Z=116.5
    shin_trimmed = [part & shin_top_cut for part in shin_orig]
    moving_skin = compound([*shin_trimmed, foot_orig])

    motor = box(MOTOR_CENTER, MOTOR_SIZE)
    motor_keepout = box(MOTOR_CENTER, (37.0, 26.72, 47.22))
    drive_pulley = cyl_x(9.0, BELT_X0, BELT_X1, DRIVE[1], DRIVE[2])
    output_pulley = cyl_x(9.0, BELT_X0, BELT_X1, KNEE[1], KNEE[2])
    belt = angled_capsule(PITCH_RADIUS+1.5, BELT_X0, BELT_X1) - \
           angled_capsule(PITCH_RADIUS-1.0, BELT_X0-.1, BELT_X1+.1)
    belt_keepout = angled_capsule(12.5, 61.0, 80.0)
    # An opposed flange arrangement leaves the motor face clear. Spline hub,
    # GT tooth forms and actual retention still need drawings/physical tests.
    drive_flange = cyl_x(11.5, 78.0, 79.5, DRIVE[1], DRIVE[2])
    output_flange = cyl_x(11.5, 61.5, 63.0, KNEE[1], KNEE[2])
    channel = posterior_channel()
    cuff_opening = box((54.0, 25.0, 123.0), (13.0, 48.0, 24.0))  # y >= 1
    shaft_clearance = cyl_x(4.6, 29.0, 87.0, KNEE[1], KNEE[2])
    bearing_clearance = compound([
        cyl_x(8.3, x, x+7, KNEE[1], KNEE[2]) for x in (31.0, 79.0)
    ])
    thigh_fixed = [part - motor_keepout - belt_keepout - channel -
                   shaft_clearance - bearing_clearance for part in thigh_orig]
    cuff_fixed = cuff_orig - cuff_opening

    shaft = cyl_x(4.0, 30.0, 86.0, KNEE[1], KNEE[2])
    bearings = compound([
        cyl_x(8, x, x+5, KNEE[1], KNEE[2]) -
        cyl_x(4.1, x-.2, x+5.2, KNEE[1], KNEE[2]) for x in (32.0, 80.0)
    ])
    seats = compound([
        cyl_x(10, x, x+6, KNEE[1], KNEE[2]) -
        cyl_x(8.1, x-.2, x+6.2, KNEE[1], KNEE[2]) for x in (31.5, 79.5)
    ])
    supports = compound([
        box((x, 23.0, 170.0), (5.0, 10.0, 20.0)) for x in (34.5, 82.5)
    ])
    # Two medial rods, a rear skin mount pad, and two keyed shaft collars.
    # The pad's intersection with the source rear shin is measured below.
    rods = [rod(x, 4.0, 108.0, 18.0, 128.0, 2.5) for x in (52.0, 56.0)]
    rods += [rod(x, 18.0, 128.0, 18.0, 155.0, 2.5) for x in (52.0, 56.0)]
    pad = box((54.0, 18.0, 108.0), (13.0, 25.0, 4.0))
    collars = [cyl_x(5, x, x+4, KNEE[1], KNEE[2]) for x in (50.0, 54.0)]
    yoke = compound([*rods, pad, *collars])

    return locals()


def run():
    g = build()
    p = g["p"]
    original_fixed, original_moving = g["original_fixed"], g["original_moving"]
    thigh_orig, cuff_orig = g["thigh_orig"], g["cuff_orig"]
    thigh_fixed, cuff_fixed = g["thigh_fixed"], g["cuff_fixed"]
    motor, belt, drive_pulley = g["motor"], g["belt"], g["drive_pulley"]
    output_pulley, shaft, yoke = g["output_pulley"], g["shaft"], g["yoke"]
    bearings, seats, supports = g["bearings"], g["seats"], g["supports"]
    fixed_skin = compound([*thigh_fixed, cuff_fixed])
    moving_skin = g["moving_skin"]
    moving_hard = compound([moving_skin, yoke])
    fixed_hard = compound([fixed_skin, motor, belt, drive_pulley, seats, supports])
    # Bearing rings are coaxial with the rotating assembly. Their distance to
    # skin/yoke is invariant under this one-axis motion; measure it once.
    bearing_noncontact_gap = moving_hard.distance_to(bearings)
    rows = []
    for angle in ANGLES:
        angle = float(angle)
        skin = turn(moving_skin, angle)
        link = turn(yoke, angle)
        moved_hard = compound([skin, link])
        # Exact BRep distance is intentionally spot checked. The enclosure
        # review already rejects this candidate; a dense swept proof is not
        # warranted for an unapproved exterior.
        distance = (moved_hard.distance_to(fixed_hard)
                    if angle in DISTANCE_CHECK_ANGLES else None)
        pair_names = ("skin_fixed_shell_overlap_mm3", "skin_motor_overlap_mm3",
                      "skin_belt_overlap_mm3", "yoke_fixed_shell_overlap_mm3",
                      "yoke_motor_overlap_mm3", "yoke_belt_overlap_mm3",
                      "yoke_bearing_seat_overlap_mm3", "yoke_support_overlap_mm3")
        pairs = {name: 0.0 for name in pair_names}
        if distance is None or distance < 1e-6:
            pairs = {
                "skin_fixed_shell_overlap_mm3": overlap(skin, fixed_skin),
                "skin_motor_overlap_mm3": overlap(skin, motor),
                "skin_belt_overlap_mm3": overlap(skin, belt),
                "yoke_fixed_shell_overlap_mm3": overlap(link, fixed_skin),
                "yoke_motor_overlap_mm3": overlap(link, motor),
                "yoke_belt_overlap_mm3": overlap(link, belt),
                "yoke_bearing_seat_overlap_mm3": overlap(link, seats),
                "yoke_support_overlap_mm3": overlap(link, supports),
            }
        total = sum(pairs.values())
        row = {"flex_deg": angle,
               **{k: round(v, 4) for k, v in pairs.items()},
               "reported_noncontact_overlap_sum_mm3": round(total, 4),
               "minimum_noncontact_hard_gap_mm":
                   round(distance, 4) if distance is not None else None}
        rows.append(row)
        if round(angle) in VIEW_ANGLES and abs(angle-round(angle)) < 1e-8:
            scene = trimesh.Scene()
            for name, shape, color in (
                ("modified_exact_source_thigh", compound(thigh_fixed), (228, 220, 204, 255)),
                ("opened_source_cuff", cuff_fixed, (97, 94, 83, 255)),
                ("original_source_shin_and_foot", skin, (237, 231, 215, 255)),
                ("rear_link_and_shin_pad", link, (199, 116, 68, 255)),
                ("independent_knee_shaft", turn(shaft, angle), (84, 90, 91, 255)),
                ("output_20T", turn(output_pulley, angle), (215, 155, 68, 255)),
                ("motor_ST3250", motor, (68, 107, 135, 200)),
                ("drive_20T", drive_pulley, (215, 155, 68, 255)),
                ("Gates_123_3MGT_15", belt, (48, 55, 56, 255)),
                ("bearings", bearings, (126, 128, 122, 255)),
                ("bearing_seats", seats, (93, 99, 98, 255)),
                ("support_ribs", supports, (111, 118, 114, 255)),
            ):
                add(scene, name, shape, color)
            scene.export(OUT / f"knee-{int(angle):03d}.glb")

    source_scene = trimesh.Scene()
    for name, shape, color in (
        ("exact_source_thigh", compound(thigh_orig), (228, 220, 204, 255)),
        ("exact_source_knee_cuff", cuff_orig, (97, 94, 83, 255)),
        ("exact_source_shin_and_foot", original_moving, (237, 231, 215, 255)),
    ):
        add(source_scene, name, shape, color)
    source_scene.export(OUT / "source-standing.glb")

    before_thigh = sum(volume(v) for v in thigh_orig)
    after_thigh = sum(volume(v) for v in thigh_fixed)
    shell_components = [len(v.solids()) for v in (*thigh_fixed, cuff_fixed)]
    moving_bounds = moving_hard.bounding_box()
    max_radius = max(math.hypot(y-KNEE[1], z-KNEE[2])
                     for y in (moving_bounds.min.Y, moving_bounds.max.Y)
                     for z in (moving_bounds.min.Z, moving_bounds.max.Z))
    min_sampled_gap = min(bearing_noncontact_gap,
                          *(r["minimum_noncontact_hard_gap_mm"] for r in rows
                            if r["minimum_noncontact_hard_gap_mm"] is not None))
    intersample_displacement = 2*max_radius*math.sin(math.radians(1.25/2))
    report = {
        "stage": "500 mm internal 1:1 belt knee; rear-slot experiment",
        "source_step": str(SOURCE.relative_to(ROOT)),
        "verdict": "UNSOLVED EXTERIOR AND UNSWEPT CLEARANCE: 49 poses have no noncontact volume overlap and 11 exact gap spot checks are positive, but no positive intersample bound was proven. The open rear throat, exposed hardware and separated look at 120 degrees fail the source-shell appearance goal.",
        "coordinate_system": "X lateral, Y rearward, Z up; mm",
        "knee_axis_xyz_mm": KNEE, "motor_axis_xyz_mm": DRIVE,
        "center_distance_mm": SPAN,
        "Gates_belt": "123-3MGT-15, 41 teeth, 3 mm pitch, 15 mm width; 20T:20T = 1:1",
        "Gates_source": "https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/korea/power-transmission-2015-products-catalogue-en.pdf",
        "motor": "Feetech ST-3250-C001; 35 X, 24.72 Y, 45.22 Z mm case; 25T OD5.9 spline",
        "motor_source": "https://www.feetechrc.com/en/562636.html",
        "motor_center_xyz_mm": MOTOR_CENTER,
        "motor_output_offset_note": "12.2 mm below box centre is a placement assumption pending drawing/adapter verification",
        "shell_modification": {
            "method": "Static motor and belt keepouts plus one linearly tapered 13 mm posterior thigh channel and 13 mm posterior cuff slit; no pose-dependent subtraction",
            "original_thigh_skin_volume_mm3": round(before_thigh, 3),
            "modified_thigh_skin_volume_mm3": round(after_thigh, 3),
            "thigh_skin_removed_mm3": round(before_thigh-after_thigh, 3),
            "original_cuff_volume_mm3": round(volume(cuff_orig), 3),
            "modified_cuff_volume_mm3": round(volume(cuff_fixed), 3),
            "individual_shell_solid_counts": shell_components,
        },
        "component_envelopes_xyz_mm": {
            "motor": bbox(motor), "belt": bbox(belt), "modified_fixed_shell": bbox(fixed_skin),
            "moving_yoke": bbox(yoke), "bearing_seats": bbox(seats),
        },
        "static_geometry": {
            "motor_vs_exact_original_skin_mm3": round(overlap(motor, original_fixed), 3),
            "belt_vs_exact_original_skin_mm3": round(overlap(belt, original_fixed), 3),
            "motor_vs_modified_skin_mm3": round(overlap(motor, fixed_skin), 4),
            "belt_vs_modified_skin_mm3": round(overlap(belt, fixed_skin), 4),
            "motor_to_modified_skin_min_gap_mm": round(gap(motor, fixed_skin), 4),
            "belt_to_modified_skin_min_gap_mm": round(gap(belt, fixed_skin), 4),
            "shaft_vs_modified_skin_mm3": round(overlap(shaft, fixed_skin), 4),
            "bearings_vs_modified_skin_mm3": round(overlap(bearings, fixed_skin), 4),
            "support_ribs_vs_modified_skin_mm3": round(overlap(supports, fixed_skin), 3),
            "bearing_seats_vs_support_ribs_mm3": round(overlap(seats, supports), 3),
            "moving_skin_and_yoke_to_bearings_axis_invariant_gap_mm": round(bearing_noncontact_gap, 4),
            "yoke_mount_pad_vs_original_rear_shin_mm3": round(overlap(g["pad"], p["R_shin_back"]), 3),
            "yoke_pad_vs_rods_mm3": round(overlap(g["pad"], compound(g["rods"])), 3),
            "yoke_rods_vs_keyed_collars_mm3": round(overlap(compound(g["rods"]), compound(g["collars"])), 3),
            "keyed_collars_vs_output_shaft_mm3": round(overlap(compound(g["collars"]), shaft), 3),
        },
        "outside_original_thigh_outer_loft_mm3": {
            "motor": 0.0, "belt": 0.0, "drive_pulley": 0.0,
            "output_pulley": 0.0, "bearings": 127.57,
            "bearing_seats": 236.64, "bearing_supports": 155.67,
            "output_shaft": 63.53,
            "note": "Separate exact-solid exterior screen against source thigh outer loft; projected components are visible at the knee, not an enclosed package.",
        },
        "motion": {
            "range_deg": [0, 120], "step_deg": 2.5,
            "poses": rows,
            "minimum_spot_checked_noncontact_gap_mm": min_sampled_gap,
            "distance_spot_check_angles_deg": DISTANCE_CHECK_ANGLES,
            "colliding_pose_count": sum(r["reported_noncontact_overlap_sum_mm3"] > .001 for r in rows),
            "maximum_moving_radial_bound_from_axis_mm": round(max_radius, 3),
            "nearest_sample_angle_error_deg": 1.25,
            "maximum_unsampled_surface_motion_mm": round(intersample_displacement, 4),
            "conservative_intersample_gap_lower_bound_mm": round(min_sampled_gap-intersample_displacement, 4),
            "continuous_positive_gap_proven": False,
            "gap_scope_note": "Exact distance was evaluated at the 11 listed angles; the other 38 rows are collision-volume checks only. Shaft-to-bearing, shaft-to-collar and pulley-to-belt mating geometry is excluded from noncontact distance.",
            "intentional_mating_interfaces_excluded": [
                "Output 20T pulley teeth to belt pitch envelope",
                "Independent 8 mm output shaft to two 688-class bearing bores and keyed collars",
                "Drive 20T pulley teeth to belt pitch envelope",
            ],
            "bearing_screen_note": "The fixed bearing rings are rotationally symmetric about the same X axis. Their standing distance to moving skin/yoke is unchanged at every knee angle; output shaft-to-bore contact is a separate mating interface.",
        },
        "aesthetic_comparison": "Exact 500 mm source standing leg is a viewer toggle. Front and standing side preserve much of the source silhouette, but the knee axis is 32.1 mm above the original cuff centre, the calf top is trimmed 2.07 mm, and a 56 mm tall open rear channel exposes orange yoke hardware. At 120 degrees the shin reads as disconnected/open-mouthed. The required overlapping fairings and flexible rear closure are unbuilt, so exterior appearance is not approved.",
        "unverified": [
            "The source-size servo case is an envelope; precise 25T spline adapter, pulley tooth/width/flange drawing, tension method and connector egress are absent.",
            "Bearing seats and rear shin pad show a geometric load path but fasteners, bearing retention, wall strength and fatigue are unverified.",
            "Rear slot needs a flexible covering that clears 0–120 degrees and preserves appearance; none is modeled.",
            "No hip, opposite leg, torso, ground, impact, torque/thermal or gait replay validation; 1:1 is a packaging option, not selected gearing.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(
        Path(__file__).with_name("swarm_belt_knee_500_v2_viewer.html").read_text())
    for name, shape in (
        ("modified-thigh", compound(thigh_fixed)), ("modified-knee-cuff", cuff_fixed),
        ("source-shin-and-foot", moving_skin), ("ST3250", motor),
        ("Gates-belt-envelope", belt), ("drive-20T", drive_pulley),
        ("output-20T", output_pulley), ("output-shaft", shaft),
        ("bearings", bearings), ("bearing-seats", seats),
        ("bearing-supports", supports), ("moving-yoke", yoke),
    ):
        bd.export_step(compound([shape]), OUT / f"{name}.step")
    bd.export_step(compound([fixed_skin, motor, drive_pulley, belt,
                             bearings, seats, supports]),
                   OUT / "fixed-assembly.step")
    bd.export_step(compound([moving_skin, yoke, shaft, output_pulley]),
                   OUT / "moving-standing-assembly.step")
    print(json.dumps({
        "artifact": str(OUT),
        "shell_solid_counts": shell_components,
        "static": report["static_geometry"],
        "colliding_poses": report["motion"]["colliding_pose_count"],
        "min_gap": report["motion"]["minimum_spot_checked_noncontact_gap_mm"],
        "max_overlap": max(r["reported_noncontact_overlap_sum_mm3"] for r in rows),
    }, indent=2), flush=True)


if __name__ == "__main__":
    run()
