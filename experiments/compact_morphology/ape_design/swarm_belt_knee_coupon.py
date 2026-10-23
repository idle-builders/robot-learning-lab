"""Gates 123-3MGT-15 / 20T:20T knee packaging coupon on the exact 350 mm sketch shell.

Run from the repository root with .venv/bin/python.  This is an occupancy and
motion experiment, not a printable joint or a claimed continuous-clearance fit.
X is lateral, Y rearward, Z upward; all lengths are millimetres.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/350mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/swarm-belt-knee-20261004"
SOURCE_500 = ROOT / "artifacts/sketch-ape-shell/exports/500mm/ape_shell_assembly.step"
AXIS = (42.0, 3.0, 86.0)  # exact source cosmetic cuff center
DRIVE = (42.0, 3.0, 117.5)
PITCH = 3.0
TEETH = 20
BELT_PITCH_LENGTH = 123.0
BELT_WIDTH = 15.0
CENTER = (BELT_PITCH_LENGTH - TEETH * PITCH) / 2.0
PITCH_RADIUS = TEETH * PITCH / (2.0 * math.pi)
PULLEY_X0, PULLEY_X1 = 58.0, 73.0
SAMPLE_ANGLES = np.arange(0, 120.0001, 2.5)
VIEW_ANGLES = (0, 30, 60, 90, 120)


def box(center, size):
    return bd.Pos(*center) * bd.Box(*size)


def cyl_x(radius, x0, x1, y, z):
    return bd.Solid.make_cylinder(radius, x1 - x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def capsule(radius, x0, x1):
    """Filled tangent envelope of two equal pitch circles in a YZ plane."""
    lower = cyl_x(radius, x0, x1, AXIS[1], AXIS[2])
    upper = cyl_x(radius, x0, x1, DRIVE[1], DRIVE[2])
    spine = box(((x0 + x1) / 2, AXIS[1], (AXIS[2] + DRIVE[2]) / 2),
                (x1 - x0, radius * 2, CENTER))
    return lower.fuse(upper, spine)


def turn(shape, degrees):
    return bd.Pos(*AXIS) * bd.Rot(float(degrees), 0, 0) * \
           bd.Pos(*(-v for v in AXIS)) * shape


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape is not None else 0.0


def overlap(left, right):
    # BRep compound Boolean can ignore children. Test every source solid.
    aa = [(s, s.bounding_box(optimal=False)) for s in left.solids()]
    bb = [(s, s.bounding_box(optimal=False)) for s in right.solids()]
    return sum(volume(a & b) for a, ab in aa for b, bbb in bb
               if aabb_gap(ab, bbb) <= 1e-8)


def gap(left, right):
    best = float("inf")
    aa = [(s, s.bounding_box(optimal=False)) for s in left.solids()]
    bb = [(s, s.bounding_box(optimal=False)) for s in right.solids()]
    for a, ab in aa:
        for b, bbb in bb:
            if aabb_gap(ab, bbb) >= best:
                continue
            best = min(best, a.distance_to(b))
            if best <= 1e-7:
                return 0.0
    return best


def aabb_gap(aa, bb):
    d = [max(bb.min.X-aa.max.X, aa.min.X-bb.max.X, 0),
         max(bb.min.Y-aa.max.Y, aa.min.Y-bb.max.Y, 0),
         max(bb.min.Z-aa.max.Z, aa.min.Z-bb.max.Z, 0)]
    return math.sqrt(sum(x*x for x in d))


def any_collision(left, right, threshold=1e-5):
    aa = [(s, s.bounding_box(optimal=False)) for s in left.solids()]
    bb = [(s, s.bounding_box(optimal=False)) for s in right.solids()]
    for a, ab in aa:
        for b, bbb in bb:
            if aabb_gap(ab, bbb) < 1e-8 and volume(a & b) > threshold:
                return True
    return False


def compound(parts):
    return bd.Compound(children=[s for part in parts for s in part.solids()])


def add(scene, name, shape, rgba):
    if volume(shape) < 1e-7:
        return
    for index, solid in enumerate(shape.solids()):
        verts, faces = solid.tessellate(.34, .30)
        mesh = trimesh.Trimesh(
            vertices=np.asarray([(v.X, v.Y, v.Z) for v in verts]),
            faces=np.asarray(faces), process=False)
        mesh.visual.vertex_colors = rgba
        node = f"{name}_{index}"
        scene.add_geometry(mesh, geom_name=node, node_name=node)


def bbox(shape):
    bb = shape.bounding_box()
    return [[round(v, 2) for v in (bb.min.X, bb.min.Y, bb.min.Z)],
            [round(v, 2) for v in (bb.max.X, bb.max.Y, bb.max.Z)]]


def capsule_at(radius, x0, x1, knee, drive):
    lower = cyl_x(radius, x0, x1, knee[1], knee[2])
    upper = cyl_x(radius, x0, x1, drive[1], drive[2])
    spine = box(((x0+x1)/2, knee[1], (knee[2]+drive[2])/2),
                (x1-x0, 2*radius, CENTER))
    return lower.fuse(upper, spine)


def larger_shell_screen():
    """Unchanged motor and belt size in the independent 500 mm source skin."""
    if not SOURCE_500.exists():
        return {"available": False, "reason": "500 mm source STEP is missing"}
    p = {part.label: part for part in bd.import_step(SOURCE_500).children}
    thigh = compound([p[f"R_thigh_{side}"] for side in ("front", "back")])
    cuff = p["R_knee_cuff"]
    shin = compound([p[f"R_shin_{side}"] for side in ("front", "back")])
    scale = 500 / 350
    knee = tuple(v * scale for v in AXIS)
    drive = (knee[0], knee[1], knee[2] + CENTER)
    servo = box((knee[0]-3, knee[1], drive[2] + 12.2), (35, 24.72, 45.22))
    x0, x1 = knee[0] + 16, knee[0] + 31
    belt_plane = capsule_at(PITCH_RADIUS + 1.5, x0, x1, knee, drive) - \
                 capsule_at(PITCH_RADIUS - 1.0, x0-.1, x1+.1, knee, drive)
    moving = []
    for deg in (0, 60, 120):
        posed = bd.Pos(*knee) * bd.Rot(deg, 0, 0) * bd.Pos(*(-v for v in knee)) * shin
        moving.append({"flex_deg": deg,
                       "untrimmed_shin_vs_exact_fixed_shell_mm3":
                       round(overlap(posed, compound([thigh, cuff])), 3),
                       "untrimmed_shin_vs_servo_mm3": round(overlap(posed, servo), 3),
                       "untrimmed_shin_vs_belt_mm3": round(overlap(posed, belt_plane), 3)})
    return {
        "available": True, "source": str(SOURCE_500.relative_to(ROOT)),
        "unchanged_motor_and_belt": True,
        "new_knee_axis_xyz_mm": [round(v, 3) for v in knee],
        "motor_vs_exact_thigh_skin_mm3": round(overlap(servo, thigh), 3),
        "belt_vs_exact_thigh_skin_mm3": round(overlap(belt_plane, thigh), 3),
        "belt_outboard_projection_past_source_shell_mm":
            round(belt_plane.bounding_box().max.X -
                  max(thigh.bounding_box().max.X, cuff.bounding_box().max.X), 3),
        "selected_source_shin_poses": moving,
        "scope": "Quick unchanged-hardware static and three-pose envelope screen only; no 500 mm engineering shell or positive-gap claim.",
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Build the exact 350 mm source shell first: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    p = {part.label: part for part in bd.import_step(SOURCE).children}
    source_thigh = compound([p[f"R_thigh_{side}"] for side in ("front", "back")])
    source_shin = compound([p[f"R_shin_{side}"] for side in ("front", "back")])
    source_cuff = p["R_knee_cuff"]
    source_foot = compound([p[f"R_{label}"] for label in
                            ("foot_dorsal", "toe_1", "toe_2", "toe_3", "sole")])
    reference_scene = trimesh.Scene()
    for name, shape, color in (
        ("unmodified_source_thigh", source_thigh, (224, 216, 199, 255)),
        ("unmodified_source_knee_cuff", source_cuff, (94, 91, 81, 255)),
        ("unmodified_source_shin", source_shin, (234, 229, 214, 255)),
        ("unmodified_source_foot", source_foot, (219, 211, 192, 255)),
    ):
        add(reference_scene, name, shape, color)
    reference_scene.export(OUT / "source-standing.glb")

    # Servo is upper-thigh fixed. 45.22 is vertical, 35 is lateral along the
    # output shaft, 24.72 is fore-aft.  Horn is 12.2 mm below box center as a
    # hypothesis; only its outer size and 25T/OD5.9 spline are source-backed.
    motor = box((39, 3, 129.7), (35.0, 24.72, 45.22))
    motor_service = box((39, 3, 129.7), (37.0, 26.72, 47.22))
    # Pulley pitch circles give exactly 2*C + 20*3 = 123 mm. Smooth pulley
    # cylinders and a 1.5 mm radial belt envelope omit tooth/flange detail.
    drive_pulley = cyl_x(9.0, PULLEY_X0, PULLEY_X1, DRIVE[1], DRIVE[2])
    knee_pulley = cyl_x(9.0, PULLEY_X0, PULLEY_X1, AXIS[1], AXIS[2])
    belt = capsule(PITCH_RADIUS + 1.5, PULLEY_X0, PULLEY_X1) - \
           capsule(PITCH_RADIUS - 1.0, PULLEY_X0 - .1, PULLEY_X1 + .1)
    flanges = compound([
        cyl_x(11.5, x0, x1, y, z)
        for y, z in ((AXIS[1], AXIS[2]), (DRIVE[1], DRIVE[2]))
        for x0, x1 in ((56.5, 58.0), (73.0, 74.5))
    ])
    # Static enclosed-belt proposal: 1.5 mm radial free space to flange
    # envelope. Its outboard x extent is visibly outside the source limb.
    belt_cover = capsule(16.0, 54.3, 78.7) - capsule(13.0, 55.5, 77.5)

    # Two separate bearing stations support the output axle. The output shaft
    # and pulley are moving, not carried by the servo's spline as a knee axle.
    shaft = cyl_x(4.0, 17.0, 85.0, AXIS[1], AXIS[2])
    bearings = compound([cyl_x(8.0, a, a + 5, AXIS[1], AXIS[2]) -
                         cyl_x(4.1, a - .2, a + 5.2, AXIS[1], AXIS[2])
                         for a in (18.0, 79.5)])
    bearing_seats = compound([cyl_x(11.0, a, a + 6, AXIS[1], AXIS[2]) -
                              cyl_x(8.15, a - .2, a + 6.2, AXIS[1], AXIS[2])
                              for a in (17.5, 79.0)])
    supports = compound([box((x, 3, 101), (5.0, 9.0, 20.0))
                         for x in (20.5, 82.0)])
    # Only the top seven millimetres of the source calf are sacrificed.
    shin_cut = box((42, 3, 38), (160, 160, 76))  # top Z=76
    shin_halves = [p[f"R_shin_{side}"] & shin_cut for side in ("front", "back")]
    shin_skin = compound(shin_halves)
    yoke = compound([
        box((x, 3, 79.5), (4.0, 12.0, 17.0)) for x in (25.5, 79.0)
    ] + [box((x, 3, 74.5), (10.0, 15.0, 5.0)) for x in (27, 74)])
    moving = compound([shin_skin, yoke, shaft, knee_pulley, source_foot])
    fixed = compound([source_thigh, source_cuff, motor, drive_pulley, belt,
                      flanges, belt_cover, bearings, bearing_seats, supports])

    # Original shell is the appearance target, not a pre-cleared mechanism.
    # Screen the unaltered geometry. No pose-based Boolean relief is applied.
    static_fit = {
        "servo_case_vs_exact_thigh_skin_mm3": round(overlap(motor, source_thigh), 3),
        "servo_1mm_service_box_vs_exact_thigh_skin_mm3": round(overlap(motor_service, source_thigh), 3),
        "drive_pulley_vs_exact_thigh_skin_mm3": round(overlap(drive_pulley, source_thigh), 3),
        "belt_vs_exact_thigh_skin_mm3": round(overlap(belt, source_thigh), 3),
        "cover_vs_exact_thigh_skin_mm3": round(overlap(belt_cover, source_thigh), 3),
        "cover_vs_exact_knee_cuff_mm3": round(overlap(belt_cover, source_cuff), 3),
        "bearing_seats_vs_source_cuff_mm3": round(overlap(bearing_seats, source_cuff), 3),
    }

    rows = []
    stationary_noncontact = compound([source_thigh, source_cuff, motor,
                                      belt, drive_pulley, belt_cover,
                                      bearing_seats, supports])
    for deg in SAMPLE_ANGLES:
        deg = float(deg)
        moving_skin = turn(compound([shin_skin, source_foot]), deg)
        moving_yoke = turn(yoke, deg)
        moving_noncontact = compound([moving_skin, moving_yoke])
        # Stationary pulley/belt have intended driving contact; the moving
        # pulley and output bearing also intentionally share the shaft axis.
        # Interference report isolates components that require free motion.
        skin_vs_source = overlap(moving_skin, compound([source_thigh, source_cuff]))
        skin_vs_cover = overlap(moving_skin, belt_cover)
        skin_vs_motor = overlap(moving_skin, motor)
        skin_vs_belt = overlap(moving_skin, belt)
        yoke_vs_thigh = overlap(moving_yoke, source_thigh)
        yoke_vs_cover = overlap(moving_yoke, belt_cover)
        measured_overlap = (skin_vs_source + skin_vs_cover + skin_vs_motor +
                            skin_vs_belt + yoke_vs_thigh + yoke_vs_cover)
        collides = measured_overlap > 1e-5 or any_collision(moving_noncontact, stationary_noncontact)
        hard_gap = 0.0 if collides else gap(moving_noncontact, stationary_noncontact)
        row = {
            "flex_deg": deg,
            "moving_skin_vs_exact_fixed_shell_mm3": round(skin_vs_source, 3),
            "moving_skin_vs_belt_cover_mm3": round(skin_vs_cover, 3),
            "moving_skin_vs_servo_mm3": round(skin_vs_motor, 3),
            "moving_skin_vs_belt_mm3": round(skin_vs_belt, 3),
            "yoke_vs_exact_thigh_mm3": round(yoke_vs_thigh, 3),
            "yoke_vs_belt_cover_mm3": round(yoke_vs_cover, 3),
            "reported_pair_overlap_sum_mm3": round(measured_overlap, 3),
            "all_noncontact_hard_geometry_collision_detected": collides,
            "minimum_noncontact_hard_geometry_gap_mm": round(hard_gap, 4),
        }
        rows.append(row)
        if round(deg) in VIEW_ANGLES and abs(deg - round(deg)) < 1e-8:
            scene = trimesh.Scene()
            for name, shape, color in (
                ("exact_source_thigh", source_thigh, (224, 216, 199, 255)),
                ("exact_source_cuff", source_cuff, (94, 91, 81, 180)),
                ("source_shin_trimmed", turn(shin_skin, deg), (234, 229, 214, 255)),
                ("source_foot", turn(source_foot, deg), (219, 211, 192, 255)),
                ("independent_knee_yoke", turn(yoke, deg), (205, 115, 66, 255)),
                ("supported_output_axle", turn(shaft, deg), (83, 86, 86, 255)),
                ("knee_20T_pitch_envelope", turn(knee_pulley, deg), (211, 145, 57, 255)),
                ("ST3250_outer_box", motor, (69, 111, 141, 185)),
                ("motor_20T_pitch_envelope", drive_pulley, (211, 145, 57, 255)),
                ("Gates_123_3MGT_15_smooth_envelope", belt, (47, 55, 56, 255)),
                ("belt_cover_candidate", belt_cover, (52, 84, 80, 105)),
                ("two_688_class_bearings", bearings, (122, 126, 121, 255)),
                ("bearing_seats", bearing_seats, (87, 91, 89, 255)),
                ("bearing_supports", supports, (94, 99, 96, 255)),
            ):
                add(scene, name, shape, color)
            scene.export(OUT / f"knee-{int(deg):03d}.glb")

    source_max_x = max(source_thigh.bounding_box().max.X,
                       source_cuff.bounding_box().max.X)
    report = {
        "stage": "1:1 Gates belt knee packaging coupon; unsolved",
        "source_shell": str(SOURCE.relative_to(ROOT)),
        "shell_is_exact": "Imported named STEP thigh, cuff, shin, and foot solids; only source shin is top-trimmed for a conceptual yoke.",
        "source_shell_warning": "The source is a cosmetic skin, with no mechanical joint or fastening features.",
        "coordinates": "X lateral, Y rearward, Z up; millimetres",
        "source_cuff_axis_xyz_mm": AXIS,
        "motor_axis_xyz_mm": DRIVE,
        "belt": {
            "manufacturer_part": "Gates PowerGrip GT3 123-3MGT-15 / 9400-5041",
            "manufacturer_catalog": "https://www.gates.com/content/dam/gates/home/knowledge-center/resource-library/catalogs/korea/power-transmission-2015-products-catalogue-en.pdf",
            "pitch_mm": PITCH, "teeth": 41, "width_mm": BELT_WIDTH,
            "pitch_length_mm": BELT_PITCH_LENGTH,
            "pulley_teeth_each": TEETH, "nominal_ratio": "1:1",
            "pitch_radius_mm": round(PITCH_RADIUS, 5),
            "nominal_center_distance_mm": CENTER,
            "formula": "L = 2 C + pi D for two equal pitch circles; D=20*3/pi; C=(123-60)/2=31.5",
            "cad_envelope": "15 mm wide smooth 1.0 mm inward/1.5 mm outward radial belt ring; teeth and tension not modeled",
        },
        "motor": {
            "model": "Feetech ST-3250-C001",
            "manufacturer_source": "https://www.feetechrc.com/en/562636.html",
            "official_case_dimensions_mm": [45.22, 24.72, 35.0],
            "modeled_xyz_mm": [35.0, 24.72, 45.22],
            "center_xyz_mm": [39, 3, 129.7],
            "output_spline": "25T, OD 5.9 mm (manufacturer); adapter unmodeled",
            "shaft_location": "12.2 mm below body-box center is a placement assumption; verify against manufacturer drawing",
            "rated_torque_at_12V_Nm": 1.569,
            "rated_torque_caveat": "Motor rating is not a proved knee requirement or thermal margin; 1:1 only screens packaging.",
        },
        "bearing_architecture": "Two 8 mm radius/4 mm bore/5 mm wide 688-class envelope rings at X=18..23 and 79.5..84.5; independent 8 mm diameter output shaft. Seat mounts and life are unverified.",
        "component_bounds_xyz_mm": {
            "source_thigh": bbox(source_thigh), "source_cuff": bbox(source_cuff),
            "servo": bbox(motor), "belt": bbox(belt), "cover": bbox(belt_cover),
            "supported_output_axle": bbox(shaft), "trimmed_shin": bbox(shin_skin),
        },
        "static_fit": static_fit,
        "appearance": {
            "exact_source_max_lateral_x_mm": round(source_max_x, 2),
            "belt_cover_max_lateral_x_mm": round(belt_cover.bounding_box().max.X, 2),
            "additional_lateral_projection_mm": round(belt_cover.bounding_box().max.X-source_max_x, 2),
            "assessment": "The broad exposed outboard belt pod breaks the sketch's compact rounded thigh/cuff seam. The candidate enclosure is not visually accepted.",
        },
        "motion": {
            "range_deg": [0, 120], "step_deg": 2.5,
            "held_out_inter_sample_angles": [float(v) for v in SAMPLE_ANGLES if float(v) % 5 != 0],
            "sampled_poses": rows,
            "max_reported_pair_overlap_sum_mm3": max(r["reported_pair_overlap_sum_mm3"] for r in rows),
            "minimum_gap_across_checked_poses_mm": min(r["minimum_noncontact_hard_geometry_gap_mm"] for r in rows),
            "colliding_pose_count": sum(r["all_noncontact_hard_geometry_collision_detected"] for r in rows),
            "positive_gap_verified": False,
            "continuous_clearance": "Rejected by sampled collisions. No moving skin was subtracted from fixed parts. The 2.5-degree grid includes intersample poses; unsampled angles have no positive-gap proof.",
            "excluded_mating_interfaces": [
                "Moving output pulley against stationary belt pitch envelope",
                "Output axle inside two bearing bores, and yoke-to-axle keying",
                "Motor drive pulley against belt pitch envelope",
            ],
        },
        "unchanged_hardware_500mm_shell_screen": larger_shell_screen(),
        "verdict": "UNSOLVED: 1:1 belt length and independent shaft lay out, but exact shell, drive, and moving limb clash; no positive gap or acceptable enclosed appearance.",
        "unresolved": [
            "The official-size servo and belt need a revised thigh cavity, mounting and service path.",
            "Output spline-to-pulley adapter, actual 3MGT tooth geometry, flanges, center tolerance and tensioning are unverified.",
            "Bearing retention, yoke/shaft keying, fasteners, cable routing and belt force/life are unverified.",
            "No hip, opposite leg, torso, ground, impact or thermal-duty screen; 1:1 is not selected as final gearing.",
        ],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("swarm_belt_knee_viewer.html").read_text())
    for label, shape in (
        ("exact-source-thigh", source_thigh), ("exact-source-knee-cuff", source_cuff),
        ("source-shin-trimmed", shin_skin), ("source-foot", source_foot),
        ("ST3250-outer-envelope", motor), ("belt-123-3MGT-15-envelope", belt),
        ("motor-20T-pulley-envelope", drive_pulley),
        ("knee-20T-pulley-envelope", knee_pulley),
        ("belt-cover-candidate", belt_cover), ("independent-output-shaft", shaft),
        ("output-bearings", bearings), ("bearing-seats", bearing_seats),
        ("bearing-supports", supports), ("moving-yoke", yoke),
    ):
        # Imported named STEP children need a fresh compound for OCC's writer.
        bd.export_step(compound([shape]), OUT / f"{label}.step")
    bd.export_step(fixed, OUT / "fixed-coupon-assembly.step")
    bd.export_step(moving, OUT / "moving-coupon-standing.step")
    print(json.dumps({
        "artifact": str(OUT), "static_fit": static_fit,
        "colliding_pose_count": report["motion"]["colliding_pose_count"],
        "max_reported_pair_overlap_sum_mm3": report["motion"]["max_reported_pair_overlap_sum_mm3"],
        "lateral_projection_mm": report["appearance"]["additional_lateral_projection_mm"],
        "verdict": report["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
