"""Compact belt-driven elbow coupon carrying the completed paw geometry.

This is a local CAD mechanism screen. It imports the paw STEP halves generated
by gripper_joint_coupon.py, then constructs the next proximal axis. The
earlier direct ST3215 candidate failed the enclosure screen. This branch
puts an XL330-sized motor above a separately supported elbow output shaft.
Physical fits, belt retention and thermal duty remain unmeasured.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

import sketch_shell as shell
from gripper_joint_coupon import build as build_paw, thumb


ROOT = Path(__file__).resolve().parents[3]
PAW_OUT = ROOT / "artifacts/ape-gripper-joint-coupon-20261003"
COMPACT_PIVOT = "--compact-pivot" in sys.argv
NESTED_COWL = "--nested-cowl" in sys.argv
OUT = ROOT / ("artifacts/ape-elbow-nested-cowl-20261003" if NESTED_COWL else
              "artifacts/ape-elbow-compact-pivot-20261003" if COMPACT_PIVOT else
              "artifacts/ape-elbow-pitch-iteration-20261003")
AXIS = (0, 96, 200)
MOTOR_CENTER = (-10, 96.5, 236.5)
MOTOR_AXIS = (-10, 112, 244)
CASE_SIZE = (20, 26, 34)
POSES = (0, 30, 60, 90, 115)


def volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    return sum(s.volume for s in shape) if isinstance(shape, bd.ShapeList) else (shape.volume if shape else 0)


def cylinder_y(radius: float, length: float, x: float, y: float, z: float) -> bd.Shape:
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(radius, length)


def flexed(shape: bd.Shape, degrees: float) -> bd.Shape:
    return bd.Pos(*AXIS) * bd.Rot(0, -degrees, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def belt_envelope(clearance: float = 0) -> bd.Shape:
    """Smooth 4 mm-wide 2GT belt solid, minus tooth detail, on pitch circles."""
    centers = ((-10, 244, 10.19), (0, 200, 20.37))
    outer = unary_union([Point(x, z).buffer(r + .69 + clearance, 32)
                         for x, z, r in centers]).convex_hull
    inner = unary_union([Point(x, z).buffer(r - .69 - clearance, 32)
                         for x, z, r in centers]).convex_hull
    y = 114.5 - clearance
    wire = lambda polygon: bd.Wire.make_polygon(
        [(x, y, z) for x, z in polygon.exterior.coords])
    return bd.extrude(bd.Face(wire(outer), [wire(inner)]),
                      amount=4 + 2 * clearance, dir=(0, 1, 0))


def side_profile(polygon, y: float, depth: float) -> bd.Shape:
    outline = bd.Wire.make_polygon([(x, y, z) for x, z in polygon.exterior.coords])
    return bd.extrude(bd.Face(outline), amount=depth, dir=(0, 1, 0))


def arc_cowl(inner_radius: float, outer_radius: float, low_deg: int,
             high_deg: int, y: float, depth: float) -> bd.Shape:
    """One rigid concentric cover sector in the elbow's x-z plane."""
    steps = 48
    arc = lambda radius, angle: (radius * math.sin(math.radians(angle)),
                                 200 + radius * math.cos(math.radians(angle)))
    angles = np.linspace(low_deg, high_deg, steps)
    outline = [arc(outer_radius, angle) for angle in angles]
    outline += [arc(inner_radius, angle) for angle in reversed(angles)]
    return side_profile(Polygon(outline), y, depth)


def mesh(shape: bd.Shape, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(.2, .2)
    result = trimesh.Trimesh(np.array([[v.X, v.Y, v.Z] for v in vertices]),
                             np.asarray(faces), process=False)
    result.merge_vertices()
    result.visual.vertex_colors = rgba
    return result


def add(scene: trimesh.Scene, name: str, shape: bd.Shape,
        rgba: tuple[int, int, int, int]) -> None:
    scene.add_geometry(mesh(shape, rgba), geom_name=name, node_name=name)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (PAW_OUT / "paw-forearm-inner-half.step").exists():
        build_paw()
    shell.build()
    original_upper = next(p.shape for p in shell.PARTS if p.name == "left_upper_arm_shell")
    paw_inner = bd.import_step(PAW_OUT / "paw-forearm-inner-half.step")
    paw_outer = bd.import_step(PAW_OUT / "paw-forearm-outer-half.step")
    # The inner bend side (positive x) ends at z=184 to clear deep flex.
    # The outside rises to z=200, covering more of the fixed pivot without
    # changing the saved gripper's motor, palm or lower cover.
    seam_cut = Polygon([(-100, 200), (-12, 200), (12, 184),
                        (100, 184), (100, 400), (-100, 400)])
    paw_top_cut = side_profile(seam_cut, -54, 300)
    # The forearm wraps around a white pivot capsule with a 2 mm radial
    # running allowance. A sphere stays at the same distance in all poses.
    pivot_relief = bd.Pos(*AXIS) * bd.Sphere(19 if COMPACT_PIVOT else 26)
    paw_inner = paw_inner - paw_top_cut - pivot_relief
    paw_outer = paw_outer - paw_top_cut - pivot_relief
    # The belt sits outside the outboard bearing but inside the paw cheek.
    # Its circular output wrap stays fixed in world coordinates as the paw
    # rotates, so a rotationally invariant relief is needed in this cover.
    paw_outer -= cylinder_y(31, 5.8, 0, 116.5, 200)
    # The big direct motor forced a 30 mm open notch. A smaller upper-arm
    # motor and a 2:1 belt instead leave only the compact output joint below.
    upper = original_upper - bd.Pos(0, 96, -42) * bd.Box(300, 300, 510)  # z <= 213
    bend_relief = bd.Pos(56, 96, 230) * bd.Box(100, 62, 36)  # x>=6, z=212..248
    upper -= bend_relief
    motor_case = bd.Pos(*MOTOR_CENTER) * bd.Box(*CASE_SIZE)
    motor_clearance = bd.Pos(*MOTOR_CENTER) * bd.Box(20.6, 26.6, 34.6)
    motor = motor_case.fuse(cylinder_y(3, 3, *MOTOR_AXIS))
    # Hollow fixed spherical pivot cover: the case is now wholly above it.
    pivot_outer_radius = 17 if COMPACT_PIVOT else 24
    pivot_cover = (bd.Pos(*AXIS) * bd.Sphere(pivot_outer_radius)) - (
        bd.Pos(*AXIS) * bd.Sphere(pivot_outer_radius - 2))
    pivot_cover &= bd.Pos(*AXIS) * bd.Box(60, 28, 60)
    pivot_cover -= cylinder_y(10.6, 36, *AXIS)
    upper_cuff = (bd.Pos(0, 96, 224) * bd.Cylinder(28, 24)) - (
        bd.Pos(0, 96, 224) * bd.Cylinder(21, 26))
    upper_cuff -= bend_relief
    fixed = upper.fuse(upper_cuff, pivot_cover) - motor_clearance
    # Two output supports close the load path into the upper arm. The yoke
    # rotates outside these seats; the 8 mm shaft is a separate moving part.
    bearing_seats = []
    bearing_envelopes = []
    for y in (80, 110):
        # igus 688: 8 mm ID / 16 mm OD / 5 mm width. The 7 mm printed
        # socket is a split-seat envelope; lips and assembly split are open.
        seat = cylinder_y(11, 7, 0, y, 200) - cylinder_y(8.2, 9, 0, y, 200)
        web = bd.Pos(0, y, 209) * bd.Box(14, 7, 18)
        bearing_seats.append(seat.fuse(web) - cylinder_y(8.2, 9, 0, y, 200))
        bearing_envelopes.append(cylinder_y(8, 5, 0, y, 200) -
                                 cylinder_y(4.05, 6, 0, y, 200))
    fixed = fixed.fuse(*bearing_seats)
    # Behind the skin these two webs tie the pivot shield and inner bearing
    # seat into the upper-arm cuff; separate floating solids would not be a
    # valid supported mechanism.
    fixed = fixed.fuse(bd.Pos(-20, 96, 213) * bd.Box(12, 8, 12),
                       bd.Pos(0, 76, 219) * bd.Box(10, 12, 18))
    # A thin fixed front wall sits just behind the swept forearm. It hides
    # the offset motor without restoring the bulky original upper-arm face.
    fixed = fixed.fuse(bd.Pos(2, 96, 233.5) * bd.Box(2, 50, 39))
    fixed -= cylinder_y(3.4, 6, *MOTOR_AXIS)
    fixed_stop_base = cylinder_y(2, 3, -14, 77.5, 200).fuse(
        bd.Pos(-12, 78.5, 200) * bd.Box(6, 3, 2))
    fixed_stops = (flexed(fixed_stop_base, -17), flexed(fixed_stop_base, 132))
    fixed = fixed.fuse(*fixed_stops)

    # Moving cheek and round 8 mm output shaft; the drive pulley lies
    # between the outer bearing and cheek. Detailed retainers are still open.
    bolt_pattern = [(0, 206), (6, 200), (0, 194), (-6, 200)]
    cheek_inner = (bd.Pos(0, 73, 185) * bd.Box(12, 5, 30)).fuse(
        cylinder_y(10, 2, 0, 74.5, 200))
    cheek_outer = (bd.Pos(0, 122, 185) * bd.Box(12, 5, 30)).fuse(
        cylinder_y(10, 2, 0, 120.5, 200))
    for x, z in bolt_pattern:
        cheek_inner -= cylinder_y(1.6, 8, x, 75, z)
        cheek_outer -= cylinder_y(1.6, 8, x, 121, z)
    cheek_inner -= cylinder_y(3.05, 8, 0, 75, 200)
    cheek_outer -= cylinder_y(4.05, 8, 0, 121, 200)
    horns = []
    for y in (74, 119):
        horn = cylinder_y(9, 3, 0, y, 200) - cylinder_y(4.05, 4, 0, y, 200)
        for x, z in bolt_pattern:
            horn -= cylinder_y(1.6, 4, x, y, z)
        horns.append(horn)
    moving_stop = cylinder_y(2, 3, -14, 75.5, 200).fuse(
        bd.Pos(-12, 74.5, 200) * bd.Box(6, 3, 2))
    moving_inner = paw_inner.fuse(cheek_inner, moving_stop)
    moving_outer = paw_outer.fuse(cheek_outer)
    # 16T cannot fasten around the XL330's Ø16 output face. A 32T driver
    # (20.37 mm pitch diameter) has room for the source 4×M2 / Ø12 pattern;
    # the 64T output is kept inside a separately cleared side pocket.
    motor_pulley = cylinder_y(10.19, 4, -10, 116.5, 244)
    for dx, dz in ((6, 0), (-6, 0), (0, 6), (0, -6)):
        motor_pulley -= cylinder_y(1.05, 6, -10 + dx, 116.5, 244 + dz)
    motor_pulley -= cylinder_y(1.1, 6, -10, 116.5, 244)
    output_pulley = cylinder_y(20.37, 4, 0, 116.5, 200) - cylinder_y(4.05, 6, 0, 116.5, 200)
    for dx, dz in ((14, 0), (-14, 0), (0, 14), (0, -14)):
        output_pulley -= cylinder_y(1.6, 6, dx, 116.5, 200 + dz)
    belt = belt_envelope()
    fixed -= cylinder_y(10.8, 5, -10, 116.5, 244)
    fixed -= cylinder_y(20.95, 5, 0, 116.5, 200)
    fixed -= belt_envelope(.45)
    # Overlapping rigid skins cover the belt from the outside. The upper
    # piece follows the slope of the arm; the round lower cheek turns with
    # the forearm. Their 1.5 mm axial gap avoids a rubbing seam.
    cap_outline = unary_union([Point(-10, 244).buffer(15, 48),
                               Point(0, 220).buffer(15, 48)]).convex_hull
    cap_outline = cap_outline.intersection(box(-40, 218, 35, 270))
    upper_belt_cap = side_profile(cap_outline, 123.5, 3)
    fixed = fixed.fuse(upper_belt_cap)
    # A tapered shell joins the upper-arm silhouette around the pivot.
    # The x=12 trim leaves the inside of the bend open; y=78..114 stays
    # inboard of the belt and output pulley. This is a rigid skin, not a
    # flexible gaiter. Its lower edge and visual seam still need review.
    cuff_outer = bd.Pos(0, 96, 212) * bd.Cone(21, 27, 42)
    cuff_inner = bd.Pos(0, 96, 212) * bd.Cone(19, 25, 42.2)
    cuff_trim = bd.Pos(-14, 96, 212) * bd.Box(52, 36, 45)
    tapered_pivot_cuff = (cuff_outer - cuff_inner) & cuff_trim
    tapered_pivot_cuff -= motor_clearance
    tapered_pivot_cuff -= cylinder_y(4.6, 38, *AXIS)
    fixed = fixed.fuse(tapered_pivot_cuff)
    if NESTED_COWL:
        # Fixed upper shutter encloses the front of the pivot while a thinner
        # forearm shutter slides radially below it through the 115-degree arc.
        # Both are rigid here: an overlap is a hard rejection, not "TPU flex".
        fixed_cowl = arc_cowl(28, 30, -30, 75, 82, 28)
        fixed = fixed.fuse(fixed_cowl)
    moving_cheek_skin = cylinder_y(24, 3, 0, 130, 200).fuse(
        cylinder_y(10, 8, 0, 126, 200))
    moving_outer = moving_outer.fuse(moving_cheek_skin)
    if NESTED_COWL:
        moving_cowl = arc_cowl(25.5, 26.5, 55, 180, 84, 24)
        moving_inner = moving_inner.fuse(moving_cowl)
    output_shaft = cylinder_y(4, 49, *AXIS)
    moving = moving_inner.fuse(moving_outer, *horns, output_pulley, output_shaft)

    direct_in_forearm = bd.Pos(0, 96, 187.5) * bd.Box(24.7, 35, 45.2)
    paw_original = paw_inner.fuse(paw_outer)
    swept = []
    for deg in range(0, 116, 5):
        arm = flexed(moving, deg)
        swept.append({"elbow_flex_deg": deg,
                      "moving_vs_fixed_mm3": round(volume(arm.intersect(fixed)), 3),
                      "moving_vs_motor_mm3": round(volume(arm.intersect(motor)), 3),
                      "paw_shell_vs_belt_mm3": round(volume(flexed(moving_inner, deg).intersect(belt)) +
                                                       volume(flexed(moving_outer, deg).intersect(belt)), 3)})
    poses = []
    for deg in POSES:
        arm = flexed(moving, deg)
        finger = flexed(thumb(35), deg)
        poses.append({"elbow_flex_deg": deg,
                      "moving_vs_fixed_mm3": round(volume(arm.intersect(fixed)), 2),
                      "moving_vs_motor_mm3": round(volume(arm.intersect(motor)), 2),
                      "moving_vs_fixed_min_gap_mm": round(arm.distance_to(fixed), 3),
                      "thumb_vs_upper_mm3": round(volume(finger.intersect(fixed)), 2)})
        clash = arm.intersect(fixed)
        poses[-1]["fixed_clash_bounds"] = [
            {"min": [round(v, 1) for v in s.bounding_box().min],
             "max": [round(v, 1) for v in s.bounding_box().max],
             "mm3": round(s.volume, 1)} for s in (clash.solids() if clash else []) if s.volume > 1]
        scene = trimesh.Scene()
        add(scene, "fixed_upper_and_motor_sleeve", fixed, (232, 222, 205, 255))
        add(scene, "moving_paw_inner", flexed(moving_inner, deg), (241, 233, 217, 255))
        add(scene, "moving_paw_outer", flexed(moving_outer, deg), (238, 229, 211, 255))
        add(scene, "gripper_open_thumb", finger, (226, 213, 189, 255))
        for index, horn in enumerate(horns):
            add(scene, f"output_hub_{index+1}", flexed(horn, deg), (158, 159, 155, 255))
        add(scene, "XL330_source_size_case", motor, (76, 113, 150, 70))
        for index, bearing in enumerate(bearing_envelopes):
            add(scene, f"igus_688_bearing_envelope_{index+1}", bearing, (96, 101, 98, 255))
        add(scene, "32T_motor_pulley", motor_pulley, (160, 154, 142, 180))
        add(scene, "64T_output_pulley", flexed(output_pulley, deg), (160, 154, 142, 180))
        add(scene, "MISUMI_188_2GT_4_belt_envelope", belt, (65, 73, 72, 190))
        scene.export(OUT / f"elbow-{deg:03d}.glb")

    bd.export_step(moving_inner, OUT / "paw-inner-with-elbow-cheek.step")
    bd.export_step(moving_outer, OUT / "paw-outer-with-elbow-cheek.step")
    bd.export_step(fixed, OUT / "upper-arm-with-elbow-sleeve.step")
    bd.export_step(upper_belt_cap, OUT / "upper-belt-cover.step")
    bd.export_step(tapered_pivot_cuff, OUT / "tapered-pivot-cuff.step")
    bd.export_step(moving_cheek_skin, OUT / "rotating-elbow-cheek-cover.step")
    bd.export_step(motor_pulley, OUT / "motor-pulley-32T-envelope.step")
    bd.export_step(output_pulley, OUT / "output-pulley-64T-envelope.step")
    bd.export_step(output_shaft, OUT / "output-shaft-envelope.step")
    bd.export_step(belt, OUT / "belt-188-2GT-4-envelope.step")
    for index, bearing in enumerate(bearing_envelopes):
        bd.export_step(bearing, OUT / f"bearing-688-envelope-{index+1}.step")
    for index, horn in enumerate(horns):
        bd.export_step(horn, OUT / f"elbow-horn-{index+1}.step")
    # Solid PLA density is an upper mass screen for this cosmetic cover;
    # print infill, screws and cable have not been selected.
    moving_cover_mass_g = moving_cheek_skin.volume * .00124
    distal_mass_g = 92 + moving_cover_mass_g
    report = {
        "stage": "elbow belt-drive packaging screen; not a cleared/print-ready joint",
        "compact_pivot_variant": COMPACT_PIVOT,
        "nested_cowl_variant": NESTED_COWL,
        "fixed_pivot_outer_radius_mm": pivot_outer_radius,
        "source_motor_url": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
        "source_motor": "ROBOTIS XL330-M288-T, 20 x 34 x 26 mm, 18 g, 0.52 Nm stall at 5 V; no published continuous torque used here",
        "source_bearing_url": "https://www.igus.com/product/xiros_radial_B180_ES_PA_C",
        "source_bearing": "igus BB-688-B180-10-ES-C envelope, 8 mm bore x 16 mm outer diameter x 5 mm width; seat retention/load life not verified",
        "source_belt_url": "https://th.misumi-ec.com/en/vona2/detail/110411276199/",
        "source_belt": "MISUMI 188-2GT-4, 94 teeth, 2 mm pitch, 4 mm wide, 1.38 mm nominal height; tooth profile and tension not modeled",
        "output_axis_xyz_mm": list(AXIS),
        "motor_case_center_mm": list(MOTOR_CENTER),
        "motor_to_elbow_map": "32T:64T GT2 concept: elbow rotation = motor rotation/2, torque ideally x2 before losses; servo zero/polarity and belt routing require assembly calibration",
        "pulley_center_distance_mm": round(math.hypot(10, 44), 3),
        "estimated_open_belt_pitch_length_mm": round(2 * math.sqrt(10**2 + 44**2 - (20.37 - 10.19)**2) + math.pi * (20.37 + 10.19) + 2 * (20.37 - 10.19) * math.asin((20.37 - 10.19) / math.hypot(10, 44)), 2),
        "estimated_center_distance_reduction_for_188mm_belt_mm": .286,
        "direct_forearm_case_vs_saved_paw_mm3": round(volume(direct_in_forearm.intersect(paw_original)), 1),
        "upper_case_vs_fixed_mm3": round(volume(motor.intersect(fixed)), 2),
        "bearings_vs_fixed_mm3": [round(volume(b.intersect(fixed)), 3) for b in bearing_envelopes],
        "bearings_vs_moving_mm3": [round(volume(b.intersect(moving)), 3) for b in bearing_envelopes],
        "belt_envelope_vs_fixed_mm3": round(volume(belt.intersect(fixed)), 3),
        "belt_envelope_vs_fixed_min_gap_mm": round(belt.distance_to(fixed), 3),
        "paw_outer_substantial_solids": len([s for s in moving_outer.solids() if s.volume > 1]),
        "stop_lug_vs_fixed_pegs_mm3": {
            str(deg): round(volume(flexed(moving_stop, deg).intersect(fixed_stops[0])) +
                            volume(flexed(moving_stop, deg).intersect(fixed_stops[1])), 3)
            for deg in (-1, 0, 115, 116)},
        "moving_parts_substantial_solids": len([s for s in moving.solids() if s.volume > 1]),
        "upper_parts_substantial_solids": len([s for s in fixed.solids() if s.volume > 1]),
        "distal_paw_core_mass_estimate_g": 92,
        "rotating_cover_solid_PLA_mass_estimate_g": round(moving_cover_mass_g, 1),
        "distal_paw_mass_estimate_g": round(distal_mass_g, 1),
        "motor_mass_g_on_upper_arm": 18,
        "static_paw_moment_at_90mm_Nm": round(distal_mass_g / 1000 * 9.81 * .09, 3),
        "static_paw_plus_20g_object_moment_at_90mm_Nm": round((distal_mass_g + 20) / 1000 * 9.81 * .09, 3),
        "motor_stall_torque_Nm_at_5V": .52,
        "motor_20pct_stall_heuristic_torque_Nm_not_a_rating": .10,
        "ideal_output_from_heuristic_Nm_before_losses": .20,
        "motor_no_load_rpm_at_5V": 103,
        "ideal_elbow_no_load_rpm": 51.5,
        "motor_to_paw_cable_route": "reserved 6 mm port at (8,100,196.5) in paw coupon; upper-arm connector bend and strain relief unverified",
        "skin_status": "oblique forearm seam clears the bend but nested lower pivot skin was rejected after appearance review; pivot remains visually exposed",
        "sweep_ten_degree": swept,
        "poses": poses,
        "open": ["verify stop pin strength and fastened retention", "verify pulley interface, belt force/tension/idler and bearing retention", "protect 6 mm cable route through flex", "close the moving gap while preserving curved arm silhouette", "physical fit and duty test"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("elbow_pitch_viewer.html").read_text())
    return report


if __name__ == "__main__":
    result = main()
    print("elbow sampled overlaps", result["poses"])
