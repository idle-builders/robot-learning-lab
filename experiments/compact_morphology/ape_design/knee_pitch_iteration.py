"""Direct, supported knee screen using the saved ankle/boot as the distal link.

This is deliberately a mechanism coupon: source-size motor, actual offset output,
two independent bearing stubs, a two-sided moving shin yoke, and a swept shell
cut. It is not a validated printed knee or a finished flexible outer skin.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from shapely.geometry import Point
from shapely.ops import unary_union

import sketch_shell as shell
from ankle_pitch_iteration import cylinder_y, volume


ROOT = Path(__file__).resolve().parents[3]
COMPACT_CAP = "--cap-compact" in sys.argv
CAPPED = "--cap" in sys.argv or COMPACT_CAP
RELOCATED = "--relocated" in sys.argv or CAPPED
OUT = ROOT / ("artifacts/ape-knee-tucked-crescent-20261003" if COMPACT_CAP else
              "artifacts/ape-knee-front-cap-20261003" if CAPPED else
              "artifacts/ape-knee-relocated-drive-20261003" if RELOCATED else
              "artifacts/ape-knee-pitch-iteration-20261003")
ANKLE = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
AXIS = (0, 41, 103)
ANGLES = tuple(range(0, 121, 10))
CASE_CENTER = (6, 41, 132) if RELOCATED else (0, 41, 115.2)
CASE_SIZE = (24.72, 35, 45.22)


def belt_envelope(clearance: float = 0) -> bd.Shape:
    """Smooth 20:20 2GT belt around the relocated knee drive, no teeth."""
    radius = 20 / math.pi  # 20 teeth * 2 mm pitch / 2*pi
    centers = ((6, 119.8), (0, 103))
    outer = unary_union([Point(x, z).buffer(radius + .69 + clearance, 32)
                         for x, z in centers]).convex_hull
    inner = unary_union([Point(x, z).buffer(radius - .69 - clearance, 32)
                         for x, z in centers]).convex_hull
    wire = lambda poly, y: bd.Wire.make_polygon([(x, y, z) for x, z in poly.exterior.coords])
    return bd.extrude(bd.Face(wire(outer, 19-clearance),
                              [wire(inner, 19-clearance)]),
                      amount=4+2*clearance, dir=(0, 1, 0))


def front_knee_cap() -> bd.Shape:
    """Thin fixed cosmetic sector over the frontal relief, not a full gaiter."""
    if COMPACT_CAP:
        sections = []
        for y, inner, outer, lo, hi in (
            (19, 24, 25, 8, 27), (24, 24, 26, 0, 35),
            (58, 24, 26, 0, 35), (63, 24, 25, 8, 27),
        ):
            angles = [lo + (hi - lo) * i / 35 for i in range(36)]
            points = [
                (outer * math.cos(math.radians(deg)), y,
                 103 + outer * math.sin(math.radians(deg))) for deg in angles
            ] + [
                (inner * math.cos(math.radians(deg)), y,
                 103 + inner * math.sin(math.radians(deg))) for deg in reversed(angles)
            ]
            sections.append(bd.Wire.make_polygon(points, close=True))
        return bd.Solid.make_loft(sections, ruled=True)
    angles = range(-35, 36, 2)
    def arc(radius: float, values) -> list[tuple[float, float, float]]:
        return [(radius * math.cos(math.radians(deg)), 18,
                 103 + radius * math.sin(math.radians(deg))) for deg in values]
    outline = arc(30, angles) + arc(28, reversed(angles))
    return bd.extrude(bd.Face(bd.Wire.make_polygon(outline, close=True)),
                      amount=46, dir=(0, 1, 0))


def pitched(part: bd.Shape, deg: float) -> bd.Shape:
    if deg == 0:
        return part
    return bd.Pos(*AXIS) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in AXIS)) * part


def colored(scene: trimesh.Scene, name: str, part: bd.Shape, color) -> None:
    try:
        vertices, faces = part.tessellate(.24, .24)
    except AttributeError:
        # A prior OCC subtraction can invalidate one cached face mesh while
        # leaving the BRep valid. Round-trip just this display copy through
        # STEP; collision and mass checks continue to use the original BRep.
        display_copy = OUT / f"_display-{name}.step"
        bd.export_step(part, display_copy)
        vertices, faces = bd.import_step(display_copy).tessellate(.24, .24)
        display_copy.unlink()
    mesh = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                           faces=np.asarray(faces), process=False)
    mesh.visual.vertex_colors = color
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    ankle_shin = bd.import_step(ANKLE / "shin-and-ankle-sleeve.step")
    boot = bd.import_step(ANKLE / "boot-and-yoke.step")
    outsole = bd.import_step(ANKLE / "oval-outsole.step")
    thigh = next(p.shape for p in shell.PARTS if p.name == "left_thigh_shell")

    # The direct motor faces upward in the thigh. Its official output is
    # 12.2 mm below case centre, therefore exactly at z=103 mm here.
    motor = bd.Pos(*CASE_CENTER) * bd.Box(*CASE_SIZE)
    motor_clearance = bd.Pos(*CASE_CENTER) * bd.Box(25.32, 35.6, 45.82)
    # The relocated motor lets the shin rise 2 mm at the seam. At 80 mm the
    # deepest bend still has a 0.2 mm3 case clash; 77 mm leaves ~2.5 mm.
    shin_top = 77 if RELOCATED else 75
    shin = ankle_shin & (bd.Pos(0, 41, shin_top-100) * bd.Box(200, 200, 200))

    # Two narrow side cheeks bridge the 28 mm gap from cut-back calf to knee.
    # Their outboard stubs run in 688 bearing envelopes fixed to the thigh;
    # inboard source-pattern discs attach to the dual 25T servo outputs.
    cheeks = []
    for y in ((14, 64) if RELOCATED else (18, 64)):
        cheek = (bd.Pos(0, y, 86) * bd.Box(13, 4, 34)).fuse(
            cylinder_y(10, 4, 0, y, 103))
        cheeks.append(cheek)
    shin_yoke = shin.fuse(*cheeks)
    horns = [cylinder_y(10, 3, 0, y, 103) for y in
             ((17, 60.5) if RELOCATED else (21.5, 60.5))]
    shafts = [cylinder_y(4, 7, 0, y, 103) for y in (12.5, 69.5)]
    motor_pulley = output_pulley = belt = None
    if RELOCATED:
        motor_pulley = cylinder_y(20/math.pi, 4, 6, 21, 119.8)
        output_pulley = cylinder_y(20/math.pi, 4, 0, 21, 103)
        belt = belt_envelope()
        # Driver uses the source 25T spline but a real horn adapter/tooth form
        # is still needed; this disc is only its pitch envelope.
        output_pulley -= cylinder_y(4.05, 6, 0, 21, 103)
        shaft_extension = cylinder_y(4, 9, 0, 18.5, 103)
        moving = shin_yoke.fuse(*horns, *shafts, shaft_extension)
    else:
        moving = shin_yoke.fuse(*horns, *shafts)

    bearings = [cylinder_y(8, 5, 0, y, 103) - cylinder_y(4.05, 6, 0, y, 103)
                for y in (9.5, 72.5)]
    bearing_housings = [
        (cylinder_y(11.5, 6, 0, y, 103) - cylinder_y(8.05, 7, 0, y, 103))
        for y in (9, 73)
    ]
    # Short overhead webs join the outboard bearing rings to the actual thigh
    # structure. Without them the visually convincing rings are floating CAD.
    support_webs = [bd.Pos(0, y, 118) * bd.Box(22, 14, 13)
                    for y in (14, 68)]
    # A swept relief is more honest than leaving the original knee seam in
    # place and counting a moving rigid shell collision as a valid joint.
    relieved_thigh = thigh.fuse(*bearing_housings, *support_webs) - motor_clearance
    if RELOCATED:
        relieved_thigh -= belt_envelope(.45)
    for deg in ANGLES:
        relieved_thigh = relieved_thigh - pitched(moving, deg)
    # Sweep Booleans can leave tiny disconnected skin chips, not manufactured
    # components. Keep the connected load path and count what was discarded.
    fixed_solids = sorted(relieved_thigh.solids(), key=lambda item: item.volume, reverse=True)
    discarded_chips_mm3 = round(sum(item.volume for item in fixed_solids[1:]), 2)
    fixed = fixed_solids[0]
    cap = front_knee_cap() if CAPPED else None
    cap_attachment_mm3 = round(volume(cap.intersect(fixed)), 2) if cap else None
    if cap:
        fixed = fixed.fuse(cap)

    poses = []
    for deg in ANGLES:
        moved = pitched(moving, deg)
        foot = pitched(boot.fuse(outsole), deg)
        poses.append({
            "knee_flex_deg": deg,
            "moving_shin_vs_fixed_thigh_mm3": round(volume(moved.intersect(fixed)), 2),
            "moving_shin_vs_motor_mm3": round(volume(moved.intersect(motor)), 2),
            "boot_vs_motor_mm3": round(volume(foot.intersect(motor)), 2),
            "moving_vs_belt_mm3": round(volume(moved.intersect(belt)), 2) if belt else None,
            "cap_vs_moving_mm3": round(volume(moved.intersect(cap)), 2) if cap else None,
            "cap_vs_boot_mm3": round(volume(foot.intersect(cap)), 2) if cap else None,
        })
        scene = trimesh.Scene()
        colored(scene, "relieved_thigh_skin_and_bearing_housings", fixed, (236, 228, 211, 255))
        colored(scene, "moving_shin_and_two_cheek_yoke", moved, (246, 236, 217, 255))
        colored(scene, "ankle_boot_and_outsole", foot, (210, 201, 182, 255))
        colored(scene, "ST3250_source_axis_motor", motor, (73, 111, 150, 190))
        for index, bearing in enumerate(bearings):
            colored(scene, f"bearing_688_{index+1}", bearing, (95, 97, 102, 255))
        if RELOCATED:
            colored(scene, "20T_motor_pulley_pitch_envelope", motor_pulley, (155, 151, 140, 220))
            colored(scene, "20T_knee_pulley_pitch_envelope", pitched(output_pulley, deg), (155, 151, 140, 220))
            colored(scene, "76_2GT_4_smooth_belt_envelope", belt, (83, 91, 90, 220))
        scene.export(OUT / f"knee-{deg:03d}.glb")

    # Equal-crank parallel-link comparison remains an unsolved hypothesis; the
    # current ToddlerBot hardware revision actually removed its parallel links.
    # Keep the older failed case/skin evidence; do not imply this is that CAD.
    shin_print_g = shin_yoke.volume * .00124
    ankle_report = json.loads((ANKLE / "report.json").read_text())
    distal_g = shin_print_g + 74.5 + ankle_report["foot_pla_plus_tpu_mass_estimate_g"]
    report = {
        "stage": ("35 cm relocated in-thigh 1:1 belt knee candidate; not a fastened or covered joint"
                  if RELOCATED else "35 cm direct knee mechanism coupon; moving yoke screened, not a finished printed joint"),
        "relocated_drive_variant": RELOCATED,
        "front_cap_variant": CAPPED,
        "front_cap_shape": "tucked tapered crescent" if COMPACT_CAP else
                           "wide cylinder sector" if CAPPED else None,
        "source_servo": "Feetech ST-3250-C001, 24.72 x 35 x 45.22 mm, 74.5 g, output 12.2 mm from case centre, 1.569 Nm rated at 12 V; stall torque excluded from duty budget",
        "servo_source": "https://www.feetechrc.com/en/562636.html",
        "bearing_source": "https://www.igus.com/product/xiros_radial_B180_ES_PA_C",
        "reference_choice": "ToddlerBot v2 removed parallel knee links in favour of a simplified knee with bearing-based motor idlers; Microban 30 cm also has a direct XL330 knee. Their CAD is not copied.",
        "toddlerbot_changelog": "https://github.com/hshi74/toddlerbot/blob/main/CHANGELOG.md",
        "microban_reference": "https://github.com/Rhoban/microban/blob/main/docs/assembly.md",
        "knee_output_axis_mm": list(AXIS),
        "motor_case_center_mm": list(CASE_CENTER),
        "motor_inside_original_thigh_interior_pct": round(
            100*volume(motor.intersect(shell.INTERIORS["left_thigh_shell"]))/motor.volume, 1),
        "motor_vs_pelvis_hip_case_mm3": round(volume(motor.intersect(
            bd.Pos(0, 41, 182.2)*bd.Box(*CASE_SIZE))), 2) if RELOCATED else None,
        "motor_to_hip_case_vertical_gap_mm": round(
            (182.2-CASE_SIZE[2]/2)-(CASE_CENTER[2]+CASE_SIZE[2]/2), 2) if RELOCATED else None,
        "motor_to_knee_map": "1:1 direct" if not RELOCATED else "20T:20T belt, 1:1 pitch direction depends on belt routing; zero/polarity to calibrate",
        "motor_output_to_knee_axis_mm": round(math.hypot(6, 16.8), 2) if RELOCATED else 0,
        "selected_belt_source": "https://jp.misumi-ec.com/vona2/detail/110302652060/" if RELOCATED else None,
        "selected_belt": "76-2GT-4, 38 teeth, 2 mm pitch, 4 mm width" if RELOCATED else None,
        "estimated_pitch_loop_mm": round(2*math.hypot(6,16.8)+40, 2) if RELOCATED else None,
        "estimated_center_adjustment_for_76mm_belt_mm": round((76-(2*math.hypot(6,16.8)+40))/2, 2) if RELOCATED else None,
        "illustrative_belt_tension_difference_N_for_3kg_30mm_knee_moment": round(
            (3*9.81*.03)/(20/math.pi/1000), 1) if RELOCATED else None,
        "illustrative_belt_tension_difference_N_at_servo_rated_torque": round(
            1.569/(20/math.pi/1000), 1) if RELOCATED else None,
        "belt_capacity_comparison_source": "https://ondrives.com/2gt-blp-4-pulley-2mm-pitch-4mm-belt-width" if RELOCATED else None,
        "belt_capacity_status": "NOT SOURCE-RATED: the 20T/4mm/76mm drive has no verified torque/pretension rating. Ondrives lists only 0.378 Nm for a larger 38T 4mm miniature pulley; that is not a transferable rating, but prevents treating this envelope as an accepted knee transmission." if RELOCATED else None,
        "belt_vs_fixed_mm3": round(volume(belt.intersect(fixed)), 2) if belt else None,
        "front_cap_fixed_attachment_mm3": cap_attachment_mm3,
        "front_cap_vs_motor_mm3": round(volume(cap.intersect(motor)), 2) if cap else None,
        "front_cap_vs_belt_mm3": round(volume(cap.intersect(belt)), 2) if cap else None,
        "motor_vs_fixed_mm3": round(volume(motor.intersect(fixed)), 2),
        "operating_range_deg": [0, 120],
        "removed_shin_above_z_mm": shin_top,
        "fixed_connected_solids": len(fixed.solids()),
        "moving_connected_solids_excluding_separate_pulley": len(moving.solids()),
        "discarded_boolean_skin_chips_mm3": discarded_chips_mm3,
        "printed_shin_yoke_mass_estimate_g_solid_PLA": round(shin_print_g, 1),
        "distal_module_mass_estimate_g_excluding_hardware_and_wires": round(distal_g, 1),
        "illustrative_gravity_moment_Nm_at_50mm_COM": round(distal_g / 1000 * 9.81 * .05, 3),
        "illustrative_whole_body_single_support_moment_Nm_at_30mm": round(3 * 9.81 * .03, 3),
        "rated_motor_torque_Nm": 1.569,
        "rated_vs_3kg_single_support_static_margin": round(1.569 / (3 * 9.81 * .03), 2),
        "direct_vs_parallel_link": "Direct 1:1 is the saved baseline. This variant offsets the same source motor upward inside the original thigh and uses a 1:1 catalog belt envelope to shrink the motor opening at the knee. A parallel link remains unevaluated; no durability or cost claim follows from this screen." if RELOCATED else "direct 1:1 was selected for this coupon because the current ToddlerBot v2 design removed its earlier knee links. The 20261002 ape parallel-link screen located no true motor axis, pin dimensions, rod rating, or swept distal ankle. This is an architecture choice for a screen, not proof of durability or cost.",
        "poses": poses,
        "unresolved": ["rigid thigh relief is an open slot: a real flexible external knee skin and strain envelope are needed", "source 25T-to-20T driver horn, true pulley teeth, 76 mm belt tension, torque capacity and retention are not designed; this belt option is not approved for knee load" if RELOCATED else "horn spline and four-bolt retention must be detailed", "688 bearing envelopes have no sourced knee load/life rating", "moving stop and hard-stop impact path", "cable slack around 120 degree flex", "servo thermal duty and actual robot mass are unmeasured", "full leg policy replay"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    bd.export_step(fixed, OUT / "relieved-thigh-and-bearing-housings.step")
    bd.export_step(shin_yoke, OUT / "shin-and-knee-yoke.step")
    if cap:
        bd.export_step(cap, OUT / "front-knee-cap-cosmetic-envelope.step")
    if RELOCATED:
        bd.export_step(motor_pulley, OUT / "20T-motor-pulley-envelope.step")
        bd.export_step(output_pulley, OUT / "20T-knee-pulley-envelope.step")
        bd.export_step(belt, OUT / "76-2GT-4-belt-envelope.step")
    (OUT / "index.html").write_text(Path(__file__).with_name("knee_pitch_viewer.html").read_text())
    return report


if __name__ == "__main__":
    result = main()
    print([(p["knee_flex_deg"], p["moving_shin_vs_fixed_thigh_mm3"],
            p["moving_shin_vs_motor_mm3"]) for p in result["poses"]])
