"""Source-axis ankle coupon for the 35 cm sketch-06 ape.

The older ankle screen put a servo box at the desired pivot, although the
Feetech output is offset within that box. This coupon checks an actual output
datum, one boot-to-shin load path, the full boot contact shape and pitch poses.
It is a CAD candidate, not a validated printed joint or motor duty model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh

import sketch_shell as shell
from make_boot_model import cad_outsole


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-ankle-pitch-iteration-20261003"
AXIS = (0, 41, 42)
ANGLES = (-38, -35, -30, -20, 0, 20, 30, 35, 38)
CONTROL_LIMIT_DEG = 35
CASE_SIZE = (24.72, 35, 45.22)
CASE_CENTER = (0, 41, 54.2)
# Official drawing: output axis 34.8 mm from one end of a 45.22 mm case.
# Flipping the case with its long axis up puts the output 12.2 mm below center.
OUTPUT_OFFSET_FROM_CASE_CENTER_MM = 12.2


def volume(shape: bd.Shape | bd.ShapeList | None) -> float:
    if shape is None:
        return 0.0
    return sum(part.volume for part in shape) if isinstance(shape, bd.ShapeList) else shape.volume


def overlap(a: bd.Shape | bd.ShapeList | None, b: bd.Shape) -> float:
    if a is None:
        return 0.0
    return sum(volume(part.intersect(b)) for part in a) if isinstance(a, bd.ShapeList) else volume(a.intersect(b))


def cylinder_y(radius: float, length: float, x: float, y: float, z: float) -> bd.Shape:
    return bd.Pos(x, y, z) * bd.Rot(90, 0, 0) * bd.Cylinder(radius, length)


def pitched(shape: bd.Shape, deg: float) -> bd.Shape:
    return bd.Pos(*AXIS) * bd.Rot(0, deg, 0) * bd.Pos(*(-v for v in AXIS)) * shape


def flexible_gaiter(deg: float) -> bd.Shape:
    """A thin deformed sleeve between the moving boot rim and fixed shin rim.

    This is only a pose-dependent clearance envelope; TPU strain, folds,
    closure and fatigue cannot be inferred from its BRep.
    """
    a = math.radians(deg)
    normal = (math.sin(a), 0, math.cos(a))
    x_axis = (math.cos(a), 0, -math.sin(a))
    lower = (-15 * math.sin(a), 41, 42 - 15 * math.cos(a))
    boot_plane = bd.Plane(origin=lower, x_dir=x_axis, z_dir=normal)
    boot_inner = bd.Plane(origin=tuple(lower[i] - normal[i] for i in range(3)),
                          x_dir=x_axis, z_dir=normal)
    outer = bd.loft([boot_plane * bd.Ellipse(28.5, 29.5),
                     bd.Plane(origin=(0, 41, 47)) * bd.Ellipse(27.5, 35)])
    inner = bd.loft([boot_inner * bd.Ellipse(27.2, 28.2),
                     bd.Plane(origin=(0, 41, 48)) * bd.Ellipse(26.2, 33.7)])
    gaiter = outer - inner
    for x in (-15, 15):
        gaiter = gaiter - pitched(bd.Pos(x, 18, 42) * bd.Box(12.6, 4, 2.6), deg)
    return gaiter


def mesh(shape: bd.Shape, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(0.18, 0.18)
    result = trimesh.Trimesh(vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
                             faces=np.asarray(faces), process=False)
    result.merge_vertices()
    result.visual.vertex_colors = rgba
    return result


def scene_add(scene: trimesh.Scene, name: str, shape: bd.Shape,
              rgba: tuple[int, int, int, int]) -> None:
    scene.add_geometry(mesh(shape, rgba), geom_name=name, node_name=name)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    boot_shell = next(p.shape for p in shell.PARTS if p.name == "left_foot_shell")
    shin_shell = next(p.shape for p in shell.PARTS if p.name == "left_shin_shell")
    reference_outsole = cad_outsole()
    # Oval sketch-following contact candidate: nearly the z=10 mm boot plan
    # bounds, unlike the old 90 x 60 mm forward-projecting rectangular pad.
    outsole = bd.Pos(7, 41, 0) * bd.extrude(bd.Ellipse(40, 33), amount=2.5)
    # These straight prototype trims replace the old intersecting skin at the
    # ankle; a later skin pass may feather their edge without moving the axis.
    boot_trim = boot_shell - bd.Pos(0, 41, 102) * bd.Box(200, 200, 150)  # z >= 27
    shin_trim = shin_shell - bd.Pos(0, 41, -22.5) * bd.Box(200, 200, 150)  # z <= 52.5

    # Direct-in-boot alternative, with true source axis at x=-11,z=27.
    # Its case is only ~62% inside the foot interior; retain as a rejected
    # comparison instead of pretending the box center is the servo output.
    direct_case = bd.Pos(1.2, 41, 27) * bd.Box(45.22, 35, 24.72)

    motor_case = bd.Pos(*CASE_CENTER) * bd.Box(*CASE_SIZE)
    # The case drawing shows dual 25T, 5.9 mm output. These cylindrical
    # envelopes check mating clearance; spline teeth and retention are not
    # represented. Their 4 mm extension is an envelope from the side view.
    output_shafts = [cylinder_y(2.95, 4, 0, y, 42) for y in (21.5, 60.5)]
    motor = motor_case.fuse(*output_shafts)
    motor_clearance = bd.Pos(*CASE_CENTER) * bd.Box(25.32, 35.6, 45.82)
    # Case is fixed to the shin. The lower sleeve masks the case where the
    # original lower shin skin must open for boot pitch. The transverse bridge
    # connects sleeve to the remaining curved shin skin.
    sleeve_outer = bd.fillet((bd.Pos(*CASE_CENTER) * bd.Box(29, 40, 48)).edges(), 2)
    horn_windows = cylinder_y(11, 70, *AXIS)
    sleeve = sleeve_outer - motor_clearance - horn_windows
    bridge = (shell.zloft([(62, -1, 41, 50, 52),
                           (68, -1, 41, 51, 53)]) - motor_clearance)
    # Two opposite radial tabs meet upper fixed pegs around +/-38 degrees. Putting
    # both stops above the pivot avoids clipping the lower moving boot/yoke.
    moving_stop_tabs = [bd.Pos(x, 18, 42) * bd.Box(12, 3, 2) for x in (-15, 15)]
    # Relieve only the cosmetic shin skin across the operating arc. The
    # motor sleeve stays intact; the fixed pegs are added AFTER this relief.
    for deg in range(-40, 41, 5):
        for x in (-15, 15):
            relief = bd.Pos(x, 18, 42) * bd.Box(14, 5, 3)
            shin_trim = shin_trim - pitched(relief, deg)
    stop_pegs = []
    for x in (-20 / np.sqrt(2), 20 / np.sqrt(2)):
        z = 55.8
        peg = cylinder_y(1.4, 6, x, 19.5, z)
        stem = bd.Pos(x, 24, z) * bd.Box(3, 4, 6)
        stop_pegs.append(peg.fuse(stem))
    fixed = shin_trim.fuse(sleeve, bridge, *stop_pegs) - motor_clearance

    # Full curved boot shell and outsole, not a thin contact plate. A low
    # internal beam connects both foot-side yoke cheeks to the underside;
    # each cheek couples to a source-size 20 mm, 4xM3 horn disc. The shaft is
    # two-sided, so fore/aft bending is resisted on both lateral faces.
    foot_beam = bd.Pos(0, 41, 8) * bd.Box(24, 56, 16)
    inner_cheek = (bd.Pos(0, 18, 27) * bd.Box(12, 4, 38)).fuse(
        cylinder_y(10, 2, 0, 19, 42))
    outer_cheek = (bd.Pos(0, 64, 27) * bd.Box(12, 4, 38)).fuse(
        cylinder_y(10, 2, 0, 63, 42))
    # Actual horn rings remain separate STEP parts. Their source drawing has
    # four M3 locations on the Ø14 pattern. Matching holes run through the
    # printed cheek flanges; bolt heads and spline teeth await a fit print.
    bolt_centres = [(0, 49), (7, 42), (0, 35), (-7, 42)]
    for x, z in bolt_centres:
        inner_cheek -= cylinder_y(1.6, 8, x, 20, z)
        outer_cheek -= cylinder_y(1.6, 8, x, 62, z)
    inner_cheek -= cylinder_y(3.05, 8, 0, 20, 42)
    outer_cheek -= cylinder_y(3.05, 8, 0, 62, 42)
    horns = []
    for y in (21.5, 60.5):
        horn = cylinder_y(10, 3, 0, y, 42) - cylinder_y(3.05, 4, 0, y, 42)
        for x, z in bolt_centres:
            horn -= cylinder_y(1.6, 4, x, y, z)
        horns.append(horn)
    foot_rigid = (boot_trim.fuse(foot_beam, inner_cheek, outer_cheek,
                                 *moving_stop_tabs) - outsole)
    # The fixed skirt hides the lower motor sleeve. Its open top/bottom are
    # important: a capped skin would falsely collide with the horn/yoke.
    skirt_outer = shell.zloft([(47, -1, 41, 55, 70), (50, -1, 41, 55, 68),
                               (54, -1, 41, 50, 55)])
    skirt_inner = shell.zloft([(46, -1, 41, 52.4, 67.4),
                               (50, -1, 41, 52.4, 65.4),
                               (55, -1, 41, 47.4, 52.4)])
    fixed_skirt = skirt_outer - skirt_inner
    for deg in range(-40, 41, 5):
        for x in (-15, 15):
            relief = bd.Pos(x, 18, 42) * bd.Box(14, 5, 3)
            fixed_skirt = fixed_skirt - pitched(relief, deg)
    # The inner-side swept slot can detach a small sliver from this thin
    # cosmetic ring. It is not a printable part, so discard that sliver.
    fixed_skirt = max(fixed_skirt.solids(), key=lambda solid: solid.volume)
    fixed = fixed.fuse(fixed_skirt)
    # The two neighboring print materials do not double-count their shared
    # CAD volume. TPU is a material choice only; its friction is unmeasured.
    body = foot_rigid.fuse(outsole, *horns)

    # The operating interval is evaluated at every whole degree, rather than
    # inferred from the nine poses chosen for the browser slider. Keep the
    # first off-range interference separate from the designed stop contact.
    sweep = []
    for deg in range(-40, 41):
        moved = pitched(body, deg)
        rigid_intersection = volume(moved.intersect(fixed))
        stop_intersection = sum(
            volume(pitched(tab, deg).intersect(peg))
            for tab in moving_stop_tabs for peg in stop_pegs)
        sweep.append({"deg": deg,
                      "rigid_intersection_mm3": round(rigid_intersection, 3),
                      "stop_intersection_mm3": round(stop_intersection, 3),
                      "motor_intersection_mm3": round(volume(moved.intersect(motor)), 3)})
    operating_sweep = [p for p in sweep if abs(p["deg"]) <= CONTROL_LIMIT_DEG]
    first_nonstop_interference = next((p for p in sweep if p["deg"] > CONTROL_LIMIT_DEG and
                                       p["rigid_intersection_mm3"] > 0.01 and
                                       p["stop_intersection_mm3"] <= 0.01), None)

    poses = []
    for deg in ANGLES:
        foot_pose = pitched(body, deg)
        gaiter = flexible_gaiter(deg) if abs(deg) <= 35 else None
        mid_gaiter = (gaiter.intersect(bd.Pos(0, 41, 38.5) * bd.Box(200, 200, 15))
                      if gaiter is not None else None)
        poses.append({"pitch_deg": deg,
                      "moving_boot_vs_fixed_shin_mm3": round(volume(foot_pose.intersect(fixed)), 2),
                      "moving_boot_vs_motor_case_mm3": round(volume(foot_pose.intersect(motor)), 2),
                      "stop_tab_vs_pegs_mm3": round(sum(
                          volume(pitched(tab, deg).intersect(peg))
                          for tab in moving_stop_tabs for peg in stop_pegs), 2),
                      "flexible_gaiter_mid_vs_fixed_mm3": round(overlap(mid_gaiter, fixed), 2) if gaiter is not None else None,
                      "flexible_gaiter_mid_vs_motor_mm3": round(overlap(mid_gaiter, motor), 2) if gaiter is not None else None,
                      "flexible_gaiter_mid_vs_moving_boot_mm3": round(overlap(mid_gaiter, foot_pose), 2) if gaiter is not None else None})
        scene = trimesh.Scene()
        scene_add(scene, "fixed_shin_and_motor_sleeve", fixed, (232, 222, 205, 255))
        scene_add(scene, "moving_boot_and_yoke", pitched(foot_rigid, deg), (241, 233, 217, 255))
        for index, horn in enumerate(horns):
            scene_add(scene, f"source_pattern_horn_{index+1}", pitched(horn, deg), (160, 157, 147, 255))
        scene_add(scene, "full_outsole", pitched(outsole, deg), (83, 91, 82, 255))
        scene_add(scene, "ST3250_source_axis_case", motor, (76, 113, 150, 180))
        if gaiter is not None:
            scene_add(scene, "pose_deformed_flexible_gaiter_candidate", gaiter, (217, 212, 198, 230))
        if abs(deg) <= CONTROL_LIMIT_DEG:
            fixed_mesh = scene.geometry["fixed_shin_and_motor_sleeve"]
            boot_mesh = trimesh.util.concatenate([scene.geometry["moving_boot_and_yoke"],
                                                   scene.geometry["full_outsole"]])
            mesh_gaps = (trimesh.proximity.closest_point(fixed_mesh, boot_mesh.vertices)[1].min(),
                         trimesh.proximity.closest_point(boot_mesh, fixed_mesh.vertices)[1].min())
            poses[-1]["sampled_mesh_boot_to_fixed_gap_mm"] = round(float(min(mesh_gaps)), 3)
        scene.export(OUT / f"ankle-{deg:+03d}.glb")

    # Contact export excludes the narrow yoke, whose high convex hull would
    # create false ground contact. It retains the whole round boot + outsole.
    contact_mesh = trimesh.util.concatenate([mesh(boot_trim, (255, 255, 255, 255)),
                                              mesh(outsole, (255, 255, 255, 255))]).convex_hull
    contact_mesh.apply_translation(-np.asarray(AXIS, dtype=float))
    contact_mesh.apply_scale(0.001)
    contact_mesh.export(OUT / "full-boot-contact-pitch42.stl")
    bd.export_step(foot_rigid, OUT / "boot-and-yoke.step")
    bd.export_step(horns[0], OUT / "inner-horn-pattern.step")
    bd.export_step(horns[1], OUT / "outer-horn-pattern.step")
    bd.export_step(outsole, OUT / "oval-outsole.step")
    bd.export_step(fixed, OUT / "shin-and-ankle-sleeve.step")
    bd.export_step(flexible_gaiter(0), OUT / "nominal-flexible-gaiter-candidate.step")

    report = {
        "stage": "35 cm ankle-pitch CAD candidate; no physical print or force verification",
        "source_servo": "Feetech ST-3250-C001 / STS3250, 45.22 x 24.72 x 35 mm, 74.5 +/- 1 g, 25T 5.9 mm dual output, 16 kg.cm rated torque and 50 kg.cm stall at 12 V",
        "source_url": "https://www.feetechrc.com/en/562636.html",
        "source_pdf_url": "https://www.feetechrc.com/Data/feetechrc/upload/file/20260622/6391772424007467984928587.pdf",
        "source_drawing_url": "https://www.feetechrc.com/Data/feetechrc/upload/image/20240120/6384135883514319991304517.png",
        "source_horn_drawing_url": "https://www.feetechrc.com/Data/feetechrc/upload/image/20240120/6384135884369006732826960.png",
        "source_torque_speed_curve_url": "https://www.feetechrc.com/Data/feetechrc/upload/image/20240120/6384135885801820125616076.png",
        "output_axis_xyz_mm": list(AXIS),
        "source_output_offset_from_case_center_mm": OUTPUT_OFFSET_FROM_CASE_CENTER_MM,
        "controlled_pitch_limit_deg": CONTROL_LIMIT_DEG,
        "shin_motor_center_xyz_mm": list(CASE_CENTER),
        "shin_motor_case_xyz_mm": list(CASE_SIZE),
        "motor_vs_shin_rigid_mm3": round(volume(motor.intersect(fixed)), 2),
        "output_shaft_diameter_mm": 5.9,
        "horn_bolt_pattern": "four M3 through-holes on 14 mm pitch circle, 3.2 mm clearance; source drawing pattern, fit unverified",
        "direct_boot_candidate_axis_xyz_mm": [-11, 41, 27],
        "direct_boot_case_inside_foot_pct": round(100 * volume(direct_case.intersect(shell.INTERIORS["left_foot_shell"])) / direct_case.volume, 1),
        "shin_case_inside_shin_pct": round(100 * volume(motor.intersect(shell.INTERIORS["left_shin_shell"])) / motor.volume, 1),
        "foot_rigid_substantial_solids": len([s for s in foot_rigid.solids() if s.volume > 1]),
        "shin_rigid_substantial_solids": len([s for s in fixed.solids() if s.volume > 1]),
        "fixed_skirt_substantial_solids": len([s for s in fixed_skirt.solids() if s.volume > 1]),
        "flexible_gaiter_nominal_volume_mm3": round(flexible_gaiter(0).volume, 1),
        "flexible_gaiter_status": "pose-deformed CAD clearance envelope only; TPU material strain, attachment and fatigue unverified",
        "foot_rigid_volume_mm3": round(foot_rigid.volume, 1),
        "outsole_volume_mm3": round(outsole.volume, 1),
        "source_motor_mass_g_on_shin": 74.5,
        "source_motor_mass_g_on_boot": 0,
        "motor_to_joint_map": "direct co-axial pitch, joint angle = motor output angle at 1:1 ratio; polarity and zero offset to calibrate after assembly",
        "material_density_sources": {
            "PLA_1.24_g_cm3": "https://eu.store.bambulab.com/nl-nl/products/pla-basic-gradient?variant=53638219006300",
            "TPU_1.22_g_cm3": "https://eu.store.bambulab.com/en-dk/products/tpu-95a-hf",
        },
        "foot_pla_plus_tpu_mass_estimate_g": round(foot_rigid.volume * 1.24 / 1000 +
                                                     outsole.volume * 1.22 / 1000, 1),
        "foot_pla_plus_tpu_mass_excludes": "horn hardware, fasteners, adhesive, wires and actual infill; do not use as final gym inertia",
        "outsole_vs_boot_rigid_mm3": round(volume(outsole.intersect(foot_rigid)), 3),
        "operating_one_degree_sweep": {
            "samples": len(operating_sweep),
            "max_boot_vs_fixed_mm3": max(p["rigid_intersection_mm3"] for p in operating_sweep),
            "max_boot_vs_motor_mm3": max(p["motor_intersection_mm3"] for p in operating_sweep),
            "max_tab_vs_peg_mm3": max(p["stop_intersection_mm3"] for p in operating_sweep),
        },
        "first_nonstop_interference_past_control_limit_deg": first_nonstop_interference,
        "off_range_sweep": [p for p in sweep if abs(p["deg"]) > CONTROL_LIMIT_DEG],
        "contact_mesh": "full CAD boot shell + 80 x 66 mm oval outsole convex hull, excluding elevated yoke",
        "reference_outsole_plan_bounds_mm": [90, 60],
        "oval_outsole_plan_bounds_mm": [80, 66],
        "outsole_plan_area_ratio_to_reference": round(outsole.volume / reference_outsole.volume, 3),
        "contact_hull_volume_cm3": round(contact_mesh.volume * 1e6, 2),
        "poses": poses,
        "sampled_mesh_gap_warning": "Vertex-to-triangle tessellated screen at displayed poses only. It does not prove exact CAD minimum clearance, print tolerance, flex-gaiter strain or servo output support rating.",
        "torque_speed_note": "Feetech rated 16 kg.cm = 1.569 Nm at 12 V, 1.4 A; no-load 75 rpm; source N-T graph falls near 51 rpm at rated torque. Not a measured dynamic ankle curve. The old gym clamps 1.569 Nm but reported near-continuous saturation during walking, so margin is unproven.",
        "static_single_support_examples": [
            {"assumed_supported_mass_kg": mass, "assumed_horizontal_com_offset_mm": 30,
             "gravity_moment_nm": round(mass * 9.81 * .03, 3),
             "fraction_of_published_rated_torque": round(mass * 9.81 * .03 / 1.569, 2)}
            for mass in (2, 3, 4)
        ],
        "still_open": ["check case mounting and horn/spline engagement in actual parts", "verify print-feasible hard-stop impact strength", "source/verify load rating of the servo's two supported outputs", "full-body policy replay with 42 mm ankle pivot and honest contact", "physical FDM fit, friction, servo heating and load"],
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "index.html").write_text(Path(__file__).with_name("ankle_pitch_viewer.html").read_text())
    return report


if __name__ == "__main__":
    result = main()
    print("ankle pitch 35 cm:", result["direct_boot_case_inside_foot_pct"],
          "% direct-boot case fit; pose overlaps", result["poses"])
