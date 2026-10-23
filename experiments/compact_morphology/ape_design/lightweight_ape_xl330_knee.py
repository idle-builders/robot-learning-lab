"""Original supported XL330-class knee coupon for the 350 mm sketch ape.

Run: .venv/bin/python experiments/compact_morphology/ape_design/lightweight_ape_xl330_knee.py
Coordinates are the sketch-shell coordinates: X right, Y rear, Z up, mm.
This is an envelope/kinematic coupon, not a print-ready or torque-qualified joint.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-xl330-knee-20261005"
AXIS = (42.0, 3.0, 99.0)
POSES = (0, 30, 60, 90, 120)


def box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> bd.Shape:
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0,
                             bd.Plane(origin=(x0, y0, z0)))


def cyl_x(radius: float, x0: float, x1: float, y: float = AXIS[1],
          z: float = AXIS[2]) -> bd.Shape:
    return bd.Solid.make_cylinder(radius, x1-x0,
                                  bd.Plane(origin=(x0, y, z), z_dir=(1, 0, 0)))


def compound(shapes) -> bd.Compound:
    return bd.Compound(children=[solid for shape in shapes for solid in shape.solids()
                                 if solid.volume > 1e-6])


def volume(shape) -> float:
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def turn(shape, deg: float):
    return (bd.Pos(*AXIS) * bd.Rot(deg, 0, 0) *
            bd.Pos(*(-v for v in AXIS)) * shape)


def metrics(shape) -> dict:
    bb = shape.bounding_box()
    return {"solids": len([s for s in shape.solids() if s.volume > 1]),
            "volume_mm3": round(volume(shape), 3),
            "bounds_mm": [[round(v, 3) for v in bb.min],
                          [round(v, 3) for v in bb.max]]}


def overlap(a, b) -> float:
    return volume(a & b)


def aabb_gap(a, b):
    return math.sqrt(sum(max(v0-u1, u0-v1, 0)**2 for u0,u1,v0,v1 in (
        (a.min.X,a.max.X,b.min.X,b.max.X),
        (a.min.Y,a.max.Y,b.min.Y,b.max.Y),
        (a.min.Z,a.max.Z,b.min.Z,b.max.Z))))


def hard_contact(a, b, threshold=1.0):
    """Exact overlap plus minimum B-rep distance for nearby solid pairs."""
    hit = 0.0
    gap = float("inf")
    right = [(s, s.bounding_box(optimal=False)) for s in b.solids()]
    for left in a.solids():
        lb = left.bounding_box(optimal=False)
        for solid, rb in right:
            coarse = aabb_gap(lb, rb)
            if coarse >= threshold and coarse >= gap:
                continue
            if coarse < 1e-7:
                hit += overlap(left, solid)
            if coarse < threshold:
                gap = min(gap, left.distance_to(solid))
            else:
                gap = min(gap, coarse)
    return round(hit, 4), round(gap, 4) if math.isfinite(gap) else None


def mesh(shape, color):
    vertices, faces = shape.tessellate(.35, .3)
    result = trimesh.Trimesh(
        np.array([(v.X, v.Y, v.Z) for v in vertices]),
        np.asarray(faces), process=False)
    result.visual.vertex_colors = color
    return result


def add(scene, name, shape, color):
    for i, solid in enumerate(shape.solids()):
        if solid.volume > .01:
            node = f"{name}_{i}"
            scene.add_geometry(mesh(solid, color), geom_name=node, node_name=node)


def geometry():
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    source = {s.label: s for s in bd.import_step(SOURCE).children}
    thigh_raw = compound(source[n] for n in ("R_thigh_front", "R_thigh_back"))
    shin_raw = compound(source[n] for n in ("R_shin_front", "R_shin_back"))
    cuff_raw = source["R_knee_cuff"]
    def outer_loft(stations):
        return bd.Solid.make_loft([bd.Wire.make_ellipse(rx, ry, bd.Plane(origin=(x,y,z)))
                                   for z,x,y,rx,ry in stations])
    thigh_outer = outer_loft([(87,42,3,16,18),(98,43,3,20,22),(124,39,3,24,25),
                              (145,35,4,22,23),(157,33,4,16,18),(163,31,4,8,10)])
    shin_outer = outer_loft([(23,46,3,21,20),(36,46,3,23,23),(57,45,4,22,23),
                             (75,43,4,19,20),(83,42,3,16,17)])
    cuff_outer = bd.Solid.make_sphere(1).scale((15,17,7)).translate((42,3,86))
    original_outer = compound((thigh_outer, shin_outer, cuff_outer))

    # Distributor drawing copy: 23 mm body along X plus 3 mm output boss;
    # output center is 9.5 mm below the top of the 34 mm body. The official
    # ROBOTIS manual confirms 20 x 34 x 26 mm, but its drawing download is
    # inaccessible here. The body is flipped so its axis is near the bottom.
    motor = box(31, 54, -7, 13, 89.5, 123.5)
    output_boss = cyl_x(8, 54, 57)
    idler_boss = cyl_x(8, 28, 31)
    motor_envelope = compound((motor, output_boss, idler_boss))
    motor_service = box(30.4, 54.6, -7.6, 13.6, 88.9, 124.1)
    motor_service = motor_service + cyl_x(8.6, 53.9, 57.6) + cyl_x(8.6, 27.4, 31.1)

    # One continuous internal frame: back plate, side rail, bottom saddle,
    # overhead bridge, and bearing socket. The two upper tabs engage the
    # source thigh's inner back wall. No claimed vendor case-screw geometry.
    rear_plate = box(25.5, 56, 14.3, 16.5, 88, 133)
    left_rail = box(25.5, 27.9, -8.5, 16.5, 91, 133)
    saddle = box(25.5, 56, -8.5, 16.5, 86.9, 88.8)
    bridge = box(25.5, 56, -8.5, 16.5, 125, 128)
    high_web = box(25.5, 56, 14.3, 16.5, 128, 142)
    anchor_a = box(25.5, 29.5, 16, 28, 134, 140)
    anchor_b = box(51, 55, 16, 28, 134, 140)
    socket = cyl_x(7.0, 25.5, 27.95) - cyl_x(4.05, 25.4, 28)
    carrier = (rear_plate + left_rail + saddle + bridge + high_web +
               anchor_a + anchor_b + socket) - cyl_x(4.05, 25.4, 28)
    # The case can be lifted out toward the open front. These holes are merely
    # two prospective cross-screw access windows through the back plate.
    for z in (113, 119):
        carrier -= cyl_x(1.2, 28.6, 30.5, y=10, z=z)

    bearing = cyl_x(4.0, 25.6, 27.9) - cyl_x(2.05, 25.5, 28)
    shim = cyl_x(4.9, 25.05, 25.5) - cyl_x(2.1, 25, 25.55)

    # Output drive disc is attached to the moving right cheek; four M2-sized
    # clearance bores sit on the drawing's 12 mm pitch circle. Its engagement
    # with the XL330 spline and central retention still need a sourced horn.
    drive_disc = cyl_x(8.0, 57.05, 59.1) - cyl_x(1.1, 57, 59.2)
    for deg in (45, 135, 225, 315):
        a = math.radians(deg)
        drive_disc -= cyl_x(1.1, 57, 59.2,
                            y=AXIS[1]+6*math.cos(a), z=AXIS[2]+6*math.sin(a))

    left_cheek = (cyl_x(6.2, 23.05, 25.05) +
                  box(23.05, 25.05, -1.5, 7.5, 77, 99) +
                  box(23.05, 25.05, 7, 22, 76.5, 79))
    right_cheek = (cyl_x(6.2, 59.1, 61.1) +
                   box(59.1, 61.1, -1.5, 7.5, 77, 99) +
                   box(59.1, 61.1, 7, 22, 76.5, 79))
    # A rear rib connects the cheeks to the actual calf skin without sweeping
    # through the servo saddle in deep flexion.
    cross_rib = box(23.05, 61.1, 21, 26, 76.2, 78.5)
    journal = cyl_x(2.0, 24.9, 27.85)
    yoke = left_cheek + right_cheek + cross_rib + journal + drive_disc

    # Keep the actual 1 mm shell. Relief is applied to the source B-reps,
    # never grown past their exterior. The black source cuff contributes only
    # a frontal, upper protective crescent; rear fold remains open.
    shin_trim = box(-100, 150, -100, 100, -10, 80)
    shin = compound((source["R_shin_front"] & shin_trim,
                     source["R_shin_back"] & shin_trim))
    moving = compound((shin, yoke))
    front_crescent = cuff_raw & box(-100, 150, -100, -5.2, 87, 110)
    # Service opening for XL330 and true kinematic relief of cheek arms and
    # calf skin. This is the minimum source-cover subtraction at 10-degree
    # samples; no continuous swept clearance is asserted between samples.
    sweep_relief = compound((turn(yoke, deg) for deg in range(0, 121, 10)))
    shin_sweep_relief = compound((turn(shin, deg) for deg in range(60, 121, 10)))
    shell_relief = compound((motor_service, sweep_relief, shin_sweep_relief))
    thigh = compound(part - shell_relief for part in
                     (source["R_thigh_front"], source["R_thigh_back"]))
    crescent = front_crescent - sweep_relief
    fixed_cover = compound((thigh, crescent))
    fixed_hard = compound((fixed_cover, carrier, motor_envelope))
    moving_hard = compound((shin, yoke))
    return locals()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = geometry()
    report = {
        "status": "rejected 120-degree mechanical and source-silhouette fit",
        "pose_pass_definition": "sampled B-rep volume overlap below 0.01 mm3; no positive manufacturing clearance implied",
        "coordinate_frame": "350 mm sketch ape, X right, Y rear, Z up, mm",
        "source_shell": str(SOURCE.relative_to(ROOT)),
        "source_wall_mm": 1,
        "xl330_manual": "https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/",
        "xl330_drawing_copy": "https://img2024.cnblogs.com/blog/2030440/202602/2030440-20260227180532067-2138822197.png",
        "manufacturer_drawing_status": "official download returned 404 on 2026-10-05; distributor copy provisional",
        "motor_published": {"case_envelope_mm": [20, 34, 26], "mass_g": 18,
                            "stall_torque_5v_Nm": .52,
                            "continuous_torque_Nm": None},
        "assumptions": ["23 mm body plus 3 mm output boss, and 9.5 mm top-to-axis from distributor drawing copy",
                        "motor mount is a generic removable saddle without confirmed case screw engagement",
                        "bearing 4 mm OD / 2 mm bore and shim are package envelopes, not selected rated components",
                        "horn/disc spline, screw depth, journal retention, cable bend and printed tolerances unverified"],
        "axis_xyz_mm": AXIS,
        "source_bounds": {"thigh": metrics(g["thigh_raw"]), "shin": metrics(g["shin_raw"]),
                          "knee_cuff": metrics(g["cuff_raw"])},
        "parts": {name: metrics(g[name]) for name in ("motor_envelope", "carrier", "bearing",
                                                  "shim", "yoke", "shin", "thigh", "crescent")},
        "source_thigh_shell_removed_pct": round(100*(volume(g["thigh_raw"])-volume(g["thigh"]))/volume(g["thigh_raw"]), 2),
        "source_shin_shell_removed_pct": round(100*(volume(g["shin_raw"])-volume(g["shin"]))/volume(g["shin_raw"]), 2),
        "source_cuff_retained_pct": round(100*volume(g["crescent"])/volume(g["cuff_raw"]), 2),
        "motor_vs_unmodified_thigh_mm3": round(overlap(g["motor_envelope"], g["thigh_raw"]), 3),
        "motor_vs_relief_thigh_mm3": round(overlap(g["motor_envelope"], g["thigh"]), 3),
        "carrier_to_unmodified_thigh_contact_mm3": round(overlap(g["carrier"], g["thigh_raw"]), 3),
        "neutral_mechanism_outside_source_outer_volume_mm3": {},
        "yoke_to_trimmed_shin_join_mm3": round(overlap(g["yoke"],g["shin"]),3),
        "poses": []}
    # Intended contacts in the drive, bearing and print-to-skin mount are
    # documented separately. All other fixed/moving contact is a hard fault.
    for name in ("carrier", "motor_envelope", "bearing", "shim", "yoke"):
        report["neutral_mechanism_outside_source_outer_volume_mm3"][name] = round(
            volume(g[name] - g["original_outer"]), 2)
    for deg in POSES:
        moving_shell = turn(g["shin"], deg)
        moving_yoke = turn(g["yoke"], deg)
        original_shin = turn(g["shin_raw"], deg)
        old_hit, old_gap = hard_contact(g["thigh_raw"], original_shin)
        cover_hit, cover_gap = hard_contact(g["fixed_cover"], moving_shell)
        structure_hit, structure_gap = hard_contact(compound((g["carrier"], g["motor_envelope"])), moving_yoke)
        structure_shin_hit, structure_shin_gap = hard_contact(
            compound((g["carrier"], g["motor_envelope"])), moving_shell)
        yoke_cover_hit, yoke_cover_gap = hard_contact(g["fixed_cover"], moving_yoke)
        report["poses"].append({"flexion_deg": deg,
            "original_thigh_vs_original_shin_overlap_mm3": old_hit,
            "original_thigh_vs_original_shin_gap_mm": old_gap,
            "relieved_cover_vs_moving_shin_overlap_mm3": cover_hit,
            "relieved_cover_vs_moving_shin_gap_mm": cover_gap,
            "fixed_structure_vs_yoke_overlap_mm3": structure_hit,
            "fixed_structure_vs_yoke_gap_mm": structure_gap,
            "fixed_structure_vs_shin_overlap_mm3": structure_shin_hit,
            "fixed_structure_vs_shin_gap_mm": structure_shin_gap,
            "relieved_cover_vs_yoke_overlap_mm3": yoke_cover_hit,
            "relieved_cover_vs_yoke_gap_mm": yoke_cover_gap,
            "hard_collision_pass": max(cover_hit, structure_hit, structure_shin_hit,
                                       yoke_cover_hit) < .01})
        scene = trimesh.Scene()
        for name, color in (("thigh", (227, 218, 194, 235)),
                            ("crescent", (72, 73, 69, 245)),
                            ("carrier", (62, 91, 87, 255)),
                            ("motor_envelope", (75, 121, 160, 190)),
                            ("bearing", (157, 153, 142, 255)),
                            ("shim", (235, 175, 67, 255))):
            add(scene, name, g[name], color)
        add(scene, "shin", moving_shell, (230, 213, 176, 235))
        add(scene, "yoke", moving_yoke, (186, 103, 62, 255))
        scene.export(OUT / f"knee-{deg:03d}.glb")
    for name in ("carrier", "motor_envelope", "bearing", "shim", "yoke", "shin", "thigh", "crescent"):
        bd.export_step(g[name], OUT / f"{name}.step")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"removed_thigh_pct": report["source_thigh_shell_removed_pct"],
                      "removed_shin_pct": report["source_shin_shell_removed_pct"],
                      "poses": report["poses"]}, indent=2))


if __name__ == "__main__":
    main()
