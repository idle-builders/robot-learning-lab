"""Exterior-first moving knee skin for the exact 350 mm sketch ape.

The source right thigh and shin are retained away from the knee. Their ends
are locally shaped around a smooth moving collar. This is an appearance and
skin-motion coupon, not an actuator, bearing, or structural design.

Run: .venv/bin/python experiments/compact_morphology/ape_design/lightweight_ape_knee_skin_form.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import build123d as bd
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/lightweight-ape-20261004/source-shell-1mm/ape_shell_assembly.step"
OUT = ROOT / "artifacts/lightweight-ape-knee-skin-form-20261005"
AXIS = (42.0, 3.0, 99.0)  # Kinematic drawing datum only; no drive is selected.
POSES = (0, 60, 115)
COLORS = {
    "thigh": (226, 221, 210, 255),
    "shin": (226, 216, 198, 255),
    "collar": (219, 211, 194, 255),
    "recess": (67, 67, 63, 255),
}


def box(x0, x1, y0, y1, z0, z1):
    return bd.Solid.make_box(x1-x0, y1-y0, z1-z0,
                             bd.Plane(origin=(x0, y0, z0)))


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape else 0.0


def overlap(a, b):
    return sum(volume(sa & sb) for sa in a.solids() for sb in b.solids())


def turn(shape, angle):
    return shape.rotate(bd.Axis(AXIS, (1, 0, 0)), angle)


def rotated_point(point, angle):
    x, y, z = point
    a = math.radians(angle)
    dy, dz = y-AXIS[1], z-AXIS[2]
    return np.array((x, AXIS[1] + dy*math.cos(a) - dz*math.sin(a),
                     AXIS[2] + dy*math.sin(a) + dz*math.cos(a)))


def ellipsoid(center, radii):
    return bd.Solid.make_sphere(1).scale(radii).translate(center)


def ellipse_loft(sections):
    return bd.Solid.make_loft([bd.Wire.make_ellipse(rx, ry,
        bd.Plane(origin=(cx, cy, z))) for z, cx, cy, rx, ry in sections])


def rear_rising_cut():
    """Retain thigh above a sculpted lower edge, high only at the rear."""
    edge_yz = [(-60, 96), (-21, 96), (-13, 96), (-2, 100),
               (10, 105), (22, 109), (60, 109)]
    points = [(0, y, z) for y, z in edge_yz]
    points += [(0, 60, 190), (0, -60, 190)]
    wire = bd.Wire.make_polygon(points, close=True)
    return bd.Solid.extrude(bd.Face(wire), (85, 0, 0))


def source_parts():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Build the exact source shell first: {SOURCE}")
    src = {s.label: s for s in bd.import_step(SOURCE).children}
    thigh_src = bd.Compound(children=[src[n] for n in ("R_thigh_front", "R_thigh_back")])
    shin_src = bd.Compound(children=[src[n] for n in ("R_shin_front", "R_shin_back")])
    # This cut reshapes the actual source panels. The legacy cuff is replaced,
    # not left underneath the new design or counted as a cover.
    thigh = bd.Compound(children=[s & rear_rising_cut() for s in thigh_src.solids()])
    shin = shin_src & box(0, 85, -60, 60, 20, 82.5)
    # A shallow, hollow moving dome receives the shin rim at its lower pole.
    # Circular Y/Z radii keep its exterior stationary under pitch, while X
    # remains narrow enough for the source thigh's broad rounded cheeks.
    collar = (ellipsoid(AXIS, (17.2, 18.2, 18.2))
              - ellipsoid(AXIS, (16.2, 17.2, 17.2)))
    collar = collar & box(0, 85, -60, 60, 81.3, 116.4)
    # Locally rebuilt top of the shin: this connects its actual source rim to
    # the rotating cap. The old flat, open end is not left on display.
    neck_outer = ellipse_loft([(81.5,42,3,16,17), (86,42,3,16,16.8),
                               (91,42,3,16.2,16.8), (95,42,3,16.5,17)])
    neck_inner = ellipse_loft([(81.4,42,3,15,16), (86,42,3,15,15.8),
                               (91,42,3,15.2,15.8), (95.1,42,3,15.5,16)])
    neck = neck_outer - neck_inner
    # Dark inner reveal is a recessed surface allocation. It has no fastener
    # or load claim. Its protrusion/possible collision is checked below.
    recess = (ellipsoid(AXIS, (16.0, 17.0, 17.0))
              - ellipsoid(AXIS, (15.2, 16.2, 16.2)))
    recess = recess & box(0, 85, 13, 60, 83, 115)
    return thigh_src, shin_src, src["R_knee_cuff"], thigh, shin, neck, collar, recess


def mesh(shape, color):
    verts, faces = shape.tessellate(.4, .25)
    m = trimesh.Trimesh(np.asarray([tuple(v) for v in verts]),
                        np.asarray(faces), process=False)
    m.visual.face_colors = color
    return m


def gusset_anchors(angle):
    """Three rear/cheek material paths, tracked in fixed and moving frames."""
    # One path on each rear cheek and one across the centre back. These are
    # material line allocations, not an isotropic TPU strain calculation.
    fixed = [(32., 24., 108.), (42., 25., 109.), (52., 24., 108.)]
    moving = [(32., 18., 103.), (42., 20., 103.), (52., 18., 103.)]
    return [(np.array(a), rotated_point(b, angle)) for a, b in zip(fixed, moving)]


def export_scene(degrees, thigh, shin, neck, collar, recess):
    scene = trimesh.Scene()
    fixed = [("thigh", thigh)]
    moving = [("shin", turn(shin, degrees)),
              ("shin", turn(neck, degrees)),
              ("collar", turn(collar, degrees)),
              ("recess", turn(recess, degrees))]
    for name, shape in fixed+moving:
        for i, solid in enumerate(shape.solids()):
            if solid.volume > .01:
                scene.add_geometry(mesh(solid, COLORS[name]),
                                   geom_name=f"{name}-{i}", node_name=f"{name}-{i}")
    # A filled rear sheet would cosmetically hide an unproved 115° path. Keep
    # the hard-skin geometry visible and report the required material travel.
    scene.export(OUT / f"knee-skin-{degrees:03d}.glb")
    # STEP includes the exact rigid skin. The conceptual flex sheet is GLB
    # only because a pose-specific surface is not a developed solid.
    labeled = []
    for name, shape in fixed+moving:
        for i, solid in enumerate(shape.solids()):
            solid.label = f"{name}_{i}"
            labeled.append(solid)
    bd.export_step(bd.Compound(children=labeled), OUT / f"knee-skin-{degrees:03d}.step")
    return scene


def project(meshes, eye, view):
    width, height = 580, 690
    scale = .275 if view == "side" else .26 if view == "threequarter" else .24
    eye = np.asarray(eye, dtype=float); eye /= np.linalg.norm(eye)
    right = np.cross((0., 0., 1.), eye); right /= np.linalg.norm(right)
    up = np.cross(eye, right)
    target = np.array((42., 25. if view in ("side", "threequarter") else 8., 92.))
    light = np.array((.35, -.52, .78)); light /= np.linalg.norm(light)
    polygons = []
    for m in meshes:
        verts = np.asarray(m.vertices)-target
        xy = np.stack((width/2 + verts@right/scale,
                       height/2 - verts@up/scale), axis=1)
        depth = verts@eye
        colors = np.asarray(m.visual.face_colors)
        for i, tri in enumerate(m.faces):
            p = xy[tri]
            if (np.all(p[:, 0] < -5) or np.all(p[:, 0] > width+5) or
                np.all(p[:, 1] < -5) or np.all(p[:, 1] > height+5)):
                continue
            lit = .73 + .27*max(0., float(m.face_normals[i]@light))
            color = tuple(int(np.clip(c*lit, 0, 255)) for c in colors[i, :3])
            polygons.append((float(depth[tri].mean()), p, color))
    polygons.sort(key=lambda row: row[0])
    image = Image.new("RGB", (width, height), "#e9e5dc")
    draw = ImageDraw.Draw(image)
    for _, pts, color in polygons:
        draw.polygon([tuple(p) for p in pts], fill=color)
    return image


def previews(degrees, scene, spans):
    views = {"front": (0, -1, .13), "side": (1, 0, .08),
             "rear": (0, 1, .13), "threequarter": (1, -1, .35)}
    meshes = list(scene.dump())
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 25)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    for view, eye in views.items():
        canvas = Image.new("RGB", (580, 778), "#f6f3ed")
        canvas.paste(project(meshes, eye, view), (0, 48))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 12), f"{degrees}°  {view}  |  moving knee skin", font=font, fill="#26312e")
        draw.text((20, 746), f"Rear path: {min(spans):.1f}–{max(spans):.1f} mm straight span",font=small,fill="#4d5650")
        canvas.save(OUT / f"preview-{degrees:03d}-{view}.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thigh_src, shin_src, cuff_src, thigh, shin, neck, collar, recess = source_parts()
    rows = []
    for angle in range(0, 116, 5):
        paths = gusset_anchors(angle)
        spans = [float(np.linalg.norm(a-b)) for a, b in paths]
        row = {"flex_deg": angle,
               "rear_material_straight_spans_mm": [round(s, 2) for s in spans]}
        if angle in POSES:
            moved_shin, moved_neck, moved_collar, moved_recess = (
                turn(s, angle) for s in (shin, neck, collar, recess))
            row.update({
                "thigh_shin_overlap_mm3": round(overlap(thigh, moved_shin), 3),
                "thigh_neck_overlap_mm3": round(overlap(thigh, moved_neck), 3),
                "thigh_collar_overlap_mm3": round(overlap(thigh, moved_collar), 3),
                "thigh_recess_overlap_mm3": round(overlap(thigh, moved_recess), 3),
            })
            scene = export_scene(angle, thigh, shin, neck, collar, recess)
            previews(angle, scene, spans)
        rows.append(row)
    neutral = rows[0]["rear_material_straight_spans_mm"]
    longest = [max(row["rear_material_straight_spans_mm"][i] for row in rows) for i in range(3)]
    shortest = [min(row["rear_material_straight_spans_mm"][i] for row in rows) for i in range(3)]
    report = {
        "intent": "Aesthetic-led moving exterior and material-path screen; no motor or mechanism selected",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_height_mm": 350, "source_wall_mm": 1,
        "kinematic_datum_axis_xyz_mm": AXIS,
        "source_thigh_volume_mm3": round(volume(thigh_src), 2),
        "retained_thigh_volume_mm3": round(volume(thigh), 2),
        "source_shin_volume_mm3": round(volume(shin_src), 2),
        "source_cuff_volume_mm3": round(volume(cuff_src), 2),
        "retained_shin_volume_mm3": round(volume(shin), 2),
        "moving_neck_volume_mm3": round(volume(neck), 2),
        "moving_collar_volume_mm3": round(volume(collar), 2),
        "recess_volume_mm3": round(volume(recess), 2),
        "gross_skin_volume_delta_mm3": round(
            sum(volume(s) for s in (thigh, shin, neck, collar, recess))
            - sum(volume(s) for s in (thigh_src, shin_src, cuff_src)), 2),
        "gross_skin_volume_delta_note": "Separate component volume sum includes overlapping joins; no net joined mass or flexible-sheet mass is claimed",
        "rear_path_neutral_mm": neutral,
        "rear_path_longest_mm": [round(x, 2) for x in longest],
        "rear_path_shortest_mm": [round(x, 2) for x in shortest],
        "illustrative_cut_length_with_2mm_allowance_mm": [round(x+2, 2) for x in longest],
        "rear_path_note": "The longest straight span is the strict developed-length lower bound. The extra 2 mm is only an illustrative allowance, not a validated routed pattern or hem.",
        "pose_rows": rows,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in report if k not in ("pose_rows",)}, indent=2))
    print("Worst measured rigid pair overlaps:", {key: max(r[key] for r in rows if key in r)
          for key in ("thigh_shin_overlap_mm3", "thigh_neck_overlap_mm3",
                      "thigh_collar_overlap_mm3", "thigh_recess_overlap_mm3")})


if __name__ == "__main__":
    main()
