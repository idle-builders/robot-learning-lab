"""Build Family 1: Parametric FDM Space-Truss & Ribbed Box-Section Lower-Leg Connectors.

Generates native STEP and STL files for all 3 lower-leg stages (bilateral R and L):
  1. Stage 1 (Shin): R/L_shin_truss_box + 4 steel compression sleeves
  2. Stage 2 (Ankle Coupler): R/L_ankle_coupler_truss_box + 10 steel compression sleeves
  3. Stage 3 (Foot Mount & Sole): R/L_foot_monocoque_truss_box + 4 steel compression sleeves
"""
from pathlib import Path
import math
from build123d import (
    Axis,
    Box,
    Circle,
    Compound,
    Cylinder,
    Edge,
    Face,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Solid,
    Vector,
    Wire,
    export_step,
    export_stl,
    extrude,
)
from common import (
    ART,
    MATERIALS,
    PRINTED,
    ALT_PRINTED,
    rel,
    sha,
    write_json,
    ref,
    fillet_2d_face,
    bevel_3d_edges,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    add_shin_shell_mounts,
)


def box_pt(a, b):
    return Box(*(b[i] - a[i] for i in range(3))).translate(tuple(0.5 * (a[i] + b[i]) for i in range(3)))


def cyl_x(r: float, x0: float, x1: float, y: float, z: float):
    return Cylinder(r, abs(x1 - x0)).rotate(Axis.Y, 90).translate((0.5 * (x0 + x1), y, z))


def cyl_y(r: float, y0: float, y1: float, x: float, z: float):
    return Cylinder(r, abs(y1 - y0)).rotate(Axis.X, 90).translate((x, 0.5 * (y0 + y1), z))


def cyl_z(r: float, z0: float, z1: float, x: float, y: float):
    return Cylinder(r, abs(z1 - z0)).translate((x, y, 0.5 * (z0 + z1)))


def strut_3d(p0, p1, width_y: float, height_z: float):
    """Create a 3D rectangular prism strut connecting p0 to p1."""
    v0 = Vector(*p0)
    v1 = Vector(*p1)
    dv = v1 - v0
    length = dv.length
    mid = (v0 + v1) * 0.5
    # Use a box aligned along X then rotated into dv
    dx, dy, dz = dv.X, dv.Y, dv.Z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    b = Box(length, width_y, height_z)
    b = b.rotate(Axis.Y, pitch).rotate(Axis.Z, yaw).translate((mid.X, mid.Y, mid.Z))
    return b


def poly_prism_x(pts_yz: list[tuple[float, float]], x0: float, x1: float):
    edges = [Edge.make_line((x0, pts_yz[i][0], pts_yz[i][1]), (x0, pts_yz[(i + 1) % len(pts_yz)][0], pts_yz[(i + 1) % len(pts_yz)][1])) for i in range(len(pts_yz))]
    return Solid.extrude(Face(Wire(edges)), Vector(x1 - x0, 0, 0))


def poly_prism_y(pts_xz: list[tuple[float, float]], y0: float, y1: float):
    edges = [Edge.make_line((pts_xz[i][0], y0, pts_xz[i][1]), (pts_xz[(i + 1) % len(pts_xz)][0], y0, pts_xz[(i + 1) % len(pts_xz)][1])) for i in range(len(pts_xz))]
    return Solid.extrude(Face(Wire(edges)), Vector(0, y1 - y0, 0))


def build_shin_truss_box():
    """Build Right Stage 1 Shin Space-Truss & Box-Section Connector + 16 Steel Sleeves
    with 100% symmetric circular hubs, 4 screws per interface (zero figure-8 holes), and 3D edge bevels."""
    # 1. Symmetric Inner cheek (X in [39.25, 41.25]) and Outer cheek (X in [78.75, 81.25])
    # Centered around Y = 4.0 mm (Y in [-9.5, 17.5]), united with Circle(10.5) at (4.0, 123.0)
    ch_poly = Polygon(
        (-9.5, 66.0),
        (17.5, 66.0),
        (17.5, 98.0),
        (14.5, 110.0),
        (14.5, 123.0),
        (-6.5, 123.0),
        (-6.5, 110.0),
        (-9.5, 98.0),
        align=None,
    )
    ch_union = (ch_poly + Pos(4.0, 123.0) * Circle(10.5)).clean().faces()[0]
    ch_face = fillet_2d_face(ch_union, 2.0, skip_xy=[(-9.5, 66.0), (17.5, 66.0), (-9.5, 98.0)], r_inner=3.0)
    tri_cut = fillet_2d_face(Polygon((-2.5, 75.5), (10.5, 75.5), (4.0, 86.5), align=None).faces()[0], 2.2)
    ch_face = (ch_face - tri_cut).clean().faces()[0]
    inner_cheek = extrude_xy_to_x(ch_face, 39.25, 41.25)
    outer_cheek = extrude_xy_to_x(ch_face, 78.75, 81.25)

    # 2. Integral cylindrical motor case standoffs for ALL 8 PA2 bolts (4 inner cheek + 4 outer cheek)
    standoffs = (
        cyl_x(3.0, 39.25, 44.0, -6.25, 69.30)
        + cyl_x(3.0, 39.25, 44.0, -6.25, 93.75)
        + cyl_x(3.0, 39.25, 44.0, 14.25, 69.30)
        + cyl_x(3.0, 39.25, 44.0, 14.25, 93.75)
        + cyl_x(3.0, 76.0, 81.25, -6.25, 69.30)
        + cyl_x(3.0, 76.0, 81.25, -6.25, 90.00)
        + cyl_x(3.0, 76.0, 81.25, 14.25, 69.30)
        + cyl_x(3.0, 76.0, 81.25, 14.25, 90.00)
    )

    # 3. Clean 2D-filleted Space-Truss & Box-Beam Bridge across X in [41.25, 78.75] (Y in [-9.5, -2.4], Z in [96.8, 108.2])
    ub_outer = Polygon((41.25, -9.5), (78.75, -9.5), (78.75, -2.4), (41.25, -2.4), align=None).faces()[0]
    ub_w1 = Pos(51.5, -5.95) * RectangleRounded(13.0, 3.0, 1.4).faces()[0]
    ub_w2 = Pos(68.5, -5.95) * RectangleRounded(13.0, 3.0, 1.4).faces()[0]
    upper_bridge = extrude_xy_to_z((ub_outer - ub_w1 - ub_w2).clean().faces()[0], 96.8, 108.2)

    # Anterior (-Y) K-truss arch in front of Ankle Pitch motor case (Y in [-9.5, -8.55], Z in [72.5, 108.2])
    fa_outer = Polygon((41.25, 72.5), (78.75, 72.5), (78.75, 108.2), (41.25, 108.2), align=None).faces()[0]
    fa_cut1 = fillet_2d_face(Polygon((46.5, 76.2), (73.5, 76.2), (60.0, 94.5), align=None).faces()[0], 2.4)
    fa_cut2 = fillet_2d_face(Polygon((44.0, 80.0), (44.0, 94.5), (55.5, 94.5), align=None).faces()[0], 2.0)
    fa_cut3 = fillet_2d_face(Polygon((76.0, 80.0), (76.0, 94.5), (64.5, 94.5), align=None).faces()[0], 2.0)
    front_arch = extrude_xy_to_y((fa_outer - fa_cut1 - fa_cut2 - fa_cut3).clean().faces()[0], -9.5, -8.55)
    front_ear_gusset_in = box_pt((41.25, -9.5, 67.2), (44.0, -4.5, 76.2))
    front_ear_gusset_out = box_pt((76.0, -9.5, 67.2), (78.75, -4.5, 76.2))

    # Upper cheek-to-bridge gussets with 2D-filleted corners
    kg_f = fillet_2d_face(
        Polygon((-9.2, 96.8), (-2.4, 96.8), (-2.4, 112.0), (-6.5, 112.0), align=None).faces()[0],
        1.5,
        skip_xy=[(-9.2, 96.8), (-2.4, 96.8)],
    )
    knee_gusset_in = extrude_xy_to_x(kg_f, 39.25, 42.3)
    knee_gusset_out = extrude_xy_to_x(kg_f, 77.8, 81.25)

    body = (
        inner_cheek
        + outer_cheek
        + standoffs
        + upper_bridge
        + front_arch
        + front_ear_gusset_in
        + front_ear_gusset_out
        + knee_gusset_in
        + knee_gusset_out
    )

    # 4. Exact functional cutouts, bores, and counterbores
    body -= cyl_x(4.0, 38.0, 83.0, 4.0, 123.0)
    body -= cyl_x(11.0, 38.0, 83.0, 4.0, 61.0)
    body -= box_pt((44.0, -8.55, 50.5), (76.0, 16.55, 96.5))
    body -= box_pt((41.25, -3.5, 50.5), (78.75, 11.5, 96.5))

    sleeves = []
    # ALL 4 M3 bolts + 4 sleeves on Knee Main Horn (X in [39.25, 41.25]) AND ALL 4 M3 bolts + 4 sleeves on Knee Rear Idler (X in [78.75, 80.75])
    for y_k, z_k in [(4.0, 116.0), (4.0, 130.0), (-3.0, 123.0), (11.0, 123.0)]:
        body -= cyl_x(2.25, 39.1, 41.4, y_k, z_k)
        body -= cyl_x(2.25, 78.6, 80.85, y_k, z_k)
        body -= cyl_x(3.0, 80.75, 82.0, y_k, z_k)
        s_in = cyl_x(2.25, 39.25, 41.25, y_k, z_k) - cyl_x(1.6, 39.1, 41.4, y_k, z_k)
        s_out = cyl_x(2.25, 78.75, 80.75, y_k, z_k) - cyl_x(1.6, 78.6, 80.9, y_k, z_k)
        sleeves.extend([s_in, s_out])

    # Separate 4 PA2 bolts + 4 sleeves on Inner Ankle Pitch Cheek (X in [39.25, 44.0]) — NO figure-8 holes!
    for y_c, z_c in [(-6.25, 69.30), (-6.25, 93.75), (14.25, 69.30), (14.25, 93.75)]:
        body -= cyl_x(1.80, 39.1, 44.1, y_c, z_c)
        s_in_c = cyl_x(1.80, 39.25, 44.0, y_c, z_c) - cyl_x(1.1, 39.1, 44.1, y_c, z_c)
        sleeves.append(s_in_c)

    # Separate 4 PA2 bolts + 4 sleeves on Outer Ankle Pitch Cheek (X in [76.0, 80.75]) — NO figure-8 holes!
    for y_c, z_c in [(-6.25, 69.30), (-6.25, 90.00), (14.25, 69.30), (14.25, 90.00)]:
        body -= cyl_x(1.80, 75.9, 80.85, y_c, z_c)
        body -= cyl_x(2.25, 80.75, 82.0, y_c, z_c)
        s_out_c = cyl_x(1.80, 76.0, 80.75, y_c, z_c) - cyl_x(1.1, 75.9, 80.85, y_c, z_c)
        sleeves.append(s_out_c)

    # 5. Integrate the 2 hard-shell mounting posts, Ø3.4 mm through-bores, and captive M3 nut pockets (`shin_front` & `shin_back`)
    body = add_shin_shell_mounts(body)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves)


def build_ankle_coupler_truss_box():
    """Build Right Stage 2 Ankle Coupler 3D Triangulated Box-Truss Yoke + 12 Steel Sleeves
    with 4 M3 bolts on Pitch Horn, 4 M3 bolts on Pitch Idler, 4 M3 bolts on Roll Horn,
    full multi-pose clearance inside `shin_front`/`shin_back` across -35..+35 deg, and 3D edge bevels."""
    # 1. Cup-Rimmed Upper Main Pitch Horn Hub + Deep-Beam Lateral Buttress (X in [74.5, 81.2], centered at Y=4, Z=61)
    mh_f = fillet_2d_face(
        (
            Polygon((-2.0, 50.5), (3.5, 33.6), (20.6, 33.6), (20.6, 50.0), (14.5, 55.5), (14.5, 61.0), (-6.5, 61.0), align=None)
            + Pos(4.0, 61.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(3.5, 33.6), (20.6, 33.6), (20.6, 50.0)],
        r_inner=2.5,
    )
    main_hub = extrude_xy_to_x(mh_f, 74.5, 81.2) - cyl_x(9.90, 74.0, 79.2, 4.0, 61.0) - box_pt((74.0, -10.0, 48.0), (76.8, 16.65, 75.0))

    # 2. Cup-Rimmed Symmetric Upper Rear Idler Hub + Deep Strut (X in [39.95, 45.0], centered at Y=4, Z=61) supporting all 4 M3 bolts!
    ih_f = fillet_2d_face(
        (
            Polygon((-2.0, 50.5), (6.5, 41.0), (20.6, 41.0), (20.6, 50.0), (14.5, 55.5), (14.5, 61.0), (-6.5, 61.0), align=None)
            + Pos(4.0, 61.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(6.5, 41.0), (20.6, 41.0), (20.6, 50.0)],
        r_inner=2.5,
    )
    idler_hub = extrude_xy_to_x(ih_f, 39.95, 45.0) - cyl_x(9.90, 41.95, 45.3, 4.0, 61.0)

    # 3. 3D Closed/Ribbed Box-Truss Arch + Lower Roll Horn Hub (Y in [16.65, 20.6], centered at X=75, Z=19)
    aw_f = fillet_2d_face(
        (
            Polygon((39.95, 41.0), (64.5, 29.0), (64.5, 19.0), (81.2, 19.0), (81.2, 50.0), (39.95, 50.0), align=None)
            + Pos(75.0, 19.0) * Circle(10.5)
        ).clean().faces()[0],
        2.0,
        skip_xy=[(39.95, 41.0), (39.95, 50.0), (81.2, 50.0)],
        r_inner=3.0,
    )
    p1 = fillet_2d_face(Polygon((43.5, 44.5), (56.5, 39.5), (56.5, 47.8), align=None).faces()[0], 1.4)
    p2 = fillet_2d_face(Polygon((59.5, 38.5), (72.0, 47.8), (59.5, 47.8), align=None).faces()[0], 1.4)
    p3 = fillet_2d_face(Polygon((62.0, 36.8), (74.5, 36.8), (74.5, 46.4), align=None).faces()[0], 1.4)
    arch_truss = (
        extrude_xy_to_y(aw_f, 16.65, 20.6)
        - extrude_xy_to_y(p1, 19.7, 22.0)
        - extrude_xy_to_y(p2, 19.7, 22.0)
        - extrude_xy_to_y(p3, 19.7, 22.0)
    )
    roll_hub_cap = cyl_y(10.5, 16.65, 20.8, 75.0, 19.0)

    body = main_hub + idler_hub + arch_truss + roll_hub_cap

    # 5. Cut exact horn central windows and 12 steel sleeve seats (4 Pitch Horn + 4 Pitch Idler + 4 Roll Horn)
    body -= cyl_y(9.90, 16.0, 18.0, 75.0, 19.0)
    body -= cyl_x(4.5, 37.0, 85.0, 4.0, 61.0)
    body -= cyl_y(4.5, 16.0, 23.5, 75.0, 19.0)
    body -= box_pt((45.0, -12.0, 46.5), (76.8, 16.65, 75.0))
    body -= box_pt((37.0, -12.0, 5.0), (74.5, 16.65, 34.5))
    body -= box_pt((74.5, -12.0, 5.0), (85.0, 16.65, 33.6))

    sleeves = []
    for y_p, z_p in [(4.0, 54.0), (4.0, 68.0), (-3.0, 61.0), (11.0, 61.0)]:
        body -= cyl_x(2.25, 74.0, 83.5, y_p, z_p)
        s = cyl_x(2.25, 79.2, 81.2, y_p, z_p) - cyl_x(1.6, 79.0, 83.5, y_p, z_p)
        sleeves.append(s)

    for y_i, z_i in [(4.0, 54.0), (4.0, 68.0), (-3.0, 61.0), (11.0, 61.0)]:
        body -= cyl_x(2.25, 38.0, 45.5, y_i, z_i)
        s = cyl_x(2.25, 39.95, 41.95, y_i, z_i) - cyl_x(1.6, 38.0, 42.1, y_i, z_i)
        sleeves.append(s)

    for x_r, z_r in [(68.0, 19.0), (82.0, 19.0), (75.0, 12.0), (75.0, 26.0)]:
        body -= cyl_y(2.25, 16.0, 22.5, x_r, z_r)
        s = cyl_y(2.25, 18.0, 20.8, x_r, z_r) - cyl_y(1.6, 17.8, 22.5, x_r, z_r)
        sleeves.append(s)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves)


def build_foot_monocoque_truss_box():
    """Build Right Stage 3 Foot Monocoque Isogrid Sole & Motor Saddle Cradle + 4 Steel Sleeves with 3D edge bevels."""
    sole_outer_full = extrude(RectangleRounded(60.0, 86.0, 22.0), 14.0).translate((67.0, -12.0, 2.0))

    # 1. Continuous flat bottom skin at Z in [2.0, 3.6] (uninterrupted Z=2.0 datum for TPU pad & 3 sensors)
    bottom_skin = extrude(RectangleRounded(60.0, 86.0, 22.0), 1.6).translate((67.0, -12.0, 2.0))

    # 2. 2D-filleted Isogrid Sole Web, Perimeter Rim & Sensor Backing Pads (Z in [2.0, 5.6])
    sole_2d = (Pos(67.0, -12.0) * RectangleRounded(60.0, 86.0, 22.0)).faces()[0]
    pockets_2d = [
        Pos(54.5, -38.5) * RectangleRounded(13.0, 17.0, 4.0).faces()[0],
        Pos(54.5, 10.5) * RectangleRounded(13.0, 12.0, 4.0).faces()[0],
        Pos(66.0, -1.5) * RectangleRounded(9.5, 16.0, 3.5).faces()[0],
        Pos(81.5, -13.0) * RectangleRounded(15.0, 22.0, 4.5).faces()[0],
    ]
    for pk in pockets_2d:
        sole_2d = (sole_2d - pk).clean().faces()[0]
    sole_isogrid = extrude_xy_to_z(sole_2d, 2.0, 5.6)

    # 3. Integrated Front & Rear A-Frame Motor Saddles with 2D-filleted buttress gussets + Medial Box-Wall Bridge
    front_wall = box_pt((38.5, 17.55, 2.0), (69.5, 20.55, 14.5))
    front_bosses = cyl_y(3.0, 14.80, 20.55, 46.0, 8.75) + cyl_y(3.0, 14.80, 20.55, 66.7, 8.75)
    fg_f = fillet_2d_face(
        Polygon((19.5, 4.5), (27.5, 4.5), (27.5, 6.5), (20.55, 14.5), (19.5, 14.5), align=None).faces()[0],
        1.5,
        skip_xy=[(19.5, 4.5), (27.5, 4.5), (19.5, 14.5), (20.55, 14.5)],
    )
    front_gussets = extrude_xy_to_x(fg_f, 38.5, 42.0) + extrude_xy_to_x(fg_f, 53.5, 59.0) + extrude_xy_to_x(fg_f, 61.5, 63.8)

    rear_wall = box_pt((38.5, -22.95, 2.0), (69.5, -19.95, 14.5))
    rear_bosses = cyl_y(3.0, -22.95, -17.20, 42.25, 8.75) + cyl_y(3.0, -22.95, -17.20, 66.7, 8.75)
    rg_f = fillet_2d_face(
        Polygon((-19.95, 4.5), (-28.5, 4.5), (-28.5, 6.5), (-22.95, 14.5), (-19.95, 14.5), align=None).faces()[0],
        1.5,
        skip_xy=[(-19.95, 4.5), (-28.5, 4.5), (-19.95, 14.5), (-22.95, 14.5)],
    )
    rear_gussets = extrude_xy_to_x(rg_f, 46.0, 50.0) + extrude_xy_to_x(rg_f, 55.0, 59.5) + extrude_xy_to_x(rg_f, 61.2, 63.8)
    medial_bridge = box_pt((37.2, -22.95, 2.0), (39.6, 20.55, 13.5))

    body = (
        bottom_skin
        + sole_isogrid
        + front_wall
        + front_bosses
        + front_gussets
        + rear_wall
        + rear_bosses
        + rear_gussets
        + medial_bridge
    ) & sole_outer_full

    # 4. Cut exact motor horn clearance cylinder (R = 11.2 mm around X=75, Z=19) and fastener bores/sleeves
    body -= cyl_y(11.2, -35.0, 35.0, 75.0, 19.0)
    body -= box_pt((39.75, -17.2, 6.3), (85.5, 14.8, 32.0))
    body -= box_pt((40.8, 14.8, 11.85), (78.5, 17.55, 28.0))
    body -= box_pt((45.3, -19.95, 11.85), (78.5, -17.2, 29.0))

    sleeves = []
    for x_f in [46.0, 66.7]:
        body -= cyl_y(1.9, 14.7, 19.65, x_f, 8.75)
        body -= cyl_y(2.4, 19.55, 29.0, x_f, 8.75)
        s = cyl_y(1.9, 14.80, 19.55, x_f, 8.75) - cyl_y(1.1, 14.7, 19.65, x_f, 8.75)
        sleeves.append(s)

    for x_r in [42.25, 66.7]:
        body -= cyl_y(1.9, -22.05, -17.1, x_r, 8.75)
        body -= cyl_y(2.4, -33.0, -21.95, x_r, 8.75)
        s = cyl_y(1.9, -21.95, -17.20, x_r, 8.75) - cyl_y(1.1, -22.05, -17.1, x_r, 8.75)
        sleeves.append(s)

    return bevel_3d_edges(body.clean(), r_outer=0.40, r_inner=0.75), Compound(children=sleeves)


def export_family():
    out_dir = ART / "truss_box" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)

    shin_r, shin_sleeves_r = build_shin_truss_box()
    coupler_r, coupler_sleeves_r = build_ankle_coupler_truss_box()
    foot_r, foot_sleeves_r = build_foot_monocoque_truss_box()

    rho_poly = PRINTED["density_g_cm3"] * 1e-3  # g/mm^3
    rho_steel = MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3  # g/mm^3

    parts_meta = []
    for stage_key, base_name, link_suffix, role, poly_r, slv_r in [
        ("stage1_shin", "shin_truss_box", "shin", "frame", shin_r, shin_sleeves_r),
        ("stage2_ankle_coupler", "ankle_coupler_truss_box", "ankle_pitch", "coupler", coupler_r, coupler_sleeves_r),
        ("stage3_foot_monocoque", "foot_monocoque_truss_box", "foot", "frame", foot_r, foot_sleeves_r),
    ]:
        for side, sgn in [("R", 1), ("L", -1)]:
            poly = poly_r if sgn == 1 else poly_r.mirror(Plane.YZ)
            slv = slv_r if sgn == 1 else slv_r.mirror(Plane.YZ)

            p_name = f"{side}_{base_name}"
            s_name = f"{side}_{base_name}_steel_sleeves"

            p_comp = Compound(children=list(poly.solids()))
            p_comp.label = p_name
            p_step = out_dir / f"{p_name}.step"
            p_stl = out_dir / f"{p_name}.stl"
            export_step(p_comp, p_step)
            export_stl(p_comp, p_stl, tolerance=0.04, angular_tolerance=0.1)

            s_comp = Compound(children=list(slv.solids()))
            s_comp.label = s_name
            s_step = out_dir / f"{s_name}.step"
            s_stl = out_dir / f"{s_name}.stl"
            export_step(s_comp, s_step)
            export_stl(s_comp, s_stl, tolerance=0.04, angular_tolerance=0.1)

            bb = p_comp.bounding_box()
            s_bb = s_comp.bounding_box()
            poly_vol = float(poly.volume)
            slv_vol = float(slv.volume)
            poly_mass_g = poly_vol * rho_poly
            slv_mass_g = slv_vol * rho_steel

            parts_meta.append({
                "stage": stage_key,
                "side": side,
                "name": p_name,
                "owning_link": f"{side}_{link_suffix}",
                "role": role,
                "native_step": rel(p_step),
                "native_step_sha256": sha(p_step),
                "mesh": rel(p_stl),
                "mesh_sha256": sha(p_stl),
                "native_valid": all(s.is_valid for s in p_comp.solids()),
                "native_solids": len(p_comp.solids()),
                "volume_mm3": round(poly_vol, 3),
                "polymer_mass_g_PAHT_CF": round(poly_mass_g, 3),
                "polymer_mass_g_PETG_CF": round(poly_vol * ALT_PRINTED["density_g_cm3"] * 1e-3, 3),
                "bounds_mm": [
                    [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)],
                    [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)],
                ],
                "sleeves": {
                    "name": s_name,
                    "native_step": rel(s_step),
                    "native_step_sha256": sha(s_step),
                    "mesh": rel(s_stl),
                    "mesh_sha256": sha(s_stl),
                    "count": len(s_comp.solids()),
                    "volume_mm3": round(slv_vol, 3),
                    "mass_g": round(slv_mass_g, 3),
                    "bounds_mm": [
                        [round(s_bb.min.X, 3), round(s_bb.min.Y, 3), round(s_bb.min.Z, 3)],
                        [round(s_bb.max.X, 3), round(s_bb.max.Y, 3), round(s_bb.max.Z, 3)],
                    ],
                },
                "total_stage_mass_g_PAHT_CF": round(poly_mass_g + slv_mass_g, 3),
            })

    summary_path = ART / "truss_box" / "cad-manifest.json"
    write_json(summary_path, {
        "family": "cove50-lower-leg-fdm-truss-box",
        "version": "r1",
        "description": "Parametric FDM Space-Truss, Ribbed Box-Section & Isogrid Monocoque Lower-Leg Connectors with Steel Compression Sleeves",
        "parts": parts_meta,
        "source_code": ref(Path(__file__)),
    })
    for row in parts_meta:
        if row["side"] == "R":
            print(
                f"{row['name']:32s} | valid={row['native_valid']} solids={row['native_solids']} "
                f"| poly={row['polymer_mass_g_PAHT_CF']:6.2f} g + sleeves={row['sleeves']['mass_g']:5.2f} g "
                f"= {row['total_stage_mass_g_PAHT_CF']:6.2f} g | bounds={row['bounds_mm']}"
            )


if __name__ == "__main__":
    export_family()
