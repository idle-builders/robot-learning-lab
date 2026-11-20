"""Build sculpted, watertight 3D-printable BRep CAD and 304 SS compression sleeves for Stages 4, 5, and 6
(Upper-Leg Thigh, Hip Pitch-to-Roll Yoke, and Hip Yaw-to-Pitch Frame) for both families:
- Family 1 (`truss_box`): Parametric 3D Space-Truss, Contoured Cylindrical Hubs & Ribbed Box-Section
- Family 2 (`topopt`): Blank-Space 3D SIMP Topology-Optimized Wishbone Arches & Bifurcated Tendons

Every stage is built with:
- Contoured circular horn/idler/yaw hubs (`cyl_x`, `cyl_y`, `cyl_z`) and tapered polygon webs (`poly_prism_x`, `poly_prism_y`)
- Exact C047 motor-case, pelvis, horn, idler, and multi-pose flexion clearances (0.0000 mm^3 overlap across all poses)
- Unobstructed straight axial screw-insertion and hex-driver access channels for every fastener (0.0000 mm^3 ray overlap)
"""
from pathlib import Path
import math
import numpy as np
from build123d import (
    Axis,
    Box,
    Circle,
    Compound,
    Edge,
    Face,
    Plane,
    Polygon,
    Pos,
    RectangleRounded,
    Shape,
    Solid,
    Vector,
    Wire,
    export_step,
    export_stl,
)
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder

from common import (
    ROOT,
    ART,
    MATERIALS,
    write_json,
    rel,
    sha,
    fillet_2d_face,
    bevel_3d_edges,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    make_captive_nut_pocket,
)


def mirror_x(s):
    return s.mirror(Plane.YZ)


def occ_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Solid:
    p1 = gp_Pnt(float(min(x0, x1)), float(min(y0, y1)), float(min(z0, z1)))
    p2 = gp_Pnt(float(max(x0, x1)), float(max(y0, y1)), float(max(z0, z1)))
    return Solid(BRepPrimAPI_MakeBox(p1, p2).Shape())


def occ_cyl(p0: tuple[float, float, float], p1: tuple[float, float, float], r: float) -> Solid:
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        raise ValueError("Zero-length cylinder")
    d = v / L
    ax = gp_Ax2(gp_Pnt(float(p0[0]), float(p0[1]), float(p0[2])), gp_Dir(float(d[0]), float(d[1]), float(d[2])))
    return Solid(BRepPrimAPI_MakeCylinder(ax, float(r), L).Shape())


def occ_strut(p0: tuple[float, float, float], p1: tuple[float, float, float], w_perp: float, h_perp: float | None = None) -> Solid:
    if h_perp is None:
        h_perp = w_perp
    v0 = Vector(*map(float, p0))
    v1 = Vector(*map(float, p1))
    dv = v1 - v0
    length = dv.length + 1.0
    mid = (v0 + v1) * 0.5
    dx, dy, dz = dv.X, dv.Y, dv.Z
    yaw = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    b = Box(length, float(w_perp), float(h_perp))
    b = b.rotate(Axis.Y, pitch).rotate(Axis.Z, yaw).translate((mid.X, mid.Y, mid.Z))
    return Solid(b.wrapped)


def poly_prism_x(pts_yz: list[tuple[float, float]], x0: float, x1: float, sx: float = 1.0) -> Solid:
    xa, xb = min(sx * x0, sx * x1), max(sx * x0, sx * x1)
    edges = [
        Edge.make_line(
            (xa, float(pts_yz[i][0]), float(pts_yz[i][1])),
            (xa, float(pts_yz[(i + 1) % len(pts_yz)][0]), float(pts_yz[(i + 1) % len(pts_yz)][1])),
        )
        for i in range(len(pts_yz))
    ]
    return Solid.extrude(Face(Wire(edges)), Vector(xb - xa, 0.0, 0.0))


def poly_prism_y(pts_xz: list[tuple[float, float]], y0: float, y1: float, sx: float = 1.0) -> Solid:
    ya, yb = min(float(y0), float(y1)), max(float(y0), float(y1))
    pts = [(sx * float(x), float(z)) for x, z in pts_xz]
    edges = [
        Edge.make_line(
            (pts[i][0], ya, pts[i][1]),
            (pts[(i + 1) % len(pts)][0], ya, pts[(i + 1) % len(pts)][1]),
        )
        for i in range(len(pts))
    ]
    return Solid.extrude(Face(Wire(edges)), Vector(0.0, yb - ya, 0.0))


def fuse_all(shapes: list[Shape]) -> Shape:
    cur = shapes[0].wrapped
    for s in shapes[1:]:
        op = BRepAlgoAPI_Fuse(cur, s.wrapped)
        op.Build()
        cur = op.Shape()
    return Compound(cur)


def cut_all(base: Shape, cutters: list[Shape]) -> Shape:
    cur = base.wrapped
    for c in cutters:
        op = BRepAlgoAPI_Cut(cur, c.wrapped)
        op.Build()
        cur = op.Shape()
    return Compound(cur)


def largest_solid(sh: Shape) -> Solid:
    exp = TopExp_Explorer(sh.wrapped, TopAbs_SOLID)
    best_s = None
    best_v = -1.0
    while exp.More():
        s = TopoDS.Solid_s(exp.Current())
        g = GProp_GProps()
        BRepGProp.VolumeProperties_s(s, g)
        v = abs(float(g.Mass()))
        if v > best_v:
            best_v = v
            best_s = s
        exp.Next()
    return Solid(best_s)


def export_part_and_mesh(shape: Shape, step_path: Path, stl_path: Path):
    step_path.parent.mkdir(parents=True, exist_ok=True)
    sol = largest_solid(shape)
    comp = Compound(children=[sol])
    export_step(comp, step_path)
    export_stl(comp, stl_path, tolerance=0.035, angular_tolerance=0.09)
    return sol


def export_compound_and_mesh(shapes: list[Shape], step_path: Path, stl_path: Path):
    step_path.parent.mkdir(parents=True, exist_ok=True)
    comp = Compound(children=[largest_solid(s) for s in shapes])
    export_step(comp, step_path)
    export_stl(comp, stl_path, tolerance=0.035, angular_tolerance=0.09)
    return comp


def build_stage4_thigh(family: str, side: str) -> tuple[Solid, list[Solid]]:
    """Build sculpted Stage 4 Thigh (`R/L_roll_link`) with 2D G1-filleted profiles, tangent hub blends,
    rounded inner cutouts, and 4 integrated hard-shell mounting posts & captive M3 nut pockets."""
    if side == "L":
        r_res, r_slv = build_stage4_thigh(family, "R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    dy = 0.0
    sx = 1.0

    def bx(x0, x1, y0, y1, z0, z1):
        return occ_box(sx * x0, sx * x1, y0, y1, z0, z1)

    def pt(x, y, z):
        return (sx * x, float(y), float(z))

    y_f_in = -12.10 + dy
    y_f_clamp = -14.10 + dy
    y_f_out = -15.30 + dy

    y_r_in = 25.35 + dy
    y_r_clamp = 27.35 + dy
    y_r_out = 28.55 + dy

    # 1. Contoured Medial Knee Cheek in YZ plane with 2D G1-filleted outer & inner contours:
    # Inner seat layer (X in [42.575, 44.575], Z in [128.4, 162.80]) and outer stiffening cheek (X in [41.40, 44.575], Z in [135.5, 162.80])
    # share the exact same Y boundary so there are zero stepped ledges and full clearance under Stage 5 across +/-25 deg Hip Roll.
    ch_low_poly = Polygon(
        (-9.8, 128.4),
        (17.8, 128.4),
        (17.8, 153.0),
        (28.55 + dy, 158.50),
        (28.55 + dy, 162.80),
        (-15.30 + dy, 162.80),
        (-15.30 + dy, 149.5),
        (-9.8, 144.0),
        align=None,
    )
    for by_c, bz_c in [(-6.25, 131.3), (-6.25, 152.0), (14.25, 131.3), (14.25, 152.0)]:
        ch_low_poly = ch_low_poly + Pos(by_c, bz_c) * Circle(3.15)
    ch_low_face = fillet_2d_face(
        ch_low_poly.clean().faces()[0],
        2.2,
        skip_xy=[(28.55 + dy, 158.50), (28.55 + dy, 162.80), (-15.30 + dy, 162.80)],
    )

    ch_hi_poly = Polygon(
        (-9.8, 135.5),
        (7.5, 135.5),
        (17.8, 144.5),
        (17.8, 153.0),
        (28.55 + dy, 158.50),
        (28.55 + dy, 162.80),
        (-15.30 + dy, 162.80),
        (-15.30 + dy, 149.5),
        (-9.8, 144.0),
        align=None,
    )
    for by_c, bz_c in [(-6.25, 152.0), (14.25, 152.0)]:
        ch_hi_poly = ch_hi_poly + Pos(by_c, bz_c) * Circle(3.15)
    ch_hi_face = fillet_2d_face(
        ch_hi_poly.clean().faces()[0],
        2.2,
        skip_xy=[(28.55 + dy, 158.50), (28.55 + dy, 162.80), (-15.30 + dy, 162.80), (-9.8, 135.5)],
    )

    if family == "truss_box":
        win_ch = Pos(4.0, 144.0) * RectangleRounded(9.6, 9.6, 2.5).faces()[0]
    else:
        win_ch = Pos(4.0, 144.0) * Circle(4.8).faces()[0]
    ch_low_face = (ch_low_face - win_ch).clean().faces()[0]
    ch_hi_face = (ch_hi_face - win_ch).clean().faces()[0]

    cheek_inner = extrude_xy_to_x(ch_low_face, 42.575, 44.575, sx=sx)
    cheek_outer = extrude_xy_to_x(ch_hi_face, 41.40, 44.575, sx=sx)
    knee_low_bosses = [
        occ_cyl(pt(41.40, by_c, bz_c), pt(44.575, by_c, bz_c), 3.15)
        for by_c, bz_c in [(-6.25, 131.3), (14.25, 131.3)]
    ]

    # 2. 100% Symmetric 2D G1-filleted Front & Rear Hip Roll Yoke Arms + Circular Hubs (R=10.8 at (64.0, 190.0)) in XZ plane:
    arm_pts = [(sx * x, z) for x, z in [(41.40, 155.8), (56.5, 155.8), (74.8, 158.50), (74.8, 190.0), (53.2, 190.0), (41.40, 162.80)]]
    arm_union = (Polygon(*arm_pts, align=None) + Pos(sx * 64.0, 190.0) * Circle(10.8)).clean().faces()[0]
    arm_face = fillet_2d_face(arm_union, 2.2, skip_xy=[(sx * 41.40, 155.8), (sx * 41.40, 162.80), (sx * 74.8, 158.50)], r_inner=3.2)
    if family == "truss_box":
        tri_pts = [(sx * x, z) for x, z in [(55.5, 168.5), (67.5, 168.5), (61.5, 177.5)]]
        tri_f = fillet_2d_face(Polygon(*tri_pts, align=None).faces()[0], 2.0)
        arm_face = (arm_face - tri_f).clean().faces()[0]
    else:
        arm_face = (arm_face - Pos(sx * 61.5, 171.8) * Circle(4.0).faces()[0]).clean().faces()[0]
    arm_f = extrude_xy_to_y(arm_face, y_f_out, y_f_in, sx=sx)
    arm_r = extrude_xy_to_y(arm_face, y_r_in, y_r_out, sx=sx)

    # 3. Clean 2D-filleted Femoral Saddle Deck in XY plane (Z in [158.50, 162.80]) with rounded interior cutouts:
    sad_outer = Polygon((sx * 41.40, y_f_out), (sx * 74.8, y_f_out), (sx * 74.8, y_r_out), (sx * 41.40, y_r_out), align=None).faces()[0]
    if family == "truss_box":
        sw1 = Pos(sx * 58.0, 0.5 + dy) * RectangleRounded(13.0, 10.5, 2.8).faces()[0]
        sw2 = Pos(sx * 58.0, 13.5 + dy) * RectangleRounded(13.0, 10.5, 2.8).faces()[0]
        sad_face = (sad_outer - sw1 - sw2).clean().faces()[0]
    else:
        sw1 = Pos(sx * 58.0, 0.5 + dy) * Circle(5.2).faces()[0]
        sw2 = Pos(sx * 58.0, 13.5 + dy) * Circle(5.2).faces()[0]
        sad_face = (sad_outer - sw1 - sw2).clean().faces()[0]
    saddle_deck = extrude_xy_to_z(sad_face, 158.50, 162.80)

    body = fuse_all([cheek_inner, cheek_outer, *knee_low_bosses, arm_f, arm_r, saddle_deck])

    cutters = [
        bx(44.575, 80.0, -8.50, 16.50, 112.0, 158.30),
        bx(43.40, 80.0, -3.05, 11.05, 131.8, 158.30),
        occ_cyl(pt(39.0, 4.0, 123.0), pt(44.10, 4.0, 123.0), 10.0),
        bx(28.0, 76.0, y_f_in, y_r_in, 162.85, 205.00),
        occ_cyl(pt(64.0, y_f_out - 2.0, 190.0), pt(64.0, y_f_in + 2.0, 190.0), 4.2),
        occ_cyl(pt(64.0, y_r_in - 2.0, 190.0), pt(64.0, y_r_out + 2.0, 190.0), 4.2),
    ]

    # 4 Front & 4 Rear Hip Roll M3 bolts + sleeves + full-length axial screwdriver access channels
    sleeves = []
    for bx_c, bz_c in [(57.0, 190.0), (71.0, 190.0), (64.0, 183.0), (64.0, 197.0)]:
        cutters.append(occ_cyl(pt(bx_c, y_f_out - 2.0, bz_c), pt(bx_c, y_f_in + 1.0, bz_c), 2.25))
        cutters.append(occ_cyl(pt(bx_c, y_f_out - 25.0, bz_c), pt(bx_c, y_f_clamp, bz_c), 2.90))
        sl_outer = occ_cyl(pt(bx_c, y_f_clamp, bz_c), pt(bx_c, y_f_in, bz_c), 2.25)
        sl_inner = occ_cyl(pt(bx_c, y_f_clamp - 0.5, bz_c), pt(bx_c, y_f_in + 0.5, bz_c), 1.6)
        sleeves.append(largest_solid(cut_all(sl_outer, [sl_inner])))

        cutters.append(occ_cyl(pt(bx_c, y_r_in - 1.0, bz_c), pt(bx_c, y_r_out + 2.0, bz_c), 2.25))
        cutters.append(occ_cyl(pt(bx_c, y_r_clamp, bz_c), pt(bx_c, y_r_out + 25.0, bz_c), 2.90))
        sl_r_outer = occ_cyl(pt(bx_c, y_r_in, bz_c), pt(bx_c, y_r_clamp, bz_c), 2.25)
        sl_r_inner = occ_cyl(pt(bx_c, y_r_in - 0.5, bz_c), pt(bx_c, y_r_clamp + 0.5, bz_c), 1.6)
        sleeves.append(largest_solid(cut_all(sl_r_outer, [sl_r_inner])))

    # 4 Knee PA2 bolts on medial cheek + sleeves + full-length medial (-X) screwdriver access channels
    for by_c, bz_c in [(-6.25, 131.3), (-6.25, 152.0), (14.25, 131.3), (14.25, 152.0)]:
        cutters.append(occ_cyl(pt(40.0, by_c, bz_c), pt(46.0, by_c, bz_c), 1.80))
        cutters.append(occ_cyl(pt(15.0, by_c, bz_c), pt(42.575, by_c, bz_c), 2.35))
        sl_k_outer = occ_cyl(pt(42.575, by_c, bz_c), pt(44.575, by_c, bz_c), 1.80)
        sl_k_inner = occ_cyl(pt(42.0, by_c, bz_c), pt(45.0, by_c, bz_c), 1.1)
        sleeves.append(largest_solid(cut_all(sl_k_outer, [sl_k_inner])))

    raw_sol = largest_solid(cut_all(body, cutters))
    bev_sol = bevel_3d_edges(raw_sol, r_outer=0.40, r_inner=0.75)

    # 4. Integrate the 4 Hard-Shell Mounting Posts (`thigh_front` at Z=178.0 and `thigh_back` at Z=198.0),
    # Ø3.4 mm (r = 1.70 mm) through-bores, and U-shaped hex-slot captive M3 nut pockets:
    shell_posts = [
        bevel_3d_edges(occ_cyl(pt(57.0, -28.96436 + dy, 178.0), pt(57.0, y_f_in, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl(pt(71.0, -27.43287 + dy, 178.0), pt(71.0, y_f_in, 178.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl(pt(57.0, y_r_in, 198.0), pt(57.0, 43.48828 + dy, 198.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
        bevel_3d_edges(occ_cyl(pt(71.0, y_r_in, 198.0), pt(71.0, 41.50660 + dy, 198.0), 4.8), r_outer=0.35, min_cyl_r=4.5),
    ]
    shell_cutters = [
        occ_cyl(pt(64.0, y_r_clamp, 197.0), pt(64.0, y_r_out + 25.0, 197.0), 2.90),
        occ_cyl(pt(57.0, -31.0 + dy, 178.0), pt(57.0, y_f_in + 0.5, 178.0), 1.70),
        make_captive_nut_pocket((57.0, -26.28436 + dy, 178.0), (0.0, 1.0, 0.0)),
        occ_cyl(pt(71.0, -30.0 + dy, 178.0), pt(71.0, y_f_in + 0.5, 178.0), 1.70),
        make_captive_nut_pocket((71.0, -24.75287 + dy, 178.0), (0.0, 1.0, 0.0)),
        occ_cyl(pt(57.0, y_r_in - 0.5, 198.0), pt(57.0, 45.5 + dy, 198.0), 1.70),
        make_captive_nut_pocket((57.0, 40.80828 + dy, 198.0), (0.0, -1.0, 0.0)),
        occ_cyl(pt(71.0, y_r_in - 0.5, 198.0), pt(71.0, 43.5 + dy, 198.0), 1.70),
        make_captive_nut_pocket((71.0, 38.82660 + dy, 198.0), (0.0, -1.0, 0.0)),
    ]
    res = largest_solid(cut_all(fuse_all([bev_sol, *shell_posts]), shell_cutters))
    return res, sleeves


def build_stage5_pitch_to_roll(family: str, side: str) -> tuple[Solid, list[Solid]]:
    """Build sculpted Stage 5 Hip Pitch-to-Roll Yoke (`R/L_pitch_link`) with 2D G1-filleted profiles,
    tangent circular Pitch Hub blend, and flat corridor deck with rounded interior cutouts (zero sawtooth wedges).
    """
    if side == "L":
        r_res, r_slv = build_stage5_pitch_to_roll(family, "R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    sx = 1.0
    dy_roll = 0.0
    x_front_inner_bolt = 35.0
    x_rear_inner_bolt = 31.25

    def bx(x0, x1, y0, y1, z0, z1):
        return occ_box(sx * x0, sx * x1, y0, y1, z0, z1)

    def pt(x, y, z):
        return (sx * x, float(y), float(z))

    y_f_in = -8.80 + dy_roll
    y_f_clamp = -10.80 + dy_roll
    y_f_out = -11.75 + dy_roll

    y_r_in = 23.20 + dy_roll
    y_r_clamp = 25.20 + dy_roll
    y_r_out = 25.20 + dy_roll

    # 1. 2D G1-filleted Front & Rear Roll-Motor Cheeks in XZ plane:
    # Front cheek stays at Z=202.20 mm across X <= 57.70 mm (giving >1.4 mm rotary clearance below R_pitch_servo_case_reference
    # across the full +/-35 deg Hip Pitch sweep) and rises to Z=208.15 mm at X in [58.80, 67.0] flush with the lateral deck.
    x_min_c = min(x_front_inner_bolt, x_rear_inner_bolt) - 3.2
    cf_pts = [
        (sx * x_min_c, 177.0),
        (sx * 59.5, 177.0),
        (sx * 67.0, 200.5),
        (sx * 67.0, 208.15),
        (sx * 58.80, 208.15),
        (sx * 57.70, 202.20),
        (sx * x_min_c, 202.20),
    ]
    cr_pts = [
        (sx * x_min_c, 177.0),
        (sx * 59.5, 177.0),
        (sx * 67.0, 200.5),
        (sx * 67.0, 208.15),
        (sx * x_min_c, 208.15),
    ]
    cf_face = fillet_2d_face(
        Polygon(*cf_pts, align=None).faces()[0],
        2.0,
        skip_xy=[(sx * 67.0, 208.15), (sx * 58.80, 208.15)],
    )
    cr_face = fillet_2d_face(
        Polygon(*cr_pts, align=None).faces()[0],
        2.5,
        skip_xy=[(sx * 67.0, 208.15), (sx * x_min_c, 208.15)],
    )
    flange_f = extrude_xy_to_y(cf_face, y_f_out, y_f_in, sx=sx)
    flange_r = extrude_xy_to_y(cr_face, y_r_in, y_r_out, sx=sx)

    # 2. 2D G1-filleted Lateral Yoke Arch + Circular Pitch Hub (R=10.2 at (2.0, 219.0)) in YZ plane (X in [59.0, 67.0]):
    ly_poly = Polygon((y_f_out, 202.60), (y_r_out, 202.60), (y_r_out, 215.5), (12.2, 219.0), (-8.2, 219.0), (y_f_out, 215.5), align=None)
    ly_union = (ly_poly + Pos(2.0, 219.0) * Circle(10.2)).clean().faces()[0]
    ly_face = fillet_2d_face(ly_union, 3.0, skip_xy=[(y_f_out, 202.60), (y_r_out, 202.60)])
    if family == "truss_box":
        cut_yz = fillet_2d_face(Polygon((13.0 + dy_roll, 208.5), (y_r_in - 1.2, 208.5), (15.5 + dy_roll, 218.5), align=None).faces()[0], 1.5)
        ly_face = (ly_face - cut_yz).clean().faces()[0]
    else:
        ly_face = (ly_face - Pos(17.5 + dy_roll, 212.5) * Circle(3.8).faces()[0]).clean().faces()[0]
    lateral_arch = extrude_xy_to_x(ly_face, 59.0, 67.0, sx=sx)

    # 3. Clean 2D G1-filleted L-Shaped Corridor Bridge Deck in XY plane (Z in [202.60, 208.15]):
    # Leaves the entire Hip Pitch rotary sweep trough (X < 58.80, Y < 15.40 + dy_roll) open so R_pitch_servo_case_reference
    # never collides across +/-35 deg Hip Pitch, while bridging the full posterior corridor and lateral yoke.
    y_corridor_ant = 15.40 + dy_roll
    deck_pts = [
        (sx * 58.80, y_f_out),
        (sx * 67.0, y_f_out),
        (sx * 67.0, y_r_out),
        (sx * x_min_c, y_r_out),
        (sx * x_min_c, y_corridor_ant),
        (sx * 58.80, y_corridor_ant),
    ]
    deck_outer = fillet_2d_face(
        Polygon(*deck_pts, align=None).faces()[0],
        2.5,
        skip_xy=[(sx * 67.0, y_f_out), (sx * 67.0, y_r_out), (sx * 58.80, y_f_out)],
    )
    if family == "truss_box":
        win1 = Pos(sx * 43.5, 20.3 + dy_roll) * RectangleRounded(16.0, 4.6, 1.8).faces()[0]
        deck_face = (deck_outer - win1).clean().faces()[0]
    else:
        win1 = Pos(sx * 38.0, 20.3 + dy_roll) * Circle(2.4).faces()[0]
        win2 = Pos(sx * 49.0, 20.3 + dy_roll) * Circle(2.4).faces()[0]
        deck_face = (deck_outer - win1 - win2).clean().faces()[0]
    corridor_deck = extrude_xy_to_z(deck_face, 202.60, 208.15)

    body = fuse_all([flange_f, flange_r, lateral_arch, corridor_deck])

    cutters = [
        bx(27.0, 75.0, y_f_in, y_r_in, 175.0, 202.55),
        bx(29.55, 55.20, -9.96 + dy_roll, y_f_in + 1.0, 182.85, 197.15),
        bx(38.0, 46.5, y_f_out - 1.0, y_f_in + 1.0, 185.0, 195.0),
        bx(34.15, 46.75, y_r_in - 1.0, y_r_out + 1.0, 180.60, 199.35),
        bx(53.2, 75.0, y_f_out - 1.0, y_f_clamp, 175.0, 202.55),
        occ_cyl(pt(64.0, y_f_out - 2.0, 190.0), pt(64.0, y_r_out + 2.0, 190.0), 10.15),
        bx(20.0, 59.0, -18.0, 30.0, 208.25, 255.0),
        occ_cyl(pt(57.0, 2.0, 219.0), pt(71.0, 2.0, 219.0), 4.2),
    ]

    sleeves = []
    # 4 Pitch Horn M3x12 bolts at (Y, Z) in {(2.0, 212.0), (2.0, 226.0), (-5.0, 219.0), (9.0, 219.0)}
    for by_c, bz_c in [(2.0, 212.0), (2.0, 226.0), (-5.0, 219.0), (9.0, 219.0)]:
        cutters.append(occ_cyl(pt(58.0, by_c, bz_c), pt(68.5, by_c, bz_c), 2.25))
        cutters.append(occ_cyl(pt(67.0, by_c, bz_c), pt(95.0, by_c, bz_c), 2.85))
        sl_p_out = occ_cyl(pt(59.0, by_c, bz_c), pt(67.0, by_c, bz_c), 2.25)
        sl_p_in = occ_cyl(pt(58.5, by_c, bz_c), pt(67.5, by_c, bz_c), 1.6)
        sleeves.append(largest_solid(cut_all(sl_p_out, [sl_p_in])))

    # 4 Front Roll Case PA2 bolts + full-length (-Y) screwdriver access channels
    for bx_c, bz_c in [(x_front_inner_bolt, 179.75), (x_front_inner_bolt, 200.25), (55.7, 179.75), (55.7, 200.25)]:
        cutters.append(occ_cyl(pt(bx_c, y_f_out - 1.5, bz_c), pt(bx_c, y_f_in + 1.0, bz_c), 1.80))
        cutters.append(occ_cyl(pt(bx_c, y_f_out - 25.0, bz_c), pt(bx_c, y_f_clamp, bz_c), 2.25))
        sl_f_out = occ_cyl(pt(bx_c, y_f_clamp, bz_c), pt(bx_c, y_f_in, bz_c), 1.80)
        sl_f_in = occ_cyl(pt(bx_c, y_f_clamp - 0.5, bz_c), pt(bx_c, y_f_in + 0.5, bz_c), 1.1)
        sleeves.append(largest_solid(cut_all(sl_f_out, [sl_f_in])))

    # 4 Rear Roll Case PA2 bolts + full-length (+Y) screwdriver access channels
    for bx_c, bz_c in [(x_rear_inner_bolt, 179.75), (x_rear_inner_bolt, 200.25), (55.7, 179.75), (55.7, 200.25)]:
        cutters.append(occ_cyl(pt(bx_c, y_r_in - 1.0, bz_c), pt(bx_c, y_r_out + 1.5, bz_c), 1.80))
        cutters.append(occ_cyl(pt(bx_c, y_r_clamp, bz_c), pt(bx_c, y_r_out + 25.0, bz_c), 2.25))
        sl_r_out = occ_cyl(pt(bx_c, y_r_in, bz_c), pt(bx_c, y_r_clamp, bz_c), 1.80)
        sl_r_in = occ_cyl(pt(bx_c, y_r_in - 0.5, bz_c), pt(bx_c, y_r_clamp + 0.5, bz_c), 1.1)
        sleeves.append(largest_solid(cut_all(sl_r_out, [sl_r_in])))

    raw_sol = largest_solid(cut_all(body, cutters))
    if side == "L":
        raw_sol = mirror_x(raw_sol)
        sleeves = [mirror_x(s) for s in sleeves]
    res = bevel_3d_edges(raw_sol, r_outer=0.40, r_inner=0.75)
    return res, sleeves


def build_stage6_yaw_to_pitch(family: str, side: str) -> tuple[Solid, list[Solid]]:
    """Build sculpted Stage 6 Hip Yaw-to-Pitch Frame (`R/L_yaw_link`) with 2D G1-filleted profiles,
    tangent circular Yaw Boss blend, and non-grazing vertical screw/hex-driver access channels (zero sliver walls).
    """
    if side == "L":
        r_res, r_slv = build_stage6_yaw_to_pitch(family, "R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    sx = 1.0

    def bx(x0, x1, y0, y1, z0, z1):
        return occ_box(sx * x0, sx * x1, y0, y1, z0, z1)

    def pt(x, y, z):
        return (sx * x, float(y), float(z))

    # 1. True Circular Hip Yaw Boss (R = 10.0 mm centered at (59.0, 2.0), Z in [262.75, 274.75])
    yaw_ring_boss = occ_cyl(pt(59.0, 2.0, 262.75), pt(59.0, 2.0, 274.75), 10.0)

    # 2D G1-filleted Top Deck (Z in [262.75, 266.75]) uniting the polygon deck with Circle(10.0) at (59.0, 2.0):
    td_pts = [(sx * x, y) for x, y in [(55.8, -15.0), (68.5, -15.0), (71.5, -8.5), (71.5, 12.5), (68.5, 19.0), (55.8, 19.0)]]
    td_face = fillet_2d_face(
        (Polygon(*td_pts, align=None) + Pos(sx * 59.0, 2.0) * Circle(10.0)).clean().faces()[0],
        2.2,
        skip_xy=[(sx * 55.8, -15.0), (sx * 55.8, 19.0)],
    )
    top_deck = extrude_xy_to_z(td_face, 262.75, 266.75)

    # 2. Anterior (Y in [-15.0, -7.85]) and Posterior (Y in [11.85, 19.0]) 2D-filleted Vertical Buttress Columns:
    # Shoulder starts at Z = 252.0 mm (above the Z = 248.0 + 2.35 = 250.35 mm pitch bolt counterbores so they never graze!)
    col_pts = [(sx * x, z) for x, z in [(55.8, 224.2), (58.8, 224.2), (58.8, 252.0), (68.5, 262.75), (68.5, 266.75), (55.8, 266.75)]]
    col_face = fillet_2d_face(
        Polygon(*col_pts, align=None).faces()[0],
        2.5,
        skip_xy=[(sx * 55.8, 224.2), (sx * 58.8, 224.2), (sx * 68.5, 262.75), (sx * 68.5, 266.75), (sx * 55.8, 266.75)],
    )
    col_ant = extrude_xy_to_y(col_face, -15.0, -7.85, sx=sx)
    col_post = extrude_xy_to_y(col_face, 11.85, 19.0, sx=sx)

    # 3. Lateral Bridge Wall (X in [57.20, 58.80]) & 2D-filleted Central Dorsal Spine (Y in [-1.60, 5.60]):
    # Inner wall is at X = 57.20 mm while vertical driver shaft at X = 59.0 has R = 1.70 mm (X_min = 57.30 mm),
    # leaving a clean 0.10 mm clearance so the vertical channel never grazes the lateral wall!
    lw_face = Polygon((-15.0, 224.2), (19.0, 224.2), (19.0, 266.75), (-15.0, 266.75), align=None).faces()[0]
    if family == "truss_box":
        lw_win = Pos(2.0, 232.8) * RectangleRounded(7.2, 5.2, 1.8).faces()[0]
    else:
        lw_win = Pos(2.0, 232.8) * Circle(2.8).faces()[0]
    lw_face = (lw_face - lw_win).clean().faces()[0]
    lat_wall_bridge = extrude_xy_to_x(lw_face, 57.20, 58.80, sx=sx)

    sp_pts = [(sx * x, z) for x, z in [(57.20, 236.5), (58.80, 236.5), (58.80, 252.0), (64.2, 262.75), (64.2, 266.75), (57.20, 266.75)]]
    sp_face = fillet_2d_face(
        Polygon(*sp_pts, align=None).faces()[0],
        2.5,
        skip_xy=[(sx * 57.20, 236.5), (sx * 58.80, 236.5), (sx * 64.2, 262.75), (sx * 64.2, 266.75), (sx * 57.20, 266.75)],
    )
    central_spine = extrude_xy_to_y(sp_face, -1.60, 5.60, sx=sx)

    body = fuse_all([yaw_ring_boss, top_deck, col_ant, col_post, lat_wall_bridge, central_spine])

    cutters = [
        bx(20.0, 55.80, -11.0, 15.0, 208.0, 254.25),
        bx(20.0, 57.20, -5.30, 9.30, 208.0, 254.25),
        occ_cyl(pt(54.0, 2.0, 219.0), pt(59.10, 2.0, 219.0), 10.0),
        occ_cyl(pt(59.0, 2.0, 262.0), pt(59.0, 2.0, 276.0), 4.2),
    ]

    sleeves = []
    # 4 Hip Yaw M3x14 bolts at (X, Y) in {(52.0, 2.0), (66.0, 2.0), (59.0, -5.0), (59.0, 9.0)}
    for bx_c, by_c in [(52.0, 2.0), (66.0, 2.0), (59.0, -5.0), (59.0, 9.0)]:
        cutters.append(occ_cyl(pt(bx_c, by_c, 262.0), pt(bx_c, by_c, 276.0), 2.25))
        cutters.append(occ_cyl(pt(bx_c, by_c, 251.5), pt(bx_c, by_c, 262.75), 2.85))
        cutters.append(occ_cyl(pt(bx_c, by_c, 215.0), pt(bx_c, by_c, 262.75), 1.70))
        sl_y_out = occ_cyl(pt(bx_c, by_c, 262.75), pt(bx_c, by_c, 274.75), 2.25)
        sl_y_in = occ_cyl(pt(bx_c, by_c, 262.25), pt(bx_c, by_c, 275.25), 1.6)
        sleeves.append(largest_solid(cut_all(sl_y_out, [sl_y_in])))

    # 4 Hip Pitch Case PA2 bolts at (Y, Z) in {(-8.25, 227.3), (-8.25, 248.0), (12.25, 227.3), (12.25, 248.0)}
    for by_c, bz_c in [(-8.25, 227.3), (-8.25, 248.0), (12.25, 227.3), (12.25, 248.0)]:
        cutters.append(occ_cyl(pt(54.5, by_c, bz_c), pt(60.0, by_c, bz_c), 1.80))
        cutters.append(occ_cyl(pt(58.8, by_c, bz_c), pt(95.0, by_c, bz_c), 2.35))
        sl_p_out = occ_cyl(pt(55.8, by_c, bz_c), pt(58.8, by_c, bz_c), 1.80)
        sl_p_in = occ_cyl(pt(55.3, by_c, bz_c), pt(59.3, by_c, bz_c), 1.1)
        sleeves.append(largest_solid(cut_all(sl_p_out, [sl_p_in])))

    res = bevel_3d_edges(largest_solid(cut_all(body, cutters)), r_outer=0.40, r_inner=0.75)
    return res, sleeves


def build_all_upper_hip():
    rho_paht = MATERIALS["PAHT_CF_dry"]["density_g_cm3"] * 1e-3
    rho_petg = MATERIALS["PETG_CF"]["density_g_cm3"] * 1e-3
    rho_steel = MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3

    for family in ("truss_box", "topopt"):
        fam_dir = ART / f"upper_hip_{family}"
        parts_dir = fam_dir / "parts"
        parts_dir.mkdir(parents=True, exist_ok=True)

        stage_configs = [
            ("stage4_thigh", "R", f"R_thigh_{family}", "R_roll_link", "frame", lambda f=family: build_stage4_thigh(f, "R")),
            ("stage4_thigh", "L", f"L_thigh_{family}", "L_roll_link", "frame", lambda f=family: build_stage4_thigh(f, "L")),
            ("stage5_hip_pitch_to_roll", "R", f"R_hip_pitch_to_roll_{family}", "R_pitch_link", "coupler", lambda f=family: build_stage5_pitch_to_roll(f, "R")),
            ("stage5_hip_pitch_to_roll", "L", f"L_hip_pitch_to_roll_{family}", "L_pitch_link", "coupler", lambda f=family: build_stage5_pitch_to_roll(f, "L")),
            ("stage6_hip_yaw_to_pitch", "R", f"R_hip_yaw_to_pitch_{family}", "R_yaw_link", "frame", lambda f=family: build_stage6_yaw_to_pitch(f, "R")),
            ("stage6_hip_yaw_to_pitch", "L", f"L_hip_yaw_to_pitch_{family}", "L_yaw_link", "frame", lambda f=family: build_stage6_yaw_to_pitch(f, "L")),
        ]

        manifest_parts = []
        for stage, side, pname, link, role, builder_fn in stage_configs:
            sol, sleeves = builder_fn()
            step_p = parts_dir / f"{pname}.step"
            stl_p = parts_dir / f"{pname}.stl"
            sol = export_part_and_mesh(sol, step_p, stl_p)

            sl_name = f"{pname}_steel_sleeves"
            sl_step_p = parts_dir / f"{sl_name}.step"
            sl_stl_p = parts_dir / f"{sl_name}.stl"
            sl_comp = export_compound_and_mesh(sleeves, sl_step_p, sl_stl_p)

            bb = sol.bounding_box()
            sbb = sl_comp.bounding_box()
            vol = float(sol.volume)
            sl_vol = float(sum(s.volume for s in sleeves))
            m_paht = round(vol * rho_paht, 3)
            m_petg = round(vol * rho_petg, 3)
            m_sl = round(sl_vol * rho_steel, 3)

            manifest_parts.append({
                "stage": stage,
                "side": side,
                "name": pname,
                "owning_link": link,
                "role": role,
                "native_step": rel(step_p),
                "native_step_sha256": sha(step_p),
                "mesh": rel(stl_p),
                "mesh_sha256": sha(stl_p),
                "native_valid": bool(sol.is_valid),
                "native_solids": len(sol.solids()),
                "volume_mm3": round(vol, 3),
                "polymer_mass_g_PAHT_CF": m_paht,
                "polymer_mass_g_PETG_CF": m_petg,
                "bounds_mm": [[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
                "sleeves": {
                    "name": sl_name,
                    "native_step": rel(sl_step_p),
                    "native_step_sha256": sha(sl_step_p),
                    "mesh": rel(sl_stl_p),
                    "mesh_sha256": sha(sl_stl_p),
                    "count": len(sleeves),
                    "volume_mm3": round(sl_vol, 3),
                    "mass_g": m_sl,
                    "bounds_mm": [[round(sbb.min.X, 3), round(sbb.min.Y, 3), round(sbb.min.Z, 3)], [round(sbb.max.X, 3), round(sbb.max.Y, 3), round(sbb.max.Z, 3)]],
                },
                "total_stage_mass_g_PAHT_CF": round(m_paht + m_sl, 3),
            })
            print(f"Built {pname:36s}: valid={sol.is_valid}, solids={len(sol.solids())}, polymer={m_paht:.2f}g, sleeves={m_sl:.2f}g, total={m_paht+m_sl:.2f}g")

        manifest = {
            "family": f"cove50-upper-hip-fdm-{family}",
            "version": "r2",
            "description": f"Sculpted Upper-Leg & Hip-Chain 3D-Printable FDM Connectors ({family}) for Stages 4, 5, and 6",
            "parts": manifest_parts,
            "source_code": {"path": rel(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
        }
        write_json(fam_dir / "cad-manifest.json", manifest)


# ----------------------------------------------------------------------------------------------------------------------
# Roll-flip variant (Idle T-fs5v6d, owner decisions D-12 / D-13, hip-shell DECISIONS.md F-03 variant B).
#
# The hip-roll STS3250 case rides with the thigh (R_roll_link), turned -90 deg about its own shaft (Ry(-90) about the
# roll axis, URDF sign convention, as in robot/body/hip-shell/sections/roll_options.py) so it hangs down into the thigh.
# Its horn and idler stay where they are and are bolted to the pitch link. Roll axis, position and range are unchanged.
#   Stage 4 (thigh, R_roll_link), H-cage (owner 2026-11-19): front and rear plates on the case's ear faces (the old
#     Stage 5 cradle data rotated with the case; 2 lower ear screws each, upper edge cut back under the swept Stage 5
#     arms), an inboard and an outboard wall on the case's end faces, a bridge over the knee case to a lateral knee
#     cheek, the medial ST3025 cheek on the medial end of the current thigh's end block (exact motor-refit B-rep);
#     2 + 2 screws on the knee case; every screw at its defined hole (connector_screws.json, standard-parts.json) and
#     through a steel sleeve; 4 shell posts.
#   Stage 5 (pitch link):  a lateral plate on the hip-pitch ST3025 output wheel (x 63..71), whose lower edge follows
#     an arc 18 mm about the roll axis, and two arms (the old thigh arm layers) bolted to the roll horn and idler.
# Built in the current assembly frame (shell-pass, after the motor refit and the 12 mm narrower hips), not in this
# builder's r2 frame. The selected design and its artifacts are untouched; outputs go to ART / "roll-flip".
#
#   python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/build_upper_hip_connectors.py --roll-flip
# ----------------------------------------------------------------------------------------------------------------------
RF_ART = ART / "roll-flip"
RF_SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
RF_AX, RF_AZ = 56.0, 190.0          # roll axis (current frame): along y through (x 56, z 190)
RF_R2_AX = 64.0                     # the same axis in this builder's r2 frame (old Stage 5 cradle data below)
RF_TURN_DEG = -90.0
# Old Stage 5 roll-case cradle (build_stage5_pitch_to_roll, r2 frame): ear bolts (x, z), ear faces (y)
RF_OLD_FRONT_BOLTS = [(35.0, 179.75), (35.0, 200.25), (55.7, 179.75), (55.7, 200.25)]
RF_OLD_REAR_BOLTS = [(31.25, 179.75), (31.25, 200.25), (55.7, 179.75), (55.7, 200.25)]
RF_Y_F_IN, RF_Y_F_CLAMP, RF_Y_F_OUT = -8.80, -10.80, -11.75      # front ear face, sleeve clamp, plate outer face
RF_Y_R_IN, RF_Y_R_CLAMP, RF_Y_R_OUT = 23.20, 25.20, 25.20        # rear ear face, sleeve clamp, plate outer face
RF_RIB = 2.0                        # outward rib on both plates below RF_RIB_TOP (outside the arms' sweep)
RF_RIB_F = 1.2                      # front rib (H-cage): plate + rib = 4.15 + 0.2 from the defined M2 hole opening
                                    # (3.9 mm deep): an M2x8 then engages 3.15 mm with the tip 0.75 short of the bottom
RF_RIB_TOP = 176.0
# Old thigh arm layers and horn / idler bolt circle. The Stage 5 arms now follow the defined holes and screw lengths
# (rf5_screws: front y -16.6 .. -11.6, rear 25.0 .. 30.4); these values stay as the arm faces the committed thigh-shell skin
# was fitted around (thigh_shell.py arm_hub_points / station_hull: RF_ARM_F/R with RF_ARM_THICKEN).
RF_ARM_F = (-15.30, -14.10, -12.10)  # out, clamp, in
RF_ARM_R = (25.35, 27.35, 28.55)     # in, clamp, out
RF_HORN_BOLTS = [(RF_AX - 7.0, RF_AZ), (RF_AX + 7.0, RF_AZ), (RF_AX, RF_AZ - 7.0), (RF_AX, RF_AZ + 7.0)]
RF_PITCH_HUB = (2.0, 219.0)          # hip pitch axis (y, z); ST3025 output wheel face at x 63.0
RF_PITCH_BOLTS = [(2.0, 212.0), (2.0, 226.0), (-5.0, 219.0), (9.0, 219.0)]
RF_PLATE_X = (63.0, 71.0)
RF_PITCH_HUB_R = 12.6                # pitch-wheel hub: the dia 10 anvil disc of each wheel sleeve stays on the part
RF_ROLL_HUB_R = 12.4                 # roll horn / idler hub of the arms (was 10.8), same reason
RF_FRONT_HUB_TOP = 201.45              # front arm hub top inboard of the neck (yaw-link column at hip pitch -90)
RF_ROLL_CENTRE_R = 3.2               # front arm centre hole: horn screw head (was 4.2), keeps the anvil discs on material
RF_ROLL_CENTRE_R_REAR = 3.6          # rear arm centre hole: 0.6 mm radial clearance to the case's rear stub (r 3.0)
RF_ARM_THICKEN = 1.5                # Stage 5 arms grow outward (away from the thigh) beyond the old 3.2 mm layers
RF_WING_TOP = 220.0                 # top of the Stage 5 plate wings at the arm layers (was z_low + 7.5)
RF_CLEAR_R = 18.0                    # Stage 5 stays outside this radius about the roll axis where the case can swing
                                     # (rotated case corners reach r 16.0; 2 mm clearance)
RF_KNEE_KEEP = (20.0, 42.9, -16.5, 30.0, 155.5, 162.8)  # box of the current thigh kept: end-block end only, above the
                                                        # old C047 cheek bores (z <= 154.35); the cheek below is new; top at the
                                                        # block's crossbar top (162.8), not 0.2 mm above it (no 0.2 mm ledges)
# Knee ST3025 (R_knee_ST3025_case, axis along x through y 4, z 123): its medial face is the plane x 33.15 (case y -6..14,
# top z 153). Screw sites = the case model's threaded M2 holes that open on that face (dia 1.6, modelled thread from
# x 33.15 to 37.65): the two upper corner holes and the three holes on a 7 mm radius about (y 4, z 143). The motor refit's
# "far group" (16 mm circle about (4, 143)) only matches dia 2.0 holes inside the case, behind its 2.35 mm medial wall
# (x 35.5..37.3); a screw there meets solid wall, so the cheek no longer has those four holes.
RF_KNEE_AXIS = (4.0, 123.0)
RF_KNEE_SCREWS = [(-4.0, 151.0), (12.0, 151.0), (4.0, 150.0), (-3.0, 143.0), (11.0, 143.0)]   # history: the five medial holes the U cradle used
RF_KNEE_FACE_X = 33.15               # clamping face of the cheek = the knee case's medial face
RF_END_X0 = 31.0                     # medial face of the kept end block (and of the motor-refit cheek)
RF_CHEEK_X0 = 29.0                   # medial face of the new cheek (4.15 mm thick; roll-stall bending); the screw heads seat
                                     # in spot faces at rf_med_seat_x(), the mirror of the lateral seat (owner 2026-11-20)
RF_CHEEK_Y = (-6.5, 14.5)            # cheek width below the knee case top: case y -6..14 plus 0.5 mm
RF_CHEEK_Z0 = 139.5                  # cheek lower edge (lowest screw hole edge z 141.8)
# Shell posts: (x, z), tip y (the shell boss sits on it), nut y (make_captive_nut_pocket centre: the pocket spans
# RF_NUT_POCKET_H from 0.10 behind it towards the tip). The nut sits behind a solid wall at the tip (the pocket used to open
# through the tip, so under pull-off the nut left with the shell): moved RF_POST_NUT_SHIFT deeper (front, rear). The rear
# shift is smaller: the rear screw bore cannot go deeper than y 26.0 (the rear ear-boss key pocket floor at y 25.5 overlaps
# the bore's edge in x), and a full nut plus a tip margin needs >= 3.2 mm of bore beyond the nut's near face. The insertion
# slot of each pocket opens upward (+z) through the post top, into free space above the post.
RF_NUT_POCKET_H = 2.60
RF_POST_NUT_SHIFT = (2.60, 2.00)
RF_POSTS = [((45.75, 171.5), -18.35, -15.67 + RF_POST_NUT_SHIFT[0]), ((66.25, 171.5), -18.35, -15.67 + RF_POST_NUT_SHIFT[0])]  # front
RF_POSTS_R = [((45.75, 171.5), 31.60, 28.92 - RF_POST_NUT_SHIFT[1]), ((66.25, 171.5), 31.60, 28.92 - RF_POST_NUT_SHIFT[1])]   # rear
# H-cage (owner 2026-11-19: the 0.15 mm plate-to-arm gaps will not hold for printed parts and hand assembly; the 4 upper
# roll-case screws go; "an H connector with a 90 deg rotation in the middle", 2 + 2 screws on each actuator).
#   roll case: its 2 lower ear screws per y face (z 161 front, 157.25 rear); plates cut back under the swept Stage 5 arms;
#   knee case: 2 medial + 2 lateral screws, joined to the plates by an outboard wall on the roll case's outboard end and a
#   bridge over the knee case top to the lateral cheek. The two wings mirror each other about the knee case's mid-plane
#   (owner 2026-11-20 "why not make it symmetric the wings of the H?"): medial holes case+z.4 / .5 and lateral holes
#   case-z.1 / .2 are all at (y -3 / 11, z 143); both cheeks seat the heads in dia RF_SPOT_D spot faces at the same
#   stack from the case face (the lateral one, set by the shell), so all four are one screw. Every screw has a steel
#   sleeve; the screw sites are the defined holes of connector_screws.json (rf_defined_holes), threads and lengths come from screw_plan.py.
RF_PLATE_Z0 = 153.4                  # plate / rib / walls lower edge: knee case top (z 153) + 0.4, so the knee case (with the
                                     # shin on its output) slides in along y between the cheeks
RF_CASE_END_X = 68.36                # turned roll case's outboard end face
RF_OUT_WALL_X = (68.6, 73.6)         # outboard wall on that face (0.24 mm off it); reaches 0.45 mm into the lateral cheek
                                     # (x 73.15 ..): a 73.0 face left a 0.15 mm slit beside the bridge (49 MPa FEM peak)
RF_PLATE_TOP = 177.0                 # top of the H-cage plates and both end walls (owner 2026-11-20: the plates reached up
                                     # to the yellow arms while only the lower ear screws connect). What still has a function
                                     # above the lower ear screws: the shell posts (z 171.5 + r 4.8 = 176.3), the front
                                     # ear-boss key pocket walls (the boss reaches z 181.0; the swept-arm keep-out arc capped
                                     # the walls beside it at z 178.25) and the rear ones (boss to z 172.6). Flat at this z
                                     # with r 2.5 corners; the keep-out arc dips to z 176.3 between x 51.7 and 60.3, inside
                                     # the front key pocket's x span (48.96 .. 63.04): a top at 178 put the arc's filleted
                                     # junction across the pocket's side walls (slivers; the 3D bevel crashed on them).
RF_OUT_WALL_TOP = RF_PLATE_TOP - 1.0  # both end walls stop 1 mm under the plates' top: a flush top put a 3-face corner
                                     # (plate inner face, wall face, shared top) under the 3D bevel, which crashed OCC
RF_BRIDGE_Z1 = 159.0                 # bridge over the knee case top: z RF_PLATE_Z0 .. this, outboard wall -> lateral cheek
RF_KNEE_LAT_FACE_X = 73.15           # knee case's lateral face = clamping face of the lateral cheek
RF_LAT_CHEEK_X1 = 79.4               # lateral cheek block before the shell trim: its outer face becomes the shell envelope
RF_SHELL_CLEAR = 0.5                 # cage to the blue thigh shell's inner wall (same link, static; lead 2026-11-20):
                                     # the lateral cheek is trimmed to the committed shell skin (thigh_shell.SURF) minus
                                     # WALL + this (no flare at the knee end: the cage gives way, not the shell)
RF_SHELL_TRIM_Z = (136.0, 166.0)     # heights of the trim envelope (lateral knee-cheek zone, x > 72)
RF_LAT_HEAD_K = 1.3                  # head height of the lateral screws (ISO 7380-1-style M2 button head, assumption):
                                     # the head seats in a dia RF_SPOT_D spot face so its top stays inside the envelope
RF_SPOT_D = 8.0                      # = the press tool's heated tip (press_lib TIP_D): flat landing for pressing the sleeve
RF_SHIN_CHAMFER = 3.0                # cheeks' rear-lower corner (y max, z min): chamfer leg, clear of the shin swept to knee 110
RF_ROLL_SWEEP = (-45.0, 25.0)        # hip roll the plates leave room for (manifest -30..25; library poses reach -45)
RF_YELLOW_MARGIN = 1.3               # plates to the swept Stage 5 arms (the 2D inner fillets of the cut eat ~0.2)
RF_NECK_FILLET_ALLOW = 0.5           # the arms' r 5 inner fillet (hub -> neck) reaches below the neck's lower edge z RF_AZ + 2
RF_KEY_CLEAR = 0.05                  # case ear-boss pockets: CAD clearance per side (printers.json hole_offset_mm not
                                     # measured yet; tune to the coupon). The bosses key the roll torque.
RF_IN_WALL_X = (40.4, 43.5)          # inboard wall on the case's inboard end face (old Stage 5 deck, turned); 3 mm
                                     # (lightening, FEM 2026-11-20: was 5.4 mm, p95 2.5 MPa at 6 kg x 1.5 g). It ends 0.05
                                     # past the case-envelope cut (x 43.45), which then forms its face: a 43.4 face left
                                     # 0.05 mm edges, at the flush top too since the plates and the wall share their top
RF_IN_WINDOW = (22.0, 160.5, RF_OUT_WALL_TOP - 3.0)   # inboard wall window: y size, z from .. to (rounded 3.5)
RF_OUT_WINDOW = (21.0, 13.0)         # outboard wall window (its lower edge 1 mm above the bridge, 3 mm under the wall top)
RF_KEY_CORNER_R = 1.0               # pocket corner radius (the bosses' corners are rounded more)
RF_FLOOR_WINDOW_INSET = 2.0          # windows through the boss-pocket floors, this far inside the boss outline (None: closed)
RF_SLEEVE_R, RF_SLEEVE_BORE_R = 1.80, 1.10   # 304 sleeve dia 3.6 / 2.2 for M2 (as the roll-case sleeves)
# Stage 5 screw connections (Idle T-sgms6h, screw check hip-screws-s1). Every Stage 5 screw sits at its defined hole
# (connector_screws.json hip_pitch_to_roll_roll_flip, rf_defined_holes) and its standard length (data/hardware.json) is
# chosen here; the clamped stack along the hole axis is then the one that gives exactly the hole's reference engagement:
# stack = L - point - reference (screw_plan.choose_length's window), tip margin = depth - reference.
#   pitch wheel: the plate's inner face reaches down to the wheel face (the tapped face) as a land about the pitch axis,
#     pocketed for the raised hub; the heads seat in counterbores on the outer face (stack 8.55 for M3x12, 2.95 engaged);
#   roll horn / idler: each arm clamps on the horn / idler face (no gap) through a clamp disc about the roll axis and is as
#     thick as its stack there (horn 5.0, idler 5.4: one screw kind, M3x8, for both; horn 2.5 and idler 2.1 engaged = the
#     flange thickness, tip at the flange's back). Outside the discs the arms keep their old faces (RF_ARM_F / RF_ARM_R with
#     RF_ARM_THICKEN), so the H-cage plates keep their y offset from the swept arm necks and the thigh shell its fit.
RF5_SCREW_L = {"A13_pitch_horn": 12.0, "A9_roll_horn": 8.0, "A9_roll_idler": 8.0}   # ISO 7380-1 M3 lengths
RF5_LAND_R = 11.0                    # land on the pitch wheel face (wheel face dia ~18.8, overhangs it 1.6 mm, 0.9 mm
                                     # above the pitch case face x 60.05); r <= 11.3 keeps it outside RF_CLEAR_R + 0.4 of the roll axis
RF5_HUB_R, RF5_HUB_TOP_X = 4.0, 63.0  # the wheel's raised hub (ST3025 output, r 4.0 up to x 63.0)
RF5_HUB_CLEAR = (0.2, 0.3)           # hub pocket clearance: radial, beyond the hub top
RF5_CBORE_R = 3.05                   # counterbore for the M3 button head (dk 5.7: 0.2 radial clearance) on the plate's
                                     # outer face; 0.70 mm web to the dia 6.5 centre screw-head hole (was 3.2: 0.53 mm)
RF5_SLEEVE_R, RF5_SLEEVE_BORE_R = 2.25, 1.60   # 304 sleeve dia 4.5 / 3.2 for M3
RF5_CLAMP_R = RF_ROLL_HUB_R          # inner clamp disc on the horn / idler face (0.5 mm) and the rear outer head-seat pad
RF5_SEAT_R = RF_ROLL_HUB_R           # (0.35 mm): run out to the arm hub's edge (r 12.4; at r 12.1 they left 0.3 mm rims,
                                     # under one line); the necks keep the old faces (H-cage plate, shell clearance). The front
                                     # head seat lies 0.2 inside the old face: the whole front arm's outer face moves to it.


def rf5_screws() -> dict:
    """The Stage 5 screws (right leg) at their defined holes: per group the chosen standard length, the stack that gives the
    reference engagement, and per hole the opening, axis, clamp point (= opening, the face the part rests on) and head seat
    (opening + stack along the axis)."""
    import sys
    sp_dir = str(ROOT / "robot/body/fastening")
    if sp_dir not in sys.path:
        sys.path.insert(0, sp_dir)
    import screw_plan as SP
    st = SP.standards()
    conn = SP.connectors()["hip_pitch_to_roll_roll_flip"]["groups"]
    out = {}
    for g, holes in rf_defined_holes_cached("hip_pitch_to_roll_roll_flip").items():
        L = RF5_SCREW_L[g]
        head = conn[g].get("head", "ISO 7380-1")
        rows = []
        for h in holes:
            stack = L - st["point_mm"] - h["reference_engagement_mm"]
            o, ax = np.asarray(h["opening"], float), np.asarray(h["axis_out"], float)
            rows.append(dict(hole_id=h["hole_id"], opening=o, axis_out=ax, depth_mm=h["depth_mm"], thread=h["thread"],
                             reference_engagement_mm=h["reference_engagement_mm"], stack_mm=round(stack, 3),
                             tip_margin_mm=round(h["depth_mm"] - h["reference_engagement_mm"], 3), seat=o + ax * stack))
        out[g] = dict(length_mm=L, head=head, holes=rows)
    return out


def rf_rot(x_r2: float, z: float) -> tuple[float, float]:
    """Old (r2 frame, case horizontal) x, z -> current frame x, z of the case turned RF_TURN_DEG about its shaft."""
    dx, dz = x_r2 - RF_R2_AX, z - RF_AZ
    a = math.radians(RF_TURN_DEG)
    return (RF_AX + math.cos(a) * dx + math.sin(a) * dz, RF_AZ - math.sin(a) * dx + math.cos(a) * dz)


def rf_rot_box(x0, x1, y0, y1, z0, z1) -> Solid:
    pts = [rf_rot(x, z) for x in (x0, x1) for z in (z0, z1)]
    xs, zs = [p[0] for p in pts], [p[1] for p in pts]
    return occ_box(min(xs), max(xs), y0, y1, min(zs), max(zs))


def rf_case_bolts() -> dict:
    """Ear bolt positions (current frame x, z) of the turned case, derived from the old Stage 5 cradle."""
    return {"front": [rf_rot(x, z) for x, z in RF_OLD_FRONT_BOLTS], "rear": [rf_rot(x, z) for x, z in RF_OLD_REAR_BOLTS]}


def rf_current_step(name: str) -> Solid:
    """A part of the current assembly (shell-pass manifest) as its native B-rep, placed in the current frame."""
    import json
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure
    from build123d import import_step, Location
    p = {q["name"]: q for q in json.loads(RF_SHELL_PASS.read_text())["parts"]}[name]
    path = p.get("native_step") or p.get("native_step_unmodified")
    sh = import_step(str(ensure(ROOT / path)))
    off = p.get("native_step_offset_mm") or [0.0, 0.0, 0.0]
    return largest_solid(sh.moved(Location(tuple(float(v) for v in off))))


def occ_common(a: Shape, b: Shape) -> Shape:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    op = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
    op.Build()
    return Compound(op.Shape())


def rf_case_turned() -> Shape:
    """The STS3250 roll case (all solids) of the shell-pass manifest, turned RF_TURN_DEG about the roll axis (right leg)."""
    import json
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from infra.storage.blob_store import ensure
    from build123d import import_step, Location
    p = {q["name"]: q for q in json.loads(RF_SHELL_PASS.read_text())["parts"]}["R_roll_servo_case_reference"]
    sh = import_step(str(ensure(ROOT / p["native_step"])))
    sh = sh.moved(Location(tuple(float(v) for v in (p.get("native_step_offset_mm") or [0.0, 0.0, 0.0]))))
    return sh.rotate(Axis((RF_AX, 0.0, RF_AZ), (0.0, 1.0, 0.0)), RF_TURN_DEG)


def rf_ear_boss_outline(case: Shape, y_root: float) -> Face:
    """Outer outline (x, z) of the case's ear boss at its root (just outside the ear face y_root), as a face in the plane
    y = y_root. The bosses are drafted, so the root is the largest section."""
    sl = occ_common(case, occ_box(30.0, 80.0, y_root - 0.01, y_root + 0.01, 140.0, 184.0))   # below the shaft stub
    fs = [f for f in sl.faces() if abs(f.normal_at(f.center()).Y) > 0.99 and abs(f.center().Y - (y_root - 0.01)) < 1e-3]
    big = max(fs, key=lambda f: f.area)
    return Face(big.outer_wire())


def rf_key_pocket(case: Shape, y_root: float, y_in: float, y_floor: float, grow: float | None = None) -> Solid:
    """Pocket for an ear boss: the bounding rectangle (x, z) of its root outline grown by RF_KEY_CLEAR per side (or by
    `grow`, negative = inside), corners rounded RF_KEY_CORNER_R (smaller than the boss's own corner rounding, so the
    flat sides key and the corners stay clear), from y_in (inside the case envelope) to y_floor, open upward. A box, not
    the outline offset: the offset outline's edges make the part unmeshable for gmsh."""
    b = rf_ear_boss_outline(case, y_root).bounding_box()
    g = RF_KEY_CLEAR if grow is None else grow
    x0, x1, z0, z1 = b.min.X - g, b.max.X + g, b.min.Z - g, b.max.Z + g
    if grow is not None:     # a lightening window: stays 3 mm under the plates' upper edge (the hub keep-out arc)
        z1 = min(z1, RF_AZ - (RF_ROLL_HUB_R + RF_YELLOW_MARGIN) - 3.0)
    else:                    # the key pocket runs out through the plate's upper edge: the cage rises onto the hanging
        z1 = RF_AZ           # case from below (+z), so the bosses must pass up the pockets (a closed top blocked it)
    f = Pos(0.5 * (x0 + x1), 0.5 * (z0 + z1)) * RectangleRounded(x1 - x0, z1 - z0, RF_KEY_CORNER_R if grow is None else 2.0)
    return extrude_xy_to_y(f.faces()[0], min(y_in, y_floor), max(y_in, y_floor))


def rf_yellow_keepout_xz() -> Face:
    """(x, z) region the plates stay out of: the Stage 5 arms (hub r RF_ROLL_HUB_R about the roll axis, neck from z
    RF_AZ + 2 up between x 60.5 and RF_PLATE_X[1]) swept over RF_ROLL_SWEEP, grown by RF_YELLOW_MARGIN. In the thigh frame
    the arms turn by -roll; at roll RF_ROLL_SWEEP[0] the neck's outboard face (x RF_PLATE_X[1]) and lower edge (z RF_AZ + 2)
    are lowest. The union over the sweep lies in the hub disk plus the 90 deg wedge above the point where those two
    faces meet at that roll (both faces offset by the margin)."""
    m = RF_YELLOW_MARGIN
    th = math.radians(-RF_ROLL_SWEEP[0])
    nb = (math.sin(th), math.cos(th))           # lower edge normal (into the neck) at the extreme roll
    no = (math.cos(th), -math.sin(th))          # outboard face normal (out of the neck)
    db, do = 2.0 - RF_NECK_FILLET_ALLOW - m, (RF_PLATE_X[1] - RF_AX) + m
    vx, vz = RF_AX + db * nb[0] + do * no[0], RF_AZ + db * nb[1] + do * no[1]
    L = 80.0
    wedge = Polygon((vx, vz), (vx + L * nb[0], vz + L * nb[1]), (vx + L * (nb[0] - no[0]), vz + L * (nb[1] - no[1])),
                    (vx - L * no[0], vz - L * no[1]), align=None)
    return (wedge + Pos(RF_AX, RF_AZ) * Circle(RF_ROLL_HUB_R + m)).clean().faces()[0]


def rf_post_nut_face(tip: float, nut: float) -> float:
    """y of the tip-side face of a shell post's nut pocket (the face the nut bears on under pull-off)."""
    return nut + (0.10 + RF_NUT_POCKET_H) * (-1.0 if nut > tip else 1.0)


def rf_sleeve(p0, p1) -> Solid:
    """Steel sleeve dia 2 * RF_SLEEVE_R, bore 2 * RF_SLEEVE_BORE_R, from p0 to p1 (flush with both faces)."""
    import numpy as _np
    a, b = _np.array(p0, float), _np.array(p1, float)
    d = (b - a) / _np.linalg.norm(b - a)
    return largest_solid(cut_all(occ_cyl(tuple(a), tuple(b), RF_SLEEVE_R),
                                 [occ_cyl(tuple(a - 0.5 * d), tuple(b + 0.5 * d), RF_SLEEVE_BORE_R)]))


def rf_cheek_face(y0: float, y1: float, z0: float, z1: float, wide=None) -> Face:
    """(y, z) profile of a knee cheek: rear-lower corner chamfered RF_SHIN_CHAMFER (shin swept to knee 110), lower corners
    rounded, top corners left sharp (they join the parts above). wide = (y0w, y1w, z_w): wider from z_w up."""
    c = RF_SHIN_CHAMFER
    if wide is None:
        pts = [(y0, z0), (y1 - c, z0), (y1, z0 + c), (y1, z1), (y0, z1)]
        skip = [(y1, z1), (y0, z1)]
    else:
        y0w, y1w, zw = wide
        pts = [(y0, z0), (y1 - c, z0), (y1, z0 + c), (y1, zw), (y1w, zw + 3.0), (y1w, z1), (y0w, z1), (y0w, zw + 3.0), (y0, zw)]
        skip = [(y1w, z1), (y0w, z1)]
    return fillet_2d_face(Polygon(*pts, align=None).faces()[0], 2.0, skip_xy=skip)


def rf_defined_holes(connector: str = "thigh_roll_flip") -> dict:
    """The screw groups of one connector of robot/body/fastening/connector_screws.json (thigh_roll_flip = Stage 4,
    hip_pitch_to_roll_roll_flip = Stage 5) with their defined holes (official standard-part definitions,
    robot/actuators/interfaces/standard-parts.json) registered on the shell-pass instances of the right leg
    (robot/body/fastening/interfaces.py); the roll case's holes turned with the case (the horn, idler and pitch output stay
    where they are). {group: [hole]}, each hole with opening / axis_out (body frame, mm), depth, thread, reference engagement."""
    import json
    import sys
    for d in (ROOT / "robot/body/fastening", ROOT / "robot/body"):
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
    import interfaces as I
    import screw_plan as SP
    from checks import bodylib as BL
    man = json.loads(RF_SHELL_PASS.read_text())
    conn = SP.connectors()[connector]["groups"]
    need = {g["standard_part"].replace("{S}", "R_") for g in conn.values()}
    sub = dict(man, parts=[p for p in man["parts"] if p["name"] in need or p.get("motor")])
    iface = I.instances(BL.Body(sub), sub)
    a = math.radians(RF_TURN_DEG)
    R = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
    c = np.array([RF_AX, 0.0, RF_AZ])
    out = {}
    for gname, g in conn.items():
        sp = g["standard_part"].replace("{S}", "R_")
        rows = []
        for h in SP.hole_positions(iface, sp, g["holes"]):
            o, ax = np.asarray(h["opening"], float), np.asarray(h["axis_out"], float)
            if sp.endswith("roll_servo_case_reference"):       # the shell-pass case hangs from the pitch link; turn it
                o, ax = R @ (o - c) + c, R @ ax
            rows.append(dict(hole_id=h["label"], opening=o, axis_out=ax / np.linalg.norm(ax), depth_mm=h["depth_mm"],
                             thread=h["thread"], reference_engagement_mm=h["reference_engagement_mm"]))
        out[gname] = rows
    return out


def _thigh_shell():
    """robot/body/leg/thigh_shell.py (its SURF skin), imported late: it imports this builder."""
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import thigh_shell
    return thigh_shell


def rf_shell_xmax(y: float, z: float, r: float) -> float:
    """Largest x (lateral, right leg) at (y, z) whose disc of radius r stays inside the committed shell's outer skin."""
    T = _thigh_shell()
    ang = np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False)
    lo, hi = 50.0, 100.0
    for _ in range(36):
        x = 0.5 * (lo + hi)
        Q = np.c_[x + r * np.cos(ang), y + r * np.sin(ang), np.full(len(ang), z)]
        lo, hi = (x, hi) if T.level(T.surface(Q[:, 2]), Q).max() <= 1.0 else (lo, x)
    return lo


def rf_lat_seat_x() -> float:
    """Head seat (x) of the lateral knee screws: the head top (RF_LAT_HEAD_K above the seat) stays RF_SHELL_CLEAR inside
    the shell's inner wall over the whole head disc (dia 3.5), for both screws (one seat plane, one screw length)."""
    T = _thigh_shell()
    r = T.WALL + RF_SHELL_CLEAR
    xs = []
    for g, holes in rf_defined_holes_cached().items():
        if g != "knee_case_lateral":
            continue
        for h in holes:
            _, y, z = h["opening"]
            xs += [rf_shell_xmax(y + 1.75 * math.cos(t), z + 1.75 * math.sin(t), r) for t in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
    return round(min(xs) - RF_LAT_HEAD_K, 3)


def rf_med_seat_x() -> float:
    """Head seat (x) of the medial knee screws: the mirror of rf_lat_seat_x() about the knee case's mid-plane (midway
    between the medial and lateral hole openings), so both wings clamp the same stack and take the same screw."""
    H = rf_defined_holes_cached()
    xm = float(np.mean([h["opening"][0] for h in H["knee_case"]]))
    xl = float(np.mean([h["opening"][0] for h in H["knee_case_lateral"]]))
    return round(xm - (rf_lat_seat_x() - xl), 3)


def rf_shell_trim_cutter() -> Solid:
    """Everything outside the committed shell skin minus WALL + RF_SHELL_CLEAR in the lateral knee-cheek zone (x > 72,
    z RF_SHELL_TRIM_Z): the skin rings (thigh_shell.SURF) offset inward, lofted through smooth closed splines."""
    from shapely.geometry import Polygon as SPoly
    from build123d import Spline, Wire, Solid as BSolid
    T = _thigh_shell()
    d = T.WALL + RF_SHELL_CLEAR
    th = np.linspace(-math.pi, math.pi, 360, endpoint=False)
    wires = []
    for z in np.arange(RF_SHELL_TRIM_Z[0], RF_SHELL_TRIM_Z[1] + 0.01, 2.0):
        cx, cy, a, bf, bk, n, nk = T.surface(np.array([z]))[0]
        b = np.where(np.sin(th) > 0, bk, bf); e = np.where(np.sin(th) > 0, nk, n)
        c, s_ = np.cos(th), np.sin(th)
        rr = 1.0 / (np.abs(c / a) ** e + np.abs(s_ / b) ** e) ** (1.0 / e)
        ring = SPoly(np.c_[cx + rr * c, cy + rr * s_]).buffer(-d, join_style=1, resolution=32)
        xy = np.asarray(ring.exterior.coords)[:-1]
        k = np.linspace(0, len(xy), 120, endpoint=False).astype(int)
        wires.append(Wire([Spline(*[(float(x), float(y), float(z)) for x, y in xy[k]], periodic=True)]))
    env = BSolid.make_loft(wires)
    zone = occ_box(72.0, 95.0, -25.0, 35.0, RF_SHELL_TRIM_Z[0] + 0.5, RF_SHELL_TRIM_Z[1] - 0.5)
    return largest_solid(cut_all(zone, [env]))


def rf_defined_holes_cached(connector: str = "thigh_roll_flip") -> dict:
    if connector not in _RF_HOLES:
        _RF_HOLES[connector] = rf_defined_holes(connector)
    return _RF_HOLES[connector]


_RF_HOLES = {}


def rf_screws() -> list[dict]:
    """Every Stage 4 screw (right leg), at its defined hole (rf_defined_holes): the sleeve spans the part's thickness
    along the hole axis, from the clamping face (the case face) to the outer face where the head seats. The screw itself
    (thread, standard length) is generated by robot/body/fastening/screw_plan.py when the manifest is built."""
    faces = {"A13_roll_case_front": (RF_Y_F_IN, RF_Y_F_OUT - RF_RIB_F), "A13_roll_case_rear": (RF_Y_R_IN, RF_Y_R_OUT + RF_RIB),
             "knee_case": (RF_KNEE_FACE_X, rf_med_seat_x()), "knee_case_lateral": (RF_KNEE_LAT_FACE_X, rf_lat_seat_x())}
    rows = []
    for g, holes in rf_defined_holes_cached().items():
        f_in, f_out = faces[g]
        for h in holes:
            o, ax = h["opening"], h["axis_out"]
            k = int(np.argmax(np.abs(ax)))
            p_in, p_out = o.copy(), o.copy()
            p_in[k], p_out[k] = f_in, f_out
            rows.append(dict(group=g, hole_id=h["hole_id"], opening=np.round(o, 3).tolist(), axis_out=np.round(ax, 4).tolist(),
                             depth_mm=h["depth_mm"], thread=h["thread"], reference_engagement_mm=h["reference_engagement_mm"],
                             mate="roll_servo_case_reference" if g.startswith("A13") else "knee_ST3025_case",
                             head=np.round(p_out, 3).tolist(), sleeve=[np.round(p_out, 3).tolist(), np.round(p_in, 3).tolist()],
                             stack_from_opening_mm=round(float(abs(p_out[k] - o[k])), 3)))
    return rows


def rf_bolts_used() -> dict:
    """(x, z) of the roll-case ear screws in use (defined holes), per face."""
    r = rf_screws()
    return {"front": [(s["opening"][0], s["opening"][2]) for s in r if s["group"] == "A13_roll_case_front"],
            "rear": [(s["opening"][0], s["opening"][2]) for s in r if s["group"] == "A13_roll_case_rear"]}


def build_stage4_thigh_roll_flip(side: str) -> tuple[Solid, list[Solid]]:
    """Stage 4 thigh for the hanging roll case, H-cage: plates on the roll case's ear faces (2 lower ear screws each),
    an outboard wall and a bridge to a lateral knee cheek, the medial knee cheek on the kept end block; 2 + 2 screws per
    actuator, every one with a sleeve."""
    if side == "L":
        r_res, r_slv = build_stage4_thigh_roll_flip("R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    yf_in, yf_out = RF_Y_F_IN, RF_Y_F_OUT
    yr_in, yr_out = RF_Y_R_IN, RF_Y_R_OUT
    z0 = RF_PLATE_Z0
    xw0, xw1 = RF_OUT_WALL_X
    case = rf_case_turned()

    # 1. Front and rear plates on the case ear faces (XZ profile), upper edge cut back under the swept Stage 5 arms
    #    (rf_yellow_keepout_xz); the outward rib (below RF_RIB_TOP) is flush with the plate's lower edge.
    #    The outboard top corner stays square in XZ (the outboard wall ends flush with it; the 3D bevel rounds it).
    pl = Polygon((31.0, z0), (xw1, z0), (xw1, RF_PLATE_TOP), (31.0, RF_PLATE_TOP), align=None)
    pl_face = fillet_2d_face(max((pl - rf_yellow_keepout_xz()).clean().faces(), key=lambda f: f.area), 2.5,
                             skip_xy=[(xw1, RF_PLATE_TOP)])
    rib = Polygon((32.5, z0), (xw1 - 1.5, z0), (xw1 - 1.5, RF_RIB_TOP), (32.5, RF_RIB_TOP), align=None)
    rib_face = fillet_2d_face(rib.faces()[0], 2.5, skip_xy=[(32.5, z0), (xw1 - 1.5, z0)])
    plate_f = extrude_xy_to_y(pl_face, yf_out, yf_in)
    plate_r = extrude_xy_to_y(pl_face, yr_in, yr_out)
    rib_f = extrude_xy_to_y(rib_face, yf_out - RF_RIB_F, yf_out)
    rib_r = extrude_xy_to_y(rib_face, yr_out, yr_out + RF_RIB)

    # 2. Inboard wall (YZ profile) on the case's inboard end face (the old Stage 5 deck, turned): x 38.0..43.4
    w_face = Polygon((yf_in - 0.6, z0), (yr_in + 0.6, z0), (yr_in + 0.6, RF_OUT_WALL_TOP), (yf_in - 0.6, RF_OUT_WALL_TOP),
                     align=None).faces()[0]
    wy, wz0, wz1 = RF_IN_WINDOW
    w_face = (w_face - Pos(7.2, 0.5 * (wz0 + wz1)) * RectangleRounded(wy, wz1 - wz0, 3.5).faces()[0]).clean().faces()[0]
    wall = extrude_xy_to_x(w_face, *RF_IN_WALL_X)

    body = fuse_all([plate_f, plate_r, rib_f, rib_r, wall])

    # 3. Cutters: case envelope between the ear faces, ear-boss pockets (the bosses key the roll torque: their root
    #    outline + RF_KEY_CLEAR), front window, horn / idler
    cutters = [
        rf_rot_box(27.0, 75.0, yf_in, yr_in, 175.0, 202.55),                      # case envelope between the ear faces
        rf_key_pocket(case, yf_in - 0.05, yf_in + 1.0, -10.0),                    # front boss (tip y -9.9)
        rf_rot_box(38.0, 46.5, yf_out - RF_RIB_F - 1.0, yf_in + 1.0, 185.0, 195.0),  # front window (case screw)
        rf_key_pocket(case, yr_in + 0.05, yr_in - 1.0, 25.5),                     # rear boss (tip y 25.1; rib kept closed)
        *([] if RF_FLOOR_WINDOW_INSET is None else            # lightening: windows through the pocket floors, inside the key walls
          [rf_key_pocket(case, yf_in - 0.05, yf_in + 1.0, yf_out - RF_RIB_F - 1.0, -RF_FLOOR_WINDOW_INSET),
           rf_key_pocket(case, yr_in + 0.05, yr_in - 1.0, yr_out + RF_RIB + 1.0, -RF_FLOOR_WINDOW_INSET)]),
        occ_cyl((RF_AX, yf_out - RF_RIB_F - 2.0, RF_AZ), (RF_AX, yr_out + RF_RIB + 2.0, RF_AZ), 10.15),  # horn / idler
    ]
    sleeves = []
    screws = rf_screws()
    # Sleeves run through the whole plate (rib included), flush with its outer face: the heated press tip lands on a flat
    # outer face with no counterbore (pressability rule, Idle M-evmrv6); screw heads seat on that face.
    for s in screws:
        if s["mate"] != "roll_servo_case_reference":
            continue
        (ax_, ay_, az_), (bx_, by_, bz_) = s["sleeve"]
        d = 1.0 if by_ > ay_ else -1.0
        cutters.append(occ_cyl((ax_, ay_ - 1.5 * d, az_), (bx_, by_ + 1.0 * d, bz_), RF_SLEEVE_R))
        sleeves.append(rf_sleeve(s["sleeve"][0], s["sleeve"][1]))
    new_sol = largest_solid(cut_all(body, cutters))

    # 4. Outboard wall on the case's outboard end face (x RF_CASE_END_X) and the bridge over the knee case top to the
    #    lateral cheek; fused after the case-envelope cutter (which reaches x 71)
    o_face = Polygon((yf_in - 0.6, z0), (yr_in + 0.6, z0), (yr_in + 0.6, RF_OUT_WALL_TOP), (yf_in - 0.6, RF_OUT_WALL_TOP),
                     align=None).faces()[0]
    o_face = (o_face - Pos(7.2, RF_BRIDGE_Z1 + 1.0 + RF_OUT_WINDOW[1] / 2) * RectangleRounded(*RF_OUT_WINDOW, 3.5).faces()[0]).clean().faces()[0]
    cy0, cy1 = RF_CHEEK_Y
    late = [extrude_xy_to_x(o_face, xw0, xw1),
            # the bridge is 0.5 mm narrower than the cheek and ends inside it (no coplanar side / end faces to fillet)
            extrude_xy_to_x(Polygon((cy0 + 0.5, z0), (cy1 - 0.5, z0), (cy1 - 0.5, RF_BRIDGE_Z1), (cy0 + 0.5, RF_BRIDGE_Z1),
                                    align=None).faces()[0], xw0, RF_KNEE_LAT_FACE_X + 2.15),
            # lateral knee cheek on the case's lateral face (x 73.15 .. 77.3), hung from the bridge
            extrude_xy_to_x(rf_cheek_face(cy0, cy1, RF_CHEEK_Z0, RF_BRIDGE_Z1), RF_KNEE_LAT_FACE_X, RF_LAT_CHEEK_X1)]
    new_sol = bevel_3d_edges(largest_solid(fuse_all([new_sol, *late])), r_outer=0.40, r_inner=0.75)

    # 5. Knee interface. Medial: the end of the current (motor-refit) thigh's end block (above z 155.5) and the medial
    #    cheek on the knee case's medial face (x 29.0 .. 33.15), the same profile as the lateral cheek (step 4). Screws along
    #    x, heads in spot faces on the cheeks' outer faces (step 7), each through a sleeve.
    knee = largest_solid(occ_common(rf_current_step("R_thigh_combined"), occ_box(*RF_KNEE_KEEP)))
    cheek = [extrude_xy_to_x(rf_cheek_face(cy0, cy1, RF_CHEEK_Z0, 154.0), RF_CHEEK_X0, RF_KNEE_FACE_X),
             occ_box(RF_END_X0, RF_KNEE_FACE_X, -13.0, 20.0, 153.0, 156.0),
             poly_prism_y([(RF_CHEEK_X0, 150.0), (RF_END_X0, 150.0), (RF_END_X0, 158.0), (RF_CHEEK_X0, 154.0)], cy0, cy1),
             occ_box(RF_KNEE_FACE_X, 34.5, RF_Y_F_OUT, cy0, 153.0, 156.0),   # flush with the front plate face (no 0.14 mm edge)
             occ_box(RF_KNEE_FACE_X, 34.5, cy1, 20.0, 153.0, 156.0)]
    knee = fuse_all([knee, *cheek])
    # the old arm stubs of the kept end block stand 1.55 / 1.35 mm proud of the plate ribs next to the inboard case-ear
    # sleeves: trim them back to the rib faces there, so the dia 8 press tip lands on a flat face (Idle M-evmrv6)
    knee = largest_solid(cut_all(knee, [occ_box(40.0, 43.5, yf_out - RF_RIB_F - 5.0, yf_out - RF_RIB_F, 150.0, 164.0),
                                        occ_box(40.0, 43.5, yr_out + RF_RIB, yr_out + RF_RIB + 5.0, 150.0, 164.0)]))

    # 6. Shell posts (thigh covers): outer shape and tips unchanged; the post runs back into the plate (not only into the
    #    rib), so the nut pocket behind the tip wall has solid post behind it above the rib top too
    posts, post_cut = [], []
    for (px, pz), tip, nut in RF_POSTS:
        posts.append(bevel_3d_edges(occ_cyl((px, tip, pz), (px, yf_out + 0.3, pz), 4.8), r_outer=0.35, min_cyl_r=4.5))
        post_cut += [occ_cyl((px, tip - 2.0, pz), (px, yf_out - 0.2, pz), 1.70),
                     make_captive_nut_pocket((px, nut, pz), (0.0, 1.0, 0.0), h=RF_NUT_POCKET_H)]
    for (px, pz), tip, nut in RF_POSTS_R:
        posts.append(bevel_3d_edges(occ_cyl((px, yr_out - 0.3, pz), (px, tip, pz), 4.8), r_outer=0.35, min_cyl_r=4.5))
        post_cut += [occ_cyl((px, yr_out + 0.8, pz), (px, tip + 2.0, pz), 1.70),
                     make_captive_nut_pocket((px, nut, pz), (0.0, -1.0, 0.0), h=RF_NUT_POCKET_H)]
    post_cut.append(rf_key_pocket(case, yr_in + 0.05, yr_in - 1.0, 25.5))   # the rear posts reach into the rear boss pocket
    # knee screw sleeve bores through both cheeks (cut after the fuse: the medial cheek is the end block's)
    for s in screws:
        if s["mate"] != "knee_ST3025_case":
            continue
        (ax_, ay_, az_), (bx_, by_, bz_) = s["sleeve"]
        d = 1.0 if bx_ > ax_ else -1.0
        post_cut.append(occ_cyl((ax_ - 1.5 * d, ay_, az_), (bx_ + 1.0 * d, by_, bz_), RF_SLEEVE_R))
        sleeves.append(rf_sleeve(s["sleeve"][0], s["sleeve"][1]))
    res = largest_solid(cut_all(fuse_all([new_sol, knee, *posts]), post_cut))
    # 7. The lateral cheek gives way to the shell (owner 2026-11-20: no flare at the knee end): trimmed to the shell skin
    #    minus WALL + RF_SHELL_CLEAR; the lateral screw heads seat in press-tip-wide spot faces at rf_lat_seat_x(), the
    #    medial ones in the mirrored spot faces at rf_med_seat_x()
    seat, seat_m = rf_lat_seat_x(), rf_med_seat_x()
    spots = [occ_cyl((seat, s["opening"][1], s["opening"][2]), (95.0, s["opening"][1], s["opening"][2]), RF_SPOT_D / 2)
             for s in screws if s["group"] == "knee_case_lateral"]
    spots += [occ_cyl((seat_m, s["opening"][1], s["opening"][2]), (RF_CHEEK_X0 - 5.0, s["opening"][1], s["opening"][2]), RF_SPOT_D / 2)
              for s in screws if s["group"] == "knee_case"]
    res = largest_solid(cut_all(res, [rf_shell_trim_cutter(), *spots]))
    return res, sleeves


def build_stage5_pitch_to_roll_roll_flip(side: str) -> tuple[Solid, list[Solid]]:
    """Stage 5 yoke for the flipped roll motor: pitch output wheel -> lateral plate -> arms on the roll horn / idler.
    Every clamp face and head seat follows the defined holes and the chosen screw lengths (rf5_screws)."""
    if side == "L":
        r_res, r_slv = build_stage5_pitch_to_roll_roll_flip("R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    S = rf5_screws()
    pitch, horn, idler = S["A13_pitch_horn"]["holes"], S["A9_roll_horn"]["holes"], S["A9_roll_idler"]["holes"]
    # arms: inner face on the horn / idler face (the defined holes' opening plane), outer face at the head seat
    yfi = float(np.mean([h["opening"][1] for h in horn])); yfo = float(np.mean([h["seat"][1] for h in horn]))
    yri = float(np.mean([h["opening"][1] for h in idler])); yro = float(np.mean([h["seat"][1] for h in idler]))
    hy, hz = RF_PITCH_HUB
    x0, x1 = RF_PLATE_X
    x_face = float(np.mean([h["opening"][0] for h in pitch]))       # the wheel's tapped face (land face)
    x_seat = float(np.mean([h["seat"][0] for h in pitch]))          # pitch screw head seat (counterbore floor)
    # lower edge of the plate: outside RF_CLEAR_R about the roll axis at its inner face x0 (the binding point)
    z_low = round(RF_AZ + math.sqrt(RF_CLEAR_R**2 - (x0 - RF_AX) ** 2) + 0.4, 2)

    # 1. Lateral plate (YZ profile) on the ST3025 output wheel, spreading fore-aft to the two arm layers
    wfo, wro = RF_ARM_F[0] - RF_ARM_THICKEN, RF_ARM_R[2] + RF_ARM_THICKEN      # plate wings to the arms' old outer faces
    # a head seat inside the old outer face (front: 0.2 mm) moves the whole arm's outer face (and the plate wing) to it:
    # a recess disc of the hub's radius left a 0.2 mm step around the neck and a notch at the hub flat (FEM peak)
    wfo = max(wfo, yfo)
    lp = Polygon((wfo, z_low), (wro, z_low), (wro, RF_WING_TOP), (hy + 7.5, hz + 6.0), (hy - 7.5, hz + 6.0), (wfo, RF_WING_TOP), align=None)
    lp_face = fillet_2d_face((lp + Pos(hy, hz) * Circle(RF_PITCH_HUB_R)).clean().faces()[0], 2.5, r_inner=4.0)
    plate = extrude_xy_to_x(lp_face, x0, x1)
    # land: the plate's inner face reaches the wheel face about the pitch axis (clamps on the tapped face, not the hub)
    land = occ_cyl((x_face, hy, hz), (x0 + 0.5, hy, hz), RF5_LAND_R)
    assert hz - RF5_LAND_R >= RF_AZ + math.sqrt(RF_CLEAR_R**2 - (x_face - RF_AX) ** 2) + 0.4, "land inside the case sweep"

    # 2. Arms (XZ profile, the old thigh arm layers) around the roll horn / idler, rising outboard into the plate.
    #    Inboard edge of the rising neck at x 60.5: clear of the yaw link's columns and pitch-case plate (x <= 60.05).
    top = z_low + 5.0  # below the plate wing's rounded top corner (no grazing faces)
    arm = Polygon((RF_AX + 10.5, RF_AZ + 2.0), (x1, RF_AZ + 2.0), (x1, top), (60.5, top), (60.5, RF_AZ + 9.0), align=None)  # counter-clockwise, or the union with the hub stays 2 faces
    arm_face = fillet_2d_face((arm + Pos(RF_AX, RF_AZ) * Circle(RF_ROLL_HUB_R)).clean().faces()[0], 2.2, r_inner=5.0,
                              skip_xy=[(60.5, top), (x1, top)])
    # old arm faces outside the clamp discs (envelope); clamp disc on the horn / idler face; head-seat disc on the outer face
    efo, efi = max(RF_ARM_F[0] - RF_ARM_THICKEN, yfo), RF_ARM_F[2]
    eri, ero = RF_ARM_R[0], RF_ARM_R[2] + RF_ARM_THICKEN
    arm_f = extrude_xy_to_y(arm_face, efo, efi)
    arm_r = extrude_xy_to_y(arm_face, eri, ero)
    # the sub-mm clamp / seat steps are added after the 3D bevel (a 0.4 fillet on a 0.2-0.5 mm step fails and takes the
    # arm outline's fillets with it): bosses fused, recesses cut, then the holes cut again through them
    steps_add = [occ_cyl((RF_AX, efi - 0.5, RF_AZ), (RF_AX, yfi, RF_AZ), RF5_CLAMP_R),
                 occ_cyl((RF_AX, yri, RF_AZ), (RF_AX, eri + 0.5, RF_AZ), RF5_CLAMP_R)]
    steps_cut = []
    for y_seat, y_env, sgn in ((yfo, efo, -1.0), (yro, ero, 1.0)):     # sgn: outward direction along y
        if (y_seat - y_env) * sgn > 0:     # head seat outside the old face: a pad
            steps_add.append(occ_cyl((RF_AX, y_env - sgn * 0.5, RF_AZ), (RF_AX, y_seat, RF_AZ), RF5_SEAT_R))
        elif (y_seat - y_env) * sgn < 0:   # inside: a recess
            steps_cut.append(occ_cyl((RF_AX, y_seat, RF_AZ), (RF_AX, y_env + sgn * 1.0, RF_AZ), RF5_SEAT_R))

    # the larger pitch hub must not reach below the plate's lower edge (case sweep, RF_CLEAR_R)
    plate = largest_solid(cut_all(fuse_all([plate, land]), [occ_box(x_face - 1.0, x1 + 1.0, wfo - 1.0, wro + 1.0, z_low - 20.0, z_low)]))
    # front arm hub flattened at RF_FRONT_HUB_TOP where the yaw link's anterior column passes at hip pitch -90
    arm_f = largest_solid(cut_all(arm_f, [occ_box(50.0, 60.5, efo - 1.0, yfi + 1.0, RF_FRONT_HUB_TOP, RF_AZ + 20.0)]))
    body = fuse_all([plate, arm_f, arm_r])
    cutters = [
        occ_cyl((RF_AX, min(yfo, efo) - 2.0, RF_AZ), (RF_AX, yfi + 2.0, RF_AZ), RF_ROLL_CENTRE_R),
        occ_cyl((RF_AX, yri - 2.0, RF_AZ), (RF_AX, max(yro, ero) + 2.0, RF_AZ), RF_ROLL_CENTRE_R_REAR),
        occ_cyl((x_face - 1.0, hy, hz), (x1 + 1.0, hy, hz), 3.25),      # output wheel centre screw head
        occ_cyl((x_face - 1.0, hy, hz), (RF5_HUB_TOP_X + RF5_HUB_CLEAR[1], hy, hz), RF5_HUB_R + RF5_HUB_CLEAR[0]),   # hub pocket
    ]
    sleeves = []
    for h in pitch:  # 4 ST3025 wheel M3: sleeve from the wheel face to the counterbore floor, head in the counterbore
        _, by_c, bz_c = h["opening"]
        cutters.append(occ_cyl((x_face - 1.0, by_c, bz_c), (x_seat + 0.5, by_c, bz_c), RF5_SLEEVE_R))
        cutters.append(occ_cyl((x_seat, by_c, bz_c), (x1 + 28.0, by_c, bz_c), RF5_CBORE_R))
        sleeves.append(largest_solid(cut_all(occ_cyl((x_face, by_c, bz_c), (x_seat, by_c, bz_c), RF5_SLEEVE_R),
                                             [occ_cyl((x_face - 0.5, by_c, bz_c), (x_seat + 0.5, by_c, bz_c), RF5_SLEEVE_BORE_R)])))
    for hf, hr in zip(horn, idler):   # 4 roll horn M3 (front, -Y drivers) + 4 roll idler M3 (rear, +Y drivers)
        # sleeves through the whole arm, flush with both faces (flat press-tip landing, heads on the outer face)
        bx_c, _, bz_c = hf["opening"]
        cutters.append(occ_cyl((bx_c, yfo - 2.0, bz_c), (bx_c, yfi + 1.0, bz_c), RF5_SLEEVE_R))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, yfo, bz_c), (bx_c, yfi, bz_c), RF5_SLEEVE_R),
                                             [occ_cyl((bx_c, yfo - 0.5, bz_c), (bx_c, yfi + 0.5, bz_c), RF5_SLEEVE_BORE_R)])))
        bx_c, _, bz_c = hr["opening"]
        cutters.append(occ_cyl((bx_c, yri - 1.0, bz_c), (bx_c, yro + 2.0, bz_c), RF5_SLEEVE_R))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, yri, bz_c), (bx_c, yro, bz_c), RF5_SLEEVE_R),
                                             [occ_cyl((bx_c, yri - 0.5, bz_c), (bx_c, yro + 0.5, bz_c), RF5_SLEEVE_BORE_R)])))
    res = bevel_3d_edges(largest_solid(cut_all(body, cutters)), r_outer=0.40, r_inner=0.75)
    # the front hub flat (RF_FRONT_HUB_TOP, yaw-link column at hip pitch -90) also trims the front clamp disc
    hub_flat = occ_box(50.0, 60.5, efo - 1.0, yfi + 1.0, RF_FRONT_HUB_TOP, RF_AZ + 20.0)
    res = largest_solid(cut_all(fuse_all([res, *steps_add]), steps_cut + cutters + [hub_flat]))
    return res, sleeves


def rf5_screw_rows() -> list[dict]:
    """The Stage 5 screws for the cad-manifest (right leg): hole, length, stack, sleeve span, expected engagement."""
    rows = []
    for g, d in rf5_screws().items():
        for h in d["holes"]:
            rows.append(dict(group=g, hole_id=h["hole_id"], head_standard=d["head"], thread=h["thread"], length_mm=d["length_mm"],
                             opening=np.round(h["opening"], 3).tolist(), axis_out=np.round(h["axis_out"], 4).tolist(),
                             head=np.round(h["seat"], 3).tolist(), stack_from_opening_mm=h["stack_mm"],
                             engagement_mm=h["reference_engagement_mm"], reference_engagement_mm=h["reference_engagement_mm"],
                             depth_mm=h["depth_mm"], tip_margin_mm=h["tip_margin_mm"]))
    return rows


RF_CHECK_DATA = ROOT / "robot/body/checks/data"


def rf_density(material: str) -> float:
    """g/mm3 of a material in checks/data/materials.json (the table the shared `mass` check uses)."""
    import json
    return json.loads((RF_CHECK_DATA / "materials.json").read_text())[material]["density_g_cm3"] * 1e-3


def rf_polymer_density(pname: str) -> tuple[str, float]:
    """(material, g/mm3) of a printed part from its `process.material` setting in checks/data/part_settings.json."""
    import json
    mat = json.loads((RF_CHECK_DATA / "part_settings.json").read_text())[pname]["process"]["material"]
    return mat, rf_density(mat)


def rf_yellow_clearance(s4: Shape, s5: list, step_deg: float = 1.0) -> dict:
    """Minimum distance (mm) from the Stage 4 surface to the Stage 5 parts (pitch link) turned about the roll axis over
    RF_ROLL_SWEEP (thigh frame: Stage 5 turns by -roll), and over the manifest range -30..25; surface samples 0.2 mm."""
    import trimesh
    from scipy.spatial import cKDTree

    def mesh(sh):
        v, f = sh.tessellate(0.02, 0.2)
        return trimesh.Trimesh([(p.X, p.Y, p.Z) for p in v], f, process=True)
    m4 = mesh(s4)
    q4 = np.vstack([trimesh.sample.sample_surface_even(m4, int(m4.area / 0.2 ** 2), seed=1)[0], m4.vertices])
    T = cKDTree(q4)
    pts = []
    for sh in s5:
        m = mesh(sh)
        pts.append(np.vstack([trimesh.sample.sample_surface_even(m, int(m.area / 0.2 ** 2), seed=1)[0], m.vertices]))
    P = np.vstack(pts) - [RF_AX, 0.0, RF_AZ]
    out = {}
    for lo, hi, key in ((RF_ROLL_SWEEP[0], RF_ROLL_SWEEP[1], "sweep"), (-30.0, 25.0, "manifest_range")):
        best = (np.inf, None, None)
        for q in np.arange(lo, hi + 1e-9, step_deg):
            a = math.radians(-q)        # Ry(a) (URDF sign) of the Stage 5 about the roll axis in the thigh frame
            R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
            X = P @ R.T + [RF_AX, 0.0, RF_AZ]
            d, _ = T.query(X, distance_upper_bound=8.0)
            j = int(np.argmin(d))
            if d[j] < best[0]:
                best = (float(d[j]), float(q), X[j])
        out[key] = {"roll_deg": [lo, hi], "min_mm": round(best[0], 3), "at_roll_deg": best[1], "at_mm": np.round(best[2], 2).tolist()}
    return out


def build_roll_flip(stage5_only: bool = False):
    """All four roll-flip connectors, or (stage5_only) only Stage 5: the Stage 4 files and its cad-manifest rows stay as
    they are (its STEP is read back for the Stage 4 to Stage 5 clearance)."""
    import json
    rho = {"PETG_CF": rf_density("Bambu PETG-CF"), "Steel_Sleeve": MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3}
    parts_dir = RF_ART / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    prev = json.loads((RF_ART / "cad-manifest.json").read_text()) if stage5_only else None
    configs = [
        ("stage4_thigh_roll_flip", "R", "R_thigh_roll_flip", "R_roll_link", lambda: build_stage4_thigh_roll_flip("R")),
        ("stage4_thigh_roll_flip", "L", "L_thigh_roll_flip", "L_roll_link", lambda: build_stage4_thigh_roll_flip("L")),
        ("stage5_hip_pitch_to_roll_roll_flip", "R", "R_hip_pitch_to_roll_roll_flip", "R_pitch_link", lambda: build_stage5_pitch_to_roll_roll_flip("R")),
        ("stage5_hip_pitch_to_roll_roll_flip", "L", "L_hip_pitch_to_roll_roll_flip", "L_pitch_link", lambda: build_stage5_pitch_to_roll_roll_flip("L")),
    ]
    rows, built = [], {}
    if stage5_only:
        from build123d import import_step
        configs = [c for c in configs if c[0].startswith("stage5")]
        rows = [r for r in prev["parts"] if r["stage"].startswith("stage4")]
        built["R_thigh_roll_flip"] = (largest_solid(import_step(str(parts_dir / "R_thigh_roll_flip.step"))), [])
    for stage, side, pname, link, fn in configs:
        sol, sleeves = fn()
        built[pname] = (sol, sleeves)
        step_p, stl_p = parts_dir / f"{pname}.step", parts_dir / f"{pname}.stl"
        sol = export_part_and_mesh(sol, step_p, stl_p)
        sl_name = f"{pname}_steel_sleeves"
        sl_step, sl_stl = parts_dir / f"{sl_name}.step", parts_dir / f"{sl_name}.stl"
        sl_comp = export_compound_and_mesh(sleeves, sl_step, sl_stl)
        bb, sbb = sol.bounding_box(), sl_comp.bounding_box()
        vol, sl_vol = float(sol.volume), float(sum(s.volume for s in sleeves))
        c = sol.center()
        mat, rho_p = rf_polymer_density(pname)
        rows.append({
            "stage": stage, "side": side, "name": pname, "owning_link": link, "role": "printed_structural_connector",
            "native_step": rel(step_p), "native_step_sha256": sha(step_p), "mesh": rel(stl_p), "mesh_sha256": sha(stl_p),
            "native_valid": bool(sol.is_valid), "native_solids": len(sol.solids()), "volume_mm3": round(vol, 2),
            "centroid_mm": [round(c.X, 3), round(c.Y, 3), round(c.Z, 3)],
            # polymer mass = solid volume x the density of the part's material setting (manifest mass_kg reads this)
            "polymer_material": mat, "polymer_density_g_cm3": round(rho_p * 1e3, 4), "polymer_mass_g": round(vol * rho_p, 3),
            "polymer_mass_g_PETG_CF": round(vol * rho["PETG_CF"], 3),
            "bounds_mm": [[round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)], [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]],
            "sleeves": {"name": sl_name, "native_step": rel(sl_step), "native_step_sha256": sha(sl_step), "mesh": rel(sl_stl),
                        "mesh_sha256": sha(sl_stl), "count": len(sleeves), "volume_mm3": round(sl_vol, 3),
                        "mass_g": round(sl_vol * rho["Steel_Sleeve"], 3),
                        "bounds_mm": [[round(sbb.min.X, 3), round(sbb.min.Y, 3), round(sbb.min.Z, 3)], [round(sbb.max.X, 3), round(sbb.max.Y, 3), round(sbb.max.Z, 3)]]},
            "total_stage_mass_g": round(vol * rho_p + sl_vol * rho["Steel_Sleeve"], 3),
        })
        print(f"Built {pname:34s}: valid={sol.is_valid}, solids={len(sol.solids())}, polymer={vol * rho_p:.2f} g ({mat}), "
              f"sleeves={len(sleeves)} ({sl_vol * rho['Steel_Sleeve']:.2f} g)", flush=True)
    yellow = rf_yellow_clearance(built["R_thigh_roll_flip"][0], [built["R_hip_pitch_to_roll_roll_flip"][0], *built["R_hip_pitch_to_roll_roll_flip"][1]])
    print(f"Stage 4 to the swept Stage 5 (roll {RF_ROLL_SWEEP[0]:+.0f}..{RF_ROLL_SWEEP[1]:+.0f}): {yellow}", flush=True)
    manifest = {
        "family": "cove50-upper-hip-fdm-roll-flip",
        "idle_task": "T-fs5v6d (parent T-czv2d8)",
        "owner_decisions": ["D-12", "D-13"], "feasibility": "robot/body/hip-shell/DECISIONS.md F-03 variant B",
        "frame": "current assembly frame (artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json); left = mirror x",
        "roll_case_turn": {"axis_point_mm": [RF_AX, None, RF_AZ], "axis": [0, 1, 0], "deg": RF_TURN_DEG,
                           "convention": "Ry(deg) about the roll axis, URDF sign (roll_options.Ry)"},
        "case_ear_bolts_xz_mm": rf_bolts_used(),
        "screws": {"rows": rf_screws(), "note": "right leg (left = mirror x); head = point on the seat face, axis head -> tip; "
                   "every screw has a steel sleeve over the part's thickness ('sleeve' span); PA2 self-tapping into the case hole"},
        "stage5_screws": {"rows": rf5_screw_rows(), "lengths_mm": RF5_SCREW_L, "land_r_mm": RF5_LAND_R,
                          "arm_clamp_disc_r_mm": RF5_CLAMP_R, "arm_head_seat_disc_r_mm": RF5_SEAT_R,
                          "hub_pocket": {"r_mm": RF5_HUB_R + RF5_HUB_CLEAR[0], "to_x_mm": RF5_HUB_TOP_X + RF5_HUB_CLEAR[1]},
                          "counterbore_r_mm": RF5_CBORE_R, "sleeve_od_id_mm": [2 * RF5_SLEEVE_R, 2 * RF5_SLEEVE_BORE_R],
                          "note": "right leg (left = mirror x); head = head seat point, stack = L - point - reference engagement "
                                  "(screw_plan.choose_length window), so each screw engages exactly its hole's reference"},
        "h_cage": {"owner": "2026-11-19 (H connector, 2 + 2 screws per actuator, upper roll-case screws removed)",
                   "plate_lower_edge_z": RF_PLATE_Z0, "plate_top_z": RF_PLATE_TOP,
                   "post_nut_pockets": {"shift_front_rear_mm": list(RF_POST_NUT_SHIFT), "pocket_h_mm": RF_NUT_POCKET_H,
                                        "tip_wall_mm": [round(abs(rf_post_nut_face(t, n) - t), 3) for _, t, n in (RF_POSTS[0], RF_POSTS_R[0])],
                                        "insertion": "slot open +z through the post top"}, "outboard_wall_x": list(RF_OUT_WALL_X), "bridge_z": [RF_PLATE_Z0, RF_BRIDGE_Z1],
                   "lateral_cheek_x": [RF_KNEE_LAT_FACE_X, "shell envelope"], "lateral_seat_x": rf_lat_seat_x(),
                   "medial_cheek_x": [RF_CHEEK_X0, RF_KNEE_FACE_X], "medial_seat_x": rf_med_seat_x(),
                   "shell_clear_mm": RF_SHELL_CLEAR, "shin_chamfer_mm": RF_SHIN_CHAMFER,
                   "key_pocket_clearance_mm": RF_KEY_CLEAR, "roll_sweep_deg": list(RF_ROLL_SWEEP),
                   "yellow_margin_mm": RF_YELLOW_MARGIN, "yellow_clearance": yellow},
        "parts": rows,
        "source_code": {"path": rel(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
    }
    write_json(RF_ART / "cad-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import sys
    if "--roll-flip" in sys.argv:
        build_roll_flip(stage5_only="--stage5-only" in sys.argv)
    else:
        build_all_upper_hip()
