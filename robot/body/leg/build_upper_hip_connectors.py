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
#   Stage 4 (thigh, R_roll_link):  the current thigh's knee interface (medial ST3025 cheek + the medial end of its end
#     block, exact motor-refit B-rep) + a U cradle for the hanging case: front and rear plates on the case's ear faces
#     (the old Stage 5 cradle data rotated with the case), an inboard wall, 4 shell posts.
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
RF_RIB_TOP = 176.0
# Old thigh arm layers (now the Stage 5 arms) and horn / idler bolt circle
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
RF_KNEE_SCREWS = [(-4.0, 151.0), (12.0, 151.0), (4.0, 150.0), (-3.0, 143.0), (11.0, 143.0)]
RF_KNEE_FACE_X = 33.15               # clamping face of the cheek = the knee case's medial face
RF_END_X0 = 31.0                     # medial face of the kept end block (and of the motor-refit cheek)
RF_CHEEK_X0 = 29.0                   # medial face of the new cheek (4.15 mm thick; roll-stall bending): screw heads seat here
RF_CHEEK_Y = (-6.5, 14.5)            # cheek width below the knee case top: case y -6..14 plus 0.5 mm
RF_CHEEK_Z0 = 139.5                  # cheek lower edge (lowest screw hole edge z 141.8)
RF_POSTS = [((45.75, 171.5), -18.35, -15.67), ((66.25, 171.5), -18.35, -15.67)]   # tips at the blue shell's inner wall   # front: (x, z), tip y, nut y
RF_POSTS_R = [((45.75, 171.5), 31.60, 28.92), ((66.25, 171.5), 31.60, 28.92)]     # rear


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


def build_stage4_thigh_roll_flip(side: str) -> tuple[Solid, list[Solid]]:
    """Stage 4 thigh for the hanging roll case: knee interface kept, U cradle on the turned case's ear faces."""
    if side == "L":
        r_res, r_slv = build_stage4_thigh_roll_flip("R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    bolts = rf_case_bolts()
    yf_in, yf_cl, yf_out = RF_Y_F_IN, RF_Y_F_CLAMP, RF_Y_F_OUT
    yr_in, yr_cl, yr_out = RF_Y_R_IN, RF_Y_R_CLAMP, RF_Y_R_OUT

    # 1. Front and rear plates on the case ear faces (XZ profile). Upper edge stays below z 186.5 so the plates stay
    #    inside r ~15 of the roll axis on the outboard side (Stage 5 plate above), lower edge just below the case.
    pl = [(31.0, 152.0), (70.0, 152.0), (70.0, 183.0), (67.0, 186.5), (40.5, 186.5), (31.0, 176.5)]
    pl_face = fillet_2d_face(Polygon(*pl, align=None).faces()[0], 2.5)
    rib = [(32.5, 153.5), (68.5, 153.5), (68.5, RF_RIB_TOP), (32.5, RF_RIB_TOP)]   # inset 1.5 mm: no grazing faces
    rib_face = fillet_2d_face(Polygon(*rib, align=None).faces()[0], 2.5)
    plate_f = extrude_xy_to_y(pl_face, yf_out, yf_in)
    plate_r = extrude_xy_to_y(pl_face, yr_in, yr_out)
    rib_f = extrude_xy_to_y(rib_face, yf_out - RF_RIB, yf_out)
    rib_r = extrude_xy_to_y(rib_face, yr_out, yr_out + RF_RIB)

    # 2. Inboard wall (YZ profile) on the case's inboard end face (the old Stage 5 deck, turned): x 38.0..43.4
    w_face = Polygon((yf_in - 0.6, 153.15), (yr_in + 0.6, 153.15), (yr_in + 0.6, 185.5), (yf_in - 0.6, 185.5), align=None).faces()[0]
    w_face = (w_face - Pos(7.2, 171.5) * RectangleRounded(17.0, 19.0, 3.5).faces()[0]).clean().faces()[0]
    wall = extrude_xy_to_x(w_face, 38.0, 43.4)

    body = fuse_all([plate_f, plate_r, rib_f, rib_r, wall])

    # 3. Old Stage 5 cradle cutters, turned with the case
    cutters = [
        rf_rot_box(27.0, 75.0, yf_in, yr_in, 175.0, 202.55),                      # case envelope between the ear faces
        rf_rot_box(29.55, 55.20, -9.96, yf_in + 1.0, 182.85, 197.15),             # front boss pocket
        rf_rot_box(38.0, 46.5, yf_out - RF_RIB - 1.0, yf_in + 1.0, 185.0, 195.0),  # front window (case screw)
        rf_rot_box(34.15, 46.75, yr_in - 1.0, 25.5, 180.60, 199.35),              # rear boss pocket (rib kept closed)
        occ_cyl((RF_AX, yf_out - RF_RIB - 2.0, RF_AZ), (RF_AX, yr_out + RF_RIB + 2.0, RF_AZ), 10.15),  # horn / idler
    ]
    sleeves = []
    # Sleeves run through the whole plate (rib included), flush with its outer face: the heated press tip lands on a
    # flat outer face with no counterbore around it (pressability rule, Idle M-evmrv6); screw heads seat on that face.
    for bx_c, bz_c in bolts["front"]:  # 4 front case PA2 + sleeves, straight -Y driver channels
        f_out = yf_out - (RF_RIB if bz_c < RF_RIB_TOP else 0.0)
        cutters.append(occ_cyl((bx_c, f_out - 1.5, bz_c), (bx_c, yf_in + 1.0, bz_c), 1.80))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, f_out, bz_c), (bx_c, yf_in, bz_c), 1.80),
                                             [occ_cyl((bx_c, f_out - 0.5, bz_c), (bx_c, yf_in + 0.5, bz_c), 1.1)])))
    for bx_c, bz_c in bolts["rear"]:   # 4 rear case PA2 + sleeves, straight +Y driver channels
        r_out = yr_out + (RF_RIB if bz_c < RF_RIB_TOP else 0.0)
        cutters.append(occ_cyl((bx_c, yr_in - 1.0, bz_c), (bx_c, r_out + 1.5, bz_c), 1.80))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, yr_in, bz_c), (bx_c, r_out, bz_c), 1.80),
                                             [occ_cyl((bx_c, yr_in - 0.5, bz_c), (bx_c, r_out + 0.5, bz_c), 1.1)])))
    new_sol = bevel_3d_edges(largest_solid(cut_all(body, cutters)), r_outer=0.40, r_inner=0.75)

    # 4. Knee interface: the medial end of the current (motor-refit) thigh's end block (above z 155.5), and a new cheek
    #    on the knee case's medial face. The motor-refit cheek kept four bored bosses of the old C047 knee pattern
    #    (y -6.25 / 14.25, z 131.3 / 152): their axes miss the case's holes (2.5 / 12.1 mm) and stand beside the case
    #    (bosses out to y -9.4 / 17.4 against case y -6 .. 14). The new cheek is the case's width and carries only the
    #    case's five open threaded holes; screws enter from the medial side (straight -x drivers), heads on x 29.0.
    knee = largest_solid(occ_common(rf_current_step("R_thigh_combined"), occ_box(*RF_KNEE_KEEP)))
    cy0, cy1 = RF_CHEEK_Y
    ck = Polygon((cy0, RF_CHEEK_Z0), (cy1, RF_CHEEK_Z0), (cy1, 154.0), (cy0, 154.0), align=None).faces()[0]
    ck = fillet_2d_face(ck, 2.0, skip_xy=[(cy1, 154.0), (cy0, 154.0)])
    cheek = [extrude_xy_to_x(ck, RF_CHEEK_X0, RF_KNEE_FACE_X),
             # above the knee case top (z 153) the cheek widens into the kept end block
             occ_box(RF_END_X0, RF_KNEE_FACE_X, -13.0, 20.0, 153.0, 156.0),
             # the 2 mm thicker medial layer tapers into the end block's medial face (x 31) above the knee case
             poly_prism_y([(RF_CHEEK_X0, 150.0), (RF_END_X0, 150.0), (RF_END_X0, 158.0), (RF_CHEEK_X0, 154.0)], cy0, cy1),
             occ_box(RF_KNEE_FACE_X, 34.5, RF_Y_F_OUT, cy0, 153.0, 156.0),   # flush with the front plate face (no 0.14 mm edge)
             occ_box(RF_KNEE_FACE_X, 34.5, cy1, 20.0, 153.0, 156.0)]
    knee = fuse_all([knee, *cheek])
    knee_holes = [occ_cyl((RF_CHEEK_X0 - 1.0, ky, kz), (RF_KNEE_FACE_X + 1.0, ky, kz), 1.20) for ky, kz in RF_KNEE_SCREWS]
    knee = largest_solid(cut_all(knee, knee_holes))
    # the old arm stubs of the kept end block stand 1.55 / 1.35 mm proud of the plate ribs next to the inboard case-ear
    # sleeves: trim them back to the rib faces there, so the dia 8 press tip lands on a flat face (Idle M-evmrv6)
    knee = largest_solid(cut_all(knee, [occ_box(40.0, 43.5, yf_out - RF_RIB - 5.0, yf_out - RF_RIB, 150.0, 164.0),
                                        occ_box(40.0, 43.5, yr_out + RF_RIB, yr_out + RF_RIB + 5.0, 150.0, 164.0)]))

    # 5. Shell posts (thigh covers are being redesigned): moved down to z 170, below the Stage 5 arms' sweep
    posts, post_cut = [], []
    for (px, pz), tip, nut in RF_POSTS:
        posts.append(bevel_3d_edges(occ_cyl((px, tip, pz), (px, yf_out - RF_RIB + 0.3, pz), 4.8), r_outer=0.35, min_cyl_r=4.5))
        post_cut += [occ_cyl((px, tip - 2.0, pz), (px, yf_out - 0.2, pz), 1.70), make_captive_nut_pocket((px, nut, pz), (0.0, 1.0, 0.0))]
    for (px, pz), tip, nut in RF_POSTS_R:
        posts.append(bevel_3d_edges(occ_cyl((px, yr_out + RF_RIB - 0.3, pz), (px, tip, pz), 4.8), r_outer=0.35, min_cyl_r=4.5))
        post_cut += [occ_cyl((px, yr_out + 0.8, pz), (px, tip + 2.0, pz), 1.70), make_captive_nut_pocket((px, nut, pz), (0.0, -1.0, 0.0))]
    res = largest_solid(cut_all(fuse_all([new_sol, knee, *posts]), post_cut))
    return res, sleeves


def build_stage5_pitch_to_roll_roll_flip(side: str) -> tuple[Solid, list[Solid]]:
    """Stage 5 yoke for the flipped roll motor: pitch output wheel -> lateral plate -> arms on the roll horn / idler."""
    if side == "L":
        r_res, r_slv = build_stage5_pitch_to_roll_roll_flip("R")
        return mirror_x(r_res), [mirror_x(s) for s in r_slv]
    yfo, yfc, yfi = RF_ARM_F
    yri, yrc, yro = RF_ARM_R
    yfo, yro = yfo - RF_ARM_THICKEN, yro + RF_ARM_THICKEN   # outer faces; clamp faces (sleeve ends) stay where they were
    hy, hz = RF_PITCH_HUB
    x0, x1 = RF_PLATE_X
    # lower edge of the plate: outside RF_CLEAR_R about the roll axis at its inner face x0 (the binding point)
    z_low = round(RF_AZ + math.sqrt(RF_CLEAR_R**2 - (x0 - RF_AX) ** 2) + 0.4, 2)

    # 1. Lateral plate (YZ profile) on the ST3025 output wheel, spreading fore-aft to the two arm layers
    lp = Polygon((yfo, z_low), (yro, z_low), (yro, RF_WING_TOP), (hy + 7.5, hz + 6.0), (hy - 7.5, hz + 6.0), (yfo, RF_WING_TOP), align=None)
    lp_face = fillet_2d_face((lp + Pos(hy, hz) * Circle(RF_PITCH_HUB_R)).clean().faces()[0], 2.5, r_inner=4.0)
    plate = extrude_xy_to_x(lp_face, x0, x1)

    # 2. Arms (XZ profile, the old thigh arm layers) around the roll horn / idler, rising outboard into the plate.
    #    Inboard edge of the rising neck at x 60.5: clear of the yaw link's columns and pitch-case plate (x <= 60.05).
    top = z_low + 5.0  # below the plate wing's rounded top corner (no grazing faces)
    arm = Polygon((RF_AX + 10.5, RF_AZ + 2.0), (x1, RF_AZ + 2.0), (x1, top), (60.5, top), (60.5, RF_AZ + 9.0), align=None)  # counter-clockwise, or the union with the hub stays 2 faces
    arm_face = fillet_2d_face((arm + Pos(RF_AX, RF_AZ) * Circle(RF_ROLL_HUB_R)).clean().faces()[0], 2.2, r_inner=5.0,
                              skip_xy=[(60.5, top), (x1, top)])
    arm_f = extrude_xy_to_y(arm_face, yfo, yfi)
    arm_r = extrude_xy_to_y(arm_face, yri, yro)

    # the larger pitch hub must not reach below the plate's lower edge (case sweep, RF_CLEAR_R)
    plate = largest_solid(cut_all(plate, [occ_box(x0 - 1.0, x1 + 1.0, yfo - 1.0, yro + 1.0, z_low - 20.0, z_low)]))
    # front arm hub flattened at RF_FRONT_HUB_TOP where the yaw link's anterior column passes at hip pitch -90
    arm_f = largest_solid(cut_all(arm_f, [occ_box(50.0, 60.5, yfo - 1.0, yfi + 1.0, RF_FRONT_HUB_TOP, RF_AZ + 20.0)]))
    body = fuse_all([plate, arm_f, arm_r])
    cutters = [
        occ_cyl((RF_AX, yfo - 2.0, RF_AZ), (RF_AX, yfi + 2.0, RF_AZ), RF_ROLL_CENTRE_R),
        occ_cyl((RF_AX, yri - 2.0, RF_AZ), (RF_AX, yro + 2.0, RF_AZ), RF_ROLL_CENTRE_R_REAR),
        occ_cyl((x0 - 1.0, hy, hz), (x1 + 1.0, hy, hz), 3.25),      # output wheel centre screw head
    ]
    sleeves = []
    for by_c, bz_c in RF_PITCH_BOLTS:  # 4 ST3025 wheel M3 + sleeves, +X driver channels
        cutters.append(occ_cyl((x0 - 1.0, by_c, bz_c), (x1 + 1.5, by_c, bz_c), 2.25))
        cutters.append(occ_cyl((x1, by_c, bz_c), (x1 + 28.0, by_c, bz_c), 2.85))
        sleeves.append(largest_solid(cut_all(occ_cyl((x0, by_c, bz_c), (x1, by_c, bz_c), 2.25),
                                             [occ_cyl((x0 - 0.5, by_c, bz_c), (x1 + 0.5, by_c, bz_c), 1.6)])))
    for bx_c, bz_c in RF_HORN_BOLTS:   # 4 roll horn M3 (front, -Y drivers) + 4 roll idler M3 (rear, +Y drivers)
        # sleeves through the whole arm, flush with its outer face (flat press-tip landing, heads on that face)
        cutters.append(occ_cyl((bx_c, yfo - 2.0, bz_c), (bx_c, yfi + 1.0, bz_c), 2.25))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, yfo, bz_c), (bx_c, yfi, bz_c), 2.25),
                                             [occ_cyl((bx_c, yfo - 0.5, bz_c), (bx_c, yfi + 0.5, bz_c), 1.6)])))
        cutters.append(occ_cyl((bx_c, yri - 1.0, bz_c), (bx_c, yro + 2.0, bz_c), 2.25))
        sleeves.append(largest_solid(cut_all(occ_cyl((bx_c, yri, bz_c), (bx_c, yro, bz_c), 2.25),
                                             [occ_cyl((bx_c, yri - 0.5, bz_c), (bx_c, yro + 0.5, bz_c), 1.6)])))
    res = bevel_3d_edges(largest_solid(cut_all(body, cutters)), r_outer=0.40, r_inner=0.75)
    return res, sleeves


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


def build_roll_flip():
    import json
    rho = {"PETG_CF": rf_density("Bambu PETG-CF"), "Steel_Sleeve": MATERIALS["Steel_Sleeve"]["density_g_cm3"] * 1e-3}
    parts_dir = RF_ART / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("stage4_thigh_roll_flip", "R", "R_thigh_roll_flip", "R_roll_link", lambda: build_stage4_thigh_roll_flip("R")),
        ("stage4_thigh_roll_flip", "L", "L_thigh_roll_flip", "L_roll_link", lambda: build_stage4_thigh_roll_flip("L")),
        ("stage5_hip_pitch_to_roll_roll_flip", "R", "R_hip_pitch_to_roll_roll_flip", "R_pitch_link", lambda: build_stage5_pitch_to_roll_roll_flip("R")),
        ("stage5_hip_pitch_to_roll_roll_flip", "L", "L_hip_pitch_to_roll_roll_flip", "L_pitch_link", lambda: build_stage5_pitch_to_roll_roll_flip("L")),
    ]
    rows = []
    for stage, side, pname, link, fn in configs:
        sol, sleeves = fn()
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
    manifest = {
        "family": "cove50-upper-hip-fdm-roll-flip",
        "idle_task": "T-fs5v6d (parent T-czv2d8)",
        "owner_decisions": ["D-12", "D-13"], "feasibility": "robot/body/hip-shell/DECISIONS.md F-03 variant B",
        "frame": "current assembly frame (artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json); left = mirror x",
        "roll_case_turn": {"axis_point_mm": [RF_AX, None, RF_AZ], "axis": [0, 1, 0], "deg": RF_TURN_DEG,
                           "convention": "Ry(deg) about the roll axis, URDF sign (roll_options.Ry)"},
        "case_ear_bolts_xz_mm": rf_case_bolts(),
        "knee_case_screws_yz_mm": {"sites": RF_KNEE_SCREWS, "axis": "x", "head_face_x": RF_CHEEK_X0,
                                   "case_face_x": RF_KNEE_FACE_X, "note": "right leg; left = mirror x"},
        "parts": rows,
        "source_code": {"path": rel(Path(__file__).resolve()), "sha256": sha(Path(__file__).resolve())},
    }
    write_json(RF_ART / "cad-manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    import sys
    if "--roll-flip" in sys.argv:
        build_roll_flip()
    else:
        build_all_upper_hip()
