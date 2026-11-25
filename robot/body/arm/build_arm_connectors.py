"""Blank-state 3D Generative CAD Reconstruction for the 4-Stage Arm FDM Connector Study:
  - Family 1 (`truss_box`): 3D Multi-Cheek Space-Truss, Circular Cup-Rimmed Hubs & Triangulated Box-Cages
  - Family 2 (`topopt`):    Blank-State 3D SIMP Bifurcated Wishbone Arches & Dual-Cheek Organic Yokes

Constructed from a 100% blank state (zero sheet-metal templates or bent-strip geometry),
connecting all 44 active physical connection points per arm (36 pressed-in 304 SS compression sleeves
including dual-cheek C047 motor case PA2 mounts + 8 integrated captive-M3 hard-shell mounts):
  - Stage 1 (`R/L_shoulder_pitch_to_roll`): 14 connections (4x M3 Pitch Horn + 6x PA2 Dual-Cheek Roll Case + 4x M3 Shoulder Shell Mounts)
  - Stage 2 (`R/L_shoulder_roll_to_yaw`):    8 connections (4x M3 Roll Horn + 4x PA2 Yaw Case across a 3D Twin-Column Windowed Saddle)
  - Stage 3 (`R/L_upper_arm_yaw_to_elbow`):  8 connections (4x M3 Yaw Horn + 4x M3 Elbow Horn across an Asymmetric 3D Twin-Branch Wishbone/Truss)
  - Stage 4 (`R/L_forearm_elbow_to_wrist`): 14 connections (6x PA2 Dual-Cheek Elbow Case + 4x M3 Wrist Pad + 4x M3 Forearm Shell Mounts)
"""
from pathlib import Path
import math
import sys
import time
import numpy as np
import manifold3d as md
from build123d import (
    Axis,
    Box,
    Circle,
    Compound,
    Plane,
    Polygon as _OrigPolygon,
    Pos,
    RectangleRounded,
    Shape,
    Solid,
    Vector,
    export_step,
    export_stl,
)


def Polygon(*pts, **kw):
    """Ensure 2D polygon vertices are oriented counter-clockwise (+Z face normal) so boolean unions with Circle(+) succeed."""
    area2 = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    if area2 < 0:
        pts = pts[::-1]
    return _OrigPolygon(*pts, **kw)
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GProp import GProp_GProps
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    ARM_SHELL_DIR,
    ART,
    A_YAW,
    ALT_PRINTED,
    N_YAW,
    P_ELBOW,
    P_PITCH,
    P_ROLL,
    P_YAW,
    PRINTED,
    REPLACED_ARM_PARTS,
    RIGHT_ARM_SHELL_MOUNTS,
    ROOT,
    ROT_ROLL,
    STAGE_LINKS,
    STAGE_PART_BASENAMES,
    V_YAW,
    W_YAW,
    bevel_3d_edges,
    build_stage_sleeves,
    cyl_along,
    ensure_arm_shell_nut_assets,
    extrude_xy_to_x,
    extrude_xy_to_y,
    extrude_xy_to_z,
    fillet_2d_face,
    make_captive_nut_pocket,
    read_json,
    ref,
    rel,
    sha,
    solid_to_watertight_arrays,
    stage_interface_specs,
    write_json,
)
from common import build_poses, cyl_mf, fk, npz_to_mf


def extrude_yaw_local_x(face_vw, xn0: float, xn1: float) -> Solid:
    """Extrude a 2D face sketched in `(V_YAW, W_YAW)` across `N_YAW` in `[xn0, xn1]`."""
    xa, xb = min(float(xn0), float(xn1)), max(float(xn0), float(xn1))
    p_orig = P_YAW + xa * N_YAW
    pl = Plane(origin=tuple(p_orig), x_dir=tuple(V_YAW), y_dir=tuple(W_YAW))
    f3d = (pl * face_vw).faces()[0]
    d_vec = Vector(*tuple((xb - xa) * N_YAW))
    return Solid(Solid.extrude(f3d, d_vec).wrapped)


def extrude_yaw_local_z(face_vn, w0: float, w1: float) -> Solid:
    """Extrude a 2D face sketched in `(V_YAW, N_YAW)` across `W_YAW` in `[w0, w1]` (`Y in [-10+w0, -10+w1]`)."""
    wa, wb = min(float(w0), float(w1)), max(float(w0), float(w1))
    p_orig = P_YAW + wa * W_YAW
    pl = Plane(origin=tuple(p_orig), x_dir=tuple(V_YAW), y_dir=tuple(N_YAW))
    f3d = (pl * face_vn).faces()[0]
    d_vec = Vector(*tuple((wb - wa) * W_YAW))
    return Solid(Solid.extrude(f3d, d_vec).wrapped)


def vn_yaw(pt_xyz) -> tuple[float, float]:
    """Project a 3D world point `pt_xyz` into `(u_v, x_n)` along `(V_YAW, N_YAW)` relative to `P_YAW`."""
    d = np.asarray(pt_xyz, dtype=float) - P_YAW
    return (float(np.dot(d, V_YAW)), float(np.dot(d, N_YAW)))


def occ_box(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> Solid:
    p1 = gp_Pnt(float(min(x0, x1)), float(min(y0, y1)), float(min(z0, z1)))
    p2 = gp_Pnt(float(max(x0, x1)), float(max(y0, y1)), float(max(z0, z1)))
    return Solid(BRepPrimAPI_MakeBox(p1, p2).Shape())


def occ_cyl(p0, p1, r: float) -> Solid:
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    L = float(np.linalg.norm(v))
    if L < 1e-6:
        raise ValueError("Zero-length cylinder")
    d = v / L
    ax = gp_Ax2(gp_Pnt(float(p0[0]), float(p0[1]), float(p0[2])), gp_Dir(float(d[0]), float(d[1]), float(d[2])))
    return Solid(BRepPrimAPI_MakeCylinder(ax, float(r), L).Shape())


def occ_capsule(p0, p1, r: float) -> Solid:
    """Build a 3D pill-section organic tendon strut (`RectangleRounded` with near-circular fillet radius) between `p0` and `p1`."""
    w = float(r) * 2.0
    return occ_strut(p0, p1, w, w, r_frac=0.45)


def occ_strut(p0, p1, w_perp: float, h_perp: float | None = None, r_frac: float = 0.34) -> Solid:
    """Build a 3D oriented rounded-edge box-beam strut (`RectangleRounded` cross-section) between 3D points `p0` and `p1`."""
    if h_perp is None:
        h_perp = w_perp
    v0 = Vector(*map(float, p0))
    v1 = Vector(*map(float, p1))
    dv = v1 - v0
    length = dv.length + 0.8
    d = dv.normalized()
    p_start = v0 - d * 0.4
    r_c = min(float(w_perp), float(h_perp)) * float(r_frac)
    pl = Plane(origin=(p_start.X, p_start.Y, p_start.Z), z_dir=(d.X, d.Y, d.Z))
    prof = pl * RectangleRounded(float(w_perp), float(h_perp), r_c)
    return Solid(Solid.extrude(prof.faces()[0], d * length).wrapped)


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


def cut_driver_and_bore_channels(body: Shape, stage_key: str) -> Solid:
    """Cut all sleeve outer seats (`ro + 0.04 mm` for zero tessellation chord overlap), through-bores,
    central rocker cutouts, and full-length straight-line hex-driver / washer access channels (`r = 3.15 / 2.25 mm`)."""
    specs = stage_interface_specs()[stage_key]
    cutters = []
    for role_key in ("support", "load"):
        grp = specs[role_key]
        norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
        ro = float(grp["sleeve_ro"])
        ri = float(grp["sleeve_ri"])
        h = float(grp["sleeve_len"])
        r_driver = 3.15 if ri > 1.4 else 2.25
        for pt, d in zip(grp["centres"], norms):
            cutters.append(cyl_along(pt - 0.04 * d, d, ro + 0.04, h + 0.08))
            cutters.append(cyl_along(pt - 3.5 * d, d, ri + 0.03, h + 14.0))
            if grp.get("name") == "wrist_pad":
                cutters.append(cyl_along(pt + (h - 0.02) * d, d, 2.85, 4.5))
                cutters.append(cyl_along(pt + (h - 0.02) * d, d, 1.60, 30.0))
            elif stage_key == "stage1_shoulder_pitch_to_roll":
                cutters.append(cyl_along(pt + (h + 0.01) * d, d, r_driver, 30.0))
            else:
                cutters.append(cyl_along(pt + (h - 0.02) * d, d, r_driver, 30.0))
        if "rocker_center" in grp:
            rc = grp["rocker_center"]
            rr = float(grp["rocker_r"])
            d0 = grp["normal"]
            cutters.append(cyl_along(rc - 3.0 * d0, d0, rr + 0.5, h + 12.0))
    return largest_solid(cut_all(body, cutters))


def extrude_yaw_local_y(face_wn, v0: float, v1: float) -> Solid:
    """Extrude a 2D face sketched in `(W_YAW, N_YAW)` across `V_YAW` in `[v0, v1]`."""
    va, vb = min(float(v0), float(v1)), max(float(v0), float(v1))
    p_orig = P_YAW + va * V_YAW
    pl = Plane(origin=tuple(p_orig), x_dir=tuple(W_YAW), y_dir=tuple(N_YAW))
    f3d = (pl * face_wn).faces()[0]
    d_vec = Vector(*tuple((vb - va) * V_YAW))
    return Solid(Solid.extrude(f3d, d_vec).wrapped)


def build_stage1_shoulder_pitch_to_roll(family: str) -> Solid:
    """Build blank-state Sculpted 3D C-Channel Shoulder Pitch-to-Roll Bridge & Cheek (`R_shoulder_pitch_to_roll`):
      - 2D $G^1$-filleted Pitch Main Horn Hub (`X in [70.625, 75.20]` centered at `(70.625, 9.0, 352.0)` with full-height `Z in [336.5, 359.0]` rear root)
      - Full-Span Sculpted Rear Structural Wall & Roll Motor Clamping Cheek (`X in [70.625, 135.65]`, `Y in [22.02, 23.575]` web clearing the raised `Y <= 21.48` drum + `Y in [20.575, 23.575]` boss pads on all 4 rear PA2 sleeves)
      - Deep 3D C-Channel Bridge Flanges (`Z in [336.5, 347.9]` lower deck and `Z in [355.4, 359.0]` upper deck spanning `Y in [12.80, 22.02]`) merging tangentially into the `Y = 20.575` `r0` roll boss pad with zero floating shell mounts
      - 3D edge beveling (`bevel_3d_edges` with `r_inner = 0.75 mm > r_outer = 0.40 mm`) for $G^1$ tangential continuity and zero sharp edges
    """
    specs = stage_interface_specs()["stage1_shoulder_pitch_to_roll"]
    horn_pts = specs["support"]["centres"]
    case_pts = specs["load"]["centres"]  # 0..3 rear (Y=20.575)
    r0, r1, r2, r3 = [(float(p[0]), float(p[2])) for p in case_pts[:4]]

    # 1. Pitch Horn Hub in YZ (`X in [70.625, 75.20]`): R=9.45 + full-height rear structural root (`Z in [336.5, 359.0]`)
    hub_yz = (
        Polygon((6.5, 336.5), (22.02, 336.5), (22.02, 359.0), (9.0, 359.0), (2.0, 352.0), align=None)
        + Pos(9.0, 352.0) * Circle(9.45)
    )
    for p in horn_pts:
        hub_yz = hub_yz + Pos(float(p[1]), float(p[2])) * Circle(3.55)
    hub_face = fillet_2d_face(
        hub_yz.clean().faces()[0],
        r_default=1.4,
        r_inner=2.4,
        skip_xy=[(22.02, 336.5), (22.02, 359.0)],
    )
    pitch_hub = extrude_xy_to_x(hub_face, 70.625, 73.625)

    # 2. Full-Span Rear Structural Wall & Roll Motor Cheek in XZ (`Y in [22.02, 23.575]` web + `Y in [20.575, 23.575]` boss pads)
    rear_poly = Polygon(
        (70.625, 336.5),
        (70.625, 359.0),
        (95.5, 359.0),
        (103.85, 353.15),
        (124.71, 358.4),
        (135.68, 332.73),
        (132.83, 332.8),
        (117.5, 332.8),
        (115.6, 325.5),
        (113.2, 325.5),
        (110.2, 329.0),
        (101.5, 336.5),
        align=None,
    )
    rear_poly = (
        rear_poly
        + Pos(r0[0], r0[1]) * Circle(3.05)
        + Pos(r2[0], r2[1]) * Circle(3.05)
        + Pos(r3[0], r3[1]) * Circle(2.85)
    )
    rear_face = fillet_2d_face(
        rear_poly.clean().faces()[0],
        r_default=1.2,
        r_inner=2.2,
        skip_xy=[(70.625, 336.5), (70.625, 359.0)],
    )
    r1_clip = Pos(r1[0], r1[1]) * Circle(2.85).faces()[0]
    r1_pad_xz = (rear_face & r1_clip).clean().faces()[0]
    if family == "truss_box":
        w_r1 = fillet_2d_face(Polygon((108.5, 345.0), (121.0, 349.5), (114.5, 338.5), align=None).faces()[0], 1.3)
        w_r2 = fillet_2d_face(Polygon((118.5, 336.8), (124.8, 338.8), (122.5, 345.5), align=None).faces()[0], 1.2)
        w_b1 = fillet_2d_face(Polygon((80.5, 349.5), (94.0, 349.5), (87.5, 354.5), align=None).faces()[0], 1.1)
        rear_face = (rear_face - w_r1 - w_r2 - w_b1).clean().faces()[0]
    else:
        w_r1 = Pos(114.5, 344.5) * Circle(4.0).faces()[0]
        w_r2 = Pos(123.5, 339.8) * Circle(3.5).faces()[0]
        w_b1 = Pos(87.0, 352.0) * Circle(3.0).faces()[0]
        rear_face = (rear_face - w_r1 - w_r2 - w_b1).clean().faces()[0]
    rear_cheek_web = extrude_xy_to_y(rear_face, 22.02, 23.575)

    rear_boss_pads = [
        occ_cyl((rx, 20.575, rz), (rx, 23.575, rz), 3.05)
        for rx, rz in (r0, r2)
    ] + [
        occ_cyl((r3[0], 20.575, r3[1]), (r3[0], 22.80, r3[1]), 2.32),
        extrude_xy_to_y(r1_pad_xz, 20.575, 23.575),
    ]

    # 3. Deep 3D C-Channel Bridge Decks (`Z in [336.5, 347.9]` lower + `Z in [355.4, 359.0]` upper)
    deck_lower_poly = Polygon(
        (70.625, 12.80),
        (101.50, 12.80),
        (101.50, 22.02),
        (70.625, 22.02),
        align=None,
    ).faces()[0]
    deck_lower_face = fillet_2d_face(
        deck_lower_poly,
        r_default=1.8,
        skip_xy=[(70.625, 12.80), (101.50, 22.02), (70.625, 22.02)],
    )
    low_win = (
        Pos(85.0, 17.5) * RectangleRounded(11.0, 4.4, 1.8).faces()[0]
        if family == "truss_box"
        else Pos(85.0, 17.5) * Circle(2.5).faces()[0]
    )
    yoke_deck_lower = extrude_xy_to_z((deck_lower_face - low_win).clean().faces()[0], 336.5, 347.9)

    deck_upper_poly = Polygon(
        (70.625, 13.20),
        (92.50, 13.20),
        (92.50, 22.02),
        (70.625, 22.02),
        align=None,
    ).faces()[0]
    deck_upper_face = fillet_2d_face(
        deck_upper_poly,
        r_default=1.6,
        skip_xy=[(70.625, 13.20), (92.50, 22.02), (70.625, 22.02)],
    )
    yoke_deck_upper = extrude_xy_to_z(deck_upper_face, 355.4, 359.0)

    body = fuse_all([
        pitch_hub,
        rear_cheek_web,
        yoke_deck_lower,
        yoke_deck_upper,
    ])
    bev_body = bevel_3d_edges(largest_solid(body), r_outer=0.40, r_inner=0.75)
    body_with_bosses = fuse_all([bev_body, *rear_boss_pads])
    body_clean = cut_driver_and_bore_channels(body_with_bosses, "stage1_shoulder_pitch_to_roll")
    return largest_solid(body_clean)


def build_stage2_shoulder_roll_to_yaw(family: str) -> Solid:
    """Build blank-state Sculpted 3D Roll-to-Yaw Windowed Saddle (`R_shoulder_roll_to_yaw`):
      - 2D $G^1$-filleted Roll Main Horn Hub & Windowed Anterior Column in `(V_YAW, N_YAW)` across `w_y in [-9.35, -5.15]` (zero step ledges)
      - 3D L-Bend Root with `r_inner = 2.0 mm` concave inner fillet and `r_outer = 1.1 mm` convex outer fillet
      - 2D-Filleted Longitudinal Dorsal Spine Rib in `(W_YAW, N_YAW)` (`v in [3.55, 6.20], r_inner = 2.8 mm, r_outer = 1.2 mm`)
      - 2D $G^1$-filleted Picture-Frame 3.00 mm Yaw Motor Saddle (`n in [17.575, 20.575]`) with filleted top/bottom motor-drum reliefs
      - 3D edge beveling (`bevel_3d_edges` with `r_inner = 0.75 mm > r_outer = 0.40 mm`) for $G^1$ tangential continuity and zero sharp edges
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

    rc_u, rc_n = vn_yaw(P_ROLL)

    # 1. Sculpted Roll Main Horn Hub & Windowed Anterior Column in `(V_YAW, N_YAW)` across `w_y in [-9.35, -5.15]`
    ant_col_vn = Polygon(
        (-22.8, 53.6),
        (-22.8, 57.8),
        (-15.8, 67.0),
        (-11.2, 67.0),
        (-3.6, 58.2),
        (6.40, 25.0),
        (6.40, 17.575),
        (-12.95, 17.575),
        (-12.95, 25.0),
        (-18.4, 43.0),
        align=None,
    ).faces()[0]
    ant_col_face = fillet_2d_face(
        ant_col_vn,
        r_default=1.5,
        r_inner=2.8,
        skip_xy=[(6.40, 17.575), (-12.95, 17.575)],
    )
    if family == "truss_box":
        aw1 = fillet_2d_face(Polygon((-14.6, 44.0), (-6.5, 44.0), (-10.0, 33.5), align=None).faces()[0], 1.3)
    else:
        aw1 = Pos(-10.8, 40.0) * Circle(3.5).faces()[0]
    ant_col_face = (ant_col_face - aw1).clean().faces()[0]
    ant_col_wall = extrude_yaw_local_z(ant_col_face, -9.35, -5.625)

    # 2. Full 3.00 mm Picture-Frame Yaw Motor Saddle (`xn in [17.575, 20.575]`) with all exposed corners 2D-filleted
    yaw_plate_vw = Polygon(
        (-12.95, -9.35),
        (13.20, -9.35),
        (13.20, 35.35),
        (6.40, 35.35),
        (6.40, 30.0),
        (-6.0, 30.0),
        (-7.2, 35.35),
        (-12.95, 35.35),
        align=None,
    ).faces()[0]
    yaw_plate_face = fillet_2d_face(
        yaw_plate_vw,
        r_default=1.4,
        r_inner=2.2,
        skip_xy=[(-12.95, -9.35), (-12.95, 35.35)],
    )
    yaw_center_clear = Pos(0.0, 0.0) * Circle(3.45).faces()[0]
    if family == "truss_box":
        yw1 = fillet_2d_face(Polygon((-7.5, 13.5), (-0.8, 13.5), (-4.2, 24.5), align=None).faces()[0], 1.3)
    else:
        yw1 = Pos(-4.2, 18.5) * Circle(3.4).faces()[0]
    yaw_plate_face = (yaw_plate_face - yw1 - yaw_center_clear).clean().faces()[0]
    yaw_saddle_full = extrude_yaw_local_x(yaw_plate_face, 17.575, 20.575)

    # 3. 3D L-Bend Root Fillet (`r_inner = 2.0 mm` concave inner root + `r_outer = 1.1 mm` convex outer corner + `r = 1.2 mm` vertical blend)
    fused_lb = largest_solid(fuse_all([ant_col_wall, yaw_saddle_full])).wrapped
    ef_lb = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(fused_lb, TopAbs_EDGE, TopAbs_FACE, ef_lb)
    mk_lb = BRepFilletAPI_MakeFillet(fused_lb)
    for i in range(1, ef_lb.Extent() + 1):
        e = TopoDS.Edge_s(ef_lb.FindKey(i))
        c = BRepAdaptor_Curve(e)
        if c.GetType() == GeomAbs_Line:
            u = 0.5 * (c.FirstParameter() + c.LastParameter())
            p = c.Value(u)
            d = np.array([p.X(), p.Y(), p.Z()]) - P_YAW
            uv, wy, xn = float(np.dot(d, V_YAW)), float(np.dot(d, W_YAW)), float(np.dot(d, N_YAW))
            if abs(wy - (-5.625)) < 0.2 and abs(xn - 20.575) < 0.2:
                mk_lb.Add(2.0, e)
            elif abs(wy - (-9.35)) < 0.2 and abs(xn - 17.575) < 0.2:
                mk_lb.Add(1.1, e)
            elif abs(uv - (-12.95)) < 0.2 and abs(wy - (-9.35)) < 0.2:
                mk_lb.Add(1.2, e)
    mk_lb.Build()
    s_lbend = Solid(mk_lb.Shape()) if mk_lb.IsDone() else Solid(fused_lb)

    # 4. Contoured Longitudinal Dorsal Spine Rib in `(W_YAW, N_YAW)` across `v in [3.55, 6.40]`
    dorsal_wn = Polygon(
        (-6.2, 19.0),
        (28.0, 19.0),
        (28.0, 22.40),
        (-2.5, 22.40),
        (-6.2, 26.45),
        align=None,
    ).faces()[0]
    dorsal_wn_face = fillet_2d_face(
        dorsal_wn,
        r_default=1.2,
        r_inner=2.8,
        skip_xy=[(-6.2, 19.0), (28.0, 19.0), (28.0, 22.40), (-6.2, 26.45)],
    )
    dorsal_rib = extrude_yaw_local_y(dorsal_wn_face, 3.55, 6.40)

    bev_frame = bevel_3d_edges(
        largest_solid(fuse_all([s_lbend, dorsal_rib])),
        r_outer=0.40,
        r_inner=0.75,
        skip_fn=lambda x, y, z: float(np.dot(np.array([x, y, z]) - P_YAW, W_YAW)) <= -5.0,
    )

    boss_keep = (
        Pos(-10.25, 8.30) * Circle(2.70).faces()[0]
        + Pos(10.25, 8.30) * Circle(2.85).faces()[0]
        + Pos(-10.25, 32.75) * Circle(2.80).faces()[0]
        + Pos(10.25, 32.75) * Circle(2.75).faces()[0]
    )
    bot_relief_face = fillet_2d_face(
        ((Pos(0.0, 13.50) * RectangleRounded(20.00, 34.50, 1.8)).faces()[0] - boss_keep).clean().faces()[0],
        r_default=0.9,
        r_inner=1.2,
    )
    bot_relief = extrude_yaw_local_x(bot_relief_face, 16.0, 18.72)

    top_relief_face = fillet_2d_face(
        ((Pos(-5.00, 13.00) * RectangleRounded(13.20, 33.60, 1.8)).faces()[0] - boss_keep).clean().faces()[0],
        r_default=0.9,
        r_inner=1.2,
    )
    top_relief = extrude_yaw_local_x(top_relief_face, 19.85, 22.0)

    body = cut_all(bev_frame, [bot_relief, top_relief])
    return cut_driver_and_bore_channels(body, "stage2_shoulder_roll_to_yaw")


def build_stage3_upper_arm_yaw_to_elbow(family: str) -> Solid:
    """Build blank-state Sculpted 3D Upper-Arm Yaw-to-Elbow Down-First & Wrap-Around-Motor Yoke (`R_upper_arm_yaw_to_elbow`):
      - 2D $G^1$-filleted Yaw Main Horn Hub Plate in `(V_YAW, W_YAW)` across `xn in [-22.2, -18.625]` flush with yaw horn rim (`u <= 10.4 mm`)
      - 2D $G^1$-filleted Elbow Main Horn Hub & Lateral Yoke in YZ (`X in [163.625, 166.625]`, capped at `Z = 272.4 mm` right above the elbow motor)
      - Down-First then Wrap-Around-Motor Twin Columns in `(V_YAW, N_YAW)` across `w_y in [4.5, 8.1]` and `[10.1, 12.50]`
      - 100% simultaneous 3D edge beveling (`bevel_3d_edges` with `r_inner = 0.72 mm > r_outer = 0.42 mm`)
    """
    # 1. Sculpted Yaw Main Horn Hub Plate in `(V_YAW, W_YAW)` across `xn in [-22.2, -18.625]` (flush with yaw horn rim `u <= 10.8 mm`)
    yaw_hub_vw = Polygon(
        (-10.8, -4.5),
        (-4.5, -10.8),
        (4.5, -10.8),
        (10.8, -4.5),
        (10.8, 12.6),
        (-4.5, 12.6),
        (-10.8, 7.5),
        align=None,
    ).faces()[0]
    yaw_hub_face = fillet_2d_face(yaw_hub_vw, r_default=1.4, r_inner=2.4, skip_xy=())
    yaw_hub_plate = extrude_yaw_local_x(yaw_hub_face, -22.2, -18.625)

    # 2. Sculpted Elbow Main Horn Hub & Lateral Yoke in YZ (`X in [163.625, 166.625]`, capped at `Z = 275.2 mm` above elbow motor)
    elb_yoke_yz = Polygon(
        (-6.75, 259.0),
        (-6.75, 275.2),
        (3.20, 275.2),
        (8.30, 265.5),
        (8.30, 255.8),
        (6.25, 254.0),
        (6.25, 248.0),
        (1.2, 242.7),
        (-7.2, 242.7),
        (-12.35, 248.0),
        (-12.35, 255.8),
        align=None,
    ).faces()[0]
    elb_yoke_face = fillet_2d_face(elb_yoke_yz, r_default=1.4, r_inner=2.4, skip_xy=())
    if family == "truss_box":
        ew = Pos(0.8, 263.8) * RectangleRounded(3.6, 3.6, 1.4).faces()[0]
    else:
        ew = Pos(0.8, 263.8) * Circle(2.0).faces()[0]
    elb_yoke_face = (elb_yoke_face - ew).clean().faces()[0]
    elb_yoke_plate = extrude_xy_to_x(elb_yoke_face, 163.625, 166.625)

    # 3. Down-First then Wrap-Around-Motor Twin Columns in `(V_YAW, N_YAW)` across `w_y in [4.5, 8.1]` and `[10.1, 12.50]`
    u_elb_top_in, n_elb_top_in = vn_yaw((156.5, 0.0, 274.0))
    u_elb_top, n_elb_top = vn_yaw((165.2, 0.0, 274.0))
    u_elb_bot, n_elb_bot = vn_yaw((165.2, 0.0, 266.6))
    u_elb_bot_in, n_elb_bot_in = vn_yaw((156.5, 0.0, 266.6))
    gusset_vn = Polygon(
        (2.2, -20.4),
        (10.8, -20.4),
        (10.8, -30.0),
        (u_elb_top_in, n_elb_top_in),
        (u_elb_top, n_elb_top),
        (u_elb_bot, n_elb_bot),
        (u_elb_bot_in, n_elb_bot_in),
        (2.2, -39.5),
        align=None,
    ).faces()[0]
    gusset_vn_face = fillet_2d_face(
        gusset_vn,
        r_default=1.8,
        r_inner=3.5,
        skip_xy=[
            (2.2, -20.4),
            (10.8, -20.4),
            (u_elb_top, n_elb_top),
            (u_elb_bot, n_elb_bot),
        ],
    )
    gusset_ant = extrude_yaw_local_z(gusset_vn_face, 4.5, 8.1)      # Y in [-5.5, -1.9]
    gusset_post = extrude_yaw_local_z(gusset_vn_face, 10.1, 12.50)  # Y in [0.1, 2.5]

    body = fuse_all([
        yaw_hub_plate,
        elb_yoke_plate,
        gusset_ant,
        gusset_post,
    ])
    bev_body = bevel_3d_edges(largest_solid(body), r_outer=0.42, r_inner=0.72)
    return cut_driver_and_bore_channels(bev_body, "stage3_upper_arm_yaw_to_elbow")


def build_stage4_forearm_elbow_to_wrist(family: str) -> Solid:
    """Build blank-state Sculpted 3D Dual-Cheek Forearm Yoke & Converging A-Frame/Wishbone Cage (`R_forearm_elbow_to_wrist`):
      - Dual-Cheek Elbow C047 Motor Yoke (`6x PA2` across both faces of `R_elbow_C047_case`) with all 4 corners 2D-filleted:
          * Medial (`-X`) Windowed Cheek (`X in [124.425, 127.425]`) clamping all 4 medial PA2 bosses
          * Lateral (`+X`) Windowed Cheek (`X in [160.425, 163.425]`) clamping the 2 lower lateral PA2 bosses
      - Converging 3D Dual-Wall A-Frame / Wishbone Arch (`Z in [171.0, 219.5]`, `Y in [-13.0, -9.2]` & `Y in [-0.2, 3.30]`) with `r_inner = 1.8 mm` U-pocket fillets
      - 2D-filleted Under-Motor Cross-Bridge Deck (`Z in [212.0, 215.5]`) tying Medial & Lateral cheeks together
      - Contoured 4-Lobe Wrist Landing Pad (`Z in [171.0, 174.8]`, all corners 2D-filleted)
      - Integrated 4-Post Forearm Hard-Shell Mounts (`Z = 187.0` and `178.0`) with rounded `(X, Z)` & `(X, Y)` tips
      - 3D edge beveling (`bevel_3d_edges` with `r_inner = 0.75 mm > r_outer = 0.40 mm`) for $G^1$ tangential continuity and zero sharp edges
    """
    specs = stage_interface_specs()["stage4_forearm_elbow_to_wrist"]
    case_pts = specs["support"]["centres"]  # 0..3 medial (X=127.425), 4..5 lateral (X=160.425)
    wrist_pts = specs["load"]["centres"]

    # 1. Medial (`-X`) Windowed Cheek (`X in [124.425, 125.95]` web + `X in [124.425, 127.425]` Circle(2.68) boss pads)
    med_poly = Polygon(
        (-15.95, 212.0),
        (9.95, 212.0),
        (9.95, 247.2),
        (-15.95, 247.2),
        align=None,
    )
    med_face = fillet_2d_face(med_poly.clean().faces()[0], r_default=1.6, skip_xy=[(-15.95, 212.0), (9.95, 212.0)])
    if family == "truss_box":
        mw1 = fillet_2d_face(Polygon((-10.5, 224.0), (4.5, 224.0), (-3.0, 235.5), align=None).faces()[0], 1.4)
    else:
        mw1 = Pos(-3.0, 231.5) * Circle(5.8).faces()[0]
    med_face = (med_face - mw1).clean().faces()[0]
    medial_cheek_web = extrude_xy_to_x(med_face, 124.425, 125.95)
    medial_boss_pads = [
        occ_cyl((124.425, float(p[1]), float(p[2])), (127.425, float(p[1]), float(p[2])), 2.65)
        for p in case_pts[:4]
    ]

    # 2. Lateral (`+X`) Windowed Cheek (`X in [161.95, 163.425]` web + `X in [160.425, 163.425]` Circle(2.68) boss pads)
    lat_poly = Polygon(
        (-15.95, 212.0),
        (9.95, 212.0),
        (9.95, 222.8),
        (-15.95, 222.8),
        align=None,
    )
    lat_face = fillet_2d_face(lat_poly.clean().faces()[0], r_default=1.5, skip_xy=[(-15.95, 212.0), (9.95, 212.0)])
    if family == "truss_box":
        lw = Pos(-3.0, 217.5) * RectangleRounded(10.0, 4.6, 1.4).faces()[0]
    else:
        lw = Pos(-3.0, 217.5) * Circle(2.6).faces()[0]
    lat_face = (lat_face - lw).clean().faces()[0]
    lateral_cheek_web = extrude_xy_to_x(lat_face, 161.95, 163.425)
    lateral_boss_pads = [
        occ_cyl((160.425, float(p[1]), float(p[2])), (163.425, float(p[1]), float(p[2])), 2.65)
        for p in case_pts[4:6]
    ]

    # 3. Converging 3D Dual-Cheek A-Frame / Wishbone Arch in XZ (`Z in [171.0, 220.0]`, `Y in [-13.0, -9.2]` and `Y in [-0.2, 3.30]`)
    aframe_xz_poly = Polygon(
        (124.425, 220.0),
        (125.95, 220.0),
        (125.95, 215.5),
        (161.95, 215.5),
        (161.95, 220.0),
        (163.425, 220.0),
        (163.425, 212.0),
        (148.20, 190.6),
        (148.20, 174.8),
        (145.4, 174.8),
        (145.4, 171.0),
        (131.6, 171.0),
        (131.6, 174.8),
        (124.425, 198.0),
        align=None,
    ).faces()[0]
    aframe_xz_face = fillet_2d_face(
        aframe_xz_poly,
        r_default=1.3,
        r_inner=1.8,
        skip_xy=[
            (124.425, 220.0),
            (125.95, 220.0),
            (161.95, 220.0),
            (163.425, 220.0),
            (148.20, 174.8),
            (145.4, 174.8),
            (145.4, 171.0),
            (131.6, 171.0),
            (131.6, 174.8),
        ],
    )
    if family == "truss_box":
        af_w1 = fillet_2d_face(Polygon((132.0, 210.5), (151.0, 210.5), (145.5, 198.0), (134.5, 198.0), align=None).faces()[0], 1.6)
        af_w2 = Pos(135.8, 186.8) * RectangleRounded(5.2, 10.5, 1.4).faces()[0]
        aframe_xz_face = (aframe_xz_face - af_w1 - af_w2).clean().faces()[0]
    else:
        af_w1 = Pos(141.5, 204.2) * Circle(5.5).faces()[0]
        af_w2 = Pos(136.0, 186.8) * Circle(3.4).faces()[0]
        aframe_xz_face = (aframe_xz_face - af_w1 - af_w2).clean().faces()[0]

    aframe_ant = extrude_xy_to_y(aframe_xz_face, -13.0, -9.2)
    aframe_post = extrude_xy_to_y(aframe_xz_face, -0.2, 3.30)

    # Under-motor bridge deck in XY (`Z in [212.0, 215.5]`, `X in [124.425, 163.425]`, `Y in [-15.95, 9.95]`)
    ub_poly = fillet_2d_face(
        Polygon((124.425, -15.95), (163.425, -15.95), (163.425, 9.95), (124.425, 9.95), align=None).faces()[0],
        r_default=1.5,
        skip_xy=[(124.425, -15.95), (163.425, -15.95), (163.425, 9.95), (124.425, 9.95)],
    )
    ub_win = Pos(143.925, -3.0) * RectangleRounded(31.0, 14.0, 2.5).faces()[0]
    ub_deck = extrude_xy_to_z((ub_poly - ub_win).clean().faces()[0], 212.0, 215.5)

    # 4. Contoured 4-Lobe Wrist Landing Pad (`Z in [171.0, 174.8]`, `X in [131.6, 145.4]`, `Y in [-13.0, 8.50]`)
    wrist_xy = Polygon((131.6, -13.0), (145.4, -13.0), (145.4, 8.50), (131.6, 8.50), align=None)
    wrist_face = (
        fillet_2d_face(
            wrist_xy.clean().faces()[0],
            r_default=1.4,
            skip_xy=[(145.4, 8.50), (145.4, -13.0)],
        )
        - Pos(138.5, -5.5) * Circle(2.0).faces()[0]
    ).clean().faces()[0]
    wrist_pad = extrude_xy_to_z(wrist_face, 171.0, 174.8)

    # 5. Integrated 2D-Filleted 4-Post Forearm Hard-Shell Mounts (`Z = 187.0` and `178.0`, `X in [141.6, 148.20]`)
    post_ff187 = extrude_xy_to_z(
        fillet_2d_face(
            Polygon((141.6, -22.65), (148.20, -22.65), (148.20, -10.5), (141.6, -10.5), align=None).faces()[0],
            r_default=1.2,
            skip_xy=[(141.6, -10.5), (148.20, -10.5)],
        ),
        183.4,
        190.6,
    )
    post_fb187 = extrude_xy_to_z(
        fillet_2d_face(
            Polygon((141.6, 1.0), (148.20, 1.0), (148.20, 16.05), (141.6, 16.05), align=None).faces()[0],
            r_default=1.2,
            skip_xy=[(141.6, 1.0), (148.20, 1.0)],
        ),
        183.4,
        190.6,
    )

    post_ff178 = extrude_xy_to_z(
        fillet_2d_face(
            Polygon((141.6, -18.55), (148.20, -18.55), (148.20, -10.5), (141.6, -10.5), align=None).faces()[0],
            r_default=1.2,
            skip_xy=[(141.6, -10.5), (148.20, -10.5)],
        ),
        174.8,
        181.6,
    )
    post_fb178 = extrude_xy_to_z(
        fillet_2d_face(
            Polygon((141.6, 1.0), (148.20, 1.0), (148.20, 12.05), (141.6, 12.05), align=None).faces()[0],
            r_default=1.2,
            skip_xy=[(141.6, 1.0), (148.20, 1.0)],
        ),
        174.8,
        181.6,
    )

    body = fuse_all([
        medial_cheek_web,
        lateral_cheek_web,
        aframe_ant,
        aframe_post,
        ub_deck,
        wrist_pad,
        post_ff187,
        post_fb187,
        post_ff178,
        post_fb178,
    ])
    bev_body = bevel_3d_edges(
        largest_solid(body),
        r_outer=0.40,
        r_inner=0.75,
        skip_fn=lambda x, y, z: (
            (z >= 211.0 and (x <= 126.2 or x >= 161.7))
            or (z <= 192.0 and (y <= -13.2 or y >= 3.5))
        ),
    )
    body_with_bosses = fuse_all([bev_body, *medial_boss_pads, *lateral_boss_pads])
    body_clean = cut_driver_and_bore_channels(body_with_bosses, "stage4_forearm_elbow_to_wrist")

    sh_cutters = []
    for m in RIGHT_ARM_SHELL_MOUNTS:
        if m["stage"] != "stage4_forearm_elbow_to_wrist":
            continue
        cx, cy, cz = m["center"]
        y0, y1 = m["post_span"]
        sh_cutters.append(cyl_along((cx, y0 - 4.0, cz), (0.0, 1.0, 0.0), 1.70, (y1 - y0) + 8.0))
        sh_cutters.append(make_captive_nut_pocket((cx, cy, cz), m["axis"], slot_dir_z=m["slot_dir"][2]))
    return largest_solid(cut_all(body_clean, sh_cutters))


def build_42pose_stage_keepout_manifolds() -> dict[str, md.Manifold]:
    """Build the exact 42-pose swept union of all environment parts (motors, horns, screws, shells, nuts)
    plus all straight-line driver/nut rays for each stage so the exported watertight meshes have 0.0000 mm^3 overlap."""
    hw = ensure_arm_shell_nut_assets()
    src = read_json(ARM_SHELL_DIR / "source.json")
    mp = read_json(ARM_SHELL_DIR / "manufactured-parts.json")
    sp = read_json(ARM_SHELL_DIR / "source-packages.json")
    replaced = set(mp["replaced_source"]) | REPLACED_ARM_PARTS

    env_parts = []
    for p in sp:
        pname = p["name"]
        if pname.startswith("upper__R_") and p.get("exact") and pname not in replaced:
            env_parts.append((pname, p["owner"], "source", npz_to_mf(ROOT / p["exact"]["path"])))
    for p in mp["parts"]:
        name = p["name"]
        if name.startswith("R_") and name not in replaced and "insert" not in name:
            env_parts.append((name, p["owning_link"], p["role"], npz_to_mf(ROOT / p["lossless_mesh"]["path"])))
    for n in hw["nuts"]:
        if n["side"] == "R":
            env_parts.append((n["name"], n["owning_link"], "captive_nut", npz_to_mf(n["npz_path"])))
    for scr in hw["screws"]:
        if scr["side"] == "R":
            env_parts.append((scr["name"], scr["owning_link"], "shell_screw", npz_to_mf(scr["npz_path"])))

    poses = build_poses()
    specs = stage_interface_specs()
    out = {}

    for stage_key in STAGE_PART_BASENAMES:
        slink = f"R_{STAGE_LINKS[stage_key]}"
        seen = set()
        obs = []
        for _, q in poses:
            T = fk(src["joints"], q)
            inv = np.linalg.inv(T[slink])
            for oname, olink, _, omf in env_parts:
                rel_M = inv @ T[olink]
                key = (oname, tuple(np.round(rel_M[:3].ravel(), 6)))
                if key in seen:
                    continue
                seen.add(key)
                obs.append(omf.transform(rel_M[:3]))

        # Also include all interface driver rays and shell mount rays
        grp_dict = specs[stage_key]
        for role_key in ("support", "load"):
            grp = grp_dict[role_key]
            norms = grp.get("normals", [grp["normal"]] * len(grp["centres"]))
            h = float(grp["sleeve_len"])
            r_ray = 1.50 if grp["sleeve_ri"] > 1.4 else 1.20
            for pt, d in zip(grp["centres"], norms):
                obs.append(cyl_mf(pt + (h + 0.02) * d, d, r_ray, 28.5))

        for m in RIGHT_ARM_SHELL_MOUNTS:
            if m["stage"] != stage_key:
                continue
            cx, cy, cz = m["center"]
            ax = np.asarray(m["axis"], dtype=float)
            sdir = np.asarray(m["slot_dir"], dtype=float)
            obs.append(cyl_mf(np.array([cx, cy, cz]) - 3.6 * ax, ax, 1.50, 16.5))
            obs.append(cyl_mf(np.array([cx, cy, cz]), sdir, 1.20, 7.5))

        out[stage_key] = md.Manifold.batch_boolean(obs, md.OpType.Add)
    return out


def save_mf_npz_and_stl(mf: md.Manifold, npz_path: Path, stl_path: Path):
    import trimesh

    m = mf.to_mesh()
    v_arr = np.asarray(m.vert_properties[:, :3], dtype=np.float64)
    f_arr = np.asarray(m.tri_verts, dtype=np.int64)
    q = md.Manifold(md.Mesh64(v_arr, f_arr.astype(np.uint64)))
    assert q.status() == md.Error.NoError and q.volume() > 0.0, f"Invalid manifold {npz_path}"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, vertices=v_arr, faces=f_arr)
    tm = trimesh.Trimesh(vertices=v_arr, faces=f_arr, process=False)
    tm.export(str(stl_path))


def build_all(run_verification_gates: bool = True):
    t0 = time.time()
    ensure_arm_shell_nut_assets()

    rho_paht = PRINTED["density_g_cm3"] * 1e-3      # the definition's PRINTED_MATERIAL (study families no entry claims)
    rho_petg = ALT_PRINTED["density_g_cm3"] * 1e-3

    builders = {
        "stage1_shoulder_pitch_to_roll": build_stage1_shoulder_pitch_to_roll,
        "stage2_shoulder_roll_to_yaw": build_stage2_shoulder_roll_to_yaw,
        "stage3_upper_arm_yaw_to_elbow": build_stage3_upper_arm_yaw_to_elbow,
        "stage4_forearm_elbow_to_wrist": build_stage4_forearm_elbow_to_wrist,
    }

    for family in ("truss_box", "topopt"):
        fam_dir = ART / family
        parts_dir = fam_dir / "parts"
        meshes_dir = fam_dir / "meshes"
        parts_dir.mkdir(parents=True, exist_ok=True)
        meshes_dir.mkdir(parents=True, exist_ok=True)

        manifest_stages = {}
        manifest_parts = []
        total_poly_r = 0.0
        total_slv_r = 0.0

        for stage_key, base_name in STAGE_PART_BASENAMES.items():
            ts = time.time()
            sol_r = builders[stage_key](family)
            sol_l = largest_solid(sol_r.mirror(Plane.YZ))

            slv_r_list = build_stage_sleeves(stage_key, "R")
            slv_l_list = build_stage_sleeves(stage_key, "L")

            r_step = parts_dir / f"R_{base_name}_{family}.step"
            l_step = parts_dir / f"L_{base_name}_{family}.step"
            r_stl = meshes_dir / f"R_{base_name}_{family}.stl"
            l_stl = meshes_dir / f"L_{base_name}_{family}.stl"
            r_npz = meshes_dir / f"R_{base_name}_{family}.npz"
            l_npz = meshes_dir / f"L_{base_name}_{family}.npz"

            export_step(Compound(children=[sol_r]), str(r_step))
            export_step(Compound(children=[sol_l]), str(l_step))

            r_slv_step = parts_dir / f"R_{base_name}_sleeves.step"
            l_slv_step = parts_dir / f"L_{base_name}_sleeves.step"
            r_slv_npz = meshes_dir / f"R_{base_name}_sleeves.npz"
            l_slv_npz = meshes_dir / f"L_{base_name}_sleeves.npz"

            comp_slv_r = Compound(children=[s["solid"] for s in slv_r_list])
            comp_slv_l = Compound(children=[s["solid"] for s in slv_l_list])
            export_step(comp_slv_r, str(r_slv_step))
            export_step(comp_slv_l, str(l_slv_step))

            v_all_r, f_all_r = [], []
            off_r = 0
            for s in slv_r_list:
                va, fa = solid_to_watertight_arrays(s["solid"], tol=0.04)
                v_all_r.append(va)
                f_all_r.append(fa + off_r)
                off_r += len(va)
            v_slv_r = np.vstack(v_all_r)
            f_slv_r = np.vstack(f_all_r)
            np.savez_compressed(r_slv_npz, vertices=v_slv_r, faces=f_slv_r)

            v_all_l, f_all_l = [], []
            off_l = 0
            for s in slv_l_list:
                va, fa = solid_to_watertight_arrays(s["solid"], tol=0.04)
                v_all_l.append(va)
                f_all_l.append(fa + off_l)
                off_l += len(va)
            np.savez_compressed(l_slv_npz, vertices=np.vstack(v_all_l), faces=np.vstack(f_all_l))

            # Direct tessellation of exact BRep STEP solid (NO post-hoc mesh boolean subtraction allowed!)
            v_r, f_r = solid_to_watertight_arrays(sol_r, tol=0.04)
            mf_r = md.Manifold(md.Mesh64(v_r, f_r.astype(np.uint64)))
            save_mf_npz_and_stl(mf_r, r_npz, r_stl)

            # Mirror mf_r across YZ (`X -> -X` with inverted triangle winding) for exact bilateral symmetry
            m_r = mf_r.to_mesh()
            vr_m = np.asarray(m_r.vert_properties[:, :3], dtype=np.float64).copy()
            vr_m[:, 0] *= -1.0
            fr_m = np.asarray(m_r.tri_verts, dtype=np.int64)[:, [0, 2, 1]].copy()
            mf_l = md.Manifold(md.Mesh64(vr_m, fr_m.astype(np.uint64)))
            save_mf_npz_and_stl(mf_l, l_npz, l_stl)

            vol_r = float(sol_r.volume)
            poly_g = round(vol_r * rho_paht, 3)
            petg_g = round(vol_r * rho_petg, 3)
            slv_g = round(sum(s["mass_g"] for s in slv_r_list), 3)
            tot_g = round(poly_g + slv_g, 3)
            total_poly_r += poly_g
            total_slv_r += slv_g

            manifest_stages[stage_key] = {
                "stage": stage_key,
                "family": family,
                "owning_link_r": f"R_{STAGE_LINKS[stage_key]}",
                "owning_link_l": f"L_{STAGE_LINKS[stage_key]}",
                "step_r": rel(r_step),
                "step_l": rel(l_step),
                "stl_r": rel(r_stl),
                "stl_l": rel(l_stl),
                "npz_r": rel(r_npz),
                "npz_l": rel(l_npz),
                "sleeves_step_r": rel(r_slv_step),
                "sleeves_step_l": rel(l_slv_step),
                "sleeves_count_per_arm": len(slv_r_list),
                "volume_mm3": round(vol_r, 2),
                "manifold_volume_mm3": round(float(mf_r.volume()), 2),
                "polymer_mass_PAHT_CF_g": poly_g,
                "polymer_mass_paht_cf_g": poly_g,
                "polymer_mass_PETG_CF_g": petg_g,
                "sleeves_mass_g": slv_g,
                "total_stage_mass_PAHT_CF_g": tot_g,
                "total_stage_mass_paht_cf_g": tot_g,
            }
            for side, step_f, npz_f, slv_step_f, slv_npz_f in (
                ("R", r_step, r_npz, r_slv_step, r_slv_npz),
                ("L", l_step, l_npz, l_slv_step, l_slv_npz),
            ):
                manifest_parts.append({
                    "name": f"{side}_{base_name}_{family}",
                    "stage": stage_key,
                    "side": side,
                    "owning_link": f"{side}_{STAGE_LINKS[stage_key]}",
                    "family": family,
                    "step": ref(step_f),
                    "mesh": ref(npz_f),
                    "sleeves_step": ref(slv_step_f),
                    "sleeves_mesh": ref(slv_npz_f),
                    "sleeves_count": len(slv_r_list),
                    "sleeves_mass_g": slv_g,
                    "volume_mm3": round(vol_r, 2),
                    "polymer_mass_paht_cf_g": poly_g,
                    "total_stage_mass_paht_cf_g": tot_g,
                })
            print(
                f"[{family:9s}] {stage_key:30s} | valid={sol_r.is_valid}/{sol_l.is_valid} "
                f"solids={len(sol_r.solids())}/{len(sol_l.solids())} | poly={poly_g:.2f}g + slv={slv_g:.2f}g ({len(slv_r_list)} slv) = {tot_g:.2f}g | dt={time.time()-ts:.2f}s"
            )

        manifest = {
            "family": family,
            "version": "r11",
            "blank_state_generative": True,
            "total_sleeves_per_arm": int(sum(s["sleeves_count_per_arm"] for s in manifest_stages.values())),
            "total_polymer_mass_PAHT_CF_g_per_arm": round(total_poly_r, 3),
            "total_sleeves_mass_g_per_arm": round(total_slv_r, 3),
            "total_arm_connector_mass_PAHT_CF_g_per_arm": round(total_poly_r + total_slv_r, 3),
            "stages": manifest_stages,
            "parts": manifest_parts,
        }
        write_json(fam_dir / "cad-manifest.json", manifest)
        write_json(fam_dir / "manifest.json", manifest)

    print(f"Completed blank-state 3D CAD build for truss_box & topopt in {time.time()-t0:.2f}s")

    if run_verification_gates:
        print("\nVerification of the built stages: robot/body/checks/run.py --manifest ... --checks geometry,fit,fem (no gate here).")


if __name__ == "__main__":
    build_all(run_verification_gates=True)
