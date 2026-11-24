"""Design kit, connectors: the B-rep building pieces of docs/design/design-kit.md section 3 (U / H / I / Y connectors
that grip each actuator from two sides, flat members, arcs about the joint axis, rounded everywhere).

The reference builder is robot/body/arm/shoulder_clevis.py (the blue and green shoulder U). A connector is built from:

  arms       flat plates on the actuator's horn face and idler face: a 2D outline (`hub_fan`, `rounded_rect`, or any face),
             rounded with `fillet_2d` (plan radii 1-3 mm, larger inside), extruded across the face normal (`extrude_y` /
             `extrude_z`)
  bar        what joins the arms: a plain plate, or an arc about the axis the connector turns about (`arc_band`), so it
             sweeps a thin shell of revolution a closed cover can hug
  hubs       about an axis: radius = bolt circle + sleeve outer radius + wall (`hub_radius`)
  roots      `root_fillet`: a concave fillet block along the corner between an arm's inner face and the bar
  sleeves    a pressed 304 sleeve in every clamped printed hole (`SLEEVES`, `sleeve`), the bore = the sleeve's outer
             radius (`bore`)
  edges      `bevel_3d` (r_outer ~0.4 touch, r_inner ~0.75 stress, sleeve bores kept)

The OCC helpers work on build123d solids. Nothing here knows a part, link or joint; positions, faces and axes come in
as arguments. Measurements only; nothing accepts or rejects.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

# 304 stainless compression sleeves (outer radius, bore radius, mm): every clamped printed hole carries one (owner
# 2026-11-18); the hole in the print is cut at the outer radius and the sleeve is pressed in (pressable: flat counter
# face and a path for the heated press, owner 2026-11-17)
SLEEVES = {"M3": (2.25, 1.60),    # 4.5 x 3.2
           "M2": (1.80, 1.10)}    # 3.6 x 2.2
HUB_WALL = 2.75                   # printed wall outside a sleeve on a hub (shoulder U: r 7 + 2.25 + 2.75 = 12)
PLAN_RADIUS = (1.0, 3.0)          # 2D plan corner radii in use (arc ends 1.0, arms and plates 3.0)
ROOT_FILLET = (1.2, 1.5)          # arm-to-bar root fillets in use
RUN_CLEAR = 1.5                   # separate bodies moving against each other: 1.5-2 mm (owner); never a slit under 1 mm
SEAT_FIT = 0.1                    # an arm seated on a horn / idler face, per side


def hub_radius(bolt_circle_r, thread="M3", wall=HUB_WALL):
    """Hub radius about an axis that carries sleeves on a bolt circle: bolt circle + sleeve outer radius + wall."""
    return bolt_circle_r + SLEEVES[thread][0] + wall


# ---------------------------------------------------------------------------------------------- OCC primitives
def _occ():
    from build123d import Solid, Compound
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse, BRepAlgoAPI_Common
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    return locals()


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box (any order of the bounds)."""
    o = _occ()
    return o["Solid"](o["BRepPrimAPI_MakeBox"](o["gp_Pnt"](min(x0, x1), min(y0, y1), min(z0, z1)),
                                               o["gp_Pnt"](max(x0, x1), max(y0, y1), max(z0, z1))).Shape())


def cyl(p0, p1, r):
    """Cylinder of radius r from p0 to p1."""
    o = _occ()
    v = np.asarray(p1, float) - np.asarray(p0, float); L = float(np.linalg.norm(v)); d = v / L
    ax = o["gp_Ax2"](o["gp_Pnt"](*map(float, p0)), o["gp_Dir"](*map(float, d)))
    return o["Solid"](o["BRepPrimAPI_MakeCylinder"](ax, float(r), L).Shape())


def boolean(kind, a, others):
    """'fuse' / 'cut' / 'common' of solid a with each of `others` in turn; a Compound."""
    o = _occ()
    op = {"fuse": o["BRepAlgoAPI_Fuse"], "cut": o["BRepAlgoAPI_Cut"], "common": o["BRepAlgoAPI_Common"]}[kind]
    cur = a.wrapped
    for s in others:
        b = op(cur, s.wrapped); b.Build(); cur = b.Shape()
    return o["Compound"](cur)


def largest(sh):
    """The largest solid of a shape."""
    return max(sh.solids(), key=lambda s: s.volume)


def transformed(shape, T):
    """shape moved by the 4x4 rigid transform T (rotation orthonormalised)."""
    from build123d import Compound
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    U, _, Vt = np.linalg.svd(T[:3, :3]); R = U @ Vt; t = T[:3, 3]
    tr = gp_Trsf()
    tr.SetValues(*[float(v) for v in (R[0, 0], R[0, 1], R[0, 2], t[0], R[1, 0], R[1, 1], R[1, 2], t[1], R[2, 0], R[2, 1], R[2, 2], t[2])])
    return largest(Compound(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape()))


def mirror_x(sh):
    """The left part of a right part (x -> -x)."""
    from build123d import Plane
    return sh.mirror(Plane.YZ)


# ---------------------------------------------------------------------------------------------- sleeves and bores
def sleeve(p_in, p_out, ro, ri):
    """Steel sleeve from p_in to p_out (flush with both faces of the part), outer radius ro, bore radius ri."""
    a, b = np.asarray(p_in, float), np.asarray(p_out, float); d = (b - a) / np.linalg.norm(b - a)
    return largest(boolean("cut", cyl(a, b, ro), [cyl(a - 0.5 * d, b + 0.5 * d, ri)]))


def bore(p_in, p_out, thread="M3", over=1.0):
    """The printed hole a sleeve is pressed into: its outer radius, `over` mm past both faces."""
    a, b = np.asarray(p_in, float), np.asarray(p_out, float); d = (b - a) / np.linalg.norm(b - a)
    return cyl(a - over * d, b + over * d, SLEEVES[thread][0])


# ---------------------------------------------------------------------------------------------- 2D outlines
def _common():
    if str(ROOT / "robot/body/leg") not in sys.path:
        sys.path.insert(0, str(ROOT / "robot/body/leg"))
    import common as C   # fillet_2d_face, bevel_3d_edges, extrude_xy_to_y / _z (the shared builder helpers)
    return C


def fillet_2d(face, r, r_inner=None, skip_xy=()):
    """Round every corner of a planar face: convex corners r, concave corners r_inner (default 1.5 r)."""
    return _common().fillet_2d_face(face, r, skip_xy=skip_xy, r_inner=r_inner)


def extrude_y(face_xz, y0, y1):
    """Extrude a face sketched in (x, z) across y0..y1 (an arm on a face normal to y)."""
    return _common().extrude_xy_to_y(face_xz, y0, y1)


def extrude_z(face_xy, z0, z1):
    """Extrude a face sketched in (x, y) across z0..z1."""
    return _common().extrude_xy_to_z(face_xy, z0, z1)


def bevel_3d(solid, r_outer=0.42, r_inner=0.75, min_cyl_r=4.6):
    """3D edge rounding: r_outer on convex edges, r_inner on concave ones; bores under min_cyl_r kept."""
    return _common().bevel_3d_edges(solid, r_outer=r_outer, r_inner=r_inner, min_cyl_r=min_cyl_r)


def rounded_rect(a0, a1, b0, b1, r):
    """Rounded rectangle a0..a1 x b0..b1 (a face) with corner radius r."""
    from build123d import Pos, RectangleRounded
    return (Pos(0.5 * (a0 + a1), 0.5 * (b0 + b1)) * RectangleRounded(a1 - a0, b1 - b0, r)).faces()[0]


def wedge(cx, cz, r, psi0, psi1, n=7):
    """Sector polygon about (cx, cz) out to r over psi0..psi1 (deg; psi from +second axis toward +first axis)."""
    from build123d import Polygon
    t = np.radians(np.linspace(psi0, psi1, n))
    return Polygon(*([(cx, cz)] + [(cx + r * np.sin(a), cz + r * np.cos(a)) for a in t]), align=None)


def arc_band(cx, cz, r_in, thickness, psi0, psi1, corner_r=1.0):
    """The cross-section of an arc bar about an axis through (cx, cz) (in the sketch plane): the ring r_in ..
    r_in + thickness over psi0..psi1, corners rounded corner_r. A face; extrude it along the axis."""
    from build123d import Circle, Pos
    ro = r_in + thickness
    band = (Pos(cx, cz) * Circle(ro) - Pos(cx, cz) * Circle(r_in)) & wedge(cx, cz, 2 * ro, psi0, psi1)
    return fillet_2d(band.faces()[0], corner_r, r_inner=corner_r)


def hub_fan(cx, cz, hub_r, fan_r, psi0, psi1, r=2.0, r_inner=4.0):
    """An arm outline: a hub of hub_r about (cx, cz) plus a fan out to fan_r over psi0..psi1 (it meets an arc bar of
    outer radius fan_r), rounded r outside and r_inner inside. A face."""
    from build123d import Circle, Pos
    fan = (Pos(cx, cz) * Circle(fan_r)) & wedge(cx, cz, 2 * fan_r, psi0, psi1)
    return fillet_2d((fan + Pos(cx, cz) * Circle(hub_r)).clean().faces()[0], r, r_inner=r_inner)


def root_fillet(corner_x, face_y, side, r, z0, z1):
    """Concave fillet block of radius r in the corner between a face normal to y at face_y (the arm's inner face, the
    bar on the side -side of it in y) and a plane normal to x at corner_x (the bar top, material below x): the box
    corner_x - r .. corner_x x face_y - side r .. face_y minus the cylinder along z at (corner_x - r, face_y - side r).
    Built in the frame the caller gives (move it with `transformed`)."""
    xc, yc = corner_x - r, face_y - side * r
    blk = box(xc, corner_x, yc, face_y, z0, z1)
    return largest(boolean("cut", blk, [cyl((xc, yc, z0 - 1.0), (xc, yc, z1 + 1.0), r)]))
