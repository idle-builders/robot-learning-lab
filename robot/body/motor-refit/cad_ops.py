"""Small exact B-rep edits for adapting r1 printed parts to new servos (OpenCASCADE via OCP).

All operations work on world-frame solids and axis-aligned planes, which is how the r1 leg and neck joints sit
(hip pitch, knee, ankle pitch and neck pitch turn about world X; neck yaw about world Z).

- stretch(shape, axis, at, delta): material beyond the plane `axis = at` (on the side delta points to) moves by
  delta; the gap is filled by the part's own cross-section at the plane, extruded (a thin slab scaled along the axis).
  This widens a clevis or a carrier without changing its other features.
- pad(shape, axis, frm, to, region): extends the part's material found just behind the plane `axis = frm` (inside
  the box `region`) up to the plane `axis = to`, e.g. to bring a mounting plate onto a new servo face.
- translate / mirror_x / box / cylinder / cut / fuse helpers.
"""
from __future__ import annotations

import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform, BRepBuilderAPI_Transform
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRepGProp import BRepGProp
from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCP.GProp import GProp_GProps
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_SOLID
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ListOfShape
from OCP.gp import gp_Ax2, gp_Dir, gp_GTrsf, gp_Mat, gp_Pnt, gp_Trsf, gp_Vec, gp_XYZ

AX = {"x": 0, "y": 1, "z": 2}
BIG = 2000.0
FUZZY = 1e-4


def box(lo, hi):
    lo = [float(v) for v in lo]; hi = [float(v) for v in hi]
    return BRepPrimAPI_MakeBox(gp_Pnt(*lo), gp_Pnt(*hi)).Shape()


def cylinder(p0, p1, r):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float); d = p1 - p0; h = float(np.linalg.norm(d))
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*p0), gp_Dir(*(d / h))), float(r), h).Shape()


def _bool(op, a, b):
    args = TopTools_ListOfShape(); args.Append(a)
    tools = TopTools_ListOfShape(); tools.Append(b)
    o = op(); o.SetArguments(args); o.SetTools(tools); o.SetFuzzyValue(FUZZY); o.SetRunParallel(False); o.Build()
    if not o.IsDone():
        raise RuntimeError(f"{op.__name__} failed")
    return o.Shape()


def common(a, b):
    return _bool(BRepAlgoAPI_Common, a, b)


def cut(a, *bs):
    for b in bs:
        a = _bool(BRepAlgoAPI_Cut, a, b)
    return a


def fuse(a, *bs):
    for b in bs:
        if b is None or volume(b) < 1e-6:
            continue
        a = _bool(BRepAlgoAPI_Fuse, a, b)
    return unify(a)


def unify(s):
    u = ShapeUpgrade_UnifySameDomain(s, True, True, False); u.Build()
    return u.Shape()


def volume(s) -> float:
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(s, g)
    return float(g.Mass())


def n_solids(s) -> int:
    e = TopExp_Explorer(s, TopAbs_SOLID); n = 0
    while e.More():
        n += 1; e.Next()
    return n


def valid(s) -> bool:
    return bool(BRepCheck_Analyzer(s).IsValid())


def translate(s, d):
    t = gp_Trsf(); t.SetTranslation(gp_Vec(*[float(v) for v in d]))
    return BRepBuilderAPI_Transform(s, t, True).Shape()


def mirror_x(s):
    from OCP.gp import gp_Ax2 as _A
    t = gp_Trsf(); t.SetMirror(_A(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)))
    return BRepBuilderAPI_Transform(s, t, True).Shape()


def _halfspace(axis, at, side):
    lo = [-BIG] * 3; hi = [BIG] * 3
    if side > 0:
        lo[AX[axis]] = at
    else:
        hi[AX[axis]] = at
    return box(lo, hi)


def _slab(axis, a0, a1, region=None):
    lo = [-BIG] * 3; hi = [BIG] * 3
    if region is not None:
        lo = list(region[0]); hi = list(region[1])
    lo[AX[axis]] = min(a0, a1); hi[AX[axis]] = max(a0, a1)
    return box(lo, hi)


def _scale_axis(s, axis, a_src0, a_src1, a_dst0, a_dst1):
    """Affine map along one axis taking [a_src0, a_src1] onto [a_dst0, a_dst1] (other axes unchanged)."""
    k = (a_dst1 - a_dst0) / (a_src1 - a_src0)
    m = gp_Mat(1, 0, 0, 0, 1, 0, 0, 0, 1)
    i = AX[axis] + 1
    m.SetValue(i, i, k)
    tr = [0.0, 0.0, 0.0]; tr[AX[axis]] = a_dst0 - k * a_src0
    g = gp_GTrsf(); g.SetVectorialPart(m); g.SetTranslationPart(gp_XYZ(*tr))
    return BRepBuilderAPI_GTransform(s, g, True).Shape()


def stretch(shape, axis: str, at: float, delta: float, slab_w: float = 0.3):
    """Move everything beyond `axis = at` (on delta's side) by delta and bridge the gap with the section at `at`."""
    if abs(delta) < 1e-6:
        return shape
    side = 1 if delta > 0 else -1
    keep = common(shape, _halfspace(axis, at, -side))
    move = common(shape, _halfspace(axis, at, side))
    d = [0.0, 0.0, 0.0]; d[AX[axis]] = delta
    moved = translate(move, d)
    # thin slab on the moving side, right at the plane, scaled to fill [at, at + delta]
    s0, s1 = (at, at + side * slab_w)
    slab = common(shape, _slab(axis, s0, s1))
    parts = [moved]
    if volume(slab) > 1e-6:
        bridge = _scale_axis(slab, axis, s0, s1, at, at + delta)
        parts.append(bridge)
    return fuse(keep, *parts)


def pad(shape, axis: str, frm: float, to: float, region, slab_w: float = 0.3):
    """Extend the material found just behind `axis = frm` (inside region) up to `axis = to`."""
    side = 1 if to > frm else -1
    s0, s1 = frm - side * slab_w, frm
    slab = common(shape, _slab(axis, s0, s1, region))
    if volume(slab) < 1e-6:
        return shape, 0.0
    ext = _scale_axis(slab, axis, s0, s1, frm, to)
    return fuse(shape, ext), volume(ext)


def stretch_region(shape, axis: str, at: float, delta: float, region):
    """stretch() applied only to the part of the shape inside the box `region` ((lo xyz), (hi xyz))."""
    rb = box(*region)
    inside = common(shape, rb)
    if volume(inside) < 1e-6:
        return shape
    outside = cut(shape, rb)
    return fuse(outside, stretch(inside, axis, at, delta))


def cut_many(shape, tools, min_keep: float = 0.5, sequential: bool = False):
    """Cut a list of tool solids in one boolean (tools fused into one compound first)."""
    if not tools:
        return shape
    v0 = volume(shape)
    if sequential:  # mutually overlapping tools (e.g. swept copies): one boolean per tool
        for t in tools:
            shape = _bool(BRepAlgoAPI_Cut, shape, t)
        return shape
    args = TopTools_ListOfShape(); args.Append(shape)
    tl = TopTools_ListOfShape()
    for t in tools:
        tl.Append(t)
    o = BRepAlgoAPI_Cut(); o.SetArguments(args); o.SetTools(tl); o.SetFuzzyValue(FUZZY); o.SetRunParallel(False); o.Build()
    res = o.Shape() if o.IsDone() else None
    if res is None or volume(res) < min_keep * v0:
        # the one-shot boolean lost the body (touching or coplanar tools); cut one tool at a time
        res = shape
        for t in tools:
            nxt = _bool(BRepAlgoAPI_Cut, res, t)
            if volume(nxt) >= min_keep * volume(res):
                res = nxt
            else:
                raise RuntimeError("cut removed more than half of the part; check the tool")
    return res


def holes(axis: str, centers, r: float, a0: float, a1: float):
    """Cylinders along a world axis from a0 to a1, one per (u, v) centre in the other two axes (in xyz order)."""
    k = AX[axis]; out = []
    for c in centers:
        p0 = [0.0, 0.0, 0.0]; p1 = [0.0, 0.0, 0.0]; o = [i for i in range(3) if i != k]
        p0[o[0]] = p1[o[0]] = c[0]; p0[o[1]] = p1[o[1]] = c[1]; p0[k] = a0; p1[k] = a1
        out.append(cylinder(p0, p1, r))
    return out


def transform(s, T4):
    """Apply a 4x4 rigid transform (numpy) to a shape."""
    t = gp_Trsf()
    t.SetValues(*[float(T4[i][j]) for i in range(3) for j in range(4)])
    return BRepBuilderAPI_Transform(s, t, True).Shape()


def largest_solid(s):
    e = TopExp_Explorer(s, TopAbs_SOLID); best, bv = None, -1
    while e.More():
        v = volume(e.Current())
        if v > bv:
            best, bv = e.Current(), v
        e.Next()
    return best


def solids(s):
    out = []; e = TopExp_Explorer(s, TopAbs_SOLID)
    while e.More():
        out.append(e.Current()); e.Next()
    return out
