"""B-rep helpers of the exact checks: multi-solid common volume and minimum distance of OCC shapes, shape transforms
(by the body's link_T), memoised optimal bounding boxes. Moved from the lower-leg study's fit_check.py without changing the maths."""
from __future__ import annotations

import numpy as np
from build123d import Compound
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepExtrema import BRepExtrema_DistShapeShape
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.gp import gp_Trsf


_MOVED: dict[int, list] = {}   # hash(moved TopoDS_Shape) -> [(moved, source shape, T)] for aabb_bound


def transform_shape(shape, T: np.ndarray):
    if np.allclose(T, np.eye(4)):
        return shape
    tr = gp_Trsf()
    tr.SetValues(*map(float, T[:3].ravel()))
    out = Compound(BRepBuilderAPI_Transform(shape.wrapped, tr, True).Shape())
    if len(_MOVED) > 20000:
        _MOVED.clear()
    _MOVED.setdefault(hash(out.wrapped), []).append((out.wrapped, shape, np.array(T, float)))
    return out


def aabb_bound(shape) -> np.ndarray:
    """A box that contains aabb(shape), cheap for shapes made by transform_shape: the 8 corners of the source's
    (memoised) optimal box, transformed. Use it to rule pairs out; aabb() stays the exact value."""
    for w, src, T in _MOVED.get(hash(shape.wrapped), []):
        if w.IsEqual(shape.wrapped):
            b = aabb(src)
            c = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ T[:3, :3].T + T[:3, 3]
            return np.array([c.min(0), c.max(0)])
    return aabb(shape)


_AABB: dict[int, list] = {}   # hash(TopoDS_Shape) -> [(shape, box)]; IsEqual = same TShape, location, orientation


def aabb(shape) -> np.ndarray:
    """Optimal bounding box (build123d default), memoised per OCC shape. The optimal box costs ~0.2 s on a
    connector and the fit loops ask for the same shapes hundreds of times; a compound's box is the union of
    its solids' boxes (identical to the compound's own optimal box), so moved compounds reuse those too."""
    w = shape.wrapped
    hit = _AABB.get(hash(w))
    if hit:
        for s, box in hit:
            if s.IsEqual(w):
                return box.copy()
    sols = shape.solids() if hasattr(shape, "solids") else []
    if (len(sols) > 1 or (len(sols) == 1 and not sols[0].wrapped.IsEqual(w))) \
            and len(shape.faces()) == sum(len(s.faces()) for s in sols):   # nothing outside the solids
        bs = np.array([aabb(s) for s in sols])
        box = np.array([bs[:, 0].min(0), bs[:, 1].max(0)])
    else:
        b = shape.bounding_box()
        box = np.array([[b.min.X, b.min.Y, b.min.Z], [b.max.X, b.max.Y, b.max.Z]], dtype=float)
    if len(_AABB) > 20000:
        _AABB.clear()
    _AABB.setdefault(hash(w), []).append((w, box))
    return box.copy()


def aabb_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(a[0] - b[1], b[0] - a[1]))))


class CallTimeout(Exception):
    """An exact OCC call did not finish within its time; its forked process was stopped."""


def bounded(fn, timeout_s):
    """fn() in a forked copy of this process, stopped after timeout_s (CallTimeout). One OCC boolean or distance cannot be
    interrupted from Python (OCP exposes no progress indicator to subclass), so a per-pair budget only holds when the
    call runs where it can be killed (T-6rr3sa: one collar x neck pitch common ran 1476 s past a 120 s budget). Costs
    about 10 ms per call (measured 2026-11-23: 47 -> 57 ms on a sphere pair, same volume). Without fork: fn() directly."""
    import os
    import pickle
    import select
    import signal
    if not hasattr(os, "fork") or timeout_s is None:
        return fn()
    if timeout_s <= 0:
        raise CallTimeout("no time left")
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:                                        # child: compute, send, leave without cleanup handlers
        try:
            os.close(r)
            try:
                out = pickle.dumps((True, fn()))
            except BaseException as e:  # noqa: BLE001
                out = pickle.dumps((False, repr(e)))
            os.write(w, out)
        finally:
            os._exit(0)
    os.close(w)
    try:
        ready, _, _ = select.select([r], [], [], timeout_s)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            raise CallTimeout(f"stopped after {timeout_s:.0f} s")
        data = b""
        while True:
            chunk = os.read(r, 1 << 16)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(r)
        os.waitpid(pid, 0)
    if not data:
        raise RuntimeError("the forked exact call died without a result")
    ok, v = pickle.loads(data)
    if not ok:
        raise RuntimeError(v)
    return v


def common_volume_mm3(s1, s2) -> float:
    if aabb_gap(aabb_bound(s1), aabb_bound(s2)) > 0.05:   # bound boxes contain the exact ones: same answer, no OCC box
        return 0.0
    b1, b2 = aabb(s1), aabb(s2)
    if aabb_gap(b1, b2) > 0.05:
        return 0.0
    solids1 = s1.solids() if hasattr(s1, "solids") and len(s1.solids()) > 0 else [s1]
    solids2 = s2.solids() if hasattr(s2, "solids") and len(s2.solids()) > 0 else [s2]
    total = 0.0
    for a in solids1:
        ba = aabb(a)
        for b in solids2:
            bb = aabb(b)
            if aabb_gap(ba, bb) > 0.05:
                continue
            op = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
            op.Build()
            if not op.IsDone():
                continue
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(op.Shape(), g)
            total += abs(float(g.Mass()))
    return total


def common_volume_area(s1, s2) -> tuple[float, float]:
    """(volume mm3, surface area mm2) of the exact common of two shapes, summed over solid pairs, as common_volume_mm3 (same
    volume). The area tells a face contact from an overlap: two solids sharing a face (a printed column standing on a board)
    can give a common whose volume is a few thousandths of a mm3 spread over the contact area; its mean thickness 2 V / A
    is then far below any print or assembly tolerance (contact_depth_mm)."""
    if aabb_gap(aabb_bound(s1), aabb_bound(s2)) > 0.05:
        return 0.0, 0.0
    b1, b2 = aabb(s1), aabb(s2)
    if aabb_gap(b1, b2) > 0.05:
        return 0.0, 0.0
    solids1 = s1.solids() if hasattr(s1, "solids") and len(s1.solids()) > 0 else [s1]
    solids2 = s2.solids() if hasattr(s2, "solids") and len(s2.solids()) > 0 else [s2]
    vol = area = 0.0
    for a in solids1:
        ba = aabb(a)
        for b in solids2:
            if aabb_gap(ba, aabb(b)) > 0.05:
                continue
            op = BRepAlgoAPI_Common(a.wrapped, b.wrapped)
            op.Build()
            if not op.IsDone():
                continue
            g = GProp_GProps()
            BRepGProp.VolumeProperties_s(op.Shape(), g)
            v = abs(float(g.Mass()))
            vol += v
            if v > 0:
                gs = GProp_GProps()
                BRepGProp.SurfaceProperties_s(op.Shape(), gs)
                area += abs(float(gs.Mass()))
    return vol, area


def contact_depth_mm(volume_mm3: float, area_mm2: float) -> float:
    """Mean thickness (mm) of a common solid: 2 V / A (a thin slab of contact area S has A ~ 2 S and V = S t)."""
    return 2.0 * volume_mm3 / area_mm2 if area_mm2 > 0 else 0.0


def min_distance_mm(s1, s2) -> float:
    op = BRepExtrema_DistShapeShape(s1.wrapped, s2.wrapped)
    op.Perform()
    return float(op.Value()) if op.IsDone() else -1.0


def closest_points(s1, s2) -> dict:
    """Exact minimum distance (mm) and where: the closest point on each shape (mm, the shapes' frame) and the feature it lies on
    (face / edge / vertex; for a face its surface type, area and the face's index in the shape's face list), from BRepExtrema."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepExtrema import BRepExtrema_IsInFace, BRepExtrema_IsOnEdge
    from OCP.GeomAbs import GeomAbs_BSplineSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    op = BRepExtrema_DistShapeShape(s1.wrapped, s2.wrapped)
    op.Perform()
    if not op.IsDone() or op.NbSolution() < 1:
        return dict(distance_mm=None)
    kinds = {GeomAbs_Plane: "plane", GeomAbs_Cylinder: "cylinder", GeomAbs_Cone: "cone", GeomAbs_Sphere: "sphere",
             GeomAbs_Torus: "torus", GeomAbs_BSplineSurface: "bspline"}

    def feature(shape, support, kind):
        if kind == BRepExtrema_IsInFace:
            fmap = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(shape, TopAbs_FACE, fmap)
            f = TopoDS.Face_s(support)
            g = GProp_GProps()
            BRepGProp.SurfaceProperties_s(f, g)
            return dict(kind="face", surface=kinds.get(BRepAdaptor_Surface(f, True).GetType(), "other"),
                        face_index=int(fmap.FindIndex(f)) - 1, face_area_mm2=round(abs(float(g.Mass())), 2))
        return dict(kind="edge" if kind == BRepExtrema_IsOnEdge else "vertex")
    p1, p2 = op.PointOnShape1(1), op.PointOnShape2(1)
    return dict(distance_mm=float(op.Value()),
                point_a_mm=[round(p1.X(), 3), round(p1.Y(), 3), round(p1.Z(), 3)], point_b_mm=[round(p2.X(), 3), round(p2.Y(), 3), round(p2.Z(), 3)],
                feature_a=feature(s1.wrapped, op.SupportOnShape1(1), op.SupportTypeShape1(1)),
                feature_b=feature(s2.wrapped, op.SupportOnShape2(1), op.SupportTypeShape2(1)))
