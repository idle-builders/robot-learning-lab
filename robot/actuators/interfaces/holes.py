"""Small round holes of a B-rep: concave cylindrical faces merged per axis (position, axis, dia, extent)."""
from __future__ import annotations

import numpy as np


def holes_of_shape(shape, r_max=2.6, r_min=0.5):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_REVERSED
    raw = []
    for f in shape.faces():
        s = BRepAdaptor_Surface(f.wrapped)
        if s.GetType() != GeomAbs_Cylinder:
            continue
        cyl = s.Cylinder(); r = cyl.Radius()
        if not (r_min <= r <= r_max):
            continue
        a = cyl.Axis(); p0 = np.array([a.Location().X(), a.Location().Y(), a.Location().Z()])
        d = np.array([a.Direction().X(), a.Direction().Y(), a.Direction().Z()]); d /= np.linalg.norm(d)
        u = 0.5 * (s.FirstUParameter() + s.LastUParameter()); v = 0.5 * (s.FirstVParameter() + s.LastVParameter())
        pf = BRepGProp_Face(f.wrapped); P = gp_Pnt(); N = gp_Vec(); pf.Normal(u, v, P, N)
        pt = np.array([P.X(), P.Y(), P.Z()]); n = np.array([N.X(), N.Y(), N.Z()])
        radial = (pt - p0) - ((pt - p0) @ d) * d
        concave = float(n @ radial) < 0          # normal points to the axis: material outside = a hole
        vs = np.array([[q.X, q.Y, q.Z] for q in f.vertices()]) if f.vertices() else pt[None]
        t = (vs - p0) @ d
        raw.append(dict(p0=p0, d=d, r=r, t0=float(t.min()), t1=float(t.max()), concave=concave,
                        span_deg=float(np.degrees(s.LastUParameter() - s.FirstUParameter()))))
    # merge faces on one axis line with the same radius (split cylinders)
    out = []
    for h in raw:
        if not h["concave"]:
            continue
        for g in out:
            if abs(g["r"] - h["r"]) < 0.02 and abs(abs(g["d"] @ h["d"]) - 1) < 1e-4:
                off = h["p0"] - g["p0"]
                if np.linalg.norm(off - (off @ g["d"]) * g["d"]) < 0.02:
                    sh = float(off @ g["d"]); sgn = 1.0 if g["d"] @ h["d"] > 0 else -1.0
                    ts = sorted([sh + sgn * h["t0"], sh + sgn * h["t1"]])
                    g["t0"] = min(g["t0"], ts[0]); g["t1"] = max(g["t1"], ts[1]); g["span_deg"] += h["span_deg"]
                    break
        else:
            out.append(dict(h))
    res = []
    for g in out:
        a = g["p0"] + g["t0"] * g["d"]; b = g["p0"] + g["t1"] * g["d"]
        res.append(dict(dia_mm=round(2 * g["r"], 3), end_a=a, end_b=b, axis=g["d"], depth_mm=round(g["t1"] - g["t0"], 3),
                        span_deg=round(min(g["span_deg"], 360.0), 1)))
    return res
