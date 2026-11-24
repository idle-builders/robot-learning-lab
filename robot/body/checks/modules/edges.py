"""edges: every sharp edge and corner of each scoped printed part's native STEP, so unfinished parts and design artefacts
are seen (owner 2026-11-19: a finished part has no sharp edges and corners).

Per B-rep edge between two faces: the dihedral deviation (angle between the face normals at the edge midpoint; 0 =
tangent). An edge is smooth below SOFT_DEG, a soft kink up to SHARP_DEG, sharp above, unless it bounds a chamfer (a
planar strip at most CHAMFER_MAX_MM wide, by area / longest edge) or a fillet face (a non-planar face tangent to a
neighbour). Convex (outer) or concave (inner) is taken from the part's mesh at the edge. Sharp edges within MATE_MM of a
neighbouring part at standing are on a mating face and reported apart (seating faces are often meant to be sharp);
part_settings <part>.edges names edges meant to be sharp: sharp_ok_near_parts (contact edges with those parts) and
sharp_ok_regions [{centre_mm, radius_mm}] (manifest frame); they are counted as intended, apart from free and mating.
With an out folder, every sharp edge and corner with its location is written to edges-<part>.json. A corner is a vertex
where three or more sharp edges meet. Also: the smallest fillet radius and chamfer width the part has. With an out
folder, one picture per part marks the free sharp edges red and the mating ones orange. Measurements and counts only."""
from __future__ import annotations

import math
import time

import numpy as np

from checks import bodylib as BL
from checks import process as P
from checks.core import tick
from checks.registry import register

SOFT_DEG, SHARP_DEG = 5.0, 20.0
CHAMFER_MAX_MM = 2.0
MATE_MM = 0.3


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and ctx.parts[n].get("native_step")]


def _normal(face, pnt):
    from OCP.BRepGProp import BRepGProp_Face
    from OCP.BRep import BRep_Tool
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt, gp_Vec
    s = BRep_Tool.Surface_s(face)
    pr = GeomAPI_ProjectPointOnSurf(pnt, s)
    if pr.NbPoints() == 0:
        return None
    u, v = pr.LowerDistanceParameters()
    p, n = gp_Pnt(), gp_Vec()
    BRepGProp_Face(face).Normal(u, v, p, n)
    a = np.array([n.X(), n.Y(), n.Z()])
    L = np.linalg.norm(a)
    return a / L if L > 1e-12 else None


def audit(step, offset):
    """Edge list of a STEP in the manifest frame: [{p (mid), a, b (ends), length, dev_deg, kind}] plus fillet / chamfer sizes."""
    from build123d import Location, import_step
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Torus
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape, TopTools_IndexedMapOfShape
    sh = import_step(str(step)).moved(Location(tuple(float(v) for v in (offset or (0, 0, 0)))))
    fmap = TopTools_IndexedMapOfShape(); TopExp.MapShapes_s(sh.wrapped, TopAbs_FACE, fmap)
    finfo = {}
    for i in range(1, fmap.Extent() + 1):
        f = TopoDS.Face_s(fmap.FindKey(i)); s = BRepAdaptor_Surface(f, True); g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g)
        longest = 0.0
        ex = TopExp_Explorer(f, TopAbs_EDGE)
        while ex.More():
            ge = GProp_GProps(); BRepGProp.LinearProperties_s(ex.Current(), ge); longest = max(longest, ge.Mass()); ex.Next()
        t = s.GetType()
        rad = s.Cylinder().Radius() if t == GeomAbs_Cylinder else (s.Torus().MinorRadius() if t == GeomAbs_Torus else None)
        finfo[i] = dict(planar=t == GeomAbs_Plane, area=g.Mass(), width=g.Mass() / max(longest, 1e-9), radius=rad)
    ef = TopTools_IndexedDataMapOfShapeListOfShape(); TopExp.MapShapesAndAncestors_s(sh.wrapped, TopAbs_EDGE, TopAbs_FACE, ef)
    edges = []
    for i in range(1, ef.Extent() + 1):
        e = TopoDS.Edge_s(ef.FindKey(i))
        if BRep_Tool.Degenerated_s(e) or ef.FindFromIndex(i).Extent() != 2:
            continue
        lst = ef.FindFromIndex(i)
        f1, f2 = TopoDS.Face_s(lst.First()), TopoDS.Face_s(lst.Last())
        if f1.IsSame(f2):                                    # seam edge of a closed surface
            continue
        c = BRepAdaptor_Curve(e)
        ge = GProp_GProps(); BRepGProp.LinearProperties_s(e, ge)
        L = ge.Mass()
        if L < 1e-4:
            continue
        pm = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
        n1, n2 = _normal(f1, pm), _normal(f2, pm)
        if n1 is None or n2 is None:
            continue
        dev = math.degrees(math.acos(np.clip(float(n1 @ n2), -1, 1)))
        i1, i2 = fmap.FindIndex(f1), fmap.FindIndex(f2)
        pa, pb = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
        poly = np.array([[q.X(), q.Y(), q.Z()] for q in (c.Value(t) for t in np.linspace(c.FirstParameter(), c.LastParameter(), 13))])
        edges.append(dict(p=np.array([pm.X(), pm.Y(), pm.Z()]), a=np.array([pa.X(), pa.Y(), pa.Z()]), b=np.array([pb.X(), pb.Y(), pb.Z()]),
                          poly=poly, length=L, dev=dev, f=(i1, i2)))
    # chamfer faces: planar strips <= CHAMFER_MAX_MM wide whose two neighbour edges both deviate (a bevel between two faces). A
    # bevel ramped off a corner splits it unevenly (72 + 18 deg): its gentle edge is a soft kink, not a sharp one, so an edge counts
    # as deviating from SOFT_DEG on (it was SHARP_DEG: the 72 deg edge of such a strip read sharp, CHAMFER_MAX_MM never applied)
    devs_by_face = {}
    for e in edges:
        for fi in e["f"]:
            devs_by_face.setdefault(fi, []).append(e["dev"])
    chamfer = {fi for fi, inf in finfo.items() if inf["planar"] and inf["width"] <= CHAMFER_MAX_MM
               and sum(1 for d in devs_by_face.get(fi, []) if SOFT_DEG <= d < 80.0) >= 2
               and any(SHARP_DEG < d < 80.0 for d in devs_by_face.get(fi, []))}
    fillet = {fi for fi, inf in finfo.items() if not inf["planar"] and inf["radius"] is not None and inf["radius"] <= 10.0
              and any(d < SOFT_DEG for d in devs_by_face.get(fi, []))}
    for e in edges:
        if e["dev"] < SOFT_DEG:
            e["kind"] = "smooth"
        elif any(fi in chamfer for fi in e["f"]):
            e["kind"] = "chamfer"
        elif e["dev"] < SHARP_DEG:
            e["kind"] = "soft kink"
        else:
            e["kind"] = "sharp"
    sizes = dict(min_fillet_radius_mm=round(min(finfo[f]["radius"] for f in fillet), 3) if fillet else None,
                 fillet_faces=len(fillet), chamfer_faces=len(chamfer),
                 min_chamfer_width_mm=round(min(finfo[f]["width"] for f in chamfer), 3) if chamfer else None)
    return edges, sizes


def convexity(m, pts):
    """Per point: True convex / False concave, from the nearest mesh edge's face_adjacency_convex."""
    from scipy.spatial import cKDTree
    mid = m.vertices[m.face_adjacency_edges].mean(1)
    ang = m.face_adjacency_angles
    keep = ang > np.radians(SOFT_DEG)                         # only creased mesh edges carry the crease's sign
    tree = cKDTree(mid[keep])
    _, j = tree.query(pts)
    return m.face_adjacency_convex[keep][j]


def picture(m, edges, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))
    light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
    shade = 0.5 + 0.45 * np.clip(m.face_normals @ light, 0, 1)
    for ax, (i, j, k), name in zip(axs, [(0, 2, 1), (1, 2, 0), (0, 1, 2)], ["front (x-z)", "side (y-z)", "top (x-y)"]):
        o = np.argsort(m.triangles_center[:, k] * (-1 if k != 2 else 1))
        ax.add_collection(PolyCollection(m.triangles[o][:, :, [i, j]], facecolors=np.c_[shade[o], shade[o], shade[o] * 1.02].clip(0, 1), edgecolors="none"))
        for kind, col, lw in (("intended", "green", 1.0), ("mating", "orange", 1.2), ("free", "red", 1.4)):
            seg = [e["poly"][:, [i, j]] for e in edges if e.get("sharp_where") == kind]
            if seg:
                ax.add_collection(LineCollection(seg, colors=col, linewidths=lw))
        ax.autoscale(); ax.set_aspect("equal"); ax.set_title(name, fontsize=8); ax.tick_params(labelsize=6)
    fig.suptitle(title + "  (red: free sharp edges, orange: on a mating face, green: named as intended)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


@register("edges", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part: every sharp edge (dihedral, length, location, convex / concave, free / on a mating face), sharp corners, smallest fillet and chamfer; a picture per part."""
    from checks.modules.overhang import _neighbour_distance
    per, cases, evidence = {}, [], []
    ev = dict(parts=0, sharp_edges_free=0, sharp_edges_on_mating_faces=0, sharp_corners_free=0, soft_kinks=0)
    for n in _targets(ctx):
        t0 = time.time()
        p = ctx.parts[n]
        edges, sizes = audit(BL.src(p["native_step"]), p.get("native_step_offset_mm"))
        m = ctx.solid(n)[0]
        sharp = [e for e in edges if e["kind"] == "sharp"]
        if sharp:
            pts = np.array([e["p"] for e in sharp])
            cvx = convexity(m, pts)
            # the nearest neighbour surface of any direction (the last pair; the facing / pressed pairs need the edge's normal)
            *_, d, who = _neighbour_distance(ctx, n, pts, np.zeros_like(pts))
            es = ctx.settings(n).get("edges") or {}
            ok_parts = set(es.get("sharp_ok_near_parts", []))
            ok_regions = [(np.asarray(r["centre_mm"], float), float(r["radius_mm"])) for r in es.get("sharp_ok_regions", [])]
            for e, c, dd, w in zip(sharp, cvx, d, who):
                e["convex"] = bool(c)
                e["mate"] = w if dd < MATE_MM else None
                intended = (w in ok_parts and dd < MATE_MM) or any(np.linalg.norm(e["p"] - c) <= r for c, r in ok_regions)
                e["sharp_where"] = "intended" if intended else ("mating" if e["mate"] else "free")
        free = [e for e in sharp if e["sharp_where"] == "free"]
        # corners: vertices (rounded to 1 um) where >= 3 free sharp edges end
        from collections import Counter
        ends = Counter(tuple(np.round(x, 3)) for e in free for x in (e["a"], e["b"]))
        corners = [np.array(k) for k, c in ends.items() if c >= 3]
        rows = sorted(free, key=lambda e: -e["dev"] * e["length"])
        r = dict(edges=len(edges), smooth=sum(e["kind"] == "smooth" for e in edges), chamfer_edges=sum(e["kind"] == "chamfer" for e in edges),
                 soft_kinks=sum(e["kind"] == "soft kink" for e in edges), sharp_free=len(free), sharp_on_mating_faces=sum(e["sharp_where"] == "mating" for e in sharp),
                 sharp_intended=sum(e["sharp_where"] == "intended" for e in sharp),
                 sharp_free_convex=sum(e["convex"] for e in free), sharp_free_concave=sum(not e["convex"] for e in free),
                 sharp_free_length_mm=round(sum(e["length"] for e in free), 1),
                 sharpest_free_deg=round(max((e["dev"] for e in free), default=0.0), 1), sharp_corners_free=len(corners), **sizes,
                 worst=[dict(dihedral_dev_deg=round(e["dev"], 1), length_mm=round(e["length"], 2), convex=e["convex"], location_mm=np.round(e["p"], 2).tolist())
                        for e in rows[:10]],
                 mating_parts=dict(Counter(e["mate"] for e in sharp if e["mate"])))
        if ctx.out_dir is not None:     # every sharp edge and corner with its location, for looking up what the picture marks
            import json as _json
            lst = ctx.out_dir / f"edges-{n}.json"
            lst.write_text(_json.dumps(dict(part=n, frame="manifest (mm)",
                edges=[dict(where=e["sharp_where"], dihedral_dev_deg=round(e["dev"], 1), length_mm=round(e["length"], 2), convex=e["convex"],
                            mating_part=e["mate"], location_mm=np.round(e["p"], 2).tolist()) for e in sorted(sharp, key=lambda e: -e["dev"] * e["length"])],
                corners=[np.round(c, 2).tolist() for c in corners]), indent=0))
            evidence.append(str(lst))
            f = ctx.out_dir / f"edges-{n}.png"
            picture(m, sharp, f, f"{n}: {r['sharp_free']} free sharp edges ({r['sharp_free_length_mm']} mm), {r['sharp_on_mating_faces']} on mating faces")
            evidence.append(str(f))
        r["seconds"] = round(time.time() - t0, 1); tick("edges", t0)
        per[n] = r
        ev["parts"] += 1; ev["sharp_edges_free"] += r["sharp_free"]; ev["sharp_edges_on_mating_faces"] += r["sharp_on_mating_faces"]
        ev["sharp_corners_free"] += r["sharp_corners_free"]; ev["soft_kinks"] += r["soft_kinks"]
        if free:
            w = r["worst"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["location_mm"], value=r["sharp_free_length_mm"], unit="mm free sharp edge length",
                              what=f"{r['sharp_free']} free sharp edges ({r['sharp_free_convex']} convex, {r['sharp_free_concave']} concave), {r['sharp_corners_free']} corners; "
                                   f"longest-worst {w['length_mm']} mm at {w['dihedral_dev_deg']} deg; smallest fillet {r['min_fillet_radius_mm']} mm"))
    cases.sort(key=lambda c: -c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=evidence,
                notes=f"B-rep edges; dihedral deviation between face normals at the edge midpoint: smooth < {SOFT_DEG:g}, soft kink < {SHARP_DEG:g}, sharp above; "
                      f"chamfer = planar strip <= {CHAMFER_MAX_MM:g} mm wide between two edges deviating >= {SOFT_DEG:g} deg, one of them > {SHARP_DEG:g} and < 80 deg; mating = within {MATE_MM} mm of a neighbour at standing; "
                      "convexity from the mesh crease at the edge; corner = >= 3 free sharp edges at one vertex")
