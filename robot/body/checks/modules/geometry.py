"""geometry: B-rep hygiene of each scoped part's native STEP: solid validity, sliver faces, micro-edges, acute wedges, tangent-continuity
(G1) kinks, unrounded profile corners, L/R mirror volume difference, sleeve count against the part's expected-hardware setting, STEP-vs-mesh volume difference, inner/outer fillet radii, extents vs the printer build volume, load-path slab areas (per-part setting), part-vs-sleeve overlap.
One parametrised check for the four per-section check_cad_rules; the face/edge audit is moved from leg/check_cad_rules.py without changing the maths.
Measurements and counts only; the old all_* / == 0 gates are not reproduced."""
from __future__ import annotations

import math
import time

import numpy as np

from checks import bodylib as BL
from checks.core import R, sha, tick
from checks.registry import register

# thresholds of the old audit, kept as named measurement definitions (they define what a counted event is, not a verdict);
# a part's own values come from its `geometry_thresholds` setting (part_settings.json, defaults there); these are the fallback
DEFAULT_THRESHOLDS = dict(sliver_mm2=0.25, micro_edge_mm=0.30, acute_normal_deg=125.0, g1_kink_deg=8.0)
# Parts we make (printed, or kind made / other: cut, machined, harness, insulator) have our own STEP and are audited. Every other part
# is bought (kind purchased / included / hardware, or unknown): its STEP is the supplier's (power supervisor board 80 MB, ROCK4D 45,
# ST3025 cases 15 each, thousands of component solids), the audit of its B-rep says nothing about our design and an unscoped run never
# finished on them (T-zfejzf, 2026-11-22: no part audited in 1800 s). Those are listed in `bought_not_audited` with their STEP size.
# The kind is the part's body definition `kind`, else its role's kind in data/roles.json (ctx.rel.kind).
MADE_KINDS = ("made", "other")


def _face_normal_at(face, pt):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Plane
    from OCP.TopAbs import TopAbs_REVERSED
    surf = BRepAdaptor_Surface(face)
    st = surf.GetType()
    rev = -1.0 if face.Orientation() == TopAbs_REVERSED else 1.0
    if st == GeomAbs_Plane:
        d = surf.Plane().Axis().Direction()
        return np.array([d.X(), d.Y(), d.Z()], dtype=float) * rev
    if st == GeomAbs_Cylinder:
        cyl = surf.Cylinder()
        ax_p, ax_d = cyl.Location(), cyl.Axis().Direction()
        a = np.array([ax_d.X(), ax_d.Y(), ax_d.Z()], dtype=float)
        v = np.array([pt.X() - ax_p.X(), pt.Y() - ax_p.Y(), pt.Z() - ax_p.Z()], dtype=float)
        rad = v - np.dot(v, a) * a
        n = np.linalg.norm(rad)
        if n > 1e-9:
            return (rad / n) * rev
    return np.array([0.0, 0.0, 1.0], dtype=float)


def brep_audit(sh, thresholds=None):
    """Face and edge audit of one build123d shape (moved from leg/check_cad_rules.audit_step_part). thresholds: sliver_mm2,
    micro_edge_mm, acute_normal_deg, g1_kink_deg (what counts as an event); missing keys take DEFAULT_THRESHOLDS."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    SLIVER_MM2, MICRO_EDGE_MM, ACUTE_NORMAL_DEG, G1_KINK_DEG = th["sliver_mm2"], th["micro_edge_mm"], th["acute_normal_deg"], th["g1_kink_deg"]
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
    from OCP.BRepGProp import BRepGProp, BRepGProp_Face
    from OCP.GeomAbs import GeomAbs_Cylinder, GeomAbs_Line, GeomAbs_Plane
    from OCP.GProp import GProp_GProps
    from OCP.gp import gp_Pnt, gp_Vec
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    exp_f = TopExp_Explorer(sh.wrapped, TopAbs_FACE)
    total_faces = sliver = cyl_faces = plane_faces = curved = 0
    min_face_area = 1e9
    worst_sliver = None
    inner_r, outer_r = [], []
    while exp_f.More():
        f = TopoDS.Face_s(exp_f.Current())
        total_faces += 1
        g = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, g)
        area = float(g.Mass())
        st = BRepAdaptor_Surface(f).GetType()
        if st == GeomAbs_Plane:
            plane_faces += 1
            if area < min_face_area:
                min_face_area = area
            if area < SLIVER_MM2:
                sliver += 1
                c = GProp_GProps(); BRepGProp.SurfaceProperties_s(f, c)
                p = c.CentreOfMass()
                if worst_sliver is None or area < worst_sliver[0]:
                    worst_sliver = (area, [round(p.X(), 2), round(p.Y(), 2), round(p.Z(), 2)])
        elif st == GeomAbs_Cylinder:
            cyl_faces += 1; curved += 1
            sa = BRepAdaptor_Surface(f, True)
            r_cyl, u_span = float(sa.Cylinder().Radius()), abs(float(sa.LastUParameter() - sa.FirstUParameter()))
            if r_cyl <= 5.5 and u_span < 5.5:        # small fillet-like cylinder: concave (inner) or convex (outer)?
                p_mid, vn = gp_Pnt(), gp_Vec()
                BRepGProp_Face(f).Normal(0.5 * float(sa.FirstUParameter() + sa.LastUParameter()), 0.5 * float(sa.FirstVParameter() + sa.LastVParameter()), p_mid, vn)
                ax = sa.Cylinder().Axis(); p0 = ax.Location()
                d_ax = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()], dtype=float)
                v_p = np.array([p_mid.X() - p0.X(), p_mid.Y() - p0.Y(), p_mid.Z() - p0.Z()], dtype=float)
                r_vec = v_p - np.dot(v_p, d_ax) * d_ax
                (inner_r if float(np.dot(np.array([vn.X(), vn.Y(), vn.Z()]), r_vec)) < 0.0 else outer_r).append(r_cyl)
        else:
            curved += 1
        exp_f.Next()
    ef = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(sh.wrapped, TopAbs_EDGE, TopAbs_FACE, ef)
    total_edges = zigzag = acute = g1 = hub = unround = 0
    min_edge, max_normal = 1e9, 0.0
    worst_zig = worst_acute = None
    for i in range(1, ef.Extent() + 1):
        edge = TopoDS.Edge_s(ef.FindKey(i))
        if BRep_Tool.Degenerated_s(edge):
            continue
        g = GProp_GProps(); BRepGProp.LinearProperties_s(edge, g)
        length = float(g.Mass())
        if length < 1e-6:
            continue
        total_edges += 1
        faces = ef.FindFromIndex(i)
        if faces.Extent() != 2:
            continue
        f1, f2 = TopoDS.Face_s(faces.First()), TopoDS.Face_s(faces.Last())
        s1, s2 = BRepAdaptor_Surface(f1), BRepAdaptor_Surface(f2)
        t1, t2 = s1.GetType(), s2.GetType()
        c = BRepAdaptor_Curve(edge)
        u = 0.5 * (c.FirstParameter() + c.LastParameter())
        pt, vt = gp_Pnt(), gp_Vec()
        c.D1(u, pt, vt)
        n1, n2 = _face_normal_at(f1, pt), _face_normal_at(f2, pt)
        ang = math.degrees(math.acos(np.clip(float(np.dot(n1, n2)), -1.0, 1.0)))
        where = [round(pt.X(), 2), round(pt.Y(), 2), round(pt.Z(), 2)]
        if t1 == GeomAbs_Plane and t2 == GeomAbs_Plane:
            min_edge = min(min_edge, length)
            if c.GetType() == GeomAbs_Line and length < MICRO_EDGE_MM and ang > 10.0:
                zigzag += 1
                if worst_zig is None or length < worst_zig[0]:
                    worst_zig = (length, where)
            if length > 0.50:
                max_normal = max(max_normal, ang)
                if ang > ACUTE_NORMAL_DEG:
                    acute += 1
                    if worst_acute is None or ang > worst_acute[0]:
                        worst_acute = (ang, where)
            if c.GetType() == GeomAbs_Line and 1.8 <= length <= 9.5 and ang > 25.0:
                d = c.Line().Direction()
                if max(abs(d.X()), abs(d.Y()), abs(d.Z())) > 0.995 and abs(float(np.dot(n1, n2))) > 0.08:
                    unround += 1
        elif (t1 == GeomAbs_Cylinder and t2 == GeomAbs_Plane) or (t1 == GeomAbs_Plane and t2 == GeomAbs_Cylinder) \
                or (t1 == GeomAbs_Cylinder and t2 == GeomAbs_Cylinder):
            if c.GetType() == GeomAbs_Line and length > 0.8:
                if ang <= G1_KINK_DEG:
                    g1 += 1
                elif (t1 == GeomAbs_Cylinder) != (t2 == GeomAbs_Cylinder):
                    fc, sc = (f1, s1) if t1 == GeomAbs_Cylinder else (f2, s2)
                    if fc.Orientation() != TopAbs_REVERSED and sc.Cylinder().Radius() >= 9.5 and ang > 15.0:
                        hub += 1
    denom = max(1, g1 + unround + hub)
    return dict(total_faces=total_faces, planar_faces=plane_faces, cylindrical_faces=cyl_faces, curved_faces=curved, total_edges=total_edges,
                sliver_faces=sliver, min_face_area_mm2=round(float(min_face_area if min_face_area < 1e8 else 0.0), 4),
                micro_edges=zigzag, min_edge_length_mm=round(float(min_edge if min_edge < 1e8 else 0.0), 4),
                acute_wedge_edges=acute, min_interior_dihedral_deg=round(180.0 - float(max_normal), 2),
                g1_tangent_transitions=g1, non_tangent_hub_steps=hub, unrounded_profile_corners=unround,
                tangential_continuity_ratio=round(g1 / denom, 4),
                inner_fillet_faces=len(inner_r), outer_fillet_faces=len(outer_r),
                mean_inner_fillet_radius_mm=round(float(np.mean(inner_r)), 3) if inner_r else None,
                mean_outer_fillet_radius_mm=round(float(np.mean(outer_r)), 3) if outer_r else None), dict(sliver=worst_sliver, micro=worst_zig, acute=worst_acute)


def _mesh_mf(ctx, name):
    """Closed manifold of a part's manifest mesh (placement offset applied) or None."""
    m = ctx.body.mesh(name)
    mf = kit_mf(m)
    return mf if mf.status().name == "NoError" and mf.volume() > 0 else None


def kit_mf(m):
    from checks.core import kit
    return kit.mf(m)


def plate_fit(ctx, name, sh):
    """Extents of the part in its print frame (process.up -> +Z, plate_yaw_deg about it) against the printer's build volume,
    for the main nozzle and for both nozzles (support from the second): per-axis margin (mm, negative = does not fit), and
    the margin at the best plate yaw (the yaw that maximises the smallest X / Y margin, 0.5 deg steps over 0-180 deg).
    Without a print direction: the old screen (sorted STEP extents against sorted build volume, orientation free)."""
    from checks import process as P
    _, prns = P.tables()
    e = P.effective(ctx, name)
    prn = prns.get(e["printer"] or "", {})
    vols = {k: prn.get(k) for k in ("build_volume_mm", "build_volume_both_nozzles_mm") if prn.get(k)}
    out = dict(printer=e["printer"], up=e["up"], up_source=e["source"]["up"], plate_yaw_deg=e["plate_yaw_deg"], build_volume_mm=vols.get("build_volume_mm"))
    if e["up"] is None:
        if sh is None:
            ext = sorted((ctx.body.mesh(name).extents).tolist(), reverse=True)
        else:
            b = sh.bounding_box()
            ext = sorted([b.size.X, b.size.Y, b.size.Z], reverse=True)
        out.update(frame="STEP frame, orientation free (no print direction)", extents_mm=[round(float(v), 2) for v in ext])
        if vols.get("build_volume_mm"):
            bed = sorted(vols["build_volume_mm"], reverse=True)
            out["bed_margin_mm"] = [round(bed[i] - ext[i], 2) for i in range(3)]
            out["min_bed_margin_mm"] = min(out["bed_margin_mm"])
        return out
    v = P.print_mesh(ctx, name, e).vertices
    ext = v.max(0) - v.min(0)
    hull = v[:, :2]
    try:
        from scipy.spatial import ConvexHull
        hull = hull[ConvexHull(hull).vertices]
    except Exception:  # noqa: BLE001
        pass
    out.update(frame="print frame (up -> +Z)", extents_mm=[round(float(x), 2) for x in ext])
    for key, bv in vols.items():
        tag = "" if key == "build_volume_mm" else "_both_nozzles"
        mg = [round(bv[i] - float(ext[i]), 2) for i in range(3)]
        best = None
        for yaw in np.arange(0.0, 180.0, 0.5):     # 0-180: the both-nozzle volume is not square
            a = np.radians(yaw)
            xy = hull @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
            w = xy.max(0) - xy.min(0)
            m = min(bv[0] - w[0], bv[1] - w[1])
            if best is None or m > best[0]:
                best = (m, yaw, w)
        out[f"bed_margin{tag}_mm"] = mg
        out[f"min_bed_margin{tag}_mm"] = min(mg)
        out[f"best_yaw{tag}_deg"] = float(best[1])
        out[f"min_bed_margin_best_yaw{tag}_mm"] = round(min(best[0], bv[2] - float(ext[2])), 2)
    return out


def loadpath_slices(mesh_mf, spec, n_default=12):
    """Slab areas along spec {axis, range_mm, n_slices}: area of each slab (mm2), parallel branches, branches that do not reach both neighbouring slabs
    (moved from arm/check_cad_rules.check_loadpath_slice_continuity). A slab that lies wholly outside the part's extent along the axis is
    counted apart (slabs_outside_part, area null): the setting's range no longer cuts the part there (a range set for an earlier geometry),
    which is not a severed load path; severed branches and the thinnest slab are over the slabs inside the part."""
    import manifold3d as md
    ax = {"x": 0, "y": 1, "z": 2}[spec["axis"]]
    n = int(spec.get("n_slices", n_default))
    s_min, s_max = spec["range_mm"]
    edges = np.linspace(s_min, s_max, n + 1)
    eps = 0.15
    bb = np.array(mesh_mf.bounding_box(), float)
    lo, hi = bb[:3] - 10.0, bb[3:] + 10.0
    p_lo, p_hi = float(bb[ax]), float(bb[3 + ax])
    inside = [bool(edges[i + 1] > p_lo and edges[i] < p_hi) for i in range(n)]
    comps_by_slab, areas = [], []
    for i in range(n):
        if not inside[i]:
            areas.append(None); comps_by_slab.append([])
            continue
        a0, a1 = float(edges[i]) - eps, float(edges[i + 1]) + eps
        size = list(hi - lo); org = list(lo)
        size[ax], org[ax] = a1 - a0, a0
        slab = mesh_mf ^ md.Manifold.cube(size).translate(org)
        areas.append(round(float(slab.volume()) / max(1e-6, a1 - a0), 2))
        comps_by_slab.append([c for c in slab.decompose() if float(c.volume()) / max(1e-6, a1 - a0) >= 1.5])
    idx = [i for i in range(n) if inside[i]]
    severed = 0
    for i in idx[1:-1]:                      # the first and last slab inside the part are its ends, not dead ends
        pu = md.Manifold.batch_boolean(comps_by_slab[i - 1], md.OpType.Add) if comps_by_slab[i - 1] else md.Manifold()
        nu = md.Manifold.batch_boolean(comps_by_slab[i + 1], md.OpType.Add) if comps_by_slab[i + 1] else md.Manifold()
        for c in comps_by_slab[i]:
            vp = float((c ^ pu).volume()) if comps_by_slab[i - 1] else 0.0
            vn = float((c ^ nu).volume()) if comps_by_slab[i + 1] else 0.0
            if vp < 1e-3 or vn < 1e-3:
                severed += 1
    a_in = [areas[i] for i in idx]
    cut = max(0.0, min(s_max, p_hi) - max(s_min, p_lo))
    return dict(axis=spec["axis"], range_mm=[s_min, s_max], part_extent_mm=[round(p_lo, 2), round(p_hi, 2)],
                range_inside_part_pct=round(100.0 * cut / max(1e-9, s_max - s_min), 1), slabs=n, slabs_outside_part=n - len(idx),
                slab_areas_mm2=areas, min_slab_area_mm2=round(min(a_in), 2) if a_in else None,
                mean_slab_area_mm2=round(float(np.mean(a_in)), 2) if a_in else None,
                max_parallel_branches=int(max([len(comps_by_slab[i]) for i in idx] or [0])), severed_or_deadend_branches=severed)


def _step(ctx, name):
    """The part's native STEP, None without one or when the manifest path resolves nowhere (31 foot-sensor / forearm-nut paths on
    22 Nov: counted in bad_events.dangling_native_step, the part is then skipped like a mesh-only part)."""
    p = ctx.parts.get(name)
    if not p or not p.get("native_step") or not BL.src(p["native_step"]).exists():
        return None
    return BL.src(p["native_step"])


def _solids(step):
    from build123d import import_step
    sh = import_step(str(step))
    return sh, list(sh.solids())


def _placed(ctx, name, step):
    """The STEP's solids as one compound in the manifest frame (native_step_offset_mm applied)."""
    from build123d import Compound, Location
    off = ctx.parts[name].get("native_step_offset_mm") or (0, 0, 0)
    return Compound(children=[s.moved(Location(tuple(float(v) for v in off))) for s in _solids(step)[1]])


def sleeve_sets(ctx, name):
    """{sleeve set: where that came from} of the sleeve sets pressed into the part: ctx.rel.sleeve_sets_of (its definition's
    joined_to {how: pressed}, else the set whose sleeves' surface touches the part), never a name."""
    return {s: ctx.rel.source(s, "sleeve_owner") for s in ctx.rel.sleeve_sets_of(name)}


def sleeve_overlap(ctx, name, pmf, sets, step):
    """Part vs its own sleeves (mm3), summed over its sleeve sets [(set, closed mesh or None, STEP or None)]: the manifest-mesh
    boolean, and where it reads above 0 and both STEPs exist, the exact OCC common volume, which is reported: a press-fit sleeve
    shares its bore's cylinder, and the two tessellations of one cylinder cross each other by a few hundredths of a mm (0.4-1.1 mm3
    over a sleeve set that is exactly 0)."""
    sets = [(sn, smf, st) for sn, smf, st in sets if smf is not None]
    if pmf is None or not sets:
        return dict(sleeve_overlap_mm3=None)
    mesh = [round(float((pmf ^ smf).volume()), 5) for _, smf, _ in sets]
    mesh_v = round(sum(mesh), 5)
    out = dict(sleeve_overlap_mm3=mesh_v, sleeve_overlap_mesh_mm3=mesh_v, sleeve_overlap_method="manifest meshes (manifold3d)")
    if mesh_v > 0 and step is not None and all(st is not None for _, _, st in sets):
        from checks import brep as L
        part = _placed(ctx, name, step)
        exact = sum(L.common_volume_mm3(part, s) for sn, _, st in sets for s in _placed(ctx, sn, st).solids())
        out.update(sleeve_overlap_mm3=round(float(exact), 5), sleeve_overlap_method="exact OCC common volume of the STEP solids (the mesh boolean read above 0)")
    return out


def mesh_part(ctx, name):
    """What geometry can measure on a printed part without a native STEP, from its own closed mesh: volume, L/R volume difference
    (mirror's mesh), sleeves found in its sleeve mesh, part-vs-sleeve overlap, extents vs the build volume, load-path slabs. The
    B-rep audit (faces, edges, fillets, solids) needs a STEP and stays null. None when the mesh is not closed."""
    pmf = _mesh_mf(ctx, name)
    if pmf is None:
        return None
    t0 = time.time()
    vol = float(pmf.volume())
    out = dict(source="own closed mesh (no native STEP): B-rep audit not measured", step_sha256=None, solids=None, invalid_solids=0,
               volume_mm3=round(vol, 2), mesh_volume_mm3=round(vol, 2), step_vs_mesh_volume_diff_pct=None)
    mirror = ctx.rel.mirror_of(name)
    mmf = _mesh_mf(ctx, mirror) if mirror in ctx.parts else None
    out.update(mirror_part=mirror if mmf is not None else None, mirror_source=ctx.rel.source(name, "mirror_of") if mmf is not None else None,
               mirror_volume_mm3=None if mmf is None else round(float(mmf.volume()), 2),
               mirror_volume_diff_pct=None if mmf is None else round(abs(vol - float(mmf.volume())) / max(1.0, vol) * 100.0, 4))
    sets = sleeve_sets(ctx, name)
    n_slv = 0
    if sets:
        import sys
        from checks.core import ROOT
        sys.path.insert(0, str(ROOT / "robot/body/fastening"))
        from mount_check import tubes
        n_slv = sum(len(tubes(ctx.body.mesh(sn))) for sn in sets)
    exp = (ctx.settings(name).get("expected_hardware") or {}).get("steel_sleeves")
    out.update(sleeve_sets=sets, sleeves=n_slv, expected_sleeves=exp, sleeves_missing=None if exp is None else exp - n_slv)
    out.update(sleeve_overlap(ctx, name, pmf, [(sn, _mesh_mf(ctx, sn), None) for sn in sets], None))
    out.update(plate_fit(ctx, name, None))
    spec = ctx.settings(name).get("loadpath_slices")
    out["loadpath"] = loadpath_slices(pmf, spec) if spec else None
    out["seconds"] = round(time.time() - t0, 1)
    return out


def audit_part(ctx, name):
    step = _step(ctx, name)
    if step is None:
        return None
    t0 = time.time()
    sh, sols = _solids(step)
    vol = sum(float(s.volume) for s in sols)
    out = dict(step_sha256=sha(step, raw=True), solids=len(sols), invalid_solids=sum(1 for s in sols if not s.is_valid), volume_mm3=round(vol, 2))
    mirror = ctx.rel.mirror_of(name)       # the definition's mirror_of, else the part whose x-mirrored box matches (never the name)
    mstep = _step(ctx, mirror) if mirror else None
    if mstep is not None:
        _, msols = _solids(mstep)
        mv = sum(float(s.volume) for s in msols)
        out.update(mirror_part=mirror, mirror_source=ctx.rel.source(name, "mirror_of"), mirror_volume_mm3=round(mv, 2),
                   mirror_invalid_solids=sum(1 for s in msols if not s.is_valid), mirror_volume_diff_pct=round(abs(vol - mv) / max(1.0, vol) * 100.0, 4))
    else:
        out.update(mirror_part=None, mirror_volume_diff_pct=None)
    sets = sleeve_sets(ctx, name)
    slv_steps = {sn: _step(ctx, sn) for sn in sets}
    n_slv = sum(len(_solids(st)[1]) for st in slv_steps.values() if st is not None)
    exp = (ctx.settings(name).get("expected_hardware") or {}).get("steel_sleeves")
    out.update(sleeve_sets=sets, sleeves=n_slv, expected_sleeves=exp, sleeves_missing=None if exp is None else exp - n_slv)
    # STEP vs manifest mesh volume (the mesh is what the other checks measure) and part vs its sleeves
    pmf = _mesh_mf(ctx, name)
    out["mesh_volume_mm3"] = None if pmf is None else round(float(pmf.volume()), 2)
    out["step_vs_mesh_volume_diff_pct"] = None if pmf is None else round(abs(vol - float(pmf.volume())) / max(1.0, vol) * 100.0, 4)
    out.update(sleeve_overlap(ctx, name, pmf, [(sn, _mesh_mf(ctx, sn), slv_steps[sn]) for sn in sets], step))
    out.update(plate_fit(ctx, name, sh))
    spec = ctx.settings(name).get("loadpath_slices")
    out["loadpath"] = loadpath_slices(pmf, spec) if (spec and pmf is not None) else None
    th = {**DEFAULT_THRESHOLDS, **(ctx.settings(name).get("geometry_thresholds") or {})}
    meas, worst = brep_audit(sh, th)
    out["thresholds"] = th
    out.update(meas)
    out["_worst"] = worst
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _applies(ctx):
    from checks import process as P
    return any(ctx.parts[n].get("native_step") or P.is_printed(ctx, n) for n in ctx.scope)


@register("geometry", _applies)
def run(ctx):
    """B-rep hygiene per scoped part (native STEP): invalid solids, sliver faces, micro-edges, acute wedges, G1 kinks, unrounded corners, L/R volume difference, sleeves vs expected."""
    t0 = time.time()
    per_part, cases, skipped = {}, [], []
    tot = dict(invalid_solids=0, sliver_faces=0, micro_edges=0, acute_wedge_edges=0, non_tangent_hub_steps=0, unrounded_profile_corners=0)
    from checks import process as P
    bought, dangling, from_mesh, not_measured = {}, [], [], {}
    for n in ctx.scope:
        if ctx.parts[n].get("native_step") and not BL.src(ctx.parts[n]["native_step"]).exists():
            dangling.append(n)
            continue
        if ctx.parts[n].get("native_step") and not (P.is_printed(ctx, n) or ctx.rel.kind(n) in MADE_KINDS):
            bought[n] = dict(role=ctx.parts[n].get("role"), kind=ctx.rel.kind(n), kind_source=ctx.rel.source(n, "kind"), step_mb=round(BL.src(ctx.parts[n]["native_step"]).stat().st_size / 1e6, 2)
                             if BL.src(ctx.parts[n]["native_step"]).exists() else None)
            continue
        r = audit_part(ctx, n)
        if r is None and P.is_printed(ctx, n):
            r = mesh_part(ctx, n)
            if r is None:
                not_measured[n] = "no native STEP and its mesh is not closed"
            else:
                per_part[n] = r
                from_mesh.append(n)
            continue
        if r is None:
            skipped.append(n)
            continue
        w = r.pop("_worst")
        per_part[n] = r
        for k in tot:
            tot[k] += r[k] if k != "invalid_solids" else r["invalid_solids"] + r.get("mirror_invalid_solids", 0)
        if w["sliver"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["sliver"][1], value=round(w["sliver"][0], 4), unit="mm2 smallest planar face", what=f"{r['sliver_faces']} sliver faces"))
        if w["micro"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["micro"][1], value=round(w["micro"][0], 4), unit="mm shortest micro-edge", what=f"{r['micro_edges']} micro-edges"))
        if w["acute"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=w["acute"][1], value=round(w["acute"][0], 1), unit="deg sharpest wedge normal angle", what=f"{r['acute_wedge_edges']} acute wedge edges"))
        if not r["solids"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=float(r.get("total_faces") or 0), unit="faces, no solid",
                              what="the native STEP holds faces but no solid: every solid-based check falls back to the mesh"))
        if r["mirror_volume_diff_pct"]:
            cases.append(dict(part=n, mating_part=r["mirror_part"], pose=None, location_mm=None, value=r["mirror_volume_diff_pct"], unit="% L/R volume difference", what="mirror volume"))
        if r["sleeves_missing"]:
            cases.append(dict(part=n, mating_part=", ".join(r["sleeve_sets"]) or None, pose=None, location_mm=None, value=r["sleeves_missing"], unit="sleeves short of expected", what=f"{r['sleeves']} of {r['expected_sleeves']}"))
    for n in from_mesh:
        r = per_part[n]
        if r["mirror_volume_diff_pct"]:
            cases.append(dict(part=n, mating_part=r["mirror_part"], pose=None, location_mm=None, value=r["mirror_volume_diff_pct"], unit="% L/R volume difference", what="mirror volume (closed meshes)"))
        if r["sleeves_missing"]:
            cases.append(dict(part=n, mating_part=", ".join(r["sleeve_sets"]) or None, pose=None, location_mm=None, value=r["sleeves_missing"], unit="sleeves short of expected", what=f"{r['sleeves']} of {r['expected_sleeves']} (sleeve mesh)"))
    for n in from_mesh + list(not_measured):
        cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=0.0, unit="B-rep faces audited", context=n in from_mesh,
                          what="no native STEP: " + ("measured from its own closed mesh (volume, mirror, sleeves, sleeve overlap, build volume, load-path slabs); "
                                                     "B-rep audit not measured" if n in from_mesh else not_measured[n] + "; nothing measured")))
    for n, r in per_part.items():
        if r.get("min_bed_margin_mm") is not None:     # context (listed in the report only) while the part fits; a negative margin is a finding
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=r["min_bed_margin_mm"], unit="mm smallest margin to build volume", context=r["min_bed_margin_mm"] >= 0,
                              what=f"extents {r['extents_mm']} ({r['frame']}, up {r['up']}) on {r['printer']}"
                                   + (f"; both nozzles {r['min_bed_margin_both_nozzles_mm']} mm" if r.get("min_bed_margin_both_nozzles_mm") is not None else "")
                                   + (f"; best yaw {r['best_yaw_deg']} deg gives {r['min_bed_margin_best_yaw_mm']} mm" if r.get("best_yaw_deg") else "")))
        if r["sleeve_overlap_mm3"]:
            cases.append(dict(part=n, mating_part=", ".join(r["sleeve_sets"]) or None, pose=None, location_mm=None, value=r["sleeve_overlap_mm3"], unit="mm3 part/sleeve overlap", what="sleeve overlap"))
        lp = r["loadpath"]
        if lp and lp["min_slab_area_mm2"] is None:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=lp["range_inside_part_pct"], unit="% of the load-path slice range inside the part",
                              what=f"the loadpath_slices setting ({lp['axis']} {lp['range_mm'][0]:g}..{lp['range_mm'][1]:g} mm) misses the part "
                                   f"({lp['axis']} {lp['part_extent_mm'][0]:g}..{lp['part_extent_mm'][1]:g} mm): set for another geometry, nothing sliced"))
        elif lp:
            stale = (f"; {lp['slabs_outside_part']} of {lp['slabs']} slabs outside the part (setting {lp['axis']} {lp['range_mm'][0]:g}..{lp['range_mm'][1]:g} mm, "
                     f"part {lp['part_extent_mm'][0]:g}..{lp['part_extent_mm'][1]:g} mm: {lp['range_inside_part_pct']} % of the range cuts it)" if lp["slabs_outside_part"] else "")
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=lp["min_slab_area_mm2"], unit="mm2 thinnest load-path slab",
                              what=f"{lp['severed_or_deadend_branches']} severed branches over the slabs inside the part" + stale))
        if r["step_vs_mesh_volume_diff_pct"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=None, value=r["step_vs_mesh_volume_diff_pct"], unit="% STEP vs mesh volume", what="STEP/mesh identity",
                              context=True))   # the summary's header already names parts whose screening solid is over 5 % off
    tick("geometry", t0)
    cases.sort(key=lambda c: (bool(c.get("context")), c["unit"] != "faces, no solid", -c["value"] if c["unit"].startswith(("%", "sleeves", "mm3")) else c["value"]))
    return dict(per_part=per_part, worst_cases=cases[:20], bought_not_audited=bought, dangling_native_step=dangling, measured_from_own_mesh=from_mesh,
                not_measured=not_measured,
                bad_events=dict(parts_audited=len(per_part) - len(from_mesh), bought_parts_not_audited=len(bought), dangling_native_step=len(dangling), invalid_solids=tot["invalid_solids"], sliver_faces=tot["sliver_faces"], micro_edges=tot["micro_edges"],
                                acute_wedge_edges=tot["acute_wedge_edges"], non_tangent_hub_steps=tot["non_tangent_hub_steps"],
                                unrounded_profile_corners=tot["unrounded_profile_corners"],
                                parts_over_build_volume=sum(1 for r in per_part.values() if (r.get("min_bed_margin_mm") or 0) < 0),
                                parts_over_build_volume_both_nozzles=sum(1 for r in per_part.values() if (r.get("min_bed_margin_both_nozzles_mm") or 0) < 0),
                                parts_without_print_direction=sum(1 for r in per_part.values() if r.get("up") is None),
                                parts_with_sleeve_overlap_over_0p05_mm3=sum(1 for r in per_part.values() if (r["sleeve_overlap_mm3"] or 0) > 0.05),
                                loadpath_severed_branches=sum((r["loadpath"] or {}).get("severed_or_deadend_branches", 0) for r in per_part.values()),
                                loadpath_slabs_outside_part=sum((r["loadpath"] or {}).get("slabs_outside_part", 0) for r in per_part.values()),
                                parts_without_native_step=len(skipped) + len(from_mesh) + len(not_measured),
                                printed_parts_measured_from_own_mesh=len(from_mesh), printed_parts_not_measured=len(not_measured),
                                steps_without_solid=sum(1 for r in per_part.values() if r["solids"] == 0)),
                evidence=[], notes="OpenCascade B-rep audit of the native STEP; event thresholds per part (setting geometry_thresholds, per_part[...].thresholds; defaults sliver < 0.25 mm2 planar face, micro-edge < 0.30 mm, acute wedge = normal angle > 125 deg on edges > 0.5 mm, "
                                   "G1 = cylinder/plane transition kink <= 8 deg); fillets = small cylinders (r <= 5.5 mm) split concave/convex; bed = extents in the print frame (process.up, plate_yaw_deg) vs the printer's build volume, main nozzle and both nozzles, plus the best plate yaw; without a print direction sorted STEP extents (orientation free); load-path slabs from the part's `loadpath_slices` setting (null = not set); sleeve overlap = manifest-mesh boolean, confirmed by the exact OCC common volume of the STEPs where the mesh reads above 0 (coincident press-fit cylinders tessellate into each other); "
                                   "mesh-only printed parts: what the closed mesh allows (volume, mirror, sleeves, sleeve overlap, build volume, load-path slabs), B-rep audit null; sleeves = solids of the STEPs of the sleeve sets pressed into the part (sleeve_sets: the definition's joined_to {how: pressed}, else the set whose sleeves touch the part) "
                                   "vs the part's expected_hardware.steel_sleeves setting (null = not set); mirror part = the definition's mirror_of, else the part whose x-mirrored box matches (mirror_source)")
