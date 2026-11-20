"""overhang: per scoped printed part with a print direction, in its print frame (process.up): downward-facing area by
overhang angle, area beyond the material's limit, the largest unsupported flat ceilings with their bridge span, and the
part of the beyond-limit area that touches a mating part at standing (support scars on a seating face).

Angle convention (stated, the data sheets give none): overhang angle of a downward face from the vertical, 0 = vertical
wall, 90 = flat ceiling. The material limit is materials.json <material>.print.max_overhang_deg (Bambu: ~70 deg for
PAHT-CF and PETG-CF), the bridge limit print.max_bridge_mm (40 / 30 mm). Faces within one layer height of the bed are on
the bed, not overhang. A flat ceiling (angle > 80 deg) region's bridge span is twice the largest distance from a point of it to
its nearest supported edge (see _span): the width of a true bridge, twice the depth of a one-sided ledge, and large for a
plate that floats on a few posts. Measurements and counts only."""
from __future__ import annotations

import time

import numpy as np

from checks import process as P
from checks.core import tick
from checks.registry import register

BANDS = (0, 30, 45, 60, 70, 80, 90)      # deg from vertical
FLAT_DEG = 80.0                           # ceiling faces for the bridge span
MATE_MM = 0.3                             # beyond-limit surface closer than this to a neighbour at standing = mating face


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


def _regions(m, sel):
    """Connected groups (face indices) of the selected faces of mesh m."""
    import networkx as nx
    idx = np.flatnonzero(sel)
    if not len(idx):
        return []
    adj = m.face_adjacency
    keep = sel[adj[:, 0]] & sel[adj[:, 1]]
    g = nx.Graph(); g.add_nodes_from(idx); g.add_edges_from(adj[keep])
    return [np.fromiter(c, int) for c in nx.connected_components(g)]


def _span(m, faces, res=0.2):
    """Bridge span of a flat-ceiling region (print frame): 2 x the largest distance (mm) from a point of the region's XY
    footprint to its nearest SUPPORTED edge, and that point. An edge is supported when the face across it reaches below the
    edge (a wall or a self-supporting slope going down towards the bed); an edge whose neighbour face only rises (the outer
    wall of a ledge or of a floating plate) holds nothing. A bridge between two supported walls reads its width; a ledge
    supported on one side reads twice its depth; a region with no supported edge reads twice its largest extent from its
    centre (it floats). Rasterised at `res` mm (reads up to ~res off)."""
    from scipy.ndimage import distance_transform_edt
    from shapely import contains_xy, polygons, union_all
    tri = m.triangles[faces][:, :, :2]
    poly = union_all(polygons(tri)).buffer(0.01).buffer(-0.01)
    if poly.is_empty:
        return 0.0, None, 0.0
    inr = np.zeros(len(m.faces), bool); inr[faces] = True
    adj, aedge = m.face_adjacency, m.face_adjacency_edges
    border = inr[adj[:, 0]] ^ inr[adj[:, 1]]
    other = np.where(inr[adj[border, 0]], adj[border, 1], adj[border, 0])
    ez = m.vertices[aedge[border]][:, :, 2].mean(1)
    fz = m.vertices[m.faces[other]][:, :, 2].min(1)
    sup = fz < ez - 0.05
    seg = m.vertices[aedge[border][sup]][:, :, :2]
    blen = np.linalg.norm(m.vertices[aedge[border]][:, 0, :2] - m.vertices[aedge[border]][:, 1, :2], axis=1)
    sup_frac = float(blen[sup].sum() / max(blen.sum(), 1e-9))
    x0, y0, x1, y1 = poly.bounds
    xs = np.arange(x0 - res, x1 + 2 * res, res); ys = np.arange(y0 - res, y1 + 2 * res, res)
    X, Y = np.meshgrid(xs, ys)
    inside = contains_xy(poly, X, Y)
    if not inside.any():
        return 0.0, None, sup_frac
    if len(seg):
        L = np.linalg.norm(seg[:, 1] - seg[:, 0], axis=1)
        k = np.maximum(2, np.ceil(L / (res / 2)).astype(int) + 1)          # sample every edge at res / 2 or finer
        pts = np.vstack([np.linspace(a, b, kk) for (a, b), kk in zip(seg, k)])
        free = np.ones_like(inside)
        ix = np.clip(np.round((pts[:, 0] - xs[0]) / res).astype(int), 0, len(xs) - 1)
        iy = np.clip(np.round((pts[:, 1] - ys[0]) / res).astype(int), 0, len(ys) - 1)
        free[iy, ix] = False
        d = distance_transform_edt(free) * res
    else:                                   # nothing below the region: it floats; distance from the footprint's centre point
        c = poly.representative_point()
        d = np.hypot(X - c.x, Y - c.y)
    d = np.where(inside, d, 0.0)
    i = np.unravel_index(int(np.argmax(d)), d.shape)
    return float(2 * d[i]), (float(X[i]), float(Y[i])), sup_frac


_TREES = {}


def _surface_tree(ctx, k, spacing=0.25, cap=600_000):
    """KD-tree of points sampled evenly on part k's solid surface (its manifest frame), about `spacing` mm apart (memory-bounded;
    distances read up to ~spacing/2 high)."""
    if k not in _TREES:
        import trimesh
        from scipy.spatial import cKDTree
        mk = ctx.solid(k)[0]
        cnt = int(min(cap, max(2000, mk.area / spacing ** 2)))
        pts, _ = trimesh.sample.sample_surface(mk, cnt, seed=1)
        _TREES[k] = cKDTree(np.vstack([pts, mk.vertices]))
    return _TREES[k]


def _neighbour_distance(ctx, n, pts_part):
    """Distance (mm) of points (part's manifest frame) to the nearest standing neighbour surface, and that neighbour's name."""
    body = ctx.body
    q = body.standing
    Ta = body.link_T(ctx.link(n), q)
    pw = pts_part @ Ta[:3, :3].T + Ta[:3, 3]
    lo, hi = pw.min(0) - 1.0, pw.max(0) + 1.0
    d = np.full(len(pw), np.inf); who = np.full(len(pw), None, dtype=object)
    from checks import bodylib as BL
    for k, p in body.parts.items():
        if k == n or (BL.is_fastener(k, p) and p.get("role") != "compression_limit_sleeves"):
            continue
        Tk = body.link_T(p["owning_link"], q)
        b = np.array(p["bounds_mm"], float)
        c = np.array([[b[i][0], b[j][1], b[kk][2]] for i in (0, 1) for j in (0, 1) for kk in (0, 1)]) @ Tk[:3, :3].T + Tk[:3, 3]
        if np.any(c.min(0) > hi) or np.any(c.max(0) < lo):
            continue
        loc = (pw - Tk[:3, 3]) @ Tk[:3, :3]                  # points in k's frame
        dk = _surface_tree(ctx, k).query(loc, distance_upper_bound=2.0)[0]
        better = dk < d
        d[better] = dk[better]; who[better] = k
    return d, who


def check_part(ctx, n):
    t0 = time.time()
    e = P.effective(ctx, n)
    mats, _ = P.tables()
    pr = (mats.get(e["material"]) or {}).get("print", {})
    lim, bridge_lim = pr.get("max_overhang_deg"), pr.get("max_bridge_mm")
    layer = e["layer_height_mm"] or 0.2
    m, T = P.print_mesh(ctx, n, e, with_transform=True)
    Tinv = np.linalg.inv(T)
    nz = m.face_normals[:, 2]
    zmin = m.vertices[m.faces].min(1)[:, 2]
    area = m.area_faces
    ang = np.degrees(np.arccos(np.clip(-nz, -1, 1)))          # 0 = facing straight down (ceiling) ... measured from the down axis
    ov = 90.0 - ang                                           # overhang angle from vertical: 90 = ceiling, 0 = vertical wall
    down = (nz < -1e-6)
    on_bed = down & (zmin < layer) & (ov > 89.0)
    over = down & ~on_bed
    bands = {f"{BANDS[i]}-{BANDS[i + 1]}": round(float(area[over & (ov >= BANDS[i]) & (ov < BANDS[i + 1] + (1e-9 if i == len(BANDS) - 2 else 0))].sum()), 1)
             for i in range(len(BANDS) - 1)}
    beyond = over & (ov > lim) if lim is not None else over & (ov > 45.0)
    out = dict(material=e["material"], up=e["up"], up_source=e["source"]["up"], layer_height_mm=layer, max_overhang_deg=lim,
               max_bridge_mm=bridge_lim, bed_contact_mm2=round(float(area[on_bed].sum()), 1),
               downward_area_by_overhang_deg_mm2=bands, beyond_limit_mm2=round(float(area[beyond].sum()), 1),
               beyond_45_mm2=round(float(area[over & (ov > 45.0)].sum()), 1), part_surface_mm2=round(float(area.sum()), 1))
    # flat ceilings: connected regions, bridge span, location (manifest frame)
    regs = []
    for f in _regions(m, over & (ov > FLAT_DEG)):
        a = float(area[f].sum())
        if a < 1.0:
            continue
        span, c, sup = _span(m, f)
        z = float(np.average(m.triangles_center[f][:, 2], weights=area[f]))
        loc = (Tinv @ np.array([c[0], c[1], z, 1.0]))[:3] if c else None
        regs.append(dict(area_mm2=round(a, 1), span_mm=round(span, 2), height_mm=round(z, 2), supported_edge_fraction=round(sup, 2), floats=sup == 0.0,
                         location_mm=None if loc is None else np.round(loc, 2).tolist()))
    regs.sort(key=lambda r: -r["span_mm"])
    out["flat_ceilings"] = regs[:12]
    out["flat_ceiling_regions"] = len(regs)
    out["longest_bridge_span_mm"] = regs[0]["span_mm"] if regs else 0.0
    out["bridges_over_limit"] = sum(1 for r in regs if bridge_lim and r["span_mm"] > bridge_lim)
    out["floating_ceilings"] = sum(1 for r in regs if r["floats"])
    # beyond-limit surface on a mating face at standing
    mate_area, mates = 0.0, {}
    bf = np.flatnonzero(beyond)
    if len(bf):
        cen = m.triangles_center[bf]
        cen_part = (cen - T[:3, 3]) @ T[:3, :3]            # print frame -> part manifest frame (T rows map part -> print)
        d, who = _neighbour_distance(ctx, n, cen_part)
        near = d < MATE_MM
        mate_area = float(area[bf][near].sum())
        for w, a in zip(who[near], area[bf][near]):
            mates[w] = mates.get(w, 0.0) + float(a)
        mates = {k: round(v, 1) for k, v in mates.items() if v >= 0.1}
    out["beyond_limit_on_mating_faces_mm2"] = round(mate_area, 1)
    out["beyond_limit_on_mating_faces_by_part_mm2"] = dict(sorted(mates.items(), key=lambda t: -t[1]))
    # worst beyond-limit region location
    big = sorted(((float(area[f].sum()), f) for f in _regions(m, beyond)), key=lambda t: -t[0])
    out["largest_beyond_limit_regions"] = [dict(area_mm2=round(a, 1), location_mm=np.round((Tinv @ np.r_[np.average(m.triangles_center[f], 0, area[f]), 1.0])[:3], 2).tolist())
                                           for a, f in big[:5]]
    out["seconds"] = round(time.time() - t0, 1)
    tick("overhang", t0)
    return out


@register("overhang", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part in its print direction: downward area by overhang angle, area beyond the material limit, flat-ceiling bridge spans, beyond-limit area on mating faces."""
    per, cases = {}, []
    ev = dict(parts=0, parts_with_beyond_limit_area=0, bridges_over_limit=0, floating_ceilings=0, parts_with_support_on_mating_faces=0)
    for n in _targets(ctx):
        r = check_part(ctx, n)
        per[n] = r
        ev["parts"] += 1
        ev["parts_with_beyond_limit_area"] += r["beyond_limit_mm2"] > 1.0
        ev["bridges_over_limit"] += r["bridges_over_limit"]
        ev["floating_ceilings"] += r["floating_ceilings"]
        ev["parts_with_support_on_mating_faces"] += r["beyond_limit_on_mating_faces_mm2"] > 1.0
        if r["largest_beyond_limit_regions"]:
            g = r["largest_beyond_limit_regions"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=g["location_mm"], value=r["beyond_limit_mm2"],
                              unit=f"mm2 downward beyond {r['max_overhang_deg']} deg from vertical",
                              what=f"needs support (up {r['up']}, {r['up_source']}); largest region {g['area_mm2']} mm2"))
        if r["flat_ceilings"]:
            f = r["flat_ceilings"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=f["location_mm"], value=f["span_mm"], unit="mm bridge span",
                              what=f"widest flat ceiling {f['area_mm2']} mm2 at {f['height_mm']} mm above the bed"
                                   + (", nothing below it (floats: needs support over its whole extent)" if f["floats"] else f", {round(100 * f['supported_edge_fraction'])} % of its edge supported")
                                   + f" (bridge limit {r['max_bridge_mm']} mm)"))
        if r["beyond_limit_on_mating_faces_mm2"] > 1.0:
            k, a = next(iter(r["beyond_limit_on_mating_faces_by_part_mm2"].items()))
            cases.append(dict(part=n, mating_part=k, pose="standing", location_mm=None, value=r["beyond_limit_on_mating_faces_mm2"],
                              unit="mm2 support on mating faces", what=f"beyond-limit area within {MATE_MM} mm of a neighbour (largest: {a} mm2)"))
    cases.sort(key=lambda c: -c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"print frame from process.up; overhang angle from vertical (0 wall, 90 ceiling), limit = the material's print.max_overhang_deg "
                      f"(data sheets give no convention; beyond_45_mm2 shows the generic 45 deg rule); bed = downward faces within one layer of z 0; "
                      f"bridge span = 2 x the largest distance inside each flat (> {FLAT_DEG:g} deg) ceiling region to its nearest supported edge (an edge whose neighbour face goes down towards the bed); mating = beyond-limit face centres "
                      f"within {MATE_MM} mm of another part's surface at standing")
