"""small_features: three small printability measurements per scoped printed part in its print direction (process.up).

Thin features (pins, ribs, tips): the part is sliced every layer height; each cross-section island's largest inscribed
circle (pole of inaccessibility, 0.005 mm) is its printable width. Islands narrower than two line widths are thin features
(under one line width they do not print); reported with height range and location. A feature one layer tall whose layer
below holds wider material under it (two line widths or more) is the top of that material, a curved crown or the last
layer of a dome, not a pin: counted apart (thin_tops_on_wider_material), with what is below every feature (`below`:
wider material / thin material / nothing / bed).
Z steps: pairs of flat faces (normal within 1 deg of vertical) less than one layer height apart in z that the slicer
cannot resolve (a chamfer, emboss or seat thinner than a layer vanishes or doubles): one over the other with a real plan
overlap (`overlap_mm2` over 0.05 mm2, no buffer), or side by side (within 0.05 mm in plan) with a step wall between them
(a near-vertical face spanning the two heights). Near-level facets side by side with no wall between them are the
tessellation of a curved surface, not a step: counted apart (near_level_facet_pairs).
Trapped support: internal voids (closed cavities, printed as trapped air or support), and the area of downward faces
beyond the material's overhang limit whose support would stand on the part itself instead of the bed (a vertical ray
down hits the part): support in pockets, bores and under ledges, harder to remove and scarring the face below.
Mesh-only parts on their full closed mesh (checks/print_solid.py). Measurements and counts only."""
from __future__ import annotations

import time

import numpy as np

from checks import print_solid as PS
from checks import process as P
from checks.core import tick
from checks.registry import register

RES = 0.05


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


def _inscribed(poly, res=RES):
    """Largest inscribed circle diameter (mm) and its centre: shapely's maximum_inscribed_circle (pole of inaccessibility) to a
    tolerance of 0.005 mm or 1/20 of the island's strip-width estimate 2 x area / perimeter, whichever is smaller. (A raster
    distance transform at 0.05 mm read the 1.26 mm H-cage cheek wall as 1.2 mm, under two 0.62 mm lines.)"""
    from shapely import maximum_inscribed_circle
    est = 2 * poly.area / max(poly.length, 1e-12)
    tol = max(1e-4, min(0.005, est / 20))
    ls = maximum_inscribed_circle(poly, tolerance=tol)
    (cx, cy), q = ls.coords[0], ls.coords[-1]
    return float(2 * np.hypot(q[0] - cx, q[1] - cy)), (float(cx), float(cy))


def thin_features(m, layer, lw):
    """Islands of the layer cross-sections narrower than 2 lw: list of {z_mm, width_mm, xy_mm, area_mm2}."""
    from shapely.geometry import MultiPolygon
    zs = np.arange(layer / 2, m.bounds[1][2], layer)
    secs = m.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=zs)
    out = []
    for z, s in zip(zs, secs):
        if s is None:
            continue
        for poly in s.polygons_full:
            geoms = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
            for g in geoms:
                if g.area <= 0:
                    continue
                # cheap screen: an island whose area / (perimeter / 2) is wide cannot be thin everywhere
                if 2 * g.area / max(g.length / 2, 1e-9) > 4 * lw and g.area > (2 * lw) ** 2 * 4:
                    continue
                w, c = _inscribed(g)
                if w < 2 * lw:
                    out.append(dict(z_mm=round(float(z), 3), width_mm=round(w, 3), xy_mm=None if c is None else [round(c[0], 2), round(c[1], 2)],
                                    area_mm2=round(float(g.area), 3), poly=g))
    return out


def below(m, poly, z, layer, lw):
    """What the layer under an island holds under it: ('bed' | 'nothing' | 'wider material' | 'thin material', width mm of
    the widest section polygon under it). Wider = an inscribed width of two line widths or more."""
    from shapely.geometry import MultiPolygon
    zb = z - layer
    if zb <= 0:
        return "bed", None
    s = m.section_multiplane(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1], heights=[zb])[0]   # same frame as thin_features
    if s is None:
        return "nothing", None
    best = None
    for pf in s.polygons_full:
        for g in (pf.geoms if isinstance(pf, MultiPolygon) else [pf]):
            if g.area <= 0 or not g.intersects(poly):
                continue
            if g.intersection(poly).area < 0.5 * poly.area:
                continue
            w = _inscribed(g)[0] if g.area < 400 * lw * lw else np.inf
            best = w if best is None else max(best, w)
    if best is None:
        return "nothing", None
    return ("wider material" if best >= 2 * lw else "thin material"), (None if not np.isfinite(best) else round(float(best), 3))


def z_steps(m, layer):
    """Pairs of flat faces closer than one layer height in z that form a step: [(dz, z_low, z_high, xy, real overlap mm2,
    kind)], kind 'overlap' (one over the other) or 'walled' (side by side with a near-vertical wall between them); and the
    count of near-level facet pairs side by side without a wall (a curved surface's tessellation)."""
    from shapely import contains_xy, polygons, union_all
    nz = m.face_normals[:, 2]
    flat = np.abs(nz) > np.cos(np.radians(1.0))
    vert = np.flatnonzero(np.abs(nz) < np.sin(np.radians(10.0)))         # near-vertical faces: step walls
    vc = m.triangles_center[vert]
    vz0, vz1 = m.vertices[m.faces[vert]][:, :, 2].min(1), m.vertices[m.faces[vert]][:, :, 2].max(1)
    zf = m.triangles_center[:, 2]
    levels, raw = {}, {}
    for z in np.unique(np.round(zf[flat], 3)):
        sel = flat & (np.abs(zf - z) < 5e-4)
        raw[float(z)] = union_all(polygons(m.triangles[sel][:, :, :2])).buffer(0)
        levels[float(z)] = raw[float(z)].buffer(0.05)
    zl = sorted(levels)
    out, level_pairs = [], 0
    for i, a in enumerate(zl):
        for b in zl[i + 1:]:
            if b - a >= layer:
                break
            if b - a < 2e-3:
                continue
            inter = levels[a].intersection(levels[b])
            if not inter.is_empty and inter.area > 0.05:
                real = raw[a].intersection(raw[b])          # plan overlap without the 0.05 mm buffer (0 = side by side)
                if real.area > 0.05:
                    c = real.representative_point()
                    out.append((round(b - a, 4), round(a, 3), round(b, 3), [round(c.x, 2), round(c.y, 2)], round(float(real.area), 2), "overlap"))
                    continue
                k = np.flatnonzero((vz0 >= a - 1e-3) & (vz1 <= b + 1e-3) & (vz1 - vz0 > 0.5 * (b - a)))
                wall = k[contains_xy(inter, vc[k, 0], vc[k, 1])] if len(k) else k
                if len(wall):
                    c = vc[wall[0]]
                    out.append((round(b - a, 4), round(a, 3), round(b, 3), [round(float(c[0]), 2), round(float(c[1]), 2)], round(float(real.area), 2), "walled"))
                else:
                    level_pairs += 1
    out.sort()
    return out, level_pairs


def trapped(m, lim, layer):
    """(internal voids [(volume_mm3, centroid)], beyond-limit area whose support stands on the part, its share, worst xy)."""
    voids = []
    try:
        for b in m.split(only_watertight=False):
            if b.volume < 0:
                voids.append((round(float(-b.volume), 2), np.round(b.centroid, 2).tolist()))
    except Exception as exc:  # noqa: BLE001
        voids = f"void search failed: {type(exc).__name__}: {exc}"
    nz = m.face_normals[:, 2]
    ov = 90.0 - np.degrees(np.arccos(np.clip(-nz, -1, 1)))
    zmin = m.vertices[m.faces].min(1)[:, 2]
    beyond = (nz < 0) & (ov > lim) & ~((zmin < layer) & (ov > 89.0))
    idx = np.flatnonzero(beyond)
    if not len(idx):
        return voids, 0.0, 0.0, None
    c = m.triangles_center[idx]
    loc, ir, _ = m.ray.intersects_location(c - [0, 0, 1e-3], np.tile([0.0, 0.0, -1.0], (len(c), 1)), multiple_hits=False)
    on_part = np.zeros(len(c), bool); on_part[ir] = True
    a = m.area_faces[idx]
    worst = None
    if on_part.any():
        j = np.argmax(np.where(on_part, a, 0))
        worst = np.round(c[j], 2).tolist()
    return voids, float(a[on_part].sum()), float(a[on_part].sum() / max(a.sum(), 1e-9)), worst


@register("small_features", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part in its print direction: thin pins / ribs per layer, Z steps under a layer height, internal voids, support standing on the part."""
    mats, _ = P.tables()
    per, cases = {}, []
    ev = dict(parts=0, thin_features_under_1_line=0, thin_features_under_2_lines=0, thin_tops_on_wider_material=0, void_search_failures=0,
              z_steps_under_layer=0, near_level_facet_pairs=0, internal_voids=0, parts_with_support_on_part=0)
    for n in _targets(ctx):
        t0 = time.time()
        e = P.effective(ctx, n)
        lw, layer = e["line_width_mm"] or 0.42, e["layer_height_mm"] or 0.2
        lim = ((mats.get(e["material"]) or {}).get("print") or {}).get("max_overhang_deg", 45.0)
        m, T = PS.print_mesh(ctx, n, e, with_transform=True)
        Ti = np.linalg.inv(T)
        back = lambda xy, z: np.round((Ti @ np.r_[xy, z, 1.0])[:3], 2).tolist()  # noqa: E731
        thin = thin_features(m, layer, lw)
        steps, level_pairs = z_steps(m, layer)
        voids, sup_on_part, share, worst = trapped(m, lim, layer)
        void_error = voids if isinstance(voids, str) else None
        voids = [] if void_error else voids
        # one feature per pin / rib: islands within 0.5 mm in plan on consecutive layers merge into a z range
        feats = []
        for t in sorted(thin, key=lambda t: t["z_mm"]):
            for f in feats:
                if abs(t["z_mm"] - f["z_to_mm"]) <= 1.5 * layer and np.hypot(t["xy_mm"][0] - f["xy_mm"][0], t["xy_mm"][1] - f["xy_mm"][1]) < 0.5:
                    f["z_to_mm"] = t["z_mm"]; f["layers"] += 1
                    if t["width_mm"] < f["width_mm"]:
                        f.update(width_mm=t["width_mm"], z_mm=t["z_mm"], xy_mm=t["xy_mm"])
                    break
            else:
                feats.append(dict(t, z_from_mm=t["z_mm"], z_to_mm=t["z_mm"], layers=1))
        for f in feats:                      # what the layer under the feature's lowest island (its first, f["poly"]) holds
            f["below"], f["below_width_mm"] = below(m, f["poly"], f["z_from_mm"], layer, lw)
        for f in feats:
            f.pop("poly", None)
        tops = [f for f in feats if f["layers"] == 1 and f["below"] == "wider material"]
        thin = sorted([f for f in feats if f not in tops], key=lambda t: t["width_mm"])
        r = dict(material=e["material"], up=e["up"], line_width_mm=lw, line_width_source=e["source"]["line_width_mm"] if e["line_width_mm"] else "no line width resolved: 0.42 mm assumed",
                 layer_height_mm=layer, solid=PS.solid(ctx, n)[1],
                 thin_tops_on_wider_material=len(tops), thin_tops=[dict(t, location_mm=back(t["xy_mm"], t["z_mm"]) if t["xy_mm"] else None)
                                                                   for t in sorted(tops, key=lambda t: t["width_mm"])[:4]],
                 thin_features_under_1_line=sum(1 for t in thin if t["width_mm"] < lw), thin_features_under_2_lines=len(thin),
                 thin_layers_under_2_lines=sum(t["layers"] for t in thin), void_search_error=void_error,
                 thinnest=[dict(t, location_mm=back(t["xy_mm"], t["z_mm"]) if t["xy_mm"] else None) for t in thin[:8]],
                 z_steps=[dict(dz_mm=s[0], z_low_mm=s[1], z_high_mm=s[2], overlap_mm2=s[4], kind=s[5], location_mm=back(s[3], s[1])) for s in steps[:10]],
                 z_steps_count=len(steps), z_steps_overlapping=sum(1 for s in steps if s[5] == "overlap"),
                 z_steps_side_by_side_walled=sum(1 for s in steps if s[5] == "walled"), near_level_facet_pairs=level_pairs, internal_voids=[dict(volume_mm3=v, location_mm=(Ti @ np.r_[c, 1.0])[:3].round(2).tolist()) for v, c in voids],
                 support_on_part_mm2=round(sup_on_part, 1), support_on_part_share=round(share, 3),
                 support_on_part_location_mm=None if worst is None else (Ti @ np.r_[worst, 1.0])[:3].round(2).tolist())
        r["seconds"] = round(time.time() - t0, 1); tick("small_features", t0)
        per[n] = r
        ev["parts"] += 1
        ev["thin_features_under_1_line"] += r["thin_features_under_1_line"]
        ev["thin_features_under_2_lines"] += r["thin_features_under_2_lines"]
        ev["void_search_failures"] += bool(void_error)
        ev["z_steps_under_layer"] += r["z_steps_count"]
        ev["near_level_facet_pairs"] += r["near_level_facet_pairs"]
        ev["thin_tops_on_wider_material"] += r["thin_tops_on_wider_material"]
        ev["internal_voids"] += len(voids)
        ev["parts_with_support_on_part"] += sup_on_part > 1.0
        if thin:
            t = r["thinnest"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=t["location_mm"], value=t["width_mm"], unit="mm thinnest feature",
                              what=f"{r['thin_features_under_2_lines']} features under 2 lines ({lw} mm), {r['thin_features_under_1_line']} under 1 line; "
                                   f"this one z {t['z_from_mm']}-{t['z_to_mm']} mm in the print frame ({t['layers']} layers, below: {t['below']}); "
                                   f"{r['thin_tops_on_wider_material']} one-layer tops on wider material apart"))
        if steps:
            s = r["z_steps"][0]
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=s["location_mm"], value=s["dz_mm"], unit="mm Z step",
                              what=f"{r['z_steps_count']} flat-face steps under one layer ({layer} mm): {r['z_steps_overlapping']} one over the other, "
                                   f"{r['z_steps_side_by_side_walled']} side by side with a step wall; this one {s['kind']}, {s['overlap_mm2']} mm2 plan overlap; "
                                   f"{r['near_level_facet_pairs']} near-level facet pairs without a wall apart"))
        for v in r["internal_voids"]:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=v["location_mm"], value=v["volume_mm3"], unit="mm3 internal void",
                              what="closed cavity inside the solid"))
        if sup_on_part > 1.0:
            cases.append(dict(part=n, mating_part=None, pose=None, location_mm=r["support_on_part_location_mm"], value=r["support_on_part_mm2"],
                              unit="mm2 support standing on the part", what=f"{round(100 * share)} % of the beyond-limit area ({lim} deg)"))
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"slices every layer height; island width = largest inscribed circle (shapely maximum_inscribed_circle, 0.005 mm); one-layer islands on wider material below "
                      f"counted apart as tops; Z steps = flat faces (within 1 deg) closer than one layer, one over the other (real plan overlap over "
                      f"0.05 mm2) or side by side with a near-vertical wall between them; near-level facet pairs without a wall counted apart; voids = inside-out shells of the solid; support on part = beyond-limit face centres whose "
                      f"downward ray hits the part; locations in the manifest frame")
