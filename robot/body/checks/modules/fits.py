"""fits: two checks on the interfaces between printed parts at standing, scores in mm, no pass/fail.

elephant_foot  per printed part in its print direction: where the bed outline (first layer) runs along a mating wall
               (within mate_mm of another part at standing) and has no chamfer. Method (mesh profile, not B-rep): the
               part is sliced at zone_mm (0.5) above the bed; every bed-outline sample (every sample_mm) takes the nearest
               point of that slice outline. Signed inset = how far the first-layer edge is pulled in from the wall at
               zone_mm (+ = chamfer or fillet, 0 = square foot, - = the wall leans out over the bed edge). The slice point decides mating (distance
               to a neighbour's surface) and the neighbour's name. Without chamfer = inset under chamfer_min_mm. Reported:
               mating outline length, length without chamfer, area (that length x zone_mm), runs with locations
               (manifest frame), mating parts, smallest inset. Bed-outline samples more than 1 mm from any wall at zone_mm
               (an undercut) are skipped.
fit_class      per scoped part and mating pair within pair_radius_mm (or with a class set): minimum distance at standing
               (negative = deepest interpenetration of either surface in the other), the class named in part_settings
               (<part>.fit_classes = {mate: class}) and its band from data/fit_classes.json, and the signed margin in mm
               (positive = distance inside the band to its nearer edge, negative = outside; also which side). Pairs
               without a class are counted (coverage), not judged. A pair on two links moves at a joint: its gap is a running
               clearance (owned by clearance / collision_sweep), listed with its distance, never given a static class from a
               seed; only a confirmed (not seeded) class on such a pair is judged (e.g. a printed bushing)."""
from __future__ import annotations

import json
import time

import numpy as np

from checks import process as P
from checks.core import DATA, tick
from checks.registry import register


def table():
    return json.loads((DATA / "fit_classes.json").read_text())


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n) and P.effective(ctx, n)["up"] is not None]


# ---------------------------------------------------------------- elephant foot

def foot(m, near, zone=0.5, mate=0.3, chamfer_min=0.2, step=0.5, layer=0.2):
    """Elephant-foot measurement on mesh m in its print frame (bed at z 0). near(points_print_frame) -> (distance, name)
    to the nearest neighbour surface. Returns the measurement dict; run points are in the print frame (key `runs_xyz`)."""
    from scipy.spatial import cKDTree
    from shapely import contains_xy, polygons, union_all
    from shapely.geometry import MultiPolygon
    nz = m.face_normals[:, 2]
    bed = (nz < -0.999) & (m.vertices[m.faces][:, :, 2].max(1) < layer)
    out = dict(zone_mm=zone, mate_mm=mate, chamfer_min_mm=chamfer_min, mating_outline_mm=0.0, no_chamfer_outline_mm=0.0, no_chamfer_area_mm2=0.0,
               runs=[], mates={}, smallest_inset_mm=None, bed_outline_mm=0.0)
    if not bed.any():
        return out
    poly = union_all(polygons(m.triangles[bed][:, :, :2])).buffer(0.01, join_style="mitre").buffer(-0.01, join_style="mitre")
    sec = m.section(plane_origin=[0, 0, zone], plane_normal=[0, 0, 1])
    if sec is None:
        return out
    wall = []
    for d in sec.discrete:
        d = np.asarray(d)[:, :2]
        seg = np.linalg.norm(np.diff(d, axis=0), axis=1)
        k = np.maximum(2, np.ceil(seg / 0.1).astype(int) + 1)
        for a, b, kk in zip(d[:-1], d[1:], k):
            wall.append(np.linspace(a, b, kk))
    wtree = cKDTree(np.vstack(wall))
    rings = []
    for g in (poly.geoms if isinstance(poly, MultiPolygon) else [poly]):
        if g.area > 0.01:
            rings += [g.exterior] + list(g.interiors)
    for ring in rings:
        L = ring.length
        out["bed_outline_mm"] += L
        n = max(3, int(round(L / step)))
        s = np.linspace(0, L, n, endpoint=False)
        b = np.array([ring.interpolate(x).coords[0] for x in s])
        dist, idx = wtree.query(b)
        wp = wtree.data[idx]
        has = dist < 1.0
        inside = contains_xy(poly, wp[:, 0], wp[:, 1])
        inset = np.where(inside, -dist, dist)       # + = bed edge pulled in from the wall (chamfer), - = the foot flares out past the wall
        q = np.c_[wp, np.full(len(wp), zone)]
        dn, who = near(q)
        mating = has & (dn < mate)
        nochamfer = mating & (inset < chamfer_min)
        ds = L / n
        out["mating_outline_mm"] += float(mating.sum() * ds)
        out["no_chamfer_outline_mm"] += float(nochamfer.sum() * ds)
        if mating.any():
            out["smallest_inset_mm"] = float(min(inset[mating].min(), out["smallest_inset_mm"] if out["smallest_inset_mm"] is not None else 1e9))
        for w, f in zip(who[mating], nochamfer[mating]):
            e = out["mates"].setdefault(w, [0.0, 0.0]); e[0] += ds; e[1] += ds if f else 0.0
        # runs of consecutive no-chamfer samples (ring is closed: wrap)
        flag = nochamfer
        if flag.all():
            runs = [np.arange(n)]
        else:
            st = int(np.flatnonzero(~flag)[0]); order = (np.arange(n) + st) % n
            runs, cur = [], []
            for i in order:
                if flag[i]:
                    cur.append(i)
                elif cur:
                    runs.append(np.array(cur)); cur = []
            if cur:
                runs.append(np.array(cur))
        for r in runs:
            mid = r[len(r) // 2]
            names = [w for w in who[r] if w is not None]
            out["runs"].append(dict(length_mm=round(len(r) * ds, 2), inset_mm=round(float(inset[r].min()), 2),
                                    mating_part=max(set(names), key=names.count) if names else None, runs_xyz=[float(b[mid][0]), float(b[mid][1]), 0.0]))
    out["no_chamfer_area_mm2"] = out["no_chamfer_outline_mm"] * zone
    return out


@register("elephant_foot", lambda ctx: bool(_targets(ctx)))
def run_foot(ctx):
    """Per printed part in its print direction: bed outline on mating walls (within mate_mm of a neighbour at standing) with and without a chamfer, area, locations, mating part."""
    from checks.modules.overhang import _neighbour_distance
    cfg = table()["elephant_foot"]
    per, cases = {}, []
    ev = dict(parts=0, parts_with_mating_bed_edge=0, parts_with_unchamfered_mating_bed_edge=0, unchamfered_runs=0)
    for n in _targets(ctx):
        t0 = time.time()
        e = P.effective(ctx, n)
        m, T = P.print_mesh(ctx, n, e, with_transform=True)
        Ti = np.linalg.inv(T)

        def near(q, T=T, n=n):
            qp = (q - T[:3, 3]) @ T[:3, :3]
            *_, d, who = _neighbour_distance(ctx, n, qp, np.zeros_like(qp))   # nearest neighbour surface of any direction
            return d, who
        r = foot(m, near, cfg["zone_mm"], cfg["mate_mm"], cfg["chamfer_min_mm"], cfg["sample_mm"], e["layer_height_mm"] or 0.2)
        for run in r["runs"]:
            run["location_mm"] = np.round((Ti @ np.r_[run.pop("runs_xyz"), 1.0])[:3], 2).tolist()
        r["runs"].sort(key=lambda x: -x["length_mm"])
        r["runs"] = r["runs"][:8]
        r["mates"] = {k: dict(outline_mm=round(v[0], 1), without_chamfer_mm=round(v[1], 1)) for k, v in r["mates"].items()}
        for k in ("mating_outline_mm", "no_chamfer_outline_mm", "no_chamfer_area_mm2", "bed_outline_mm"):
            r[k] = round(r[k], 2 if k != "no_chamfer_area_mm2" else 2)
        if r["smallest_inset_mm"] is not None:
            r["smallest_inset_mm"] = round(r["smallest_inset_mm"], 2)
        r.update(up=e["up"], up_source=e["source"]["up"], seconds=round(time.time() - t0, 1))
        per[n] = r
        ev["parts"] += 1
        ev["parts_with_mating_bed_edge"] += r["mating_outline_mm"] > 0
        ev["parts_with_unchamfered_mating_bed_edge"] += r["no_chamfer_outline_mm"] > 0
        ev["unchamfered_runs"] += len([x for x in r["runs"]])
        if r["no_chamfer_outline_mm"] > 0:
            top = r["runs"][0]
            cases.append(dict(part=n, mating_part=top["mating_part"], pose="standing", location_mm=top["location_mm"], value=r["no_chamfer_outline_mm"],
                              unit="mm of bed outline on a mating wall without chamfer",
                              what=f"{r['no_chamfer_area_mm2']} mm2 in the first {cfg['zone_mm']} mm, of {r['mating_outline_mm']} mm mating outline; longest run {top['length_mm']} mm, inset {top['inset_mm']} mm ({e['up']}, {e['source']['up']})"))
        tick("elephant_foot", t0)
    cases.sort(key=lambda c: -c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"mesh profile (not B-rep): part sliced at {cfg['zone_mm']} mm above the bed, signed inset of that wall from the first-layer outline at every "
                      f"{cfg['sample_mm']} mm; mating = wall within {cfg['mate_mm']} mm of another part at standing (neighbour surface sampled ~0.25 mm: reads up to ~0.12 high); "
                      f"chamfer = inset >= {cfg['chamfer_min_mm']} mm; area = length x zone; locations in the manifest frame")


# ---------------------------------------------------------------- clearance per fit class

def margin(c, lo, hi):
    """Signed margin (mm) of clearance c against the band [lo, hi]: distance to the nearer band edge, negative outside."""
    return min(c - lo, hi - c)


def penetration(ma, mb, n=4000, seed=1):
    """Deepest interpenetration (mm) of two overlapping meshes: the largest distance of a surface sample of one that lies inside the
    other to the other's surface (both ways). 0 when none is inside."""
    import trimesh
    best = 0.0
    for a, b in ((ma, mb), (mb, ma)):
        pts = np.vstack([trimesh.sample.sample_surface(a, n, seed=seed)[0], a.vertices])
        inside = b.contains(pts) if b.is_watertight else np.zeros(len(pts), bool)
        if inside.any():
            q = pts[inside]
            if len(q) > 1500:
                q = q[np.random.default_rng(seed).choice(len(q), 1500, replace=False)]
            best = max(best, float(trimesh.proximity.closest_point(b, q)[1].max()))
    return best


def judge(c, cls, classes):
    b = classes[cls]
    return dict(fit_class=cls, band_mm=[b["min_mm"], b["max_mm"]], margin_mm=round(margin(c, b["min_mm"], b["max_mm"]), 3),
                side="inside" if margin(c, b["min_mm"], b["max_mm"]) >= 0 else ("too tight" if c < b["min_mm"] else "too loose"))


@register("fit_class", lambda ctx: bool(ctx.scope))
def run_fit_class(ctx):
    """Per scoped part and mating pair at standing: minimum distance (negative = interference depth) against the pair's fit class band as a signed margin in mm; pairs without a class counted."""
    from checks.modules.gaps import check_gaps
    cfg = table()
    classes, rad = cfg["classes"], cfg["pair_radius_mm"]
    for n in ctx.scope:
        ctx.solid(n)
    raw = check_gaps(ctx.body, ctx.scope, ctx.sol, ctx.mf)
    per, cases, seen = {}, [], set()
    ev = dict(pairs_measured=0, pairs_with_class=0, pairs_without_class=0, pairs_outside_band=0, unknown_class_names=0)
    for k, g in raw.items():
        a, n = k.split(" x ")
        spec = ctx.settings(a).get("fit_classes", {}).get(n) or ctx.settings(n).get("fit_classes", {}).get(a)
        # a class is "slide" or {"class": "slide", "seeded": "<why, not yet confirmed>"}
        cls = spec.get("class") if isinstance(spec, dict) else spec
        cls_src = (f"seeded ({spec['seeded']})" if isinstance(spec, dict) and spec.get("seeded") else "part setting") if spec else None
        running = not g["same_link"]
        if running and cls and cls_src.startswith("seeded"):
            ev["seeded_classes_ignored_on_running_pairs"] = ev.get("seeded_classes_ignored_on_running_pairs", 0) + 1
            cls = None
        near = g["distance_mm"] <= rad or g["overlap_mm3"] > 0
        if not near and not cls:
            continue
        if cls and cls not in classes:
            ev["unknown_class_names"] += 1
            cls = None
        t0 = time.time()
        c = g["distance_mm"]
        if g["overlap_mm3"] > 1e-3:
            c = -_depth(ctx, a, n)
        rec = dict(clearance_mm=round(float(c), 3), overlap_mm3=g["overlap_mm3"], same_link=g["same_link"])
        ev["pairs_measured"] += 1
        if cls:
            rec.update(judge(c, cls, classes)); ev["pairs_with_class"] += 1
            band_src = classes[cls].get("source", "reference value, not measured on this printer")
            rec.update(class_source=cls_src, band_source=band_src)
            ev["pairs_outside_band"] += rec["margin_mm"] < 0
            ev["pairs_with_seeded_class"] = ev.get("pairs_with_seeded_class", 0) + cls_src.startswith("seeded")
            cases.append(dict(part=a, mating_part=n, pose="standing", location_mm=None, value=rec["margin_mm"], unit="mm margin to the fit-class band",
                              what=f"{cls} band {rec['band_mm']} mm ({band_src}), clearance {rec['clearance_mm']} mm: {rec['side']}; class: {cls_src}"))
        elif running:
            rec.update(fit_class=None, running_clearance=True); ev["running_clearance_pairs"] = ev.get("running_clearance_pairs", 0) + 1
        else:
            rec["fit_class"] = None; ev["pairs_without_class"] += 1
        per.setdefault(a, {})[n] = rec
    cases.sort(key=lambda c: c["value"])
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=[],
                notes=f"clearance = minimum distance at standing (gaps logic, search 6 mm); on overlap = minus the deepest sample of either surface inside the other (sampled, reads low). "
                      f"Pairs within {rad} mm, or with a class, are listed; band from data/fit_classes.json (reference values, not measured on this printer); margin = distance to the nearer band edge, negative outside; "
                      f"pairs without a class are counted (pairs_without_class), not judged; pairs on two links are running clearances (running_clearance_pairs; "
                      f"a seeded class on them is ignored, a confirmed one judged); classes set per pair in part_settings <part>.fit_classes")


def _depth(ctx, a, n):
    """Interpenetration depth of a and n at standing, both in their manifest frames moved to the world frame."""
    body = ctx.body
    q = body.standing
    out = []
    for k in (a, n):
        m = ctx.solid(k)[0].copy()
        m.apply_transform(body.link_T(ctx.link(k), q))
        out.append(m)
    return penetration(out[0], out[1])
