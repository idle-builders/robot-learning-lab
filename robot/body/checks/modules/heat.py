"""heat: per scoped printed part, standing, against every heat source (servo case / motor, battery cell, power board; data/heat_sources.json,
every temperature labelled with its source): contact area (mm2, surface within screen.contact_mm of the source), closest distance (mm),
area within screen.near_mm, the source's worst-case temperature, the part's material HDT (0.45 / 1.8 MPa) and glass transition
(materials.json print.*, resolved by process.effective so `--process material=...` cross-evaluates) and the margin in C (limit minus
source temperature). Migrated from the torso thermal_evaluate.py (its HDT margin = HDT - carrier wall temperature); the coupled
thermal simulation itself is not part of this screen. Measurements and counts only."""
from __future__ import annotations

import json
import time

import numpy as np

from checks import bodylib as BL
from checks import process as P
from checks.core import DATA, tick
from checks.registry import register
from checks.modules.overhang import _surface_tree

SPACING_MM = 0.5                 # sample spacing on the printed part


def table():
    return json.loads((DATA / "heat_sources.json").read_text())


def sources(ctx, tab=None):
    """{part: (kind, kind entry)} of the heat sources in the manifest: the parts whose body definition declares
    `heat_source: {kind: ...}` (robot/**/<builder>.bom.yaml, via ctx.settings); the kinds (temperatures and their sources)
    are the shared table data/heat_sources.json."""
    tab = tab or table()
    out = {}
    for n in ctx.parts:
        k = (ctx._settings.get(n, {}).get("heat_source") or {}).get("kind")
        if k:
            out[n] = (k, tab["kinds"][k])
    return out


def _targets(ctx):
    return [n for n in ctx.scope if P.is_printed(ctx, n)]


def limits(ctx, n):
    """Material heat values of printed part n (None when it has no known material)."""
    e = P.effective(ctx, n)
    mats, _ = P.tables()
    pr = (mats.get(e["material"]) or {}).get("print", {})
    h = pr.get("hdt_C_045_18MPa") or [None, None]
    return dict(material=e["material"], material_source=e["source"]["material"], hdt_045_C=h[0], hdt_18_C=h[1], glass_transition_C=pr.get("glass_transition_C"))


def _margins(lim, T):
    f = lambda v: None if v is None else round(v - T, 1)
    return dict(margin_hdt_045_C=f(lim["hdt_045_C"]), margin_hdt_18_C=f(lim["hdt_18_C"]), margin_tg_C=f(lim["glass_transition_C"]))


def measure(ctx, n, pts, areas, tab=None, srcs=None):
    """Per heat source within search_mm of the part (points `pts` in the part's manifest frame carrying `areas` mm2 each) at standing."""
    tab = tab or table(); srcs = srcs if srcs is not None else sources(ctx, tab)
    sc = tab["screen"]; contact, near, srch = sc["contact_mm"], sc["near_mm"], sc["search_mm"]
    q = ctx.body.standing
    Ta = ctx.body.link_T(ctx.link(n), q)
    pw = pts @ Ta[:3, :3].T + Ta[:3, 3]
    lo, hi = pw.min(0) - srch, pw.max(0) + srch
    out = {}
    for k, (kind, ent) in srcs.items():
        if k == n:
            continue
        p = ctx.parts[k]
        Tk = ctx.body.link_T(p["owning_link"], q)
        b = np.array(p["bounds_mm"], float)
        c = np.array([[b[i][0], b[j][1], b[kk][2]] for i in (0, 1) for j in (0, 1) for kk in (0, 1)]) @ Tk[:3, :3].T + Tk[:3, 3]
        if np.any(c.min(0) > hi) or np.any(c.max(0) < lo):
            continue
        loc = (pw - Tk[:3, 3]) @ Tk[:3, :3]
        d = _surface_tree(ctx, k).query(loc, distance_upper_bound=srch)[0]
        if not np.isfinite(d).any():
            continue
        i = int(np.argmin(d))
        out[k] = dict(kind=kind, temperature_C=ent["temperature_C"], temperature_source=ent["temperature_source"],
                      contact_area_mm2=round(float(areas[d < contact].sum()), 1), closest_mm=round(float(d[i]), 2),
                      area_within_near_mm2=round(float(areas[d < near].sum()), 1),
                      closest_point_mm=np.round(pw[i], 1).tolist())
    return out


def check_part(ctx, n, tab=None, srcs=None):
    import trimesh
    t0 = time.time()
    tab = tab or table()
    m = ctx.solid(n)[0]
    cnt = int(max(2000, m.area / SPACING_MM ** 2))
    pts, _ = trimesh.sample.sample_surface(m, cnt, seed=1)
    areas = np.full(len(pts), m.area / len(pts))
    lim = limits(ctx, n)
    rows = measure(ctx, n, pts, areas, tab, srcs)
    for r in rows.values():
        r.update(_margins(lim, r["temperature_C"]))
    rows = dict(sorted(rows.items(), key=lambda t: (t[1]["closest_mm"], -t[1]["contact_area_mm2"])))
    nr = tab["screen"]["near_mm"]
    exp = [r for r in rows.values() if r["closest_mm"] < nr or r["contact_area_mm2"] > 0]
    mg = [r["margin_hdt_18_C"] for r in exp if r["margin_hdt_18_C"] is not None]
    out = dict(**lim, sources=rows, sources_in_contact=sum(r["contact_area_mm2"] > 0 for r in rows.values()),
               sources_within_near=len(exp), contact_area_total_mm2=round(sum(r["contact_area_mm2"] for r in rows.values()), 1),
               smallest_margin_hdt_18_C_within_near=min(mg) if mg else None, seconds=round(time.time() - t0, 1))
    tick("heat", t0)
    return out


@register("heat", lambda ctx: bool(_targets(ctx)))
def run(ctx):
    """Per printed part at standing: contact area and closest distance to servo cases, battery cells and power boards, their temperature, the material's HDT / Tg and the margin in C."""
    tab = table()
    srcs = sources(ctx, tab)
    per, cases = {}, []
    ev = dict(parts=0, parts_in_contact_with_a_source=0, parts_within_near_of_a_source=0, contact_pairs=0, near_pairs=0)
    for n in _targets(ctx):
        r = check_part(ctx, n, tab, srcs)
        per[n] = r
        ev["parts"] += 1
        ev["parts_in_contact_with_a_source"] += r["sources_in_contact"] > 0
        ev["parts_within_near_of_a_source"] += r["sources_within_near"] > 0
        for k, s in r["sources"].items():
            ev["contact_pairs"] += s["contact_area_mm2"] > 0
            ev["near_pairs"] += s["closest_mm"] < tab["screen"]["near_mm"]
            if s["closest_mm"] < tab["screen"]["near_mm"] or s["contact_area_mm2"] > 0:
                cases.append(dict(part=n, mating_part=k, pose="standing", location_mm=s["closest_point_mm"], value=s["margin_hdt_18_C"],
                                  unit="C margin to HDT 1.8 MPa", what=f"{s['kind']} {s['temperature_C']:g} C, contact {s['contact_area_mm2']} mm2, closest {s['closest_mm']} mm; "
                                  f"{r['material']}: HDT {r['hdt_045_C']}/{r['hdt_18_C']} C, Tg {r['glass_transition_C']} C"))
    cases.sort(key=lambda c: (c["value"] is None, c["value"] if c["value"] is not None else 0))
    return dict(per_part=per, worst_cases=cases, bad_events=ev, evidence=["robot/body/checks/data/heat_sources.json"],
                notes=f"standing; part surface sampled every {SPACING_MM} mm, distance to the source's surface samples (0.25 mm) so values read up to ~0.3 mm high; contact = within "
                      f"{tab['screen']['contact_mm']} mm, near = within {tab['screen']['near_mm']} mm, measured up to {tab['screen']['search_mm']} mm; source temperatures are worst cases "
                      f"labelled in heat_sources.json (assumptions where marked); margin = HDT (1.8 / 0.45 MPa) or Tg minus the source temperature, from materials.json print.*; no thermal simulation")
