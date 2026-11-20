"""clearance: scoped parts against their neighbours on other links over the pose set: new overlap vs standing (mm3) and minimum surface distance (mm, capped at search_mm). Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register


def check_clearance(body, changed, others, sol, mf, poses, SEARCH_MM=5.0):
    """others: {name: part} candidate neighbours; only those on a link other than the scoped part's are paired."""
    link = {n: body.parts[n]["owning_link"] for n in set(others) | set(changed)}
    pairs = sorted({tuple(sorted((a, b))) for a in changed for b in set(others) | set(changed)
                    if a != b and link[a] != link[b]})
    for n in {x for pr in pairs for x in pr}:
        if n not in mf:
            t0 = time.time(); sol[n] = screen_solid(body, n, changed); mf[n] = kit.mf(sol[n][0]); tick("solids (others)", t0)
    t0 = time.time()

    def at(q):
        Tq = {lk: body.link_T(lk, q)[:3, :4] for lk in {link[n] for pr in pairs for n in pr}}
        posed = {n: mf[n].transform(Tq[link[n]]) for n in {x for pr in pairs for x in pr}}
        bb = {n: np.array(g.bounding_box()).reshape(2, 3) for n, g in posed.items()}
        out = {}
        for a, b in pairs:
            gap = np.maximum(0.0, np.maximum(bb[a][0] - bb[b][1], bb[b][0] - bb[a][1]))
            if np.linalg.norm(gap) > SEARCH_MM:
                continue
            v = (posed[a] ^ posed[b]).volume()
            d = 0.0 if v > 1e-3 else posed[a].min_gap(posed[b], SEARCH_MM)
            out[(a, b)] = (v, min(d, SEARCH_MM))
        return out
    base = at(poses[0][2])
    worst, by_fam = {}, {}
    for name, fam, q in poses:
        res = at(q)
        for (a, b), (v, d) in res.items():
            k = f"{a} x {b}"
            dv = v - base.get((a, b), (0.0, 0.0))[0]
            w = worst.setdefault(k, dict(standing_overlap_mm3=round(base.get((a, b), (0.0, SEARCH_MM))[0], 1),
                                         max_new_overlap_mm3=0.0, pose_of_max=None, min_distance_mm=SEARCH_MM,
                                         pose_of_min=None, poses_new_overlap_over_20mm3=0))
            if dv > w["max_new_overlap_mm3"]:
                w["max_new_overlap_mm3"], w["pose_of_max"] = round(dv, 1), name
            if d < w["min_distance_mm"]:
                w["min_distance_mm"], w["pose_of_min"] = round(d, 3), name
            if dv > 20.0:
                w["poses_new_overlap_over_20mm3"] += 1
                f = by_fam.setdefault(k, {}).setdefault(fam, dict(max_new_overlap_mm3=0.0, poses_over_20mm3=0))
                f["poses_over_20mm3"] += 1
                f["max_new_overlap_mm3"] = max(f["max_new_overlap_mm3"], round(dv, 1))
    tick("clearance sweep", t0)
    worst = dict(sorted(worst.items(), key=lambda t: (-t[1]["max_new_overlap_mm3"], t[1]["min_distance_mm"])))
    return dict(poses=len(poses), pairs=len(pairs), search_mm=SEARCH_MM, by_pair=worst, new_overlap_by_family=by_fam,
                note="overlap = mesh boolean volume of closed screening solids; distance = mesh min_gap, capped at search_mm")


def _applies(ctx):
    return bool(ctx.scope)


@register("clearance", _applies)
def run(ctx):
    """New overlap vs standing (mm3) and minimum distance (mm) of each scoped part against its neighbours on other links over the pose set."""
    changed = set(ctx.scope)
    for n in ctx.scope:
        ctx.solid(n)
    raw = check_clearance(ctx.body, changed, ctx.neighbours(), ctx.sol, ctx.mf, ctx.poses, ctx.search_mm)
    per_part, cases = {}, []
    for k, w in raw["by_pair"].items():
        a, b = k.split(" x ")
        for part, mate in ((a, b), (b, a)):
            if part in changed:
                per_part.setdefault(part, {})[mate] = w
    for part, mates in per_part.items():
        for mate, w in mates.items():
            if part in changed and mate in changed and part > mate:
                continue
            cases.append(dict(part=part, mating_part=mate, pose=w["pose_of_max"] if w["max_new_overlap_mm3"] > 0 else w["pose_of_min"],
                              value=w["max_new_overlap_mm3"] if w["max_new_overlap_mm3"] > 0 else w["min_distance_mm"],
                              unit="mm3 new overlap" if w["max_new_overlap_mm3"] > 0 else "mm min distance",
                              what="new overlap vs standing" if w["max_new_overlap_mm3"] > 0 else "closest approach"))
    cases.sort(key=lambda c: (c["unit"] != "mm3 new overlap", -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    for c in cases[:5]:
        c["location_mm"] = ctx.locate(c["part"], c["mating_part"], c["pose"])
    raw_pairs = raw["by_pair"]
    return dict(per_part=per_part, worst_cases=cases[:20],
                bad_events=dict(pairs_with_new_overlap=sum(1 for w in raw_pairs.values() if w["max_new_overlap_mm3"] > 0.1),
                                pose_pair_new_overlaps_over_20mm3=sum(w["poses_new_overlap_over_20mm3"] for w in raw_pairs.values())),
                new_overlap_by_family=raw["new_overlap_by_family"], evidence=[],
                notes=f"{raw['poses']} poses ({ctx.poses_name}), {raw['pairs']} pairs, distance capped at {raw['search_mm']} mm; " + raw["note"])
