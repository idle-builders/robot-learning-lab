"""gaps: standing, every part (any link) within 6 mm of a scoped part: overlap (mm3) and distance (mm). Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, kit, screen_solid, step_mesh, tick
from checks.registry import register


def check_gaps(body, changed, sol, mf):
    """Fasteners are left out, their sleeve sets kept (process.relations: definition, role class sleeve_set)."""
    from checks import process as P
    rel = P.relations(body)
    t0 = time.time()
    q = body.standing
    out, memo = {}, {}
    for a in changed:
        Ta = body.link_T(body.parts[a]["owning_link"], q)[:3, :4]
        ga = mf[a].transform(Ta); ba = np.array(ga.bounding_box()).reshape(2, 3)
        for n, p in body.parts.items():
            if n == a or rel.is_fastener(n) and not rel.is_sleeve_set(n):
                continue
            b = np.array(p["bounds_mm"], float)
            Tn = body.link_T(p["owning_link"], q)
            c = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ Tn[:3, :3].T + Tn[:3, 3]
            if np.linalg.norm(np.maximum(0.0, np.maximum(ba[0] - c.max(0), c.min(0) - ba[1]))) > 6.0:
                continue
            if n not in mf:
                t1 = time.time(); sol[n] = screen_solid(body, n, changed); mf[n] = kit.mf(sol[n][0]); tick("solids (others)", t1)
            key = (n, a) if n < a else (a, n)
            if key in memo:                     # both parts scoped (unscoped run): the pair was measured from the other side
                v, d = memo[key]
            else:
                gn = mf[n].transform(Tn[:3, :4])
                v = (ga ^ gn).volume()
                d = 0.0 if v > 1e-3 else ga.min_gap(gn, 6.0)
                memo[key] = (v, d)
            if d < 6.0:
                out[f"{a} x {n}"] = dict(same_link=p["owning_link"] == body.parts[a]["owning_link"], overlap_mm3=round(v, 3),
                                         distance_mm=round(min(d, 6.0), 3), solid=sol[n][1])
    tick("gaps (standing)", t0)
    return dict(sorted(out.items(), key=lambda t: (t[1]["distance_mm"], -t[1]["overlap_mm3"])))


@register("gaps", lambda ctx: bool(ctx.scope))
def run(ctx):
    """Standing: every part within 6 mm of a scoped part: overlap (mm3) and distance (mm)."""
    for n in ctx.scope:
        ctx.solid(n)
    raw = check_gaps(ctx.body, ctx.scope, ctx.sol, ctx.mf)
    per_part, cases = {}, []
    for k, g in raw.items():
        a, n = k.split(" x ")
        per_part.setdefault(a, {})[n] = g
        cases.append(dict(part=a, mating_part=n, pose="standing", value=g["distance_mm"], unit="mm distance",
                          overlap_mm3=g["overlap_mm3"], what="gap at standing", location_mm=None))
    for c in cases[:5]:
        c["location_mm"] = ctx.locate(c["part"], c["mating_part"], "standing") if "standing" in [p[0] for p in ctx.poses] else None
    return dict(per_part=per_part, worst_cases=cases[:20],
                bad_events=dict(parts_overlapping_at_standing=sum(1 for g in raw.values() if g["overlap_mm3"] > 0.1)),
                evidence=[], notes="standing pose, search radius 6 mm; overlap = mesh boolean volume")
