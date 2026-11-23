"""static_audit: standing overlap (mm3) of every scoped part against every other non-hardware part whose bounds meet it (exact mesh booleans).
Moved from motor-refit/static_audit.py; limited to the scope (all pairs of the whole body = no scope selector)."""
from __future__ import annotations

import time

import numpy as np

from checks.core import R, kit, tick
from checks.registry import register


@register("static_audit", lambda ctx: bool(ctx.scope))
def run(ctx):
    """Standing: overlap volume (mm3) of each scoped part against every other non-fastener part (bounds within 0.2 mm), exact mesh booleans.
    Left out: fasteners and the role class audit_skip (glue, harness, cut material: data/roles.json), as ctx.rel says."""
    t0 = time.time()
    body = ctx.body
    others = [n for n in body.parts if not ctx.rel.is_fastener(n) and not ctx.rel.in_class(n, "audit_skip")]
    bb = {n: np.array(body.parts[n]["bounds_mm"], float) for n in others}
    rows, per_part, unmeasurable = {}, {}, 0
    for a in ctx.scope:
        if a not in bb:
            continue
        for b in others:
            if b == a or (b in ctx.scope and b < a):
                continue
            A, B = bb[a], bb[b]
            if np.any(A[0] > B[1] + 0.2) or np.any(B[0] > A[1] + 0.2):
                continue
            for n in (a, b):
                ctx.solid(n)
            v = (ctx.mf[a] ^ ctx.mf[b]).volume()
            if v > 0.05:
                r = dict(overlap_mm3=round(float(v), 2), link_a=ctx.link(a), link_b=ctx.link(b))
                rows[(a, b)] = r
                per_part.setdefault(a, {})[b] = r
                if b in ctx.scope:
                    per_part.setdefault(b, {})[a] = r
    tick("static audit", t0)
    cases = [dict(part=a, mating_part=b, pose="standing", location_mm=None, value=r["overlap_mm3"], unit="mm3 overlap", what="standing overlap")
             for (a, b), r in sorted(rows.items(), key=lambda kv: -kv[1]["overlap_mm3"])]
    for c in cases[:5]:
        c["location_mm"] = ctx.locate(c["part"], c["mating_part"], "standing") if "standing" in [p[0] for p in ctx.poses] else None
    return dict(per_part=per_part, worst_cases=cases[:20], bad_events=dict(overlapping_pairs_standing=len(rows)), evidence=[],
                notes="closed screening solids (native STEP for scoped parts), manifold3d boolean volume, pairs reported above 0.05 mm3; same-link and cross-link alike")
