"""shell_offset: inner-wall offset (mm) of a cover shell from the parts it covers: distance from inner-surface samples of the shell to the covered
parts (KD-tree over dense surface samples; error <= offset_resolution_mm). Covered parts, axis and lowest z come from the shell's part setting
`covers`. Moved from the wall-offset part of leg/thigh_shell.check; the overlap of shells with neighbours over poses is the `clearance` check."""
from __future__ import annotations

import time

import numpy as np

from checks.core import tick
from checks.registry import register


def _has(ctx, n):
    return bool(ctx.settings(n).get("covers"))


@register("shell_offset", lambda ctx: any(_has(ctx, n) for n in ctx.scope))
def run(ctx):
    """Inner-wall distance (min / p10 / median mm, share within 3 mm) of a cover shell from the parts it covers."""
    import trimesh
    from scipy.spatial import cKDTree
    t0 = time.time()
    per_part, cases = {}, []
    for n in [n for n in ctx.scope if _has(ctx, n)]:
        cv = ctx.settings(n)["covers"]
        cov = trimesh.util.concatenate([ctx.body.mesh(c) for c in cv["parts"] if c in ctx.parts])
        S = np.vstack([cov.vertices, trimesh.sample.sample_surface(cov, 600000)[0]])
        tree, spacing = cKDTree(S), float(np.sqrt(cov.area / 600000))
        m = ctx.body.mesh(n)
        P, fi = trimesh.sample.sample_surface(m, 20000)
        cx, cy = cv["axis_xy_mm"]
        inward = -m.face_normals[fi]
        facing = np.einsum("ij,ij->i", inward, (np.c_[np.full(len(P), cx), np.full(len(P), cy), P[:, 2]] - P)) > 0
        P = P[facing & (P[:, 2] > cv["z_min_mm"])]
        d, j = tree.query(P)
        per_part[n] = dict(covered_parts=[c for c in cv["parts"] if c in ctx.parts], offset_resolution_mm=round(spacing, 2), samples=int(len(P)),
                           min_mm=round(float(d.min()), 2), p10_mm=round(float(np.percentile(d, 10)), 2), median_mm=round(float(np.median(d)), 2),
                           share_within_3mm_pct=round(100 * float((d <= 3.0).mean()), 1))
        cases.append(dict(part=n, mating_part=None, pose="standing", location_mm=np.round(P[int(np.argmin(d))], 1).tolist(), value=per_part[n]["min_mm"],
                          unit="mm min inner-wall offset", what="closest inner wall to a covered part"))
    tick("shell offset", t0)
    cases.sort(key=lambda c: c["value"])
    return dict(per_part=per_part, worst_cases=cases, bad_events=dict(shells_measured=len(per_part)), evidence=[],
                notes="inner wall = samples whose inward normal faces the setting's axis, above its z_min; distance to dense surface samples of the covered parts")
