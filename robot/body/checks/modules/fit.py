"""fit: exact B-rep fit of each scoped part against its neighbours (native STEP solids) over the pose set (use --poses fit for the
roll_flip_checks pose list): overlap (mm3) and minimum distance (mm), worst pose of each. Same-link pairs are measured once, standing.
Moved from roll_flip_checks.run_fit / fit_check_upper_hip (without their screw and driver rays, which belong to the screw checks)."""
from __future__ import annotations

import time

import numpy as np

from checks.core import R, tick
from checks.registry import register

NEAR_MM = 4.0   # distance is exact below this, the box gap above (as in roll_flip_checks)


def _shape(ctx, n, cache):
    if n not in cache:
        p = ctx.parts[n]
        if not (p.get("native_step") or p.get("native_step_unmodified")):
            cache[n] = None
        else:
            turned = str(p.get("roll_flip", "")).startswith("turned")
            try:
                sh = R.brep(n, ctx.parts, turned=turned, keep_case_solids=bool((ctx.settings(n).get("exact_fit") or {}).get("min_solid_mm3")))
                if getattr(sh, "_wrapped", None) is None or not sh.solids():
                    raise ValueError("STEP gave an empty shape")
                cache[n] = sh
            except Exception as exc:  # noqa: BLE001  (a neighbour's STEP missing on disk and in the bucket must not end the check)
                cache[n] = None
                cache.setdefault("_unreadable", {})[n] = f"{type(exc).__name__}: {str(exc)[-120:]}"
    return cache[n]


def exact_pairs(ctx):
    from checks import brep as L
    t0 = time.time()
    shapes, T = {}, {}
    link = {n: ctx.link(n) for n in list(ctx.scope) + list(ctx.neighbours())}
    pairs, no_step = [], set()
    for a in ctx.scope:
        for b in list(ctx.neighbours()) + [s for s in ctx.scope if s != a]:
            if (b, a) in pairs and b in ctx.scope:
                continue
            pairs.append((a, b))
    for n in {x for pr in pairs for x in pr}:
        if _shape(ctx, n, shapes) is None:
            no_step.add(n)
    pairs = [(a, b) for a, b in pairs if a not in no_step and b not in no_step]
    worst, cache, standing = {}, {}, {}
    for pname, fam, q in ctx.poses:
        T = {lk: ctx.body.link_T(lk, q) for lk in set(link.values())}
        for a, b in pairs:
            if link[a] == link[b] and standing.get((a, b)) is not None:
                continue                      # same-link parts never move against each other
            Trel = np.linalg.inv(T[link[a]]) @ T[link[b]]
            key = (a, b, tuple(np.round(Trel[:3], 5).ravel()))
            w = worst.setdefault((a, b), dict(same_link=link[a] == link[b], max_overlap_mm3=0.0, pose_of_max=None, min_distance_mm=1e9,
                                              pose_of_min=None, poses_overlapping=0, poses=0))
            if key in cache:
                v, d = cache[key]
            else:
                moved = L.transform_shape(shapes[b], Trel)
                glo = L.aabb_gap(L.aabb(shapes[a]), L.aabb_bound(moved))
                if glo >= NEAR_MM and glo >= w["min_distance_mm"]:
                    v, d = 0.0, float("inf")      # far, cannot lower this pair's minimum (not cached: depends on w)
                else:
                    gap = L.aabb_gap(L.aabb(shapes[a]), L.aabb(moved))
                    v = L.common_volume_mm3(shapes[a], moved) if gap <= 0.05 else 0.0
                    d = L.min_distance_mm(shapes[a], moved) if gap < NEAR_MM else round(gap, 3)
                    cache[key] = (v, d)
            if link[a] == link[b]:
                standing[(a, b)] = True
            w["poses"] += 1
            if v > 1e-3:
                w["poses_overlapping"] += 1
            if v > w["max_overlap_mm3"]:
                w["max_overlap_mm3"], w["pose_of_max"] = round(v, 3), pname
            if d < w["min_distance_mm"]:
                w["min_distance_mm"], w["pose_of_min"] = round(d, 3), pname
    tick("fit (exact)", t0)
    ctx.fit_unreadable_steps = shapes.get("_unreadable", {})
    return worst, sorted(no_step)


@register("fit", lambda ctx: any(ctx.parts[n].get("native_step") for n in ctx.scope))
def run(ctx):
    """Exact B-rep overlap (mm3) and minimum distance (mm) of each scoped part against its neighbours over the pose set; worst pose of each."""
    worst, no_step = exact_pairs(ctx)
    per_part, cases = {}, []
    for (a, b), w in worst.items():
        w = dict(w, min_distance_mm=None if w["min_distance_mm"] >= 1e8 else w["min_distance_mm"])
        for part, mate in ((a, b), (b, a)):
            if part in ctx.scope:
                per_part.setdefault(part, {})[mate] = w
        pose = w["pose_of_max"] if w["max_overlap_mm3"] > 0 else w["pose_of_min"]
        cases.append(dict(part=a, mating_part=b, pose=pose, location_mm=None,
                          value=w["max_overlap_mm3"] if w["max_overlap_mm3"] > 0 else (w["min_distance_mm"] if w["min_distance_mm"] is not None else NEAR_MM),
                          unit="mm3 overlap" if w["max_overlap_mm3"] > 0 else "mm min distance",
                          what="exact overlap" if w["max_overlap_mm3"] > 0 else "closest approach (exact)"))
    cases.sort(key=lambda c: (c["unit"] != "mm3 overlap", -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    for c in cases[:5]:
        c["location_mm"] = ctx.locate(c["part"], c["mating_part"], c["pose"])
    return dict(per_part=per_part, worst_cases=cases[:20],
                bad_events=dict(pairs_overlapping=sum(1 for w in worst.values() if w["max_overlap_mm3"] > 1e-3),
                                pose_pair_overlaps=sum(w["poses_overlapping"] for w in worst.values()),
                                neighbours_without_native_step=len(no_step),
                                steps_unreadable=len(getattr(ctx, "fit_unreadable_steps", {}))),
                steps_unreadable=getattr(ctx, "fit_unreadable_steps", {}),
                evidence=[], notes=f"{len(ctx.poses)} poses ({ctx.poses_name}); exact OpenCascade common volume and BRepExtrema distance on native STEP; "
                                   f"distance exact below {NEAR_MM} mm, box gap above; parts without native STEP not measured: {no_step[:8]}")
