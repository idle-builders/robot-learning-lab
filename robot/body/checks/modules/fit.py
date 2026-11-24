"""fit: exact B-rep fit of each scoped part against its neighbours (native STEP solids) over the pose set (use --poses fit for the
roll_flip_checks pose list): overlap (mm3) and minimum distance (mm), worst pose of each. Same-link pairs are measured once, standing.
Poses that set a joint beyond its declared range (body definitions `joints:`, core.declare_ranges: the fit list's stress angles, knee
105-125, hip yaw 35-40, pitch to -90, roll to -45, and its combined extremes) are measured as the others and reported apart per pair
(beyond_declared_range), not in per_part / worst_cases.
Moved from roll_flip_checks.run_fit / fit_check_upper_hip (without their screw and driver rays, which belong to the screw checks)."""
from __future__ import annotations

import sys
import time

import numpy as np

from checks.core import R, beyond_declared, range_report, tick
from checks.registry import register

NEAR_MM = 4.0   # distance is exact below this, the box gap above (as in roll_flip_checks)
PAIR_BUDGET_S = 120.0   # exact OCC seconds per pair before it stops (run.py --fit-pair-budget-s); each call runs forked and is stopped at the budget (brep.bounded)


def _shape(ctx, n, cache):
    if n not in cache:
        p = ctx.parts[n]
        if not (p.get("native_step") or p.get("native_step_unmodified")):
            cache[n] = None
        else:
            turned = str(p.get("roll_flip", "")).startswith("turned")     # the manifest's roll-flip turn of the case
            try:   # exact_fit.min_solid_mm3 (part settings / body definition): only STEP solids of at least this volume
                sh = R.brep(n, ctx.parts, turned=turned, min_solid_mm3=float((ctx.settings(n).get("exact_fit") or {}).get("min_solid_mm3") or 0.0))
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
    budget = float(getattr(ctx, "fit_pair_budget_s", None) or PAIR_BUDGET_S)
    worst, cache, standing, spent, apart, beyond_poses = {}, {}, {}, {}, {}, []
    n_pose = len(ctx.poses)
    print(f"fit: {len(pairs)} pairs x {n_pose} poses, {budget:g} s exact work per pair at most", file=sys.stderr, flush=True)
    for ip, (pname, fam, q) in enumerate(ctx.poses):
        T = {lk: ctx.body.link_T(lk, q) for lk in set(link.values())}
        beyond = beyond_declared(ctx.body, q)
        if beyond:
            beyond_poses.append(dict(pose=pname, family=fam, joints=beyond))
        exact = 0
        for a, b in pairs:
            if link[a] == link[b] and standing.get((a, b)) is not None:
                continue                      # same-link parts never move against each other
            w = worst.setdefault((a, b), dict(same_link=link[a] == link[b], max_overlap_mm3=0.0, pose_of_max=None, min_distance_mm=1e9,
                                              pose_of_min=None, poses_overlapping=0, poses=0, exact_seconds=0.0, timed_out=False))
            if w["timed_out"]:
                continue
            stress = bool(beyond) and link[a] != link[b]        # a stress pose: same measurement, kept apart from the design poses
            tgt = apart.setdefault((a, b), dict(max_overlap_mm3=0.0, pose_of_max=None, joints_at_max=None, min_distance_mm=1e9,
                                                pose_of_min=None, poses_overlapping=0, poses=0)) if stress else w
            Trel = np.linalg.inv(T[link[a]]) @ T[link[b]]
            key = (a, b, tuple(np.round(Trel[:3], 5).ravel()))
            if key in cache:
                v, d = cache[key]
            else:
                # box test before moving the B-rep: the corners of b's box under Trel bound the moved box (a transform copies the shape)
                bb = L.aabb(shapes[b])
                c = np.array([[bb[i][0], bb[j][1], bb[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ Trel[:3, :3].T + Trel[:3, 3]
                glo = L.aabb_gap(L.aabb(shapes[a]), np.array([c.min(0), c.max(0)]))
                if glo >= NEAR_MM and glo >= tgt["min_distance_mm"]:
                    v, d = 0.0, float("inf")      # far, cannot lower this pair's minimum (not cached: depends on w)
                else:
                    t1 = time.time()
                    moved = L.transform_shape(shapes[b], Trel)
                    gap = L.aabb_gap(L.aabb(shapes[a]), L.aabb(moved))
                    try:        # each exact call in a forked process stopped at what is left of the pair's budget
                        v = L.bounded(lambda: L.common_volume_mm3(shapes[a], moved), budget - w["exact_seconds"]) if gap <= 0.05 else 0.0
                        d = (L.bounded(lambda: L.min_distance_mm(shapes[a], moved), budget - w["exact_seconds"] - (time.time() - t1))
                             if gap < NEAR_MM else round(gap, 3))
                    except L.CallTimeout as e:
                        w["exact_seconds"] = round(w["exact_seconds"] + time.time() - t1, 2)
                        w["timed_out"], w["stopped_in_call"] = True, f"pose {pname}: {e}"
                        print(f"fit: pair {a} x {b}: one exact call at pose {pname} {e}, over its {budget:g} s budget after {w['poses']} poses",
                              file=sys.stderr, flush=True)
                        continue
                    cache[key] = (v, d)
                    exact += 1
                    w["exact_seconds"] = round(w["exact_seconds"] + time.time() - t1, 2)
                    if w["exact_seconds"] > budget:
                        w["timed_out"] = True     # stop this pair; its values cover the poses reached (w["poses"] + 1)
                        print(f"fit: pair {a} x {b} over its {budget:g} s budget after {w['poses'] + 1} poses", file=sys.stderr, flush=True)
            if link[a] == link[b]:
                standing[(a, b)] = True
            w["poses"] += 1
            if stress:
                w = tgt
                w["poses"] += 1
                if v > w["max_overlap_mm3"]:
                    w["joints_at_max"] = beyond
            if v > 1e-3:
                w["poses_overlapping"] += 1
            if v > w["max_overlap_mm3"]:
                w["max_overlap_mm3"], w["pose_of_max"] = round(v, 3), pname
            if d < w["min_distance_mm"]:
                w["min_distance_mm"], w["pose_of_min"] = round(d, 3), pname
        print(f"fit: pose {ip + 1}/{n_pose} {pname}: {exact} exact pair evaluations, {time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    tick("fit (exact)", t0)
    ctx.fit_unreadable_steps = shapes.get("_unreadable", {})
    ctx.fit_beyond = dict(poses=beyond_poses, contacts={f"{a} x {b}": dict(w, min_distance_mm=None if w["min_distance_mm"] >= 1e8 else w["min_distance_mm"])
                                                        for (a, b), w in sorted(apart.items(), key=lambda t: (-t[1]["max_overlap_mm3"], t[1]["min_distance_mm"]))})
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
    fb = getattr(ctx, "fit_beyond", dict(poses=[], contacts={}))
    ranges, rcount, rfind = range_report(ctx.body, {j for _, _, q in ctx.poses for j in q})
    return dict(per_part=per_part, worst_cases=cases[:20], beyond_declared_range=fb, joint_ranges=ranges, range_findings=rfind,
                bad_events=dict(pairs_overlapping=sum(1 for w in worst.values() if w["max_overlap_mm3"] > 1e-3),
                                beyond_declared_range_poses=len(fb["poses"]),
                                beyond_declared_range_pairs_overlapping=sum(1 for w in fb["contacts"].values() if w["max_overlap_mm3"] > 1e-3),
                                **rcount,
                                pose_pair_overlaps=sum(w["poses_overlapping"] for w in worst.values()),
                                neighbours_without_native_step=len(no_step),
                                pairs_timed_out=sum(1 for w in worst.values() if w["timed_out"]),
                                steps_unreadable=len(getattr(ctx, "fit_unreadable_steps", {}))),
                steps_unreadable=getattr(ctx, "fit_unreadable_steps", {}),
                pairs_timed_out={f"{a} x {b}": dict(poses_reached=w["poses"], of_poses=len(ctx.poses), exact_seconds=w["exact_seconds"], stopped_in_call=w.get("stopped_in_call"))
                                 for (a, b), w in worst.items() if w["timed_out"]},
                evidence=[], notes=f"{len(ctx.poses)} poses ({ctx.poses_name}), {len(fb['poses'])} of them beyond a declared range: measured, in beyond_declared_range, not in per_part / worst_cases; a pair over {getattr(ctx, 'fit_pair_budget_s', None) or PAIR_BUDGET_S:g} s of exact work stops (pairs_timed_out: poses reached); exact OpenCascade common volume and BRepExtrema distance on native STEP; "
                                   f"distance exact below {NEAR_MM} mm, box gap above; parts without native STEP not measured: {no_step[:8]}")
