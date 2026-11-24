"""collision_sweep: part-to-part collision over a joint's range: the joint over its full range, each neighbouring joint over its range at the
joint's ends and middle, and pairwise range corners (motor-refit pose_set). Pairs of non-fastener parts near the joint that involve a scoped part:
max overlap (mm3, exact mesh boolean), minimum clearance (mm), poses overlapping, worst pose. Same-link pairs once at neutral. Pairs of parts on
two different limbs (the other leg, an arm) are left out, as in range_sweep: their contacts are WBC self-collision limits; counted per joint.
Poses that are controller-limit corners (data/pose_sets.json controller_limit_corners, owner decision D-15: e.g. all three hip joints at
their range ends at once) are measured too, but their contacts are reported apart (controller_limit_contacts), not as design overlaps.
Ranges: the declared ones (body definitions `joints:`, core.declare_ranges; joint_ranges and range_findings in the result): every pose
is inside them. A joint without a range is not swept (joints_without_range) and a neighbour without one is not posed.
Near-contacts (sampled clearance under REFINE_MM) are refined around the sampled minimum: finer angles and the exact B-rep distance
(contacts[].refined: value, pose, method; the worst cases carry the refined value). Moved from motor-refit/collision_sweep.py."""
from __future__ import annotations

import sys
import time

import numpy as np

from checks import bodylib as BL
from checks.core import R, range_report, tick
from checks.modules.range_sweep import Prep, controller_limit, in_class, scope_joints
from checks.registry import register

F = BL
REFINE_MM = 1.0       # pairs whose sampled clearance is below this are refined (finer angles, exact closest-point distance)
REFINE_GRID = 9       # first pass: this many angles over +-one sweep step around the sampled minimum, per joint of the pose
REFINE_TOL_DEG = 0.05  # then golden-section search down to this
REFINE_CALL_S = 60.0  # one exact distance call stops after this (brep.bounded); the pair falls back to the closed-mesh distance
REFINE_PAIR_S = 180.0  # exact seconds per refined pair at most


class _Dist:
    """Distance of one pair at a pose {joint: deg}: the exact OpenCascade B-rep distance on the native STEP solids (fit's shapes)
    when both have one, else manifold3d min_gap on the closed screening meshes (not surface samples). Memoised per pose."""

    def __init__(self, ctx, a, b, shapes):
        from checks.modules.fit import _shape
        self.ctx, self.a, self.b, self.memo, self.spent, self.calls = ctx, a, b, {}, 0.0, 0
        self.sa, self.sb = _shape(ctx, a, shapes), _shape(ctx, b, shapes)
        self.method = "exact B-rep distance (OpenCascade BRepExtrema)" if self.sa is not None and self.sb is not None else "closed-mesh distance (manifold3d min_gap)"
        self.la, self.lb = ctx.body.parts[a]["owning_link"], ctx.body.parts[b]["owning_link"]

    def __call__(self, pose):
        from checks import brep as L
        k = tuple(sorted((j, round(v, 4)) for j, v in pose.items()))
        if k in self.memo:
            return self.memo[k]
        body = self.ctx.body
        Trel = np.linalg.inv(body.link_T(self.la, pose)) @ body.link_T(self.lb, pose)      # b in a's neutral frame
        d = None
        if self.method.startswith("exact"):
            t1 = time.time()
            try:
                moved = L.transform_shape(self.sb, Trel)
                d = L.bounded(lambda: L.min_distance_mm(self.sa, moved), min(REFINE_CALL_S, REFINE_PAIR_S - self.spent))
            except (L.CallTimeout, RuntimeError) as e:
                self.method = f"closed-mesh distance (manifold3d min_gap; exact call stopped: {str(e)[:60]})"
            self.spent += time.time() - t1
        if d is None or d < 0:
            ma, mb = Prep.get(self.ctx, self.a).mf, Prep.get(self.ctx, self.b).mf
            if ma is None or mb is None:        # no closed mesh either: this pair keeps its sampled value
                self.method = "not refined (no native STEP and no closed mesh)"
                d = float("inf")
            else:
                mb = mb.transform(Trel[:3, :4])
                d = 0.0 if (ma ^ mb).volume() > 1e-3 else float(ma.min_gap(mb, 2 * REFINE_MM + 1.0))
        self.calls += 1
        self.memo[k] = float(d)
        return float(d)


def refine_pair(ctx, row, jn, step, shapes):
    """Refined minimum distance of a near-contact pair: around the sweep pose of its sampled minimum, each joint the pose sets
    (the swept joint jn on +-step, a neighbour on +-2 step, both clipped to the declared range) is searched on a grid of REFINE_GRID
    angles, then by golden section to REFINE_TOL_DEG, with the exact distance (_Dist). Poses at a controller-limit corner are skipped.
    Same-link pairs: the exact distance at standing only."""
    t0 = time.time()
    dist = _Dist(ctx, row["a"], row["b"], shapes)
    pose = dict(row.get("pose_of_min") or {})
    if row["same_link"]:
        d = dist({})
        return None if not np.isfinite(d) else dict(min_distance_mm=round(d, 3), pose={}, method=dist.method, evaluations=dist.calls, seconds=round(time.time() - t0, 1))
    pose.setdefault(jn, 0.0)
    body = ctx.body
    best = dist(pose)
    for j in list(pose):
        r = body.joints[j].get("range_deg")
        if not r:
            continue
        g = step if j == jn else 2 * step
        lo, hi = max(min(r), pose[j] - g), min(max(r), pose[j] + g)
        f = lambda a: dist(dict(pose, **{j: float(a)})) if not controller_limit(body, dict(pose, **{j: float(a)})) else 1e9  # noqa: E731
        xs = np.linspace(lo, hi, REFINE_GRID)
        ys = [f(x) for x in xs]
        i = int(np.argmin(ys))
        a, b = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
        gr = (np.sqrt(5) - 1) / 2
        c, d = b - gr * (b - a), a + gr * (b - a)
        while b - a > REFINE_TOL_DEG and dist.spent < REFINE_PAIR_S:
            if f(c) < f(d):
                b, d = d, c
                c = b - gr * (b - a)
            else:
                a, c = c, d
                d = a + gr * (b - a)
        cand = [(ys[i], xs[i]), (f(c), c), (f(d), d)]
        y, x = min(cand)
        if y < best:
            best, pose = y, dict(pose, **{j: float(x)})
    if not np.isfinite(best):
        return None
    return dict(min_distance_mm=round(best, 3), pose={k: round(v, 2) for k, v in pose.items()}, method=dist.method,
                evaluations=dist.calls, seconds=round(time.time() - t0, 1))


def sweep_joint(ctx, jn, step, reach_mm=95.0, margin=3.0):
    body, t0 = ctx.body, time.time()
    limb = body.limb_of()
    other_limb = set()
    o = np.array(body.joints[jn]["origin_world_mm"], float)
    names = []
    for n, p in body.parts.items():
        if F.is_fastener(n, p) or in_class(body, n, "no_collision"):   # role class (data/roles.json): glue, harness, sensor layers
            continue
        b = np.array(p["bounds_mm"], float)
        if np.linalg.norm(0.5 * (b[0] + b[1]) - o) < reach_mm + 0.5 * np.linalg.norm(b[1] - b[0]):
            names.append(n)
    scope = set(ctx.scope)
    poses, adj = F.pose_set(body, jn, step)
    poses = [{}] + poses
    results, at_limit, cache = {}, {}, {}
    for pi, pose in enumerate(poses):
        corner = controller_limit(body, pose)
        rows_of = at_limit if corner else results      # D-15: a controller-limit corner's contacts are listed apart
        Tl = {}
        lt = lambda lk: Tl.setdefault(lk, body.link_T(lk, pose))  # noqa: E731
        for i, a in enumerate(names):
            pa = body.parts[a]
            for b in names[i + 1:]:
                if a not in scope and b not in scope:
                    continue
                pb = body.parts[b]
                la, lb = limb.get(pa["owning_link"]), limb.get(pb["owning_link"])
                if la and lb and la != lb:
                    other_limb.add((a, b))
                    continue
                same = pa["owning_link"] == pb["owning_link"]
                if same and pi > 0:
                    continue
                T_rel = np.linalg.inv(lt(pb["owning_link"])) @ lt(pa["owning_link"])
                key = (a, b, tuple(np.round(T_rel, 4).ravel()))
                if key not in cache:
                    A = Prep.get(ctx, a)
                    cache[key] = F.measure(A.mf, A.tree.data, A.bounds, T_rel, Prep.get(ctx, b), margin)
                r = cache[key]
                if r is None:
                    continue
                vol, clear, method = r
                v = 0.0 if vol != vol else vol
                row = rows_of.setdefault((a, b), dict(a=a, b=b, link_a=pa["owning_link"], link_b=pb["owning_link"], same_link=same,
                                                       max_overlap_mm3=0.0, min_clearance_mm=1e9, poses_overlapping=0, worst_pose=None, method=method,
                                                       pose_of_min=None))
                if v > 0.05:
                    row["poses_overlapping"] += 1
                if v > row["max_overlap_mm3"] or row["worst_pose"] is None:
                    row["worst_pose"] = pose
                row["max_overlap_mm3"] = max(row["max_overlap_mm3"], v)
                if clear < row["min_clearance_mm"]:
                    row["min_clearance_mm"], row["pose_of_min"] = clear, pose
    rows = sorted(results.values(), key=lambda r: (-r["max_overlap_mm3"], r["min_clearance_mm"]))
    rows = [r for r in rows if r["max_overlap_mm3"] > 0.05 or r["min_clearance_mm"] < 1.0]
    shapes = {}
    for r in rows:              # near-contacts: surface samples ~0.7 mm apart over-state them; refine around the sampled minimum
        if r["max_overlap_mm3"] <= 0.05 and r["min_clearance_mm"] < REFINE_MM:
            r["refined"] = refine_pair(ctx, r, jn, step, shapes)
            if r["refined"] is None:
                del r["refined"]
                continue
            print(f"collision_sweep: {jn}: {r['a']} x {r['b']} sampled {r['min_clearance_mm']:.3f} mm -> refined "
                  f"{r['refined']['min_distance_mm']} mm at {r['refined']['pose']} ({r['refined']['evaluations']} evaluations, "
                  f"{r['refined']['seconds']} s)", file=sys.stderr, flush=True)
    lim = sorted((dict(r, design_poses_max_overlap_mm3=round(results.get((r["a"], r["b"]), {}).get("max_overlap_mm3", 0.0), 3))
                  for r in at_limit.values() if r["max_overlap_mm3"] > 0.05), key=lambda r: -r["max_overlap_mm3"])
    return dict(poses=len(poses), parts=len(names), adjacent=[x["name"] for x in adj], contacts=rows, other_limb_pairs_left_out=len(other_limb),
                controller_limit_poses=sum(1 for p in poses if controller_limit(body, p)), controller_limit_contacts=lim,
                seconds=round(time.time() - t0, 1))


def merge_pair_cases(cases):
    """One worst case per pair: the same pair seen from several joints is listed once, with the joints that see it."""
    by = {}
    for c in cases:
        k = frozenset((c["part"], c["mating_part"]))
        by.setdefault(k, []).append(c)
    out = []
    for cs in by.values():
        w = min(cs, key=lambda c: (c["unit"] != "mm3 overlap", -c["value"] if c["unit"].startswith("mm3") else c["value"]))
        joints = [c["what"].replace(" sweep", "").replace(", same link", "") for c in cs]
        same = ", same link" if "same link" in w["what"] else ""
        out.append(dict(w, what=f"{', '.join(dict.fromkeys(joints))} sweep{same}"))
    return out


@register("collision_sweep", lambda ctx: bool(ctx.scope) and bool(getattr(ctx, "joints", None) or scope_joints(ctx)))
def run(ctx):
    """Per joint that moves a scoped part: pair overlap (mm3) and clearance (mm) over the joint range and neighbouring-joint combinations."""
    asked = getattr(ctx, "joints", None) or scope_joints(ctx)
    joints = [jn for jn in asked if ctx.body.joints[jn].get("range_deg")]      # a joint without a range is not swept, it is listed
    step = getattr(ctx, "step_deg", None) or 5.0
    per_joint, per_part, cases = {}, {}, []
    for jn in joints:
        r = sweep_joint(ctx, jn, step)
        per_joint[jn] = r
        for c in r["contacts"]:
            for n in (c["a"], c["b"]):
                if n in ctx.scope:
                    per_part.setdefault(n, {}).setdefault(jn, []).append(c)
            ref = c.get("refined")
            if ref:             # the refined value and its pose (sampled value kept next to it)
                cases.append(dict(part=c["a"], mating_part=c["b"], pose=str(ref["pose"]) if ref["pose"] else None, location_mm=None,
                                  value=ref["min_distance_mm"], unit="mm min clearance", sampled_mm=round(c["min_clearance_mm"], 3),
                                  method=ref["method"], what=f"{jn} sweep" + (", same link" if c["same_link"] else "")))
                continue
            cases.append(dict(part=c["a"], mating_part=c["b"], pose=str({k: round(v, 1) for k, v in c["worst_pose"].items()}) if c["worst_pose"] else None,   # same-link pairs: no pose
                              location_mm=None,
                              value=c["max_overlap_mm3"] if c["max_overlap_mm3"] > 0.05 else c["min_clearance_mm"],
                              unit="mm3 overlap" if c["max_overlap_mm3"] > 0.05 else "mm min clearance", what=f"{jn} sweep" + (", same link" if c["same_link"] else "")))
    cases = merge_pair_cases(cases)
    limit = {}                                          # worst controller-limit contact per pair over the joints that see it
    for jn, r in per_joint.items():
        for c in r["controller_limit_contacts"]:
            k = f"{c['a']} x {c['b']}"
            if c["max_overlap_mm3"] > limit.get(k, {}).get("max_overlap_mm3", -1):
                limit[k] = dict(max_overlap_mm3=round(c["max_overlap_mm3"], 2), pose={j: round(v, 1) for j, v in c["worst_pose"].items()},
                                design_poses_max_overlap_mm3=c["design_poses_max_overlap_mm3"], joint=jn)
    limit = dict(sorted(limit.items(), key=lambda t: -t[1]["max_overlap_mm3"]))
    cases.sort(key=lambda c: (c["unit"] != "mm3 overlap", -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    # distinct pairs (a pair seen from several joints counts once), as the merged worst cases list them
    over = {frozenset((c["a"], c["b"])): c["same_link"] for r in per_joint.values() for c in r["contacts"] if c["max_overlap_mm3"] > 0.05}
    n_over = len(over)
    ranges, rcount, rfind = range_report(ctx.body, list(asked) + [a for r in per_joint.values() for a in r["adjacent"]])
    return dict(per_part=per_part, per_joint=per_joint, worst_cases=cases[:20], controller_limit_contacts=limit,
                joint_ranges=ranges, range_findings=rfind,
                bad_events=dict(joints_swept=len(joints), joints_not_swept_without_range=len(asked) - len(joints), **rcount, overlapping_pairs=n_over,
                                overlapping_pairs_across_links=sum(1 for same in over.values() if not same),
                                controller_limit_contact_pairs=len(limit),
                                other_limb_pairs_left_out=sum(r["other_limb_pairs_left_out"] for r in per_joint.values())),
                evidence=[], notes=f"step {step} deg; poses = declared joint range + neighbours' declared ranges + corners (joint_ranges: range_source per joint); pairs limited to those with a scoped part; pairs on two different limbs left out (WBC self-collision limits, as range_sweep; other_limb_pairs_left_out, summed over joints); "
                                   "contacts at controller-limit corners (pose_sets.json controller_limit_corners, D-15) measured and listed apart in controller_limit_contacts, "
                                   "not in worst_cases or overlapping_pairs; contacts listed above 0.05 mm3 or below 1 mm clearance; clearance sampled on surface points ~0.7 mm apart (over-states near-contacts), "
                                   f"so a pair below {REFINE_MM} mm is refined (contacts[].refined, used in worst_cases): around the pose of its sampled minimum each joint of the pose "
                                   f"on {REFINE_GRID} angles over +-one step (+-2 steps for a neighbour), then golden section to {REFINE_TOL_DEG} deg, inside the declared range, "
                                   "with the exact B-rep distance on the native STEP (closed-mesh min_gap when a part has none)")
