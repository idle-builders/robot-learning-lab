"""collision_sweep: part-to-part collision over a joint's range: the joint over its full range, each neighbouring joint over its range at the
joint's ends and middle, and pairwise range corners (motor-refit pose_set). Pairs of non-fastener parts near the joint that involve a scoped part:
max overlap (mm3, exact mesh boolean), minimum clearance (mm), poses overlapping, worst pose. Same-link pairs once at neutral. Pairs of parts on
two different limbs (the other leg, an arm) are left out, as in range_sweep: their contacts are WBC self-collision limits; counted per joint.
Moved from motor-refit/collision_sweep.py."""
from __future__ import annotations

import time

import numpy as np

from checks import bodylib as BL
from checks.core import R, tick
from checks.modules.range_sweep import Prep, scope_joints
from checks.registry import register

F = BL
SKIP_ROLES = {"cut_material", "adhesive_allocation", "authored_harness", "candidate_sensor", "produced_backing"}


def sweep_joint(ctx, jn, step, reach_mm=95.0, margin=3.0):
    body, t0 = ctx.body, time.time()
    limb = body.limb_of()
    other_limb = set()
    o = np.array(body.joints[jn]["origin_world_mm"], float)
    names = []
    for n, p in body.parts.items():
        if F.is_fastener(n, p) or p.get("role") in SKIP_ROLES:
            continue
        b = np.array(p["bounds_mm"], float)
        if np.linalg.norm(0.5 * (b[0] + b[1]) - o) < reach_mm + 0.5 * np.linalg.norm(b[1] - b[0]):
            names.append(n)
    scope = set(ctx.scope)
    poses, adj = F.pose_set(body, jn, step)
    poses = [{}] + poses
    results, cache = {}, {}
    for pi, pose in enumerate(poses):
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
                row = results.setdefault((a, b), dict(a=a, b=b, link_a=pa["owning_link"], link_b=pb["owning_link"], same_link=same,
                                                       max_overlap_mm3=0.0, min_clearance_mm=1e9, poses_overlapping=0, worst_pose=None, method=method))
                if v > 0.05:
                    row["poses_overlapping"] += 1
                if v > row["max_overlap_mm3"] or row["worst_pose"] is None:
                    row["worst_pose"] = pose
                row["max_overlap_mm3"] = max(row["max_overlap_mm3"], v)
                row["min_clearance_mm"] = min(row["min_clearance_mm"], clear)
    rows = sorted(results.values(), key=lambda r: (-r["max_overlap_mm3"], r["min_clearance_mm"]))
    rows = [r for r in rows if r["max_overlap_mm3"] > 0.05 or r["min_clearance_mm"] < 1.0]
    return dict(poses=len(poses), parts=len(names), adjacent=[x["name"] for x in adj], contacts=rows, other_limb_pairs_left_out=len(other_limb),
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
    joints = getattr(ctx, "joints", None) or scope_joints(ctx)
    step = getattr(ctx, "step_deg", None) or 5.0
    per_joint, per_part, cases = {}, {}, []
    for jn in joints:
        r = sweep_joint(ctx, jn, step)
        per_joint[jn] = r
        for c in r["contacts"]:
            for n in (c["a"], c["b"]):
                if n in ctx.scope:
                    per_part.setdefault(n, {}).setdefault(jn, []).append(c)
            cases.append(dict(part=c["a"], mating_part=c["b"], pose=str({k: round(v, 1) for k, v in c["worst_pose"].items()}) if c["worst_pose"] else None,   # same-link pairs: no pose
                              location_mm=None,
                              value=c["max_overlap_mm3"] if c["max_overlap_mm3"] > 0.05 else c["min_clearance_mm"],
                              unit="mm3 overlap" if c["max_overlap_mm3"] > 0.05 else "mm min clearance", what=f"{jn} sweep" + (", same link" if c["same_link"] else "")))
    cases = merge_pair_cases(cases)
    cases.sort(key=lambda c: (c["unit"] != "mm3 overlap", -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    # distinct pairs (a pair seen from several joints counts once), as the merged worst cases list them
    over = {frozenset((c["a"], c["b"])): c["same_link"] for r in per_joint.values() for c in r["contacts"] if c["max_overlap_mm3"] > 0.05}
    n_over = len(over)
    return dict(per_part=per_part, per_joint=per_joint, worst_cases=cases[:20],
                bad_events=dict(joints_swept=len(joints), overlapping_pairs=n_over,
                                overlapping_pairs_across_links=sum(1 for same in over.values() if not same)),
                evidence=[], notes=f"step {step} deg; poses = joint range + neighbours' ranges + corners; pairs limited to those with a scoped part; pairs on two different limbs left out (WBC self-collision limits, as range_sweep); contacts listed above 0.05 mm3 or below 1 mm clearance")
