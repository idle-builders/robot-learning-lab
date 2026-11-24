"""clearance: scoped parts against their neighbours on other links over the pose set: new overlap vs standing (mm3) and minimum surface distance (mm, capped at search_mm).
Pairs of parts on two different limbs (the other leg, an arm; limbs from the manifest's joint tree) are left out, as in range_sweep and
collision_sweep: their contacts are WBC self-collision limits; they are counted and listed (other_limb_pairs). Poses that are
controller-limit corners (data/pose_sets.json controller_limit_corners, owner decision D-15) are measured, their new overlaps listed
apart (controller_limit_contacts), not as design overlaps. Poses that set a joint beyond its declared range (body definitions `joints:`,
core.declare_ranges: the A-11 / A-12 leg library, stress ends, fit extremes) are measured and reported apart (beyond_declared_range: per
pose its joints, declared range and value; per pair the measurements), outside by_pair / worst_cases. Raw function moved from leg/fast_check.py."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from checks.core import ROOT, R, beyond_declared, kit, range_report, screen_solid, step_mesh, tick
from checks.modules.range_sweep import controller_limit
from checks.registry import register


def check_clearance(body, changed, others, sol, mf, poses, SEARCH_MM=5.0):
    """others: {name: part} candidate neighbours; only those on a link other than the scoped part's are paired."""
    link = {n: body.parts[n]["owning_link"] for n in set(others) | set(changed)}
    limb = body.limb_of()
    pairs = sorted({tuple(sorted((a, b))) for a in changed for b in set(others) | set(changed)
                    if a != b and link[a] != link[b]})
    two_limbs = lambda a, b: bool(limb.get(link[a]) and limb.get(link[b]) and limb[link[a]] != limb[link[b]])  # noqa: E731
    other_limb = [pr for pr in pairs if two_limbs(*pr)]
    pairs = [pr for pr in pairs if not two_limbs(*pr)]
    for n in {x for pr in pairs for x in pr}:
        if n not in mf:
            t0 = time.time(); sol[n] = screen_solid(body, n, changed); mf[n] = kit.mf(sol[n][0]); tick("solids (others)", t0)
    t0 = time.time()

    T0 = {lk: body.link_T(lk, poses[0][2]) for lk in {link[n] for pr in pairs for n in pr}}
    base = None

    def at(q):
        """(overlap, distance) per pair within SEARCH_MM at pose q. A pair whose two links keep the relative transform they have at the
        first pose (standing) has its first-pose result (an unscoped run: torso pairs over leg poses); only moved pairs are measured."""
        Tq = {lk: body.link_T(lk, q) for lk in T0}
        lp = {(link[a], link[b]) for a, b in pairs}
        lmoved = lp if base is None else {(la, lb) for la, lb in lp if not np.allclose(np.linalg.solve(Tq[la], Tq[lb]),
                                                                                        np.linalg.solve(T0[la], T0[lb]), rtol=0.0, atol=1e-9)}
        moved = [(a, b) for a, b in pairs if (link[a], link[b]) in lmoved]
        out = {} if base is None else {pr: r for pr, r in base.items() if (link[pr[0]], link[pr[1]]) not in lmoved}
        posed = {n: mf[n].transform(Tq[link[n]][:3, :4]) for n in {x for pr in moved for x in pr}}
        bb = {n: np.array(g.bounding_box()).reshape(2, 3) for n, g in posed.items()}
        for a, b in moved:
            gap = np.maximum(0.0, np.maximum(bb[a][0] - bb[b][1], bb[b][0] - bb[a][1]))
            if np.linalg.norm(gap) > SEARCH_MM:
                continue
            v = (posed[a] ^ posed[b]).volume()
            d = 0.0 if v > 1e-3 else posed[a].min_gap(posed[b], SEARCH_MM)
            out[(a, b)] = (v, min(d, SEARCH_MM))
        return out
    base = at(poses[0][2])
    worst, by_fam, at_limit, beyond_poses, apart = {}, {}, {}, [], {}
    for name, fam, q in poses:
        res = at(q)
        beyond = beyond_declared(body, q)
        corner = controller_limit(body, q)
        if beyond:
            beyond_poses.append(dict(pose=name, family=fam, joints=beyond, controller_limit_corner=corner or None))
        for (a, b), (v, d) in res.items():
            k = f"{a} x {b}"
            dv = v - base.get((a, b), (0.0, 0.0))[0]
            if beyond:                                   # a stress pose beyond a declared range: measured, listed apart
                w = apart.setdefault(k, dict(max_new_overlap_mm3=0.0, pose_of_max=None, joints_at_max=None, min_distance_mm=SEARCH_MM,
                                             pose_of_min=None, poses_new_overlap_over_20mm3=0))
                if dv > w["max_new_overlap_mm3"]:
                    w["max_new_overlap_mm3"], w["pose_of_max"], w["joints_at_max"] = round(dv, 1), name, beyond
                if d < w["min_distance_mm"]:
                    w["min_distance_mm"], w["pose_of_min"] = round(d, 3), name
                w["poses_new_overlap_over_20mm3"] += dv > 20.0
                continue
            if corner:                                   # D-15: a controller-limit corner's contacts are listed apart
                if dv > at_limit.get(k, {}).get("max_new_overlap_mm3", 0.1):
                    at_limit[k] = dict(max_new_overlap_mm3=round(dv, 1), pose=name, joints_at_range_ends=corner)
                continue
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
    apart = dict(sorted(((k, w) for k, w in apart.items() if w["max_new_overlap_mm3"] > 0.1 or w["min_distance_mm"] < SEARCH_MM),
                        key=lambda t: (-t[1]["max_new_overlap_mm3"], t[1]["min_distance_mm"])))
    ranges, rcount, rfind = range_report(body, {j for _, _, q in poses for j in q})
    return dict(poses=len(poses), pairs=len(pairs), search_mm=SEARCH_MM, by_pair=worst, new_overlap_by_family=by_fam,
                beyond_declared_range=dict(poses=beyond_poses, contacts=apart), joint_ranges=ranges, range_findings=rfind, range_counts=rcount,
                other_limb_pairs=[f"{a} x {b}" for a, b in other_limb],
                controller_limit_poses=[name for name, _, q in poses if controller_limit(body, q) and not beyond_declared(body, q)],
                controller_limit_contacts=dict(sorted(at_limit.items(), key=lambda t: -t[1]["max_new_overlap_mm3"])),
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
                                pose_pair_new_overlaps_over_20mm3=sum(w["poses_new_overlap_over_20mm3"] for w in raw_pairs.values()),
                                other_limb_pairs_left_out=len(raw["other_limb_pairs"]),
                                controller_limit_contact_pairs=len(raw["controller_limit_contacts"]),
                                beyond_declared_range_poses=len(raw["beyond_declared_range"]["poses"]),
                                beyond_declared_range_pairs_with_new_overlap=sum(1 for w in raw["beyond_declared_range"]["contacts"].values()
                                                                                 if w["max_new_overlap_mm3"] > 0.1),
                                **raw["range_counts"]),
                new_overlap_by_family=raw["new_overlap_by_family"], other_limb_pairs=raw["other_limb_pairs"],
                controller_limit_poses=raw["controller_limit_poses"], controller_limit_contacts=raw["controller_limit_contacts"],
                beyond_declared_range=raw["beyond_declared_range"], joint_ranges=raw["joint_ranges"], range_findings=raw["range_findings"], evidence=[],
                notes=f"{raw['poses']} poses ({ctx.poses_name}), {raw['pairs']} pairs, distance capped at {raw['search_mm']} mm; "
                      f"{len(raw['other_limb_pairs'])} pairs on two different limbs left out (other_limb_pairs: WBC self-collision limits, as range_sweep / collision_sweep); "
                      f"{len(raw['controller_limit_poses'])} controller-limit corner poses (pose_sets.json controller_limit_corners, D-15): their new overlaps are in "
                      "controller_limit_contacts, not in by_pair / worst_cases; "
                      f"{len(raw['beyond_declared_range']['poses'])} poses set a joint beyond its declared range (joint_ranges): measured, in "
                      "beyond_declared_range (poses with joint, declared range, value; contacts per pair), not in by_pair / worst_cases; " + raw["note"])
