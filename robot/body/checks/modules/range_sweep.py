"""range_sweep: how far each joint turns before the scoped parts meet their own limb or the body. Per joint, the joint alone over a wide grid
(other joints at --hold, default neutral); at each angle every part on the moving side measured against every part on the fixed chain back to the
torso: exact mesh-boolean overlap (mm3) and sampled clearance (mm); overlap counts above what the pair already has at neutral. Reports the first
contact angle each side and the overlap against angle. Moved from motor-refit/range_sweep.py; pairs are limited to those with a scoped part."""
from __future__ import annotations

import json
import re
import time

import numpy as np

from checks import bodylib as BL
from checks.core import DATA, R, tick
from checks.registry import register

F = BL
SKIP_ROLES = {"cut_material", "adhesive_allocation", "authored_harness", "candidate_sensor", "produced_backing"}
WIDE = {k: tuple(v) for k, v in json.loads((DATA / "pose_sets.json").read_text())["joint_sweep_wide_deg"].items() if not k.startswith("_")}
COUNT_MM3 = 0.5
OWN_SERVO = {k: tuple(v) for k, v in json.loads((DATA / "pose_sets.json").read_text())["own_servo_part_prefixes"].items() if not k.startswith("_")}


def kind(name):
    return re.sub(r"^[RL]_", "", name)


def own_servo_pair(a, b, joint):
    side = joint[0] if joint[:2] in ("R_", "L_") else ""
    pats = [p.format(s=side) for p in OWN_SERVO.get(kind(joint), ())]
    return any(a.startswith(p) for p in pats) and any(b.startswith(p) for p in pats)


def subtree(body, link):
    out, stack = set(), [link]
    while stack:
        lk = stack.pop(); out.add(lk)
        stack.extend(j["child"] for j in body.joints.values() if j["parent"] == lk)
    return out


def chain_to_root(body, link):
    out = {link}
    while link in body.child_joint:
        link = body.child_joint[link]["parent"]; out.add(link)
    return out


def scope_joints(ctx):
    """Joints that move a scoped part: the joint whose child link, or an ancestor joint's subtree, owns it (the chain from the root to the part's link)."""
    links = {ctx.link(n) for n in ctx.scope}
    out = []
    for jn, j in ctx.body.joints.items():
        if j.get("active") is False or kind(jn) not in WIDE:
            continue
        if links & subtree(ctx.body, j["child"]):
            out.append(jn)
    return out


class Prep:
    cache = {}

    @classmethod
    def get(cls, ctx, n):
        if n not in cls.cache:
            cls.cache[n] = F.Prepped(ctx.solid(n)[0])
        return cls.cache[n]


def sweep_joint(ctx, jn, step, held, margin=3.0):
    body, t0 = ctx.body, time.time()
    j = body.joints[jn]
    lo, hi = WIDE[kind(jn)]
    mlinks, flinks = subtree(body, j["child"]), chain_to_root(body, j["parent"])
    usable = lambda n: not F.is_fastener(n, body.parts[n]) and body.parts[n].get("role") not in SKIP_ROLES  # noqa: E731
    moving = [n for n, p in body.parts.items() if p["owning_link"] in mlinks and usable(n)]
    fixed = [n for n, p in body.parts.items() if p["owning_link"] in flinks and usable(n)]
    o = np.array(j["origin_world_mm"], float)
    reach = lambda n: np.linalg.norm(0.5 * np.sum(body.parts[n]["bounds_mm"], axis=0) - o) - 0.5 * np.linalg.norm(  # noqa: E731
        np.subtract(*body.parts[n]["bounds_mm"][::-1]))
    swing = max(np.linalg.norm(np.array(body.parts[n]["bounds_mm"]) - o, axis=1).max() for n in moving) if moving else 0
    fixed = [n for n in fixed if reach(n) < swing + margin]
    scope = set(ctx.scope)
    angles = sorted(set(np.round(np.arange(lo, hi + 1e-6, step), 3).tolist()) | {0.0})
    T_fixed = {lk: body.link_T(lk, held) for lk in flinks}
    base, curve, pairs = {}, [], {}
    for ang in [0.0] + angles:
        q = dict(held, **{jn: float(ang)})
        T_mov = {lk: body.link_T(lk, q) for lk in mlinks}
        tot, servo_tot, clear_min, worst = 0.0, 0.0, 1e9, None
        for a in moving:
            for b in fixed:
                if a not in scope and b not in scope:
                    continue
                A = Prep.get(ctx, a)
                T_rel = np.linalg.inv(T_fixed[body.parts[b]["owning_link"]]) @ T_mov[body.parts[a]["owning_link"]]
                r = F.measure(A.mf, A.tree.data, A.bounds, T_rel, Prep.get(ctx, b), margin)
                if r is None:
                    continue
                vol, clear, _ = r
                vol = 0.0 if vol != vol else vol
                if ang == 0.0 and (a, b) not in base:
                    base[(a, b)] = vol
                grow = max(0.0, vol - base.get((a, b), 0.0))
                if own_servo_pair(a, b, jn):
                    servo_tot += grow
                    continue
                clear_min = min(clear_min, clear if vol <= base.get((a, b), 0.0) + 0.01 else 0.0)
                if grow > COUNT_MM3:
                    tot += grow
                    row = pairs.setdefault(f"{a} x {b}", dict(a=a, b=b, angles=[]))
                    row["angles"].append([float(ang), round(grow, 2)])
                    if worst is None or grow > worst[1]:
                        worst = (f"{a} x {b}", round(grow, 2))
        if ang != 0.0 or not curve:
            curve.append(dict(deg=float(ang), overlap_mm3=round(tot, 2), servo_model_mm3=round(servo_tot, 2),
                              min_clearance_mm=round(clear_min, 2) if clear_min < 1e9 else None, worst_pair=worst))
    curve = sorted({c["deg"]: c for c in curve}.values(), key=lambda c: c["deg"])
    neg = [c["deg"] for c in curve if c["deg"] < 0 and c["overlap_mm3"] > 0]
    pos = [c["deg"] for c in curve if c["deg"] > 0 and c["overlap_mm3"] > 0]
    return dict(swept_deg=[lo, hi], assembly_range_deg=j.get("range_deg"), moving_parts=len(moving), fixed_parts=len(fixed),
                contact_free_deg=[max(neg) + step if neg else lo, min(pos) - step if pos else hi],
                first_contact_deg=[max(neg) if neg else None, min(pos) if pos else None], curve=curve,
                pairs=sorted(pairs.values(), key=lambda r: -max(x[1] for x in r["angles"])),
                neutral_static_overlap_mm3=round(sum(v for (a, b), v in base.items() if not own_servo_pair(a, b, jn)), 2),
                seconds=round(time.time() - t0, 1))


@register("range_sweep", lambda ctx: bool(ctx.scope) and bool(getattr(ctx, "joints", None) or scope_joints(ctx)))
def run(ctx):
    """Per joint that moves a scoped part: first-contact angle each side and overlap (mm3) against angle, for pairs involving a scoped part."""
    joints = getattr(ctx, "joints", None) or scope_joints(ctx)
    step = getattr(ctx, "step_deg", None) or 2.5
    held = getattr(ctx, "held", {})
    per_joint, cases, n_pairs = {}, [], 0
    for jn in joints:
        r = sweep_joint(ctx, jn, step, held)
        per_joint[jn] = r
        n_pairs += len(r["pairs"])
        for row in r["pairs"][:3]:
            first = min(row["angles"], key=lambda x: abs(x[0]))
            cases.append(dict(part=row["a"], mating_part=row["b"], pose=f"{jn} {first[0]:+g} deg", location_mm=None,
                              value=max(x[1] for x in row["angles"]), unit="mm3 max overlap growth", what=f"{jn} first contact at {first[0]:+g} deg"))
    cases.sort(key=lambda c: -c["value"])
    per_part = {}
    for jn, r in per_joint.items():
        for row in r["pairs"]:
            for n in (row["a"], row["b"]):
                if n in ctx.scope:
                    per_part.setdefault(n, {}).setdefault(jn, []).append(dict(pair=f"{row['a']} x {row['b']}", first_contact_deg=min(row["angles"], key=lambda x: abs(x[0]))[0],
                                                                         max_overlap_mm3=max(x[1] for x in row["angles"])))
    return dict(per_part=per_part, per_joint=per_joint, worst_cases=cases[:20],
                bad_events=dict(joints_swept=len(joints), contact_pairs=n_pairs,
                                joints_with_contact_inside_assembly_range=sum(
                                    1 for jn, r in per_joint.items() if r["assembly_range_deg"] and any(
                                        c["overlap_mm3"] > 0 and r["assembly_range_deg"][0] <= c["deg"] <= r["assembly_range_deg"][1] for c in r["curve"]))),
                evidence=[], notes=f"step {step} deg; overlap growth above the neutral pose counts above {COUNT_MM3} mm3; other limbs left out (their contacts are WBC self-collision limits); "
                                   "pairs limited to those with a scoped part; shell covers are included when they are in the manifest")
