"""range_sweep: how far each joint turns before the scoped parts meet their own limb or the body. Per joint, the joint alone over a wide grid
(other joints at --hold, default neutral); at each angle every part on the moving side measured against every part on the fixed chain back to the
torso: exact mesh-boolean overlap (mm3) and sampled clearance (mm); overlap counts above what the pair already has at neutral. Reports the first
contact angle each side and the overlap against angle. Moved from motor-refit/range_sweep.py; pairs are limited to those with a scoped part.
Ranges: each joint's declared range (body definitions `joints:`, core.declare_ranges), recorded per joint with its source (range_source).
The sweep goes deliberately beyond it (joint_sweep_wide_deg: the training tape's extent); angles outside the declared range are measured,
marked (beyond_declared_range in the curve) and their contacts reported apart (per joint and top-level beyond_declared_range), not as
worst cases or contact pairs. A joint without a range is not swept (joints_without_range)."""
from __future__ import annotations

import json
import time

import numpy as np

from checks import bodylib as BL
from checks.core import DATA, R, beyond_declared, joint_mirrors, range_report, tick
from checks.relations import ROLES
from checks.registry import register

F = BL
_PS = json.loads((DATA / "pose_sets.json").read_text())
WIDE = {k: tuple(v) for k, v in _PS["joint_sweep_wide_deg"].items() if not k.startswith("_")}
WIDE_MARGIN_DEG = float(_PS["joint_sweep_wide_deg"]["_margin_deg"])
COUNT_MM3 = 0.5
OWN_SERVO_TOL_MM = float(_PS["own_servo"]["tol_mm"])
LIMIT = _PS["controller_limit_corners"]


def in_class(body, n, cls):
    """Is part n in role class cls (data/roles.json)? Through the run's Relations when there is one."""
    rel = BL.RELATIONS
    return rel.in_class(n, cls) if rel is not None else body.parts[n].get("role") in ROLES["classes"][cls]["roles"]


def wide_range(body, jn):
    """(lo, hi) deg range_sweep turns joint jn over: the hull of its declared range (core.declare_ranges) and its stress extent: its own
    joint_sweep_wide_deg entry, else its mirror joint's (angles x sign, core.joint_mirrors), else the declared range widened by
    _margin_deg each side (pose_sets.json). None for a joint without a range (not swept)."""
    r = body.joints[jn].get("range_deg")
    if not r:
        return None
    m, sg = joint_mirrors(body).get(jn, (None, None))
    w = WIDE[jn] if jn in WIDE else tuple(sorted(sg * a for a in WIDE[m])) if m in WIDE else \
        (min(r) - WIDE_MARGIN_DEG, max(r) + WIDE_MARGIN_DEG)
    return (min(min(w), min(r)), max(max(w), max(r)))


def controller_limit(body, q):
    """The joints of a controller-limit corner at pose q (data/pose_sets.json controller_limit_corners, owner decision D-15): every
    joint of a group, or of its mirror group (core.joint_mirrors), at (or past) an end of its declared range (core.declare_ranges) at
    once; [] when q is no such corner. Joints q leaves out stand at 0; a joint without a range makes no corner."""
    tol = LIMIT["end_tol_deg"]
    at_end = lambda j: q.get(j, 0.0) <= min(body.joints[j]["range_deg"]) + tol or q.get(j, 0.0) >= max(body.joints[j]["range_deg"]) - tol  # noqa: E731
    jm = joint_mirrors(body)
    for grp in LIMIT["groups"]:
        for js in (list(grp), [jm.get(j, (None,))[0] for j in grp]):
            if all(j in body.joints and body.joints[j].get("range_deg") and at_end(j) for j in js):
                return js
    return []


def _axis_span(o, a, b, tol):
    """(t0, t1) of the line o + t a inside box b widened by tol, or None when the line misses it."""
    lo, hi = b[0] - tol, b[1] + tol
    t0, t1 = -np.inf, np.inf
    for i in range(3):
        if abs(a[i]) < 1e-12:
            if o[i] < lo[i] or o[i] > hi[i]:
                return None
        else:
            u, v = (lo[i] - o[i]) / a[i], (hi[i] - o[i]) / a[i]
            t0, t1 = max(t0, min(u, v)), min(t1, max(u, v))
    return (t0, t1) if t0 <= t1 else None


def own_servo(body, jn):
    """The joint's own servo parts (case, horn, idler), from the joint tree and geometry, not names: on the joint's parent or child
    link, the servo_case parts (role class, data/roles.json) whose box holds the joint origin, and the servo_output parts the joint
    axis passes through within such a case's extent along the axis (boxes widened by own_servo.tol_mm, pose_sets.json). A part
    whose body definition declares spec.servo_of_joint belongs to that joint's servo only."""
    cache = body.__dict__.setdefault("_own_servo", {})
    if jn in cache:
        return cache[jn]
    j, rel, tol = body.joints[jn], BL.RELATIONS, OWN_SERVO_TOL_MM
    o, a = np.array(j["origin_world_mm"], float), np.array(j["axis_world_neutral"], float)
    a = a / np.linalg.norm(a)
    declared = {}
    for n in body.parts:
        d = ((rel.claims.get(n) or {}).get("spec") or {}).get("servo_of_joint") if rel is not None else None
        if d:
            declared[n] = d
    near = {n: np.array(p["bounds_mm"], float) for n, p in body.parts.items() if p["owning_link"] in (j["parent"], j["child"])
            and n not in declared}
    cases = [n for n, b in near.items() if in_class(body, n, "servo_case") and np.all(o >= b[0] - tol) and np.all(o <= b[1] + tol)]
    spans = [s for s in (_axis_span(o, a, near[c], tol) for c in cases) if s]
    outs = []
    for n, b in near.items():
        h = _axis_span(o, a, b, tol) if in_class(body, n, "servo_output") else None
        if h and any(h[0] <= s[1] + tol and h[1] >= s[0] - tol for s in spans):
            outs.append(n)
    cache[jn] = set(cases) | set(outs) | {n for n, d in declared.items() if d == jn}
    return cache[jn]


def own_servo_pair(body, a, b, joint):
    s = own_servo(body, joint)
    return a in s and b in s


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
        if j.get("active") is False:
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
    lo, hi = wide_range(body, jn)
    dr = j["range_deg"]
    mlinks, flinks = subtree(body, j["child"]), chain_to_root(body, j["parent"])
    usable = lambda n: not F.is_fastener(n, body.parts[n]) and not in_class(body, n, "no_collision")  # noqa: E731
    moving = [n for n, p in body.parts.items() if p["owning_link"] in mlinks and usable(n)]
    fixed = [n for n, p in body.parts.items() if p["owning_link"] in flinks and usable(n)]
    o = np.array(j["origin_world_mm"], float)
    reach = lambda n: np.linalg.norm(0.5 * np.sum(body.parts[n]["bounds_mm"], axis=0) - o) - 0.5 * np.linalg.norm(  # noqa: E731
        np.subtract(*body.parts[n]["bounds_mm"][::-1]))
    swing = max(np.linalg.norm(np.array(body.parts[n]["bounds_mm"]) - o, axis=1).max() for n in moving) if moving else 0
    fixed = [n for n in fixed if reach(n) < swing + margin]
    scope = set(ctx.scope)
    angles = sorted(set(np.round(np.arange(lo, hi + 1e-6, step), 3).tolist()) | {0.0} | {float(min(dr)), float(max(dr))})
    T_fixed = {lk: body.link_T(lk, held) for lk in flinks}
    base, curve, pairs = {}, [], {}
    for ang in [0.0] + angles:
        q = dict(held, **{jn: float(ang)})
        beyond = [b["joint"] for b in beyond_declared(body, q)]       # the swept or a held joint outside its declared range
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
                if own_servo_pair(body, a, b, jn):
                    servo_tot += grow
                    continue
                clear_min = min(clear_min, clear if vol <= base.get((a, b), 0.0) + 0.01 else 0.0)
                if grow > COUNT_MM3:
                    tot += grow
                    row = pairs.setdefault(f"{a} x {b}", dict(a=a, b=b, angles=[], beyond_declared_angles=[]))
                    row["beyond_declared_angles" if beyond else "angles"].append([float(ang), round(grow, 2)])
                    if worst is None or grow > worst[1]:
                        worst = (f"{a} x {b}", round(grow, 2))
        if ang != 0.0 or not curve:
            curve.append(dict(deg=float(ang), overlap_mm3=round(tot, 2), servo_model_mm3=round(servo_tot, 2),
                              min_clearance_mm=round(clear_min, 2) if clear_min < 1e9 else None, worst_pair=worst,
                              controller_limit=controller_limit(body, q),      # D-15 corner (--hold): measured, not a first contact
                              beyond_declared_range=beyond))                    # stress angle: measured, reported apart
    curve = sorted({c["deg"]: c for c in curve}.values(), key=lambda c: c["deg"])
    neg = [c["deg"] for c in curve if c["deg"] < 0 and c["overlap_mm3"] > 0 and not c["controller_limit"]]
    pos = [c["deg"] for c in curve if c["deg"] > 0 and c["overlap_mm3"] > 0 and not c["controller_limit"]]
    ineg = [d for d in neg if min(dr) <= d]
    ipos = [d for d in pos if d <= max(dr)]
    design = [r for r in pairs.values() if r["angles"]]
    apart = [dict(pair=f"{r['a']} x {r['b']}", first_contact_deg=min(r["beyond_declared_angles"], key=lambda x: abs(x[0]))[0],
                  max_overlap_mm3=max(x[1] for x in r["beyond_declared_angles"]), angles=r["beyond_declared_angles"])
             for r in pairs.values() if r["beyond_declared_angles"]]
    bc = [c for c in curve if c["beyond_declared_range"]]
    return dict(swept_deg=[lo, hi], assembly_range_deg=list(dr), declared_range_deg=list(dr), range_source=body.range_source.get(jn),
                moving_parts=len(moving), fixed_parts=len(fixed),
                contact_free_deg=[max(neg) + step if neg else lo, min(pos) - step if pos else hi],
                first_contact_deg=[max(neg) if neg else None, min(pos) if pos else None],
                first_contact_inside_declared_deg=[max(ineg) if ineg else None, min(ipos) if ipos else None], curve=curve,
                pairs=sorted(design, key=lambda r: -max(x[1] for x in r["angles"])),
                beyond_declared_range=dict(angles=len(bc), deg=[[lo, float(min(dr))], [float(max(dr)), hi]],
                                           max_overlap_mm3=max((c["overlap_mm3"] for c in bc), default=0.0),
                                           pairs=sorted(apart, key=lambda r: -r["max_overlap_mm3"])),
                neutral_static_overlap_mm3=round(sum(v for (a, b), v in base.items() if not own_servo_pair(body, a, b, jn)), 2),
                seconds=round(time.time() - t0, 1))


@register("range_sweep", lambda ctx: bool(ctx.scope) and bool(getattr(ctx, "joints", None) or scope_joints(ctx)))
def run(ctx):
    """Per joint that moves a scoped part: first-contact angle each side and overlap (mm3) against angle, for pairs involving a scoped part."""
    asked = getattr(ctx, "joints", None) or scope_joints(ctx)
    step = getattr(ctx, "step_deg", None) or 2.5
    held = getattr(ctx, "held", {})
    joints = [jn for jn in asked if wide_range(ctx.body, jn) is not None]      # a joint without a range is not swept, it is listed
    ranges, rcount, rfind = range_report(ctx.body, list(asked) + list(held))
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
    apart = {}                                          # contacts only beyond a declared range: per pair, the joint that shows the most
    for jn, r in per_joint.items():
        for row in r["beyond_declared_range"]["pairs"]:
            if row["max_overlap_mm3"] > apart.get(row["pair"], {}).get("max_overlap_mm3", -1):
                apart[row["pair"]] = dict(joint=jn, declared_range_deg=r["declared_range_deg"], first_contact_deg=row["first_contact_deg"],
                                          max_overlap_mm3=row["max_overlap_mm3"])
    apart = dict(sorted(apart.items(), key=lambda t: -t[1]["max_overlap_mm3"]))
    per_part = {}
    for jn, r in per_joint.items():
        for row in r["pairs"]:
            for n in (row["a"], row["b"]):
                if n in ctx.scope:
                    per_part.setdefault(n, {}).setdefault(jn, []).append(dict(pair=f"{row['a']} x {row['b']}", first_contact_deg=min(row["angles"], key=lambda x: abs(x[0]))[0],
                                                                         max_overlap_mm3=max(x[1] for x in row["angles"])))
    return dict(per_part=per_part, per_joint=per_joint, worst_cases=cases[:20], joint_ranges=ranges, range_findings=rfind,
                beyond_declared_range=apart,
                bad_events=dict(joints_swept=len(joints), contact_pairs=n_pairs,
                                joints_with_contact_inside_assembly_range=sum(
                                    1 for jn, r in per_joint.items() if any(
                                        c["overlap_mm3"] > 0 and not c["beyond_declared_range"] for c in r["curve"])),
                                beyond_declared_range_contact_pairs=len(apart), joints_not_swept_without_range=len(asked) - len(joints), **rcount),
                evidence=[], notes=f"step {step} deg; overlap growth above the neutral pose counts above {COUNT_MM3} mm3; other limbs left out (their contacts are WBC self-collision limits); "
                                   "ranges are the declared ones (joint_ranges: range_source per joint, range_findings); angles beyond a declared range (the stress extent of "
                                   "joint_sweep_wide_deg) are measured, marked in the curve and their contacts listed in beyond_declared_range, not in worst_cases or contact_pairs; "
                                   "pairs limited to those with a scoped part; shell covers are included when they are in the manifest; "
                                   "angles where the held and swept joints make a controller-limit corner (pose_sets.json controller_limit_corners, D-15) are measured, "
                                   "marked in the curve (controller_limit) and not taken as first contact")
