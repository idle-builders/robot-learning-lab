"""fastener_motion: every screw (head, shank, protruding tip) against the non-fastener parts on OTHER links over the pose set (and, on request, the
range of the joints given with --joints): overlap (mm3, exact manifold boolean, split head / shank / tip) and minimum distance (mm) per screw with
the worst pose; plus screws whose head sits inside a part of their own link at standing (stale models). clearance, fit and collision_sweep skip
fasteners, which hid the STS3250 case-ear PA2 heads against the hip part on roll-flip Stage 4."""
from __future__ import annotations

import time

import numpy as np

from checks import bodylib as BL
from checks import screwlib as SL
from checks.core import kit, tick
from checks.registry import register

TIP_MM = 3.0            # the tip zone: last 3 mm of the screw
HEAD_IN_PLATE_MM3 = 1.0  # head volume inside a same-link part at standing above this = a screw model placed inside its own plate
OVERLAP_MM3 = 1e-3


def _zones(mf, g):
    """Head / shank / tip manifolds of a screw (frame of its mesh) split along its axis; None for a screw without geometry."""
    if g is None:
        return None
    d, h = np.asarray(g["_d"]), np.asarray(g["_head"])
    s0 = float(h @ d)
    head = mf.trim_by_plane((-d).tolist(), -(s0 + g["head_height_mm"]))
    tip = mf.trim_by_plane(d.tolist(), s0 + g["length_mm"] - TIP_MM)
    shank = mf.trim_by_plane(d.tolist(), s0 + g["head_height_mm"]).trim_by_plane((-d).tolist(), -(s0 + g["length_mm"] - TIP_MM))
    return dict(head=head, shank=shank, tip=tip)


def _bbox(mf):
    return np.array(mf.bounding_box()).reshape(2, 3)


def _poses(ctx):
    """[(name, q)]: the pose set, plus the range poses of the joints named with --joints (opt in)."""
    P = [(n, q) for n, _, q in ctx.poses]
    joints = getattr(ctx, "joints", None)
    if joints:
        st, step = ctx.body.standing, getattr(ctx, "step_deg", None) or 10.0
        for jn in joints:
            ps, _ = BL.pose_set(ctx.body, jn, step)
            P += [(f"range {jn}: " + ", ".join(f"{k} {v:+.0f}" for k, v in p.items()), {**st, **p}) for p in ps]
    return P


def _screw_names(ctx, touching):
    links = {ctx.link(n) for n in ctx.scope}
    names = {k for k, p in ctx.parts.items() if SL.mount_check().kind(p) == "screw" and p["owning_link"] in links}
    return sorted(names | set(touching))


@register("fastener_motion", lambda ctx: bool(ctx.scope))
def run(ctx):
    """Per screw of the scoped links or touching a scoped part: overlap (mm3, head / shank / tip) and minimum distance (mm) against non-fastener parts on other links over the poses, worst pose; screws whose head sits inside a part of their own link."""
    t0 = time.time()
    body, parts, MC = ctx.body, ctx.parts, SL.mount_check()
    touching = SL.screws_of_scope(ctx)
    names = _screw_names(ctx, touching)
    poses = _poses(ctx)
    search = ctx.search_mm
    rel = ctx.rel
    hw = lambda n: rel.is_fastener(n) and not rel.is_sleeve_set(n)    # noqa: E731  sleeve sets are part of the printed part they are pressed into
    # other links: the parts that collide as rigid bodies (role class no_collision left out) and no fasteners (a sleeve set is a few
    # tubes in a hole of the part it sits in; its boolean against a screw costs minutes); body shells only on request
    cand = [n for n in parts if not rel.is_fastener(n) and not rel.in_class(n, "no_collision") and (ctx.include_shells or not rel.is_shell(n))]
    corners = {n: np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) for n in cand for b in [parts[n]["bounds_mm"]]}
    Tc = {}

    def link_T(lk, qi):
        return Tc.setdefault((lk, qi), body.link_T(lk, poses[qi][1]))

    def wbox(c, lk, qi):
        T = link_T(lk, qi)
        w = c @ T[:3, :3].T + T[:3, 3]
        return w.min(0), w.max(0)

    per_screw, unmeasured = {}, []
    for sn in names:
        p = parts[sn]
        try:
            m = body.mesh(sn)
            mf = kit.mf(m)
            if mf.is_empty():
                raise ValueError("empty manifold")
            g = MC.screw_geometry(m)
        except Exception as exc:  # noqa: BLE001
            unmeasured.append(f"{sn}: {type(exc).__name__}: {exc}")
            continue
        zones = _zones(mf, g)
        lk = p["owning_link"]
        b = m.bounds
        sc = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
        rows, cache = {}, {}
        for qi, (pn, q) in enumerate(poses):
            slo, shi = wbox(sc, lk, qi)
            Ts = link_T(lk, qi)
            for n in cand:
                ln = parts[n]["owning_link"]
                if ln == lk:
                    continue
                lo, hi = wbox(corners[n], ln, qi)
                if np.linalg.norm(np.maximum(0.0, np.maximum(slo - hi, lo - shi))) > search:
                    continue
                Trel = np.linalg.inv(link_T(ln, qi)) @ Ts      # screw into the part's frame
                key = (n, tuple(np.round(Trel, 3).ravel()))
                if key not in cache:
                    ctx.solid(n)
                    sm = mf.transform(Trel[:3, :4])
                    v = float((sm ^ ctx.mf[n]).volume())
                    z, cen = None, None
                    if v > OVERLAP_MM3:
                        zt = _zones_t(zones, Trel)
                        z = {k: float((zt[k] ^ ctx.mf[n]).volume()) for k in zt}
                        cen = np.round(kit.tm(sm ^ ctx.mf[n]).center_mass @ link_T(ln, qi)[:3, :3].T + link_T(ln, qi)[:3, 3], 2).tolist()
                    d = 0.0 if v > OVERLAP_MM3 else float(sm.min_gap(ctx.mf[n], search))
                    cache[key] = (v, min(d, search), z, cen)
                v, d, z, cen = cache[key]
                r = rows.setdefault(n, dict(part=n, link=ln, max_overlap_mm3=0.0, pose_of_max=None, zones_at_max=None, location_mm=None,
                                            min_distance_mm=search, pose_of_min=None, poses_overlapping=0, poses_measured=0))
                r["poses_measured"] += 1
                if v > OVERLAP_MM3:
                    r["poses_overlapping"] += 1
                if v > OVERLAP_MM3 and v > r["max_overlap_mm3"]:
                    r.update(max_overlap_mm3=round(v, 3), pose_of_max=pn, _qi=qi, _Trel=Trel, zones_at_max={k: round(x, 3) for k, x in z.items()}, location_mm=cen)
                if d < r["min_distance_mm"]:
                    r.update(min_distance_mm=round(d, 3), pose_of_min=pn)
        for r in rows.values():       # worst pose again with the part's full manifest mesh (the screening solid is decimated / coarser)
            Trel = r.pop("_Trel", None); r.pop("_qi", None)
            if Trel is not None:
                try:
                    r["exact_overlap_at_max_mm3"] = round(float((mf.transform(Trel[:3, :4]) ^ kit.mf(body.mesh(r["part"]))).volume()), 3)
                except Exception:  # noqa: BLE001
                    r["exact_overlap_at_max_mm3"] = None
        # head against parts of its own link at standing (neutral frame: same link, no pose)
        own = []
        for n in parts:
            pn_ = parts[n]
            if n == sn or hw(n) or pn_["owning_link"] != lk or rel.in_class(n, "audit_skip"):
                continue
            nb = np.array(pn_["bounds_mm"], float)
            if np.any(m.bounds[0] > nb[1]) or np.any(m.bounds[1] < nb[0]):
                continue
            ctx.solid(n)
            v = float((zones["head"] ^ ctx.mf[n]).volume())
            if v > 1e-3:
                own.append(dict(part=n, head_overlap_mm3=round(v, 3), role=pn_.get("role")))
        hv = float(zones["head"].volume())
        own.sort(key=lambda r: -r["head_overlap_mm3"])
        per_screw[sn] = dict(link=lk, touches=touching[sn]["touched"] if sn in touching else [], head_height_mm=g["head_height_mm"],
                             head_dia_mm=round(2 * g["r_head_mm"], 2), length_mm=g["length_mm"], head_volume_mm3=round(hv, 2),
                             others=sorted(rows.values(), key=lambda r: (-r["max_overlap_mm3"], r["min_distance_mm"]))[:6],
                             head_inside_own_link_parts=own,
                             stale_head_in_plate=bool(own and own[0]["head_overlap_mm3"] > HEAD_IN_PLATE_MM3))
    tick("fastener_motion", t0)
    cases, per_part = [], {}
    for sn, r in per_screw.items():
        for owner in (r["touches"] or [f"link {r['link']}"]):
            per_part.setdefault(owner, {})[sn] = r
        for o in r["others"]:
            hit = o["max_overlap_mm3"] > OVERLAP_MM3
            cases.append(dict(part=sn, mating_part=o["part"], pose=o["pose_of_max"] if hit else o["pose_of_min"],
                              location_mm=o["location_mm"] if hit else None, value=o["max_overlap_mm3"] if hit else o["min_distance_mm"],
                              unit="mm3 overlap" if hit else "mm min distance",
                              what=("zones " + str(o["zones_at_max"]) + f", full-mesh {o.get('exact_overlap_at_max_mm3')} mm3" + f", {o['poses_overlapping']}/{o['poses_measured']} poses") if hit else "closest approach"))
        if r["stale_head_in_plate"]:
            o = r["head_inside_own_link_parts"][0]
            cases.append(dict(part=sn, mating_part=o["part"], pose="standing", location_mm=None, value=o["head_overlap_mm3"],
                              unit="mm3 head inside own plate", what=f"head volume {r['head_volume_mm3']} mm3, same link"))
    # poses outside a joint's manifest range are named so: a hit there is beyond the robot's motion
    import json as _json
    rng = {j["name"]: j.get("range_deg") for j in _json.loads(open(ctx.manifest).read()).get("joints", []) if j.get("range_deg")}
    qs = dict(poses)

    def outside(pose_name):
        q = qs.get(pose_name) or {}
        bad = [f"{j} {v:g} (range {rng[j][0]:g}..{rng[j][1]:g})" for j, v in q.items() if j in rng and not rng[j][0] - 1e-6 <= v <= rng[j][1] + 1e-6]
        return bad
    for c in cases:
        bad = outside(c.get("pose")) if c.get("pose") and c["pose"] != "standing" else []
        if bad:
            c["pose_outside_joint_range"] = bad
            c["what"] += "; pose outside the joint range: " + ", ".join(bad)
    order = {"mm3 overlap": 0, "mm3 head inside own plate": 1}         # collisions first, stale screw models after
    cases.sort(key=lambda c: (order.get(c["unit"], 2), bool(c.get("pose_outside_joint_range")), -c["value"] if c["unit"].startswith("mm3") else c["value"]))
    n_ov = sum(1 for r in per_screw.values() if any(o["max_overlap_mm3"] > OVERLAP_MM3 for o in r["others"]))
    return dict(per_part=per_part, worst_cases=cases[:30],
                bad_events=dict(screws=len(per_screw), screws_overlapping_other_link=n_ov,
                                screws_head_inside_own_plate=sum(r["stale_head_in_plate"] for r in per_screw.values()),
                                screws_unmeasured=len(unmeasured)),
                evidence=[], unmeasured=unmeasured,
                notes=f"{len(poses)} poses ({ctx.poses_name}{' + joint ranges' if getattr(ctx, 'joints', None) else ''}); screw = scoped-link screws + screws touching "
                      f"a scoped part; others = non-fastener parts on other links within {search:g} mm; overlap = manifold boolean of the screw mesh with the "
                      f"screening solid, split head / shank / tip (last {TIP_MM:g} mm); distance capped at {search:g} mm; head inside own plate = head volume "
                      f"overlap with a same-link part at standing above {HEAD_IN_PLATE_MM3:g} mm3")


def _zones_t(zones, Trel):
    return {k: v.transform(Trel[:3, :4]) for k, v in zones.items()}
