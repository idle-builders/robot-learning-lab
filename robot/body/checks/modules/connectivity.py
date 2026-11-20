"""connectivity: the connection graph of the whole manifest, so a floating shell, a forgotten bracket or a missing screw
shows up (owner 2026-11-19: no part unconnected; every part linked through the assembly chain to the core).

Nodes: every non-hardware part. Edges:
  screw     a screw passes through or bears on both parts: points on a ring just outside its shank (r_shank + RING_MM)
            along its length, and under its head, lie inside each part's solid
  sleeve    <part>_steel_sleeves is pressed into <part>
  joint     for each joint, the servo case / motor parts and the horn / output parts of its two links within 5 mm
  contact   the two parts are within CONTACT_MM at standing (touching, bonded, seated, or just resting)
The core is part_settings _core.part, else the largest printed part on the root link. Reported: parts with no edge,
connected groups that do not reach the core, parts that reach the core only through contact edges, and per scoped part
its shortest chain to the core with the edge kinds (fastened edges preferred). Manifest-wide; the scope selects whose
chains are listed. Measurements and counts only."""
from __future__ import annotations

import heapq
import sys
import time

import numpy as np

from checks import bodylib as BL
from checks.core import ROOT, tick
from checks.registry import register
from checks.screwlib import ring_points

RING_MM = 0.8
CONTACT_MM = 0.2
CASE_ROLES = {"motor", "actuator", "purchased_servo_case_family_reference"}
HORN_ROLES = {"output", "output_horn", "horn", "purchased_servo_output_family_reference"}
COST = dict(screw=1.0, sleeve=1.0, joint=1.0, contact=10.0)    # path search prefers fastened chains


def _corners(p, T=None):
    b = np.array(p["bounds_mm"], float)
    return np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _near(b1, b2, m):
    return not (np.any(b1[0] > b2[1] + m) or np.any(b2[0] > b1[1] + m))


def build_graph(ctx):
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import kind, screw_geometry
    body = ctx.body
    parts = body.parts
    nodes = [n for n, p in parts.items() if kind(p) is None and (not BL.is_fastener(n, p) or p.get("role") == "compression_limit_sleeves")]
    q0 = body.standing
    T = {lk: body.link_T(lk, q0) for lk in {p["owning_link"] for p in parts.values()}}

    def world_bounds(n):
        c = _corners(parts[n]); Tl = T[parts[n]["owning_link"]]
        w = c @ Tl[:3, :3].T + Tl[:3, 3]
        return np.array([w.min(0), w.max(0)])
    WB = {n: world_bounds(n) for n in nodes}
    edges = {}

    def add(a, b, k, via):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        edges.setdefault(key, {}).setdefault(k, []).append(via)
    # sleeves
    for n in nodes:
        s = f"{n}_steel_sleeves"
        if s in parts and s in WB:
            add(n, s, "sleeve", s)
    # screws: ring points in the manifest (neutral) frame of the screw, mapped to world at standing
    t0 = time.time()
    for k, p in parts.items():
        if kind(p) != "screw":
            continue
        try:
            g = screw_geometry(BL.load_mesh(p["mesh"]))
        except Exception:  # noqa: BLE001
            continue
        pts = ring_points(g, RING_MM)
        Ts = T[p["owning_link"]]
        pw = pts @ Ts[:3, :3].T + Ts[:3, 3]
        lo, hi = pw.min(0), pw.max(0)
        hit = []
        for n in nodes:
            if not _near(WB[n], np.array([lo, hi]), 0.5):
                continue
            Tn = T[parts[n]["owning_link"]]
            loc = (pw - Tn[:3, 3]) @ Tn[:3, :3]
            m = ctx.solid(n)[0]
            try:
                inside = m.contains(loc)
            except Exception:  # noqa: BLE001
                continue
            if inside.sum() >= 2:
                hit.append(n)
        for i, a in enumerate(hit):
            for b in hit[i + 1:]:
                add(a, b, "screw", k)
    tick("connectivity screws", t0)
    # joints: case / horn parts of the two links of each joint
    joints = [(jj.get("parent"), jj.get("child"), jj.get("name")) for jj in json_joints(ctx)]
    for par, chi, name in joints:
        cands = [n for n in nodes if parts[n]["owning_link"] in (par, chi)]
        cases = [n for n in cands if parts[n].get("role") in CASE_ROLES]
        horns = [n for n in cands if parts[n].get("role") in HORN_ROLES]
        for a in cases:
            for b in horns:
                if parts[a]["owning_link"] != parts[b]["owning_link"] and _near(WB[a], WB[b], 5.0):
                    add(a, b, "joint", name)
    # contact at standing (manifold min_gap on the screening solids)
    t0 = time.time()
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if not _near(WB[a], WB[b], CONTACT_MM):
                continue
            ma, mb = ctx.solid(a), ctx.solid(b)
            Ta, Tb = T[parts[a]["owning_link"]], T[parts[b]["owning_link"]]
            try:
                fa, fb = ctx.mf[a].transform(Ta[:3, :4]), ctx.mf[b].transform(Tb[:3, :4])
                dist = 0.0 if (fa ^ fb).volume() > 1e-3 else fa.min_gap(fb, CONTACT_MM + 0.01)
            except Exception:  # noqa: BLE001
                continue
            if dist <= CONTACT_MM:
                add(a, b, "contact", round(float(dist), 3))
    tick("connectivity contact", t0)
    return nodes, edges


def json_joints(ctx):
    import json
    man = json.loads(open(ctx.manifest).read())
    return man.get("joints", [])


@register("connectivity", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """Connection graph of the manifest (screws, sleeves, joints, contact): unconnected parts, groups not reaching the core, contact-only parts, chain to the core per part."""
    import json
    from checks import process as P
    t0 = time.time()
    nodes, edges = build_graph(ctx)
    parts = ctx.body.parts
    core = (ctx._settings.get("_core") or {}).get("part")
    if not core:
        man = json.loads(open(ctx.manifest).read())
        root = man.get("root_link", "torso")
        pr = [n for n in nodes if parts[n]["owning_link"] == root and parts[n].get("role") in ("printed_structural_carrier", "printed_structural_connector")]
        core = max(pr, key=lambda n: ctx.solid(n)[0].volume) if pr else None
    adj = {n: [] for n in nodes}
    for (a, b), ks in edges.items():
        if a in adj and b in adj:
            c = min(COST[k] for k in ks)
            adj[a].append((b, c, sorted(ks))); adj[b].append((a, c, sorted(ks)))
    # shortest (fastened-preferring) chains from the core
    dist, prev = {core: 0.0}, {core: None}
    pq = [(0.0, core)]
    while pq:
        dd, u = heapq.heappop(pq)
        if dd > dist[u]:
            continue
        for v, c, ks in adj[u]:
            if dd + c < dist.get(v, float("inf")):
                dist[v] = dd + c; prev[v] = (u, ks); heapq.heappush(pq, (dd + c, v))

    def chain(n):
        out = []
        while prev.get(n):
            u, ks = prev[n]; out.append(dict(part=n, via="+".join(ks))); n = u
        return list(reversed(out))
    isolated = sorted(n for n in nodes if not adj[n])
    unreached = sorted(n for n in nodes if n not in dist and adj[n])
    contact_only = sorted(n for n in nodes if n in dist and dist[n] >= COST["contact"])
    # break points: the first contact-only link on each contact-only part's chain from the core (what a screw is missing for)
    breaks = {}
    for n in contact_only:
        prev_part = core
        for c in chain(n):
            if c["via"] == "contact":
                breaks.setdefault((prev_part, c["part"]), set()).add(n)
                break
            prev_part = c["part"]
    breaks = sorted(((a, b, len(v)) for (a, b), v in breaks.items()), key=lambda t: -t[2])
    # groups that do not reach the core
    seen, groups = set(dist), []
    for n in unreached:
        if n in seen:
            continue
        comp, stack = [], [n]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u); comp.append(u); stack += [v for v, _, _ in adj[u]]
        groups.append(sorted(comp))
    per = {}
    for n in ctx.scope:
        if n not in adj:
            continue
        per[n] = dict(printed=P.is_printed(ctx, n), edges={v: ks for v, _, ks in adj[n]}, reaches_core=n in dist,
                      fastened_chain=n in dist and dist[n] < COST["contact"], chain_to_core=chain(n) if n in dist else None)
    cases = [dict(part=n, mating_part=None, pose=None, location_mm=None, value=0.0, unit="connections", what=f"no screw, sleeve, joint or contact ({parts[n].get('role')})")
             for n in isolated]
    cases += [dict(part=g[0], mating_part=None, pose=None, location_mm=None, value=float(len(g)), unit="parts in a group not reaching the core",
                   what=", ".join(g[:6]) + (" ..." if len(g) > 6 else "")) for g in groups]
    cases += [dict(part=b, mating_part=a, pose="standing", location_mm=None, value=float(k), unit="parts beyond a contact-only link",
                   what="first link on the chain to the core held by contact only (no screw, sleeve or joint between the two)") for a, b, k in breaks]
    kinds = {}
    for ks in edges.values():
        for k in ks:
            kinds[k] = kinds.get(k, 0) + 1
    tick("connectivity", t0)
    return dict(per_part=per, core=core, worst_cases=cases, bad_events=dict(parts=len(nodes), isolated_parts=len(isolated), groups_not_reaching_core=len(groups),
                parts_not_reaching_core=len(unreached), parts_reaching_core_by_contact_only=len(contact_only), contact_only_break_points=len(breaks), **{f"edges_{k}": v for k, v in kinds.items()}),
                evidence=[], contact_only_breaks=[dict(from_part=a, to_part=b, parts_beyond=k) for a, b, k in breaks], groups_not_reaching_core=groups, isolated=isolated, contact_only=contact_only,
                notes=f"screw edge = >= 2 of the ring points (r_shank + {RING_MM} mm, 8 x 8 along the shank, 8 under the head) inside the part's solid at standing; "
                      f"contact = within {CONTACT_MM} mm at standing (screening solids); joint = case / motor and horn / output parts of a joint's two links within 5 mm; "
                      f"core = {core}; chains prefer fastened edges (contact costs {COST['contact']:g})")
