"""connectivity: the connection graph of the whole manifest, so a floating shell, a forgotten bracket or a missing screw
shows up (owner 2026-11-19: no part unconnected; every part linked through the assembly chain to the core).

Nodes: every non-hardware part. Edges:
  screw     a screw holds both parts: the part encircles the shank (rings at several radii beyond the shank, so a
            hole with a pressed steel sleeve counts too) or the head bears on it
            (a threaded spacer / standoff that two screws enter joins the parts of both screws)
  sleeve    <part>_steel_sleeves is pressed into <part>
  joint     for each joint, the servo case / motor parts and the horn / output parts of its two links within 5 mm
  joined    the parts are one piece once made: printed together (two nozzles, one print) or bonded, as the part's body
            definition declares it (`joined_to: [{part, how: co_printed | bonded}]` on its parts entry); never from names
  contact   the two parts are within CONTACT_MM at standing (touching, seated, or just resting)
Every part is screened with the coarse neighbour solid (core.screen_solid with no part treated as changed: native STEP at
0.1 mm for the STEP roles, else the part's own closed mesh), whatever the scope. Taking ctx.solid() instead gave every
scoped part the fine 0.03 mm STEP tessellation, and an unscoped run scopes all 190 non-hardware parts: 54 of them had no
0.03 mm cache entry, among them supplier STEPs of 78 / 44 / 26 / 22 MB (power supervisor board, ROCK4D, B0196, B0410)
whose thousands of component solids are each tessellated and unioned; the first one alone ran past 170 s, and the run
did not finish in 2400 s (T-c4sy9d, 2026-11-20). The 19 Nov 27.6 s runs were scoped to one part.
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
from checks import screwlib as SL
from checks.registry import register

RING_RADII_MM = (0.3, 0.8, 1.5, 2.5)   # beyond the shank radius: 0.8 alone falls into the gap of a pressed steel sleeve
STATION_MM = 0.5                        # ring spacing along the shank
ENCIRCLE = 5                            # of 8 ring angles inside the part at one station and radius: the part surrounds the shank
HEAD_BEAR = 3                           # of 8 points just under the head inside the part: the head bears on it
CONTACT_MM = 0.2
CASE_ROLES = {"motor", "actuator", "purchased_servo_case_family_reference"}
HORN_ROLES = {"output", "output_horn", "horn", "purchased_servo_output_family_reference"}
COST = dict(screw=1.0, sleeve=1.0, joint=1.0, joined=1.0, contact=10.0)    # path search prefers fastened chains


def _corners(p, T=None):
    b = np.array(p["bounds_mm"], float)
    return np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])


def _near(b1, b2, m):
    return not (np.any(b1[0] > b2[1] + m) or np.any(b2[0] > b1[1] + m))


def screw_probe_points(g):
    """Probe points of a screw (mount_check.screw_geometry) in its own frame: rings of 8 around the shank every
    STATION_MM along it (a 1.6 mm board between coarser stations was missed) at the RING_RADII_MM radii beyond the
    shank, and around the head at mid-height 0.4 mm beyond its radius (a countersink or counterbore that holds it);
    then rings of 8 under the head (0.3 and 0.8 mm below its seat: some screw models stand 0.5 mm proud of the part
    they clamp) at half-way and 90 % between shank and head radius.
    Returns (points, number of ring points)."""
    d, h = np.asarray(g["_d"], float), np.asarray(g["_head"], float)
    u = np.cross(d, [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]); u /= np.linalg.norm(u); w = np.cross(d, u)
    ang = np.radians(np.arange(0, 360, 45))
    ring = np.array([np.cos(a) * u + np.sin(a) * w for a in ang])
    rs, rh, hh = g["r_shank_mm"], g["r_head_mm"], g["head_height_mm"]
    ts = np.linspace(hh + 0.2, g["length_mm"] - 0.2, max(2, int(np.ceil((g["length_mm"] - hh - 0.4) / STATION_MM)) + 1))
    shank = [h + t * d + (rs + dr) * r for t in ts for dr in RING_RADII_MM for r in ring]
    shank += [h + 0.5 * hh * d + (rh + 0.4) * r for r in ring]
    head = [h + (hh + dz) * d + (rs + f * max(0.3, rh - rs)) * r for dz in (0.3, 0.8) for f in (0.5, 0.9) for r in ring]
    return np.array(shank + head), len(shank)


def screw_from_bounds(p, nominal):
    """Screw geometry for a screw row without a usable mesh (L/R_wrist_M3x8_1..4 carry bounds only): the axis along the
    longest extent of its bounds through their centre, shank radius from the thread in its name, no head (rings only)."""
    th = nominal(p["name"]).get("thread", "")
    try:
        rs = float(th[1:]) / 2.0 if th.upper().startswith("M") else None
    except ValueError:
        rs = None
    if not rs or not p.get("bounds_mm"):
        return None
    b = np.array(p["bounds_mm"], float); ext = b[1] - b[0]; i = int(np.argmax(ext))
    d = np.eye(3)[i]; h = b.mean(0) - 0.5 * ext[i] * d
    return {"_d": d, "_head": h, "head_height_mm": 0.0, "length_mm": float(ext[i]), "r_shank_mm": rs, "r_head_mm": rs}


def solid(ctx, n):
    """(probe mesh, how, manifold) of part n: the coarse screening solid shared with the screw checks (screwlib.screen)."""
    return SL.screen(ctx, n)


def build_graph(ctx):
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    from mount_check import kind, nominal, screw_geometry
    body = ctx.body
    screws_from_bounds, screws_unprobed = ctx.__dict__.setdefault("_conn_screws_from_bounds", []), ctx.__dict__.setdefault("_conn_screws_unprobed", [])
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
    # screws: rings around the shank at several radii, and under the head, in the screw's manifest (neutral) frame,
    # mapped to world at standing; a part is held by the screw when it encircles the shank or bears the head
    t0 = time.time()
    # checks.bodylib.Body keeps only rows with a mesh: screw rows without one (L/R_wrist_M3x8_1..4) come from the raw manifest
    raw = {p["name"]: p for p in json_manifest(ctx).get("parts", []) if p.get("name") not in parts and p.get("owning_link") in T}
    # threaded standoffs / spacers are hardware, not nodes: a spacer that two screws enter joins the parts of both
    # (B0196 / B0410 camera: camera screw into an M2 F-F standoff, plate screw into its other end)
    spacers = [n for n, p in parts.items() if kind(p) == "spacer"]
    for n in spacers:
        WB[n] = world_bounds(n)
    via_spacer = {}
    for k, p in list(parts.items()) + list(raw.items()):
        # a screw named without its length (fan_M3_11_430) is "other_hardware" to mount_check.kind: a thread in the name
        # and a screw-shaped mesh make it a screw here
        if kind(p) != "screw" and not (kind(p) == "other_hardware" and nominal(k).get("thread")):
            continue
        try:
            g = screw_geometry(BL.load_mesh(p["mesh"]))
        except Exception:  # noqa: BLE001
            g = screw_from_bounds(p, nominal)
            if g is None:
                screws_unprobed.append(k)
                continue
            screws_from_bounds.append(k)
        pts, shank_n = screw_probe_points(g)
        Ts = T[p["owning_link"]]
        pw = pts @ Ts[:3, :3].T + Ts[:3, 3]
        lo, hi = pw.min(0), pw.max(0)
        hit = []
        for n in nodes + spacers:
            if not _near(WB[n], np.array([lo, hi]), 0.5):
                continue
            Tn = T[parts[n]["owning_link"]]
            loc = (pw - Tn[:3, 3]) @ Tn[:3, :3]
            m = solid(ctx, n)[0]
            try:
                inside = m.contains(loc)
            except Exception:  # noqa: BLE001
                continue
            rings = inside[:shank_n].reshape(-1, 8).sum(1)          # per ring (station x radius, head): angles inside, of 8
            head = inside[shank_n:].reshape(-1, 8).sum(1)
            if rings.max(initial=0) >= ENCIRCLE or head.max(initial=0) >= HEAD_BEAR:
                hit.append(n)
        for sp in [n for n in hit if n in spacers]:
            via_spacer.setdefault(sp, []).append((k, [n for n in hit if n not in spacers]))
        hit = [n for n in hit if n not in spacers]
        for i, a in enumerate(hit):
            for b in hit[i + 1:]:
                add(a, b, "screw", k)
    for sp, sc in via_spacer.items():
        if len(sc) < 2:
            continue
        for i, (ka, ha) in enumerate(sc):
            for kb, hb in sc[i + 1:]:
                for a in ha:
                    for b in hb:
                        add(a, b, "screw", f"{ka} + {sp} + {kb}")
    ctx.__dict__["_conn_spacer_joins"] = {sp: [k for k, _ in sc] for sp, sc in via_spacer.items() if len(sc) >= 2}
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
    # joined: co-printed / bonded pairs as the body definitions declare them
    for a, b, how in joined_pairs(nodes):
        add(a, b, "joined", how)
    # contact at standing (manifold min_gap on the screening solids)
    t0 = time.time()
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if not _near(WB[a], WB[b], CONTACT_MM):
                continue
            Ta, Tb = T[parts[a]["owning_link"]], T[parts[b]["owning_link"]]
            try:
                fa, fb = solid(ctx, a)[2].transform(Ta[:3, :4]), solid(ctx, b)[2].transform(Tb[:3, :4])
                dist = 0.0 if (fa ^ fb).volume() > 1e-3 else fa.min_gap(fb, CONTACT_MM + 0.01)
            except Exception:  # noqa: BLE001
                continue
            if dist <= CONTACT_MM:
                add(a, b, "contact", round(float(dist), 3))
    tick("connectivity contact", t0)
    return nodes, edges


def joined_pairs(nodes):
    """[(part, other part, how)] from the `joined_to` lists of the body definitions' parts entries (robot/**/<builder>.bom.yaml;
    `part` a name or glob of other nodes, `how` co_printed / bonded). No definitions readable: none."""
    import fnmatch
    try:
        sys.path.insert(0, str(ROOT / "robot/body/bom"))
        import definitions as DEF
        claims = DEF.load().claims(list(nodes))
    except Exception:  # noqa: BLE001
        return []
    out = []
    for n, e in claims.items():
        for j in e.get("joined_to") or []:
            for m in nodes:
                if m != n and fnmatch.fnmatchcase(m, j["part"]):
                    out.append((n, m, j.get("how", "joined")))
    return out


def json_manifest(ctx):
    import json
    if "_conn_manifest" not in ctx.__dict__:
        man = json.loads(open(ctx.manifest).read())
        if isinstance(man.get("parts"), dict):
            man["parts"] = [{"name": k, **v} for k, v in man["parts"].items()]
        ctx.__dict__["_conn_manifest"] = man
    return ctx.__dict__["_conn_manifest"]


def json_joints(ctx):
    return json_manifest(ctx).get("joints", [])


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
        core = max(pr, key=lambda n: solid(ctx, n)[0].volume) if pr else None
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
                   what="first link on the chain to the core held by contact only (no screw, sleeve, joint or declared joining between the two)") for a, b, k in breaks]
    kinds = {}
    for ks in edges.values():
        for k in ks:
            kinds[k] = kinds.get(k, 0) + 1
    tick("connectivity", t0)
    hows = {}
    for n, (_, h, _) in ctx.__dict__.get("_conn_solids", {}).items():
        hows.setdefault(h, []).append(n)
    return dict(spacers_joining_screws=ctx.__dict__.get("_conn_spacer_joins", {}), screws_probed_from_bounds=ctx.__dict__.get("_conn_screws_from_bounds", []), screws_not_probed=ctx.__dict__.get("_conn_screws_unprobed", []),
                solids_by_kind={h: len(v) for h, v in hows.items()}, solids_approximate={h: sorted(v) for h, v in hows.items() if "voxel" in h or "hull" in h or "approximate" in h},
                per_part=per, core=core, worst_cases=cases, bad_events=dict(parts=len(nodes), isolated_parts=len(isolated), groups_not_reaching_core=len(groups),
                parts_not_reaching_core=len(unreached), parts_reaching_core_by_contact_only=len(contact_only), contact_only_break_points=len(breaks), **{f"edges_{k}": v for k, v in kinds.items()},
                screws_probed_from_bounds=len(ctx.__dict__.get("_conn_screws_from_bounds", [])), screws_not_probed=len(ctx.__dict__.get("_conn_screws_unprobed", []))),
                evidence=[], contact_only_breaks=[dict(from_part=a, to_part=b, parts_beyond=k) for a, b, k in breaks], groups_not_reaching_core=groups, isolated=isolated, contact_only=contact_only,
                notes=f"screw edge = the part encircles the shank (>= {ENCIRCLE} of 8 ring points inside at one station, every {STATION_MM} mm, rings at r_shank + {RING_RADII_MM} mm) or the head "
                      f"or bears the head (>= {HEAD_BEAR} of 8 points 0.3 / 0.8 mm under the head) at standing; "
                      f"joined = co-printed / bonded as the body definitions declare (joined_to); contact = within {CONTACT_MM} mm at standing (screening solids); joint = case / motor and horn / output parts of a joint's two links within 5 mm; "
                      f"core = {core}; chains prefer fastened edges (contact costs {COST['contact']:g})")
