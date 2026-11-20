"""assembly: walk an assembly order (data/assembly_order.json) and measure, step by step, whether each part can be inserted
along its direction and each screw reached by its driver, given only what is already assembled at that step.

Order file: assemblies -> {name: {base: [parts present at the start], steps: [...]}}. A step names a `part` (name, glob or
list of them; several parts move together as one rigid group, e.g. a part with the steel sleeves pressed into it) or a `screw`
(name, glob or list), a `direction` (manifest-frame vector, the direction the part travels while it goes in; the part comes
from final position minus t x direction; or "screw axis" = head toward tip of each screw) and `seeded` (true = drafted from the
connection graph, not yet confirmed by the designer) with `derived` (how).
Frame: standing pose (link transforms at standing), which is the manifest frame.

Part step: the group is translated back along -direction in 1 mm samples (0 < t <= max_mm, data defaults) and its overlap volume
(exact boolean of the screening solids) is measured against every part assembled before it. Reported: first_contact_mm (the
largest sampled t with overlap, refined by bisection to 0.01 mm: where the part first touches coming in from outside), the
maximum overlap along the path (mm3) with the part that has it, and the overlap at the final position (seat, t = 0). Thin
obstacles thinner than the sample step can be stepped over; a refinement of the first hit only.
Screw step: driver path (driver_access.measure_screw: tool blade cylinder from the head outward, data/tools.json) against the
parts assembled at that step only; `closed_in_by` lists parts named later in the order that would block the same path.
Also per screw the parts it touches that are not assembled yet at its step. Coverage: parts and screws on the assembly's links
that no step names. Measurements only."""
from __future__ import annotations

import fnmatch
import json
import time

import numpy as np

from checks import bodylib as BL
from checks import screwlib as SL
from checks.core import DATA, tick
from checks.registry import register

ORDER_FILE = DATA / "assembly_order.json"
EPS_MM3 = 1e-3


def load_order(path=None):
    return json.loads((path or ORDER_FILE).read_text())


def names_of(spec):
    return [spec] if isinstance(spec, str) else list(spec)


def resolve(parts, spec):
    """Part names in `parts` matching a name / glob / list of them, in manifest order."""
    pats = names_of(spec)
    return [n for n in parts if any(fnmatch.fnmatchcase(n, p) for p in pats)]


# ------------------------------------------------------------------------------------------------ pure measurement
def _aabb(m):
    b = np.asarray(m.bounding_box() if hasattr(m, "bounding_box") else m.bounds(), float).reshape(-1)
    return b[:3], b[3:]


def _moved(m, d, t):
    return m.translate([float(x) for x in (-t * np.asarray(d))])


def sweep_insertion(moving, direction, obstacles, max_mm=60.0, step_mm=1.0, refine_mm=0.01):
    """Straight insertion of `moving` (manifold, final position) along `direction` against `obstacles` {name: manifold}, same frame.
    The part sits at final position - t x direction for t = 0 .. max_mm. Returns dict(seat_overlap_mm3 (t = 0), max_overlap_mm3,
    max_overlap_at_mm, max_overlap_part, first_contact_mm (None = free path), first_part, blocked_samples, samples, by_part)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    lo, hi = _aabb(moving)
    obs = {}
    for n, o in obstacles.items():
        olo, ohi = _aabb(o)
        # swept box of the mover over 0..max_mm
        s0, s1 = lo - max_mm * np.maximum(d, 0), hi - max_mm * np.minimum(d, 0)
        if np.all(olo <= s1) and np.all(ohi >= s0):
            obs[n] = o

    def overlap(t, only=None):
        mv = _moved(moving, d, t)
        out = {}
        for n, o in (only or obs).items():
            try:
                v = float((mv ^ o).volume())
            except Exception:  # noqa: BLE001
                continue
            if v > EPS_MM3:
                out[n] = v
        return out
    seat = overlap(0.0)
    ts = np.arange(step_mm, max_mm + 1e-9, step_mm)
    hits, by_part, best = [], {}, (0.0, None, None)
    for t in ts:
        ov = overlap(float(t))
        tot = sum(ov.values())
        if tot > 0:
            hits.append((float(t), ov))
            for n, v in ov.items():
                if v > by_part.get(n, (0, 0))[0]:
                    by_part[n] = (v, float(t))
            if tot > best[0]:
                best = (tot, float(t), max(ov, key=ov.get))
    first_t, first_part = None, None
    if hits:
        t_hit, ov = hits[-1]                      # largest sampled t that overlaps
        first_part = max(ov, key=ov.get)
        lo_t, hi_t = t_hit, t_hit + step_mm        # hi_t does not overlap (or is past max_mm)
        while t_hit < max_mm - 1e-9 and hi_t - lo_t > refine_mm:
            mid = 0.5 * (lo_t + hi_t)
            if overlap(mid, {n: obs[n] for n in ov}):
                lo_t = mid
            else:
                hi_t = mid
        first_t = round(0.5 * (lo_t + hi_t), 2) if t_hit < max_mm - 1e-9 else float(max_mm)   # still overlapping at max_mm: the path is not clear within the range
    return dict(seat_overlap_mm3=round(sum(seat.values()), 3), seat_overlap_by_part={k: round(v, 3) for k, v in seat.items()},
                max_overlap_mm3=round(best[0], 3), max_overlap_at_mm=best[1], max_overlap_part=best[2], overlap_at_max_mm3=round(sum(hits[-1][1].values()), 3) if hits and hits[-1][0] >= max_mm - 1e-9 else 0.0,
                first_contact_mm=first_t, first_part=first_part, blocked_samples=len(hits), samples=len(ts),
                by_part=sorted(({"part": n, "max_mm3": round(v, 3), "at_mm": a} for n, (v, a) in by_part.items()), key=lambda r: -r["max_mm3"]))


# ------------------------------------------------------------------------------------------------ the walk
def _world(ctx, T, n):
    ctx.solid(n)
    return ctx.mf[n].transform(T[ctx.body.parts[n]["owning_link"]][:3, :4])


def _screw_info(ctx, T, k, node_names, WB):
    """geometry, world transform and the parts (of node_names) a screw touches (ring points inside their solids, as screws_of_scope)."""
    MC = SL.mount_check()
    p = ctx.body.parts[k]
    g = MC.screw_geometry(BL.load_mesh(p["mesh"]))
    Ts = T[p["owning_link"]]
    pw = SL.ring_points(g, SL.NEAR_MM) @ Ts[:3, :3].T + Ts[:3, 3]
    lo, hi = pw.min(0) - 0.5, pw.max(0) + 0.5
    touched = []
    for n in node_names:
        if np.any(WB[n][0] > hi) or np.any(WB[n][1] < lo):
            continue
        Tn = T[ctx.body.parts[n]["owning_link"]]
        try:
            inside = ctx.solid(n)[0].contains((pw - Tn[:3, 3]) @ Tn[:3, :3])
        except Exception:  # noqa: BLE001
            continue
        if inside.sum() >= 2:
            touched.append(n)
    return dict(geometry=g, touched=touched, world=Ts)


def walk(ctx, order, name, scope=None, max_mm=None, step_mm=None):
    """Measure every step of assembly `name`; returns dict(steps [...], coverage {...})."""
    from checks.modules import driver_access as DA
    asm = order["assemblies"][name]
    dflt = order.get("defaults", {})
    max_mm = max_mm or dflt.get("max_mm", 60.0)
    step_mm = step_mm or dflt.get("step_mm", 1.0)
    parts = ctx.body.parts
    T = SL.standing_T(ctx)
    nodes_all = SL.solid_nodes(ctx)
    WB = {n: SL.world_bounds(ctx, T, n) for n in nodes_all}
    allb = np.array(list(WB.values()))
    box = (allb[:, 0].min(0), allb[:, 1].max(0))
    tdata = SL.tools()
    # resolve every step first (later parts are needed for closed_in_by)
    res = []
    for st in asm["steps"]:
        if "part" in st:
            res.append(("part", [n for n in resolve(parts, st["part"]) if n in nodes_all]))
        else:
            res.append(("screw", [n for n in resolve(parts, st["screw"]) if SL.mount_check().kind(parts[n]) == "screw"]))
    named_parts = [n for k, ns in res if k == "part" for n in ns]
    assembled = [n for n in resolve(parts, asm.get("base", [])) if n in nodes_all]
    named_screws = [n for k, ns in res if k == "screw" for n in ns]
    out = []
    for i, (st, (kind, ns)) in enumerate(zip(asm["steps"], res)):
        t0 = time.time()
        row = dict(step=i + 1, id=st.get("id"), kind=kind, names=ns, direction=st.get("direction"), seeded=bool(st.get("seeded")),
                   derived=st.get("derived"), assembled_before=len(assembled))
        if not ns:
            row["error"] = "step names nothing in this manifest"
            out.append(row); continue
        if kind == "part":
            d = np.asarray(st["direction"], float)
            obst = {n: _world(ctx, T, n) for n in assembled}
            per = {}
            for n in ns:
                per[n] = sweep_insertion(_world(ctx, T, n), d, obst, max_mm, step_mm)
            # a group moves as one: sum its members' overlaps per sample is not available from per-part sweeps, so report the worst member
            worst = max(per.values(), key=lambda r: (r["first_contact_mm"] or 0.0, r["max_overlap_mm3"]))
            row.update(members=per, first_contact_mm=worst["first_contact_mm"], first_part=worst["first_part"],
                       max_overlap_mm3=max(r["max_overlap_mm3"] for r in per.values()), overlap_at_max_mm3=max(r["overlap_at_max_mm3"] for r in per.values()), seat_overlap_mm3=round(sum(r["seat_overlap_mm3"] for r in per.values()), 3),
                       blocked_samples=max(r["blocked_samples"] for r in per.values()), path_mm=max_mm)
            assembled += [n for n in ns if n not in assembled]
        else:
            later = [n for n in named_parts if n not in assembled]
            screws = {}
            for k in ns:
                info = _screw_info(ctx, T, k, nodes_all, WB)
                r = DA.measure_screw(ctx, k, info, T, assembled, WB, box, tdata)
                rl = DA.measure_screw(ctx, k, info, T, assembled + later, WB, box, tdata) if "error" not in r else r
                if "error" not in r:
                    r["closed_in_by"] = sorted({b["part"] for b in rl.get("by_part", [])} - {b["part"] for b in r.get("by_part", [])})
                    r["closed_in_mm3"] = round(rl["obstruction_mm3"] - r["obstruction_mm3"], 3)
                r["touches_not_yet_assembled"] = [n for n in info["touched"] if n not in assembled]
                screws[k] = r
            ok = [r for r in screws.values() if "error" not in r]
            row.update(screws=screws, obstruction_mm3=max([r["obstruction_mm3"] for r in ok], default=0.0),
                       first_distance_mm=min([r["first_distance_mm"] for r in ok if r["first_distance_mm"] is not None], default=None),
                       first_part=next((r["first_part"] for r in ok if r["first_part"]), None),
                       obstruction_within_min_length_mm3=max([r["obstruction_within_min_length_mm3"] for r in ok], default=0.0),
                       closed_in_by=sorted({p for r in ok for p in r.get("closed_in_by", [])}),
                       touches_not_yet_assembled=sorted({p for r in screws.values() for p in r["touches_not_yet_assembled"]}))
        row["seconds"] = round(time.time() - t0, 1)
        out.append(row)
    links = {parts[n]["owning_link"] for n in named_parts}
    un_parts = [n for n in nodes_all if parts[n]["owning_link"] in links and n not in named_parts and n not in assembled]
    un_screws = []
    MC = SL.mount_check()
    for k, p in parts.items():
        if MC.kind(p) == "screw" and p["owning_link"] in links and k not in named_screws:
            un_screws.append(k)
    return dict(steps=out, coverage=dict(links=sorted(links), parts_not_named=un_parts, screws_not_named=un_screws))


def _selected(ctx, order, scope):
    parts = ctx.body.parts
    sel = {}
    for name, asm in order["assemblies"].items():
        named = set(resolve(parts, asm.get("base", [])))
        for st in asm["steps"]:
            named |= set(resolve(parts, st.get("part", []))) | set(resolve(parts, st.get("screw", [])))
        if named & set(scope):
            sel[name] = named
    return sel


@register("assembly", lambda ctx: ORDER_FILE.exists() and bool(ctx.body.parts))
def run(ctx):
    """Assembly order walk (data/assembly_order.json): per step the insertion path of a part (first contact mm, overlap mm3 vs what is assembled) or the driver path of a screw at that step, plus parts the order never names."""
    t0 = time.time()
    order = load_order()
    sel = _selected(ctx, order, ctx.scope)
    per, cases, evidence_rows = {n: dict(steps=[]) for n in ctx.scope}, [], {}
    for name in sel:
        w = walk(ctx, order, name)
        evidence_rows[name] = w
        for r in w["steps"]:
            touched = set(r["names"]) | {p for s in r.get("screws", {}).values() for p in s.get("touches", [])}
            for n in ctx.scope:
                if n in touched:
                    per[n]["steps"].append(dict(assembly=name, **{k: v for k, v in r.items() if k not in ("members", "screws")}))
            if r.get("first_contact_mm"):
                cases.append(dict(part=(r["names"] or [None])[0], mating_part=r["first_part"], pose="standing", location_mm=None,
                                  value=r["max_overlap_mm3"], unit="mm3 max overlap along the insertion path",
                                  what=f"{name} step {r['step']} ({r['id']}): first contact {r['first_contact_mm']} mm from the final position along {r['direction']}"))
            elif r.get("obstruction_mm3", 0) > 0:
                cases.append(dict(part=(r["names"] or [None])[0], mating_part=r["first_part"], pose="standing", location_mm=None,
                                  value=r["obstruction_within_min_length_mm3"], unit="mm3 in the first driver-path length",
                                  what=f"{name} step {r['step']} ({r['id']}): driver path blocked {r['obstruction_mm3']} mm3, first {r['first_distance_mm']} mm from the head; closed in later by {r['closed_in_by']}"))
    cases.sort(key=lambda c: -c["value"])
    steps = [r for w in evidence_rows.values() for r in w["steps"]]
    cov = {k: sorted({x for w in evidence_rows.values() for x in w["coverage"][k]}) for k in ("parts_not_named", "screws_not_named")}
    in_any = set().union(*sel.values()) if sel else set()
    for n in ctx.scope:
        per[n]["named_in_assembly"] = n in in_any
    tick("assembly", t0)
    return dict(per_part=per, worst_cases=cases[:30],
                bad_events=dict(assemblies=len(sel), steps=len(steps), part_steps_with_contact_on_path=sum(1 for r in steps if r.get("first_contact_mm")),
                                screw_steps_with_obstruction=sum(1 for r in steps if r.get("obstruction_mm3", 0) > 0),
                                screw_steps_closed_in_by_later_part=sum(1 for r in steps if r.get("closed_in_by")),
                                seeded_steps=sum(1 for r in steps if r.get("seeded")),
                                parts_not_named=len(cov["parts_not_named"]), screws_not_named=len(cov["screws_not_named"]),
                                scoped_parts_in_no_assembly=sum(1 for n in ctx.scope if not per[n]["named_in_assembly"])),
                evidence=[], assemblies=evidence_rows, coverage=cov,
                notes="straight insertion along each step's direction in 1 mm samples to 60 mm (data/assembly_order.json defaults), exact booleans of screening solids "
                      "against the parts assembled earlier in the order (plus its base); screw steps = driver_access path against the assembled parts only; standing pose; "
                      "orders are seeded drafts until the designer confirms them")
