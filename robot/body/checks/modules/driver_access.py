"""driver_access: can a driver reach the head of every screw that touches a scoped part, in the assembled standing state?

For each screw (the screws whose shank or head seat lies in a scoped part or its steel sleeves, as in the connectivity graph)
the tool path is a cylinder of the driver's blade diameter along the screw axis from the head top outward, long enough to
clear the body (data/tools.json: drive and blade per head type and thread, with sources; assumptions are labelled; the head
type from the screw's head standard: screw_plan, else its body definition's spec). Measured against every other
non-hardware part and steel sleeve at standing (coarse screening solids, screwlib.screen): obstruction volume (mm3) over the
whole path and over the first 40 mm the blade and hand need, the share of the path volume it fills, and the first
obstructing part with its distance along the path from the head top. Body shells go on last (the assembly check's
covered_later_by): their obstruction is reported apart (shell_obstruction_mm3, covered_later_by), not in the headline. A screw whose head
cannot be reached shows as a large obstruction at a short distance, not as a verdict. Other screws' heads are not in the path
solids. Parts on another limb than the screw's are left out (posed away or not attached while it is driven). Generalises the arm / roll-flip insertion rays of the earlier per-study scripts and the disc rays of mount_check."""
from __future__ import annotations

import time

import numpy as np

from checks import bodylib as BL
from checks import screwlib as SL
from checks.core import tick
from checks.registry import register

SHELL_ROLE = "body_shell"     # the body shells go on last (the assembly check: covered_later_by): their obstruction is reported apart


def measure_screw(ctx, name, info, T, nodes, WB, box, tdata):
    MC = SL.mount_check()
    p = ctx.body.parts[name]
    g = info["geometry"]
    Ts = info["world"]
    R_, t_ = Ts[:3, :3], Ts[:3, 3]
    head = R_ @ np.asarray(g["_head"]) + t_
    out = -(R_ @ np.asarray(g["_d"]))
    spec = MC.screw_spec(p)          # screw_plan, else the body definition's spec, else the name
    thread = spec.get("thread") or MC.nominal(name).get("thread")
    ht = SL.head_type(name, g, thread, tdata, spec.get("head_standard"))
    tl = SL.tool_for(ht, thread, tdata)
    row = dict(thread=thread, head_standard=spec.get("head_standard"), spec_source=spec.get("spec_source"), head_type=ht,
               head_world_mm=np.round(head, 2).tolist(), out_axis=np.round(out, 3).tolist(), touches=info["touched"])
    if tl is None:
        row["error"] = f"no tool in data/tools.json for {ht} {thread}"
        return row
    size, blade, kind, src = tl
    pc = tdata["path"]
    length = min(pc["max_length_mm"], max(pc["min_length_mm"], SL.exit_distance(box, head, out) + 1.0))
    radius = blade / 2.0
    # candidates: parts whose standing bounds the path comes within the radius of (axis samples every 5 mm)
    ts = np.arange(0.0, length + 5.0, 5.0)
    samp = head[None] + ts[:, None] * out[None]
    cand, other_limb = {}, []
    limb = ctx.body.limb_of()
    own = limb.get(p["owning_link"])
    for n in nodes:
        lo, hi = WB[n]
        if np.any((samp > lo - radius - 0.5).all(1) & (samp < hi + radius + 0.5).all(1)):
            lk = ctx.body.parts[n]["owning_link"]
            if own and limb.get(lk) and limb[lk] != own:
                other_limb.append(n)              # on another limb: posed away or not attached while this screw is driven
                continue
            cand[n] = T[lk]
    solids, shells = {}, {}
    for n, Tn in cand.items():
        mf = SL.screen(ctx, n)[2]           # the coarse screening solid: an unscoped run must not tessellate every supplier STEP at 0.03 mm
        (shells if ctx.body.parts[n].get("role") == SHELL_ROLE else solids)[n] = mf.transform(Tn[:3, :4])
    res = SL.driver_path(head, out, radius, length, solids, pc["start_offset_mm"])
    near = SL.driver_path(head, out, radius, min(length, pc["min_length_mm"]), solids, pc["start_offset_mm"])   # the blade and the hand behind it
    sh = SL.driver_path(head, out, radius, length, shells, pc["start_offset_mm"])
    row.update(obstruction_within_min_length_mm3=near["obstruction_mm3"], first_distance_within_min_length_mm=near["first_distance_mm"],
               drive=kind, drive_size=size, blade_dia_mm=blade, path_length_mm=round(length, 1), tool_source=src,
               candidates=len(cand), other_limb_parts_left_out=sorted(other_limb), **res,
               shell_obstruction_mm3=sh["obstruction_mm3"], shell_first_distance_mm=sh["first_distance_mm"],
               covered_later_by=[b["part"] for b in sh["by_part"]], shell_by_part=sh["by_part"])
    return row


@register("driver_access", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """Driver path along every scoped screw's axis from the head outward at standing: obstruction volume (mm3), first obstructing part and its distance."""
    t0 = time.time()
    tdata = SL.tools()
    T = SL.standing_T(ctx)
    found = SL.screws_of_scope(ctx, T)
    nodes = SL.solid_nodes(ctx)           # body shells too: a cover over a screw is the obstruction that matters most
    WB = {n: SL.world_bounds(ctx, T, n) for n in nodes}
    allb = np.array([SL.world_bounds(ctx, T, n) for n in SL.solid_nodes(ctx)])
    box = (allb[:, 0].min(0), allb[:, 1].max(0))
    rows = {}
    for k, info in sorted(found.items()):
        try:
            rows[k] = measure_screw(ctx, k, info, T, nodes, WB, box, tdata)
        except Exception as e:  # noqa: BLE001
            rows[k] = dict(error=repr(e), touches=info["touched"])
    per, cases = {n: dict(screws={}) for n in ctx.scope}, []
    for k, r in rows.items():
        for n in r["touches"]:
            per[n]["screws"][k] = r
        if r.get("obstruction_mm3", 0) > 0:
            ml = tdata["path"]["min_length_mm"]
            cases.append(dict(part=r["touches"][0], mating_part=r["first_part"], pose="standing", location_mm=r["head_world_mm"],
                              value=r["obstruction_within_min_length_mm3"], unit=f"mm3 in the first {ml:g} mm of the driver path", total_mm3=r["obstruction_mm3"],
                              what=f"screw {k} ({r['head_type']}, {r['drive']} {r['drive_size']}, blade {r['blade_dia_mm']} mm): first obstruction {r['first_distance_mm']} mm from the head, "
                                   f"{r['obstruction_mm3']} mm3 ({r['blocked_share_pct']} % of the path) over {r['path_length_mm']} mm to clear the body"))
    cases.sort(key=lambda c: (-c["value"], -c["total_mm3"]))
    for n, d in per.items():
        v = [r for r in d["screws"].values() if "error" not in r]
        d.update(screws_measured=len(v), max_obstruction_mm3=max([r["obstruction_mm3"] for r in v], default=0.0),
                 max_obstruction_within_min_length_mm3=max([r['obstruction_within_min_length_mm3'] for r in v], default=0.0),
                 min_first_distance_mm=min([r["first_distance_mm"] for r in v if r["first_distance_mm"] is not None], default=None))
    errs = [k for k, r in rows.items() if "error" in r]
    shelled = [r for r in rows.values() if r.get("shell_obstruction_mm3", 0) > 0]
    for n, d in per.items():
        d["max_shell_obstruction_mm3"] = max([r.get("shell_obstruction_mm3", 0.0) for r in d["screws"].values() if "error" not in r], default=0.0)
    tick("driver_access", t0)
    return dict(per_part=per, worst_cases=cases[:20], bad_events=dict(screws=len(rows), screws_with_obstruction_within_tool_length=sum(1 for c in cases if c['value'] > 0), screws_with_obstruction_anywhere_on_path=len(cases),
                screws_with_obstruction_within_10mm=sum(1 for r in rows.values() if r.get("first_distance_mm") is not None and r["first_distance_mm"] < 10.0),
                screws_not_measured=len(errs), screws_whose_path_a_body_shell_covers=len(shelled)),
                evidence=[], screws_not_measured=errs,
                notes=f"cylinder of the tool blade diameter from {tdata['path']['start_offset_mm']} mm above the head top outward, "
                      f"{tdata['path']['min_length_mm']:g} mm to the exit of the body's bounding box (max {tdata['path']['max_length_mm']:g}, a long precision driver blade, assumption); standing state; "
                      f"parts on another limb than the screw's left out (other_limb_parts_left_out); "
                      f"obstruction = sum of exact booleans against each part's coarse screening solid, body shells apart (shell_obstruction_mm3, covered_later_by: they go on last; a voxel-filled open shell can be off in volume, see solid_volume_off_over_5pct); "
                      "other screw heads not in the path; tool data and its sources in data/tools.json (blade diameters are assumptions)")
