"""screws: the screw connections of the scoped parts, measured by robot/body/fastening/mount_check.py (its functions, not a copy).

Screws that touch a scoped part or the sleeve sets pressed into it (screwlib.screws_of_scope, checks.relations) are analysed by mount_check.run_screws in the
manifest's neutral frame (screws, standard parts, sleeves and printed parts share it; the screw geometry does not depend on the
pose). Per screw: thread engagement against the official reference of the standard part's tapped hole (shortfall mm), hole
centre offset from the screw axis (terminal and clamped holes), clamp-stack gaps (head seat and each layer pair, mm, < 0 =
interpenetration), the sleeve under the head (bearing ring mm, sleeve offset). Worst cases carry the head position and the
mating part. Reverse view restricted to the scope: sleeves of scoped parts without a screw, defined mount holes of scoped
standard parts and which screws use them. The driver / insertion rays of mount_check are left to driver_access."""
from __future__ import annotations

import re
import sys
import time
import types

import numpy as np

from checks import screwlib as SL
from checks.core import WORK, ROOT, tick
from checks.registry import register

REF = None


def _gap(c):
    return c["min_gap_mm"]


def screw_summary(r, touched):
    """The numbers of one mount_check row, flattened to the measurements this check reports."""
    t = r.get("terminal") or {}
    ref = t.get("reference_engagement_mm")
    holes = [h["axis_offset_mm"] for h in r.get("clamped_holes", []) if h["axis_offset_mm"] is not None]
    offs = ([t["axis_offset_mm"]] if t.get("axis_offset_mm") is not None else []) + holes
    seat = r.get("head_seat")
    layers = [dict(upper=c["upper"], lower=c["lower"], min_gap_mm=c["min_gap_mm"], max_gap_mm=c.get("max_gap_mm"), interface=c["interface"],
                   level_gap_mm=c.get("level_gap_mm"), bears=c.get("bears"))
              for c in r.get("contacts", [])]
    return dict(thread=r.get("thread"), nominal_length_mm=r.get("nominal_length_mm"), head_standard=r.get("head_standard"),
                spec_source=r.get("spec_source"), touches=touched, head_mm=r["head_mm"], axis=r["axis"], length_mm=r["length_mm"],
                terminal=dict(kind=t.get("kind"), part=t.get("part"), engagement_mm=t.get("engagement_mm"), reference_engagement_mm=ref,
                              shortfall_mm=None if ref is None or t.get("engagement_mm") is None else round(ref - t["engagement_mm"], 3),
                              axis_offset_mm=t.get("axis_offset_mm"), angle_deg=t.get("angle_deg")) if t else None,
                hole_offset_max_mm=round(max(offs), 3) if offs else None, clamped_holes=r.get("clamped_holes", []),
                head_seat_gap_mm=None if seat is None else seat["min_gap_mm"], head_seat_part=None if seat is None else seat["lower"],
                layers=layers, sleeves=r.get("sleeves", []),
                disc_ray_driver_blocked_pct_shells_off=(r.get("driver_path_without_shells") or {}).get("blocked_pct"),
                findings=[f for f in r.get("findings", []) if not f["kind"].endswith("path_blocked_pct")])


CHUNK = 30    # screws per mount_check process, as mount_check.run does for the whole manifest


def run_mount_check(MC, manifest, only, n, out):
    """mount_check.run_screws on the selected screws; above CHUNK screws in chunks of CHUNK, each in its own process.
    One process for all of them holds the full mesh of every part within driver reach of any screw (the whole body, the
    587k-face power board and the shells among them) plus two embree scenes over them: an unscoped run (about 250 screws)
    passed 6 GB within 50 s and did not finish in 1800 s (T-c4sy9d, 2026-11-20). Embree also leaks per scene."""
    import json
    import subprocess
    mp = MC.Path(manifest).resolve()
    if n <= CHUNK:
        return MC.run_screws(mp, only, out, None)
    rows, secs = [], 0.0
    for a in range(0, n, CHUNK):
        po = out.parent / f"screws-mount-check-part-{a}.json"
        subprocess.run([sys.executable, "-B", str(MC.__file__), "--manifest", str(mp), "--only", only, "--part", f"{a}:{a + CHUNK}",
                        "--out", str(po)], check=True)
        r = json.loads(po.read_text()); po.unlink()
        rows += r["rows"]; secs += r.get("seconds", 0.0)
        print(f"  mount_check {min(a + CHUNK, n)}/{n} screws", flush=True)
    body = MC.F.Body(json.loads(mp.read_text()))
    rep = dict(manifest=str(mp.relative_to(MC.ROOT)), screws=len(rows), seconds=round(secs, 1), chunks=-(-n // CHUNK),
               rows=rows, by_connector=MC.by_connector(rows, body))
    out.write_text(json.dumps(rep, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return rep


@register("screws", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """Screw connections of the scoped parts via mount_check: engagement vs the standard part's reference, hole offset, clamp gaps, sleeve under the head, reverse view (unused sleeves / mount holes) in scope."""
    t0 = time.time()
    MC = SL.mount_check()
    found = SL.screws_of_scope(ctx)
    tick("screws touching scope", t0)
    if not found:
        return dict(per_part={n: dict(screws={}) for n in ctx.scope}, worst_cases=[], bad_events=dict(screws=0), evidence=[], notes="no screw touches the scope")
    only = "^(?:" + "|".join(re.escape(k) for k in sorted(found)) + ")$"
    out = (ctx.out_dir or WORK / "screws") / "screws-mount-check.json"     # per label: two runs at once never share the file
    t1 = time.time()
    rep = run_mount_check(MC, ctx.manifest, only, len(found), out)
    tick("mount_check.run_screws", t1)
    rows = {r["name"]: r for r in rep["rows"]}
    per, cat = {n: dict(screws={}) for n in ctx.scope}, {k: [] for k in ("engagement", "offset", "gap", "bearing", "other")}
    for k, r in rows.items():
        if "error" in r:
            continue
        mine = [n for n in found[k]["touched"] if n in r["stack"] or any(s in r["stack"] for s in ctx.rel.sleeve_sets_of(n))
                or (r.get("terminal") or {}).get("part") == n]
        if not mine:
            continue
        s = screw_summary(r, mine)
        head = np.round(found[k]["world"][:3, :3] @ np.asarray(r["head_mm"]) + found[k]["world"][:3, 3], 2).tolist()
        for n in mine:
            per[n]["screws"][k] = s
        part = mine[0]
        t = s["terminal"]
        mate = t["part"] if t else None
        if t and t["shortfall_mm"] is not None:
            cat["engagement"].append(dict(part=part, mating_part=mate, pose="standing", location_mm=head, value=max(0.0, t["shortfall_mm"]), unit="mm engagement shortfall",
                                          what=f"screw {k}: engaged {t['engagement_mm']} mm of the reference {t['reference_engagement_mm']} mm ({t['kind']})"))
        if s["hole_offset_max_mm"] is not None:
            cat["offset"].append(dict(part=part, mating_part=mate, pose="standing", location_mm=head, value=s["hole_offset_max_mm"], unit="mm hole centre offset from the screw axis",
                                      what=f"screw {k}: terminal {t['axis_offset_mm'] if t else None}, clamped " + ", ".join(f"{h['part']} {h['axis_offset_mm']}" for h in s["clamped_holes"])))
        gaps = ([(s["head_seat_gap_mm"], f"head seat on {s['head_seat_part']}")] if s["head_seat_gap_mm"] is not None else []) + \
               [(c["min_gap_mm"], f"{c['upper']} / {c['lower']}" + ("" if c["interface"] else " (overlap)")) for c in s["layers"]
                if not c["interface"] or c["bears"] is not False]       # a sleeve end over a counterbore while its part bears: not the stack's gap
        if gaps:
            v, w = max(gaps, key=lambda g: abs(g[0]))
            cat["gap"].append(dict(part=part, mating_part=mate, pose="standing", location_mm=head, value=abs(v), unit="mm clamp-stack gap (signed in the text)", what=f"screw {k}: {v:+g} mm at {w}"))
        for sl in s["sleeves"]:
            if sl.get("head_bearing_ring_mm") is not None:
                cat["bearing"].append(dict(part=sl["printed_part"], mating_part=sl["sleeve"], pose="standing", location_mm=head, value=sl["head_bearing_ring_mm"],
                                           unit="mm head bearing ring on the sleeve (low is bad)", what=f"screw {k}: head on sleeve {sl['sleeve']}"))
            elif sl["sleeve"] is None:
                cat["other"].append(dict(part=sl["printed_part"], mating_part=None, pose="standing", location_mm=head, value=sl["clamped_len_mm"],
                                         unit="mm clamped printed hole without a sleeve", what=f"screw {k}"))
    cat["engagement"].sort(key=lambda c: -c["value"]); cat["offset"].sort(key=lambda c: -c["value"]); cat["gap"].sort(key=lambda c: -c["value"])
    cat["bearing"].sort(key=lambda c: c["value"]); cat["other"].sort(key=lambda c: -c["value"])
    cases = []
    for i in range(5):                       # worst of each measure first, so the summary shows every measure
        for kk in ("engagement", "offset", "gap", "bearing", "other"):
            if i < len(cat[kk]):
                cases.append(cat[kk][i])
    # reverse view in scope: sleeves of scoped parts, defined holes of scoped standard parts
    import fit_check as F
    import json
    man = json.loads(open(ctx.manifest).read())
    fbody = F.Body(man)
    iface = MC.I.instances(fbody, man)
    scoped = set(ctx.scope)
    owner = {s: n for n in ctx.scope for s in ctx.rel.sleeve_sets_of(n)}        # sleeve set -> the scoped part it is pressed into
    sc_parts = {s: fbody.parts[s] for s in owner if s in fbody.parts}
    shim = types.SimpleNamespace(parts=sc_parts)
    iface_scope = dict(iface, holes=[h for h in iface["holes"] if h["part"] in scoped])
    good = [r for r in rep["rows"] if "error" not in r]
    cache = {}
    rv = MC.reverse_view(shim, good, iface_scope, lambda n: cache.setdefault(n, fbody.mesh(n)))
    for n in ctx.scope:
        per[n]["reverse"] = dict(sleeves=[s for s in rv["sleeves"] if owner.get(s["part"]) == n],
                                 interface_faces=[f for f in rv["interface_faces"] if f["part"] == n])
    for s in rv["sleeves"]:
        if s["nearest_screw"] is None:
            cases.append(dict(part=owner[s["part"]], mating_part=None, pose=None, location_mm=s["centre_mm"], value=s["length_mm"],
                              unit="mm steel sleeve with no screw through it", what=f"sleeve {s['index']} od {s['od_mm']} / id {s['id_mm']} mm"))
    unused_faces = [f for f in rv["interface_faces"] if f["used"] < f["holes"]]
    errs = [k for k, r in rows.items() if "error" in r]
    mine_rows = [r for k, r in rows.items() if "error" not in r and any(k in p["screws"] for p in per.values())]
    no_spec = sorted(r["name"] for r in mine_rows if not r.get("spec_source"))          # thread / length declared nowhere
    unknown = sorted({c for r in mine_rows for c, x in r["parts"].items() if x.get("hardware") == "unknown"})   # hardware of no known kind
    allf = [f for s in (x for p in per.values() for x in p["screws"].values()) for f in s["findings"]]
    tick("screws", t0)
    return dict(per_part=per, worst_cases=cases[:30], evidence=[],
                bad_events=dict(screws=len({k for p in per.values() for k in p['screws']}), screws_with_engagement_shortfall=sum(1 for c in cat["engagement"] if c["value"] > 1e-6),
                                screws_with_hole_offset_over_0p1mm=sum(1 for c in cat["offset"] if c["value"] > 0.1),
                                screws_with_gap_over_0p05mm=sum(1 for c in cat["gap"] if c["value"] > 0.05),
                                sleeves_without_screw=sum(1 for s in rv["sleeves"] if s["nearest_screw"] is None), mount_hole_faces_with_unused_holes=len(unused_faces),
                                screws_not_analysed=len(errs), screws_without_spec=len(no_spec), hardware_kind_unknown=len(unknown)),
                screws_not_analysed=errs, screws_without_spec=no_spec, hardware_kind_unknown=unknown,
                notes="mount_check.run_screws on the screws that touch the scope (ring points r_shank + 0.8 mm inside the part or its sleeve); per-screw numbers are the "
                      "full mount_check numbers (neutral frame, same functions); engagement shortfall = reference - engaged (>0 = short); offsets are screw axis vs hole centre; "
                      "gaps signed (< 0 interpenetration); reverse view restricted to scoped parts and their sleeves; driver / insertion access is the driver_access check; "
                      "screw kind, thread, length and head standard from the body definition / screw_plan (screws_without_spec: declared nowhere; hardware_kind_unknown: "
                      "hardware in a screw's stack whose kind neither a definition, a screw_plan nor its shape says: declare them)")
