"""Draft a seeded assembly order for one chain from the connectivity graph (checks/modules/connectivity.py) and write it into
data/assembly_order.json (merging with the assemblies already there). Every step is marked seeded with how it was derived; the
designer confirms or reorders them by editing the file. Run:
  PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/assembly_seed.py --name right_leg --links R_yaw_link,R_pitch_link,R_roll_link,R_shin,R_ankle_pitch,R_foot --base lower__R_yaw_servo_case_reference
Order: parts by their fastened distance from the base (Dijkstra on the connectivity edges, contact edges cost 10, ties by name),
each printed part together with its steel sleeves; after a part the screws whose touched chain parts are all in place, grouped
by name stem. Part direction: the mean screw direction of the screws that join it to what is placed (head to tip when the
part carries the head, else reversed), else toward the centroid of its placed neighbours snapped to the nearest axis."""
from __future__ import annotations

import argparse
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import bodylib as BL  # noqa: E402
from checks import screwlib as SL  # noqa: E402
from checks.core import DATA, ROOT  # noqa: E402
from checks.ctx import Ctx  # noqa: E402
from checks.modules import connectivity as CN  # noqa: E402
from checks.modules.assembly import ORDER_FILE  # noqa: E402

SKIP_ROLES = {"adhesive_allocation", "candidate_sensor", "produced_backing", "cut_stock", "produced_holder"}


def snap(v):
    i = int(np.argmax(np.abs(v)))
    out = np.zeros(3); out[i] = np.sign(v[i]); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--links", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--manifest")
    a = ap.parse_args()
    from checks.core import load_body  # noqa: F401
    man = Path(a.manifest) if a.manifest else ROOT / json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
    ctx = Ctx(man, [])
    parts = ctx.body.parts
    links = set(a.links.split(","))
    nodes, edges = CN.build_graph(ctx)
    base = a.base.split(",")
    core = (ctx._settings.get("_core") or {}).get("part")
    if core:
        base.append(core)
    chain = [n for n in nodes if (parts[n]["owning_link"] in links and parts[n].get("role") not in SKIP_ROLES and not n.endswith("_steel_sleeves")) and n not in base]
    adj = {}
    for (x, y), ks in edges.items():
        c = min(CN.COST[k] for k in ks)
        adj.setdefault(x, []).append((y, c, ks)); adj.setdefault(y, []).append((x, c, ks))
    dist = {b: 0.0 for b in base}
    pq = [(0.0, b) for b in base]; heapq.heapify(pq)
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, c, _ in adj.get(u, []):
            if d + c < dist.get(v, 1e9):
                dist[v] = d + c; heapq.heappush(pq, (d + c, v))
    order = sorted([n for n in chain if n in dist], key=lambda n: (dist[n], n))
    unreached = [n for n in chain if n not in dist]
    T = SL.standing_T(ctx)
    MC = SL.mount_check()
    # screws: geometry + touched parts (graph screw edges)
    screw_parts = {}
    for (x, y), ks in edges.items():
        for k in ks.get("screw", []):
            screw_parts.setdefault(k, set()).update((x, y))
    sdir = {}
    for k in screw_parts:
        g = MC.screw_geometry(BL.load_mesh(parts[k]["mesh"]))
        Ts = T[parts[k]["owning_link"]]
        sdir[k] = (Ts[:3, :3] @ np.asarray(g["_d"]), Ts[:3, :3] @ np.asarray(g["_head"]) + Ts[:3, 3])
    import trimesh

    def centroid(n):
        Tn = T[parts[n]["owning_link"]]
        return ctx.solid(n)[0].bounds.mean(0) @ Tn[:3, :3].T + Tn[:3, 3]

    def carries_head(n, k):
        """Does part n hold the screw head (closest of the screw's touched parts to the point just under the head)?"""
        d, h = sdir[k]
        pt = h + 1.0 * d
        best, bn = 1e9, None
        for m in screw_parts[k]:
            Tm = T[parts[m]["owning_link"]]
            loc = (pt - Tm[:3, 3]) @ Tm[:3, :3]
            try:
                dd = abs(trimesh.proximity.closest_point(ctx.solid(m)[0], loc[None])[1][0])
            except Exception:  # noqa: BLE001
                dd = np.linalg.norm(centroid(m) - pt)
            if dd < best:
                best, bn = dd, m
        return bn == n
    steps, placed, done_screws = [], set(base), set()
    chain_set = set(order) | set(base)
    for n in order:
        mine = [k for k, ps in screw_parts.items() if n in ps and (ps - {n}) & placed and k not in done_screws]
        if mine:
            vs = [sdir[k][0] * (1 if carries_head(n, k) else -1) for k in mine]
            v = np.mean(vs, 0)
            v = (v if np.linalg.norm(v) > 0.2 else vs[0]) / np.linalg.norm(v if np.linalg.norm(v) > 0.2 else vs[0])   # opposite screws cancel: first screw's direction
            how = f"mean screw direction of {len(mine)} screw(s) joining it to placed parts ({', '.join(sorted(set(re.sub(r'_\\d+$', '', k) for k in mine)))}), head to tip when the part carries the head, else reversed"
        else:
            nb = [m for m, _, _ in adj.get(n, []) if m in placed]
            tgt = np.mean([centroid(m) for m in nb], 0) if nb else centroid(n)
            v = snap(tgt - centroid(n)) if nb else np.array([0, 0, -1.0])
            how = f"no screw to placed parts: toward the centroid of placed neighbours ({', '.join(nb[:3]) or 'none'}), snapped to the nearest axis"
        s = f"{n}_steel_sleeves"
        grp = [n] + ([s] if s in parts else [])
        steps.append(dict(id=f"p{len(steps) + 1:02d}", part=grp[0] if len(grp) == 1 else grp, direction=[round(float(x), 3) for x in v], seeded=True,
                          derived=f"graph order, fastened distance {dist[n]:g} from the base; direction: {how}"))
        placed.update(grp)
        ready = sorted(k for k, ps in screw_parts.items() if k not in done_screws and (ps & chain_set) and (ps & chain_set) <= placed and parts[k]["owning_link"] in links | {parts[n]["owning_link"]})
        stems = {}
        for k in ready:
            stems.setdefault(re.sub(r"_\d+(_\d+)?$", "", k), []).append(k)
        for stem, ks in stems.items():
            glob = ks if len(ks) > 1 else ks[0]       # explicit names: a stem glob can match another set
            steps.append(dict(id=f"s{len(steps) + 1:02d}", screw=glob, direction="screw axis", seeded=True,
                              derived=f"{len(ks)} screw(s) whose touched chain parts are all placed after {n}; driver path vs placed parts"))
            done_screws.update(ks)
    asm = dict(doc=f"Seeded draft from the connection graph: parts of links {sorted(links)} ordered by fastened distance from the base; unreached by the graph: {unreached}. Designer: confirm or reorder, delete `seeded` once confirmed.",
               base=base, steps=steps)
    cur = json.loads(ORDER_FILE.read_text()) if ORDER_FILE.exists() else dict(
        _doc="Assembly order per subassembly, walked by the assembly check (docs/operations/working-rules.md: editing). Step: part (name / glob / list = one rigid group) or screw (name / glob / list), direction (manifest-frame vector the part travels while it goes in; or 'screw axis'), seeded (drafted, not confirmed), derived (how).",
        defaults=dict(max_mm=60.0, step_mm=1.0), assemblies={})
    cur["assemblies"][a.name] = asm
    ORDER_FILE.write_text(json.dumps(cur, indent=1) + "\n")
    print(f"{a.name}: {len(steps)} steps, {len(order)} parts, {len(done_screws)} screws, unreached {unreached}, screws never placed {sorted(set(screw_parts) - done_screws)[:10]}")


main()
