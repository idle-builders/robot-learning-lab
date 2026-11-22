"""Draft assembly steps for the body definitions (robot/**/<builder>.bom.yaml) from the connectivity graph
(checks/modules/connectivity.py). It never edits a definition: it writes proposals, per owning definition file, into
.cache/work/bom/assembly-proposals/<builder>.proposal.yaml for the owner to review and paste into its `assembly.steps`.

  PYTHONPATH=robot/body .venv/bin/python -B robot/body/checks/assembly_seed.py propose [--names a,b] [--search] [--manifest PATH]

A subassembly's `plan` (in its definition) says what belongs to it: `links`, `include` / `exclude` globs, `roles`,
`exclude_roles`, `last_roles` (default body_shell), `mirror_of` (an empty subassembly starts as the mirror of the other side:
R_ <-> L_ in the names, x negated). Every non-hardware part goes to the first subassembly in build order whose plan takes it.
For what no declared step names it proposes:
- parts by fastened distance from what is present (Dijkstra on the connectivity edges, contact edges cost 10, ties by name),
  each right after the step that places its predecessor on that path (`after`), each printed part with its steel sleeves;
  parts of `last_roles` at the end; parts the graph does not reach before them;
- screws (parts declared as hardware with spec kind screw) right after the first step at which every part they join is in
  place (graph screw edges, else ring points inside the coarse solids; a screw joining nothing after the part nearest its head);
- nuts, washers, spacers and inserts as `with` on the step of the screw whose axis runs through them, else of the part they
  sit in, else of the nearest part; parts without a mesh as `with` on the step of the nearest part.
Part direction: the mean screw direction of the screws that join it to what is placed (head to tip when the part carries the
head, else reversed), else toward the centroid of its placed neighbours snapped to an axis. --search: for blocked seeded
steps, the clearest of the seeded and the six axis directions by the assembly check's own sweep (proposed as changed_steps).
Whether a step is supported is measured by the assembly check and shown in ASSEMBLY.md, not stored."""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import heapq
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks import bodylib as BL  # noqa: E402
from checks import screwlib as SL  # noqa: E402
from checks.core import ROOT  # noqa: E402
from checks.modules import assembly as A  # noqa: E402

SKIP_ROLES = set()          # every part goes in somewhere (owner 2026-11-22: 0 parts without a step)
AXES = [np.array(v, float) for v in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1])]


def snap(v):
    i = int(np.argmax(np.abs(v)))
    out = np.zeros(3); out[i] = np.sign(v[i]); return out


def mirror_name(n):
    return re.sub(r"(^|_)([RL])_", lambda m: m.group(1) + ("L" if m.group(2) == "R" else "R") + "_", n)


def mirror_spec(spec, names):
    """Mirror a part spec (name / glob / list); names that do not exist on the other side are dropped."""
    out = []
    for s in A.names_of(spec):
        ms = mirror_name(s)
        if any(fnmatch.fnmatchcase(n, ms) for n in names):
            out.append(ms)
    return out[0] if len(out) == 1 and isinstance(spec, str) else out


def stem(k):
    return re.sub(r"_\d+(_\d+)?$", "", k)


class Seeder:
    def __init__(self, man_path):
        from checks.ctx import Ctx
        from checks.modules import connectivity as CN
        self.ctx = ctx = Ctx(man_path, [])
        self.parts = ctx.body.parts
        self.raw = {p["name"]: p for p in CN.json_manifest(ctx)["parts"]}
        self.T = SL.standing_T(ctx)
        self.nodes, edges = CN.build_graph(ctx)
        self.node_set = set(self.nodes)
        self.adj = {}
        self.screw_parts = {}
        for (x, y), ks in edges.items():
            c = min(CN.COST[k] for k in ks)
            self.adj.setdefault(x, []).append((y, c)); self.adj.setdefault(y, []).append((x, c))
            for k in ks.get("screw", []):
                for kk in k.split(" + "):        # joined through a threaded spacer: "screw_a + spacer + screw_b"
                    if kk in self.parts:
                        self.screw_parts.setdefault(kk, set()).update((x, y))
        self.WB = {n: SL.world_bounds(ctx, self.T, n) for n in self.nodes}
        self.screws = sorted(n for n, p in self.raw.items() if A.hardware_kind(p) == "screw")
        self.geo = {}
        MC = SL.mount_check()
        for k in self.screws:
            p = self.parts.get(k)
            if p is None:
                continue                          # no mesh: handled as a mesh-less part
            try:
                g = MC.screw_geometry(BL.load_mesh(p["mesh"]))
            except Exception:  # noqa: BLE001
                continue
            Ts = self.T[p["owning_link"]]
            d = Ts[:3, :3] @ np.asarray(g["_d"]); h = Ts[:3, :3] @ np.asarray(g["_head"]) + Ts[:3, 3]
            self.geo[k] = (d / np.linalg.norm(d), h, float(g.get("length_mm") or g.get("_len") or 0.0), g)
            if k not in self.screw_parts:          # not in the graph (proxy screws): ring points inside the coarse solids
                try:
                    info = A._screw_info(ctx, self.T, k, self.nodes, self.WB)
                    if info["touched"]:
                        self.screw_parts[k] = set(info["touched"])
                except Exception:  # noqa: BLE001
                    pass
        self._cent = {}

    # ---------------------------------------------------------------- geometry helpers
    def centroid(self, n):
        if n not in self._cent:
            if n in self.WB:
                self._cent[n] = self.WB[n].mean(0)
            else:
                p = self.raw[n]
                b = p.get("bounds") or p.get("bounds_mm") or p.get("aabb")
                if b is None:
                    self._cent[n] = None
                else:
                    b = np.asarray(b, float).reshape(2, 3)
                    Tn = self.T.get(p.get("owning_link"))
                    c = b.mean(0)
                    self._cent[n] = Tn[:3, :3] @ c + Tn[:3, 3] if Tn is not None else c
        return self._cent[n]

    def hw_centre(self, n):
        p = self.parts.get(n)
        if p is not None and p.get("mesh"):
            try:
                m = BL.load_mesh(p["mesh"])
                Tn = self.T[p["owning_link"]]
                return Tn[:3, :3] @ np.asarray(m.bounds, float).mean(0) + Tn[:3, 3]
            except Exception:  # noqa: BLE001
                pass
        return self.centroid(n)

    def carries_head(self, n, k):
        d, h, _, _ = self.geo[k]
        pt = h + 1.0 * d
        best, bn = 1e9, None
        for m in self.screw_parts[k]:
            Tm = self.T[self.parts[m]["owning_link"]]
            loc = (pt - Tm[:3, 3]) @ Tm[:3, :3]
            try:
                import trimesh
                dd = abs(trimesh.proximity.closest_point(SL.screen(self.ctx, m)[0], loc[None])[1][0])
            except Exception:  # noqa: BLE001
                dd = np.linalg.norm(self.centroid(m) - pt)
            if dd < best:
                best, bn = dd, m
        return bn == n

    def nearest_node(self, pt, among):
        best, bn = 1e18, None
        for n in among:
            lo, hi = self.WB[n]
            dd = float(np.linalg.norm(np.maximum(0, np.maximum(lo - pt, pt - hi))))
            if dd < best:
                best, bn = dd, n
        return bn, best

    # ---------------------------------------------------------------- assignment
    def takes(self, plan, n):
        p = self.raw[n]
        if plan.get("exclude") and any(fnmatch.fnmatchcase(n, g) for g in plan["exclude"]):
            return False
        if plan.get("include") and any(fnmatch.fnmatchcase(n, g) for g in plan["include"]):
            return True
        if plan.get("roles") and p.get("role") not in plan["roles"]:
            return False
        if plan.get("exclude_roles") and p.get("role") in plan["exclude_roles"]:
            return False
        return p.get("owning_link") in set(plan.get("links", []))

    def assign(self, order, seq):
        """{assembly: [node names]} for every node (first plan in build order that takes it); and nodes no plan takes."""
        out, left = {a: [] for a in seq}, []
        for n in self.nodes:
            if n.endswith("_steel_sleeves") and n[: -len("_steel_sleeves")] in self.node_set:
                continue                          # goes with its part
            for a in seq:
                if self.takes(order["assemblies"][a].get("plan", {}), n):
                    out[a].append(n); break
            else:
                left.append(n)
        return out, left

    # ---------------------------------------------------------------- seeding one assembly
    def seed(self, order, name, mine, present, search=False, walker=None):
        a = order["assemblies"][name]
        plan = a.get("plan", {})
        parts = self.parts
        steps = list(a.get("steps", []))
        named = set(A.named_in(list(self.raw), a))
        placed0 = set(present) | set(A.resolve(parts, a.get("base", [])))
        last_roles = set(plan.get("last_roles", ["body_shell"]))
        missing = [n for n in mine if n not in named and n not in placed0]
        # Dijkstra from what is present
        dist, pred = {b: 0.0 for b in placed0 if b in self.adj or b in self.node_set}, {}
        pq = [(0.0, b) for b in dist]; heapq.heapify(pq)
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, c in self.adj.get(u, []):
                if d + c < dist.get(v, 1e9):
                    dist[v] = d + c; pred[v] = u; heapq.heappush(pq, (d + c, v))
        early = sorted([n for n in missing if n in dist and parts[n].get("role") not in last_roles], key=lambda n: (dist[n], n))
        unreached = sorted(n for n in missing if n not in dist and parts[n].get("role") not in last_roles)
        last = sorted([n for n in missing if parts[n].get("role") in last_roles], key=lambda n: (dist.get(n, 1e9), n))
        # where each existing step places its parts
        def step_parts(st):
            return [n for n in A.resolve(parts, st.get("part")) if n in self.node_set]

        def place_after(n):
            """Index after which part n's step goes: after the step that places its path predecessor (or the last step of an
            inserted part with a smaller distance whose predecessor is the same), else at the start."""
            p = pred.get(n)
            while p is not None and p not in placed0:
                for i, st in enumerate(steps):
                    if p in step_parts(st):
                        return i
                p = pred.get(p)
            return -1
        new_ids = []
        for n in early:
            i = place_after(n)
            j = i + 1
            while j < len(steps) and steps[j].get("_new") and steps[j].get("_dist", 0) <= dist[n]:
                j += 1                                # keep inserted siblings in distance order
            st = self.part_step(n, dist.get(n), pred.get(n))
            st["_new"], st["_dist"] = True, dist[n]
            steps.insert(j, st); new_ids.append(n)
        for n in unreached + last:
            st = self.part_step(n, dist.get(n), pred.get(n), unreached=n in unreached)
            st["_new"] = True
            steps.append(st)
        # directions of the new steps from the screws that join them to what is placed before them
        placed = set(placed0)
        for st in steps:
            ns = step_parts(st)
            if st.pop("_new", False):
                st.pop("_dist", None)
                st["direction"], how = self.direction(ns[0], placed)
                st["derived"] += f"; direction: {how}"
            placed.update(ns)
        a["steps"] = steps
        return [n for n in missing]

    def part_step(self, n, dist, pred, unreached=False):
        s = f"{n}_steel_sleeves"
        grp = [n] + ([s] if s in self.parts else [])
        how = ("not reached by the connection graph (no screw, joint or contact edge to what is present): placed before the covers"
               if unreached else f"graph order, fastened distance {dist:g} from what is present (via {pred})" if dist is not None
               else "covers last (role in last_roles)")
        return dict(id=None, part=grp[0] if len(grp) == 1 else grp, direction=None, seeded=True, derived=how)

    def direction(self, n, placed):
        mine = [k for k, ps in self.screw_parts.items() if n in ps and (ps - {n}) & placed and k in self.geo]
        if mine:
            vs = [self.geo[k][0] * (1 if self.carries_head(n, k) else -1) for k in mine]
            v = np.mean(vs, 0)
            v = v if np.linalg.norm(v) > 0.2 else vs[0]                   # opposite screws cancel: first screw's direction
            v = v / np.linalg.norm(v)
            return [round(float(x), 3) for x in v], (f"mean screw direction of {len(mine)} screw(s) joining it to placed parts "
                                                      f"({', '.join(sorted({stem(k) for k in mine}))}), head to tip when the part carries the head, else reversed")
        nb = [m for m, _ in self.adj.get(n, []) if m in placed]
        if nb and n in self.WB:
            tgt = np.mean([self.centroid(m) for m in nb], 0)
            return [float(x) for x in snap(tgt - self.centroid(n))], f"toward the centroid of placed neighbours ({', '.join(sorted(nb)[:3])}), snapped to the nearest axis"
        return [0.0, 0.0, -1.0], "no placed neighbour: straight down (-z)"

    # ---------------------------------------------------------------- screws and hardware over the whole order
    def place_screws(self, order, seq, keep_existing=True):
        """Insert every screw no step names right after the first step (build order) at which all parts it joins are placed."""
        names = list(self.raw)
        named = set()
        for a in seq:
            for st in order["assemblies"][a]["steps"]:
                named |= set(A.resolve(names, st.get("screw"))) | set(A.resolve(names, st.get("with")))
        todo = [k for k in self.screws if k not in named and k in self.parts]
        placed, where, plan_new = set(), {}, {}
        for a in seq:
            asm = order["assemblies"][a]
            placed |= set(A.resolve(self.parts, asm.get("base", [])))
            for i, st in enumerate(asm["steps"]):
                ns = [n for n in A.resolve(self.parts, st.get("part")) if n in self.node_set]
                placed |= set(ns)
                for n in ns:
                    where[n] = (a, i)
                ready = [k for k in todo if self.screw_parts.get(k) and self.screw_parts[k] <= placed and ns and self.screw_parts[k] & set(ns)]
                for k in ready:
                    plan_new.setdefault((a, i), []).append(k); todo.remove(k)
        # screws joining nothing (or parts never placed): after the step of the part nearest the head
        rest = {}
        for k in todo:
            ps = self.screw_parts.get(k)
            cand = [n for n in (ps or []) if n in where] or [n for n in where]
            pt = self.geo[k][1] if k in self.geo else self.centroid(k)
            n, dd = self.nearest_node(pt, cand) if pt is not None else (None, None)
            if n is None:
                rest.setdefault(None, []).append(k); continue
            plan_new.setdefault(where[n], []).append(k)
            rest.setdefault("joins nothing in the graph" if not ps else "joins parts never placed", []).append(k)
        notes = {k: why for why, ks in rest.items() if why for k in ks}
        for a in seq:
            asm = order["assemblies"][a]
            out = []
            for i, st in enumerate(asm["steps"]):
                out.append(st)
                ks = sorted(plan_new.get((a, i), []))
                stems = {}
                for k in ks:
                    stems.setdefault(stem(k), []).append(k)
                after = (A.names_of(st.get("part")) or ["?"])[0]
                for s, group in stems.items():
                    why = {notes.get(k) for k in group} - {None}
                    out.append(dict(id=None, screw=group if len(group) > 1 else group[0], direction="screw axis", seeded=True,
                                    derived=(f"{len(group)} screw(s) whose joined parts are all placed after {after}" if not why else
                                             f"{len(group)} screw(s) that {', '.join(sorted(why))} (connection graph): placed after the nearest part {after}")
                                    + "; driver path vs placed parts"))
            asm["steps"] = out
        return rest.get(None, [])

    def attach_hardware(self, order, seq):
        """`with` lists: nuts / washers / spacers / inserts on the step of the screw through them, else of the part they sit in,
        else of the nearest placed part; mesh-less parts and screws on the step of the nearest part."""
        names = list(self.raw)
        named = set()
        screw_step, part_step = {}, {}
        for a in seq:
            for st in order["assemblies"][a]["steps"]:
                hit = A.named_in(names, dict(steps=[st]))
                named |= set(hit)
                for k in A.resolve(names, st.get("screw")):
                    screw_step[k] = st
                for n in A.resolve(names, st.get("part")):
                    part_step[n] = st
            named |= set(A.resolve(names, order["assemblies"][a].get("base", [])))
        nodes = set(self.nodes)
        todo = [n for n in names if n not in named and (A.coverage_class(self.raw[n], nodes) != "part" or n not in self.parts)]
        log = {}
        for n in todo:
            c = self.hw_centre(n)
            target, why = None, None
            if c is not None and n in self.parts:
                best = (1e9, None)
                for k, (d, h, L, g) in self.geo.items():
                    if k not in screw_step:
                        continue
                    v = c - h
                    t = float(v @ d)
                    r = float(np.linalg.norm(v - t * d))
                    reach = max(L, 2.0 * float(g.get("r_shank_mm", 1.5)) * 8)
                    if r < 2.0 and -6.0 <= t <= reach + 6.0 and r < best[0]:
                        best = (r, k)
                if best[1]:
                    target, why = screw_step[best[1]], f"on the axis of {best[1]} ({best[0]:.2f} mm off)"
            if target is None and c is not None:
                inside = []
                for m in part_step:
                    if m not in self.WB:
                        continue
                    lo, hi = self.WB[m]
                    if np.all(c >= lo - 0.5) and np.all(c <= hi + 0.5):
                        Tm = self.T[self.parts[m]["owning_link"]]
                        try:
                            if SL.screen(self.ctx, m)[0].contains(((c - Tm[:3, 3]) @ Tm[:3, :3])[None])[0]:
                                inside.append(m)
                        except Exception:  # noqa: BLE001
                            pass
                if inside:
                    m = min(inside, key=lambda m: float(np.prod(self.WB[m][1] - self.WB[m][0])))
                    target, why = part_step[m], f"centre inside {m}"
                else:
                    m, dd = self.nearest_node(c, [m for m in part_step if m in self.WB])
                    if m:
                        target, why = part_step[m], f"nearest part {m} ({dd:.1f} mm)"
            if target is None:
                log[n] = "no placement: no bounds or mesh"
                continue
            w = A.names_of(target.get("with"))
            target["with"] = sorted(set(w) | {n})
            log[n] = why
        return log


def new_ids(order):
    """Ids for the steps the proposer added (id None): <assembly>:new-<n>, n past every id already used."""
    for an, a in order["assemblies"].items():
        k = 0
        for st in a["steps"]:
            if st.get("id") is None:
                k += 1
                st["id"] = f"{an}:new-{k:03d}"
                st["_new"] = True


def mirror_into(order, src, dst, names):
    """Steps of `dst` = the steps of `src` mirrored (R_ <-> L_, x negated); mirrored names that do not exist are dropped."""
    out = []
    for st in order["assemblies"][src]["steps"]:
        m = dict(st)
        for key in ("part", "screw", "with"):
            if key in st:
                ms = mirror_spec(st[key], names)
                if ms:
                    m[key] = ms
                else:
                    m.pop(key)
        if "part" not in m and "screw" not in m:
            continue
        if isinstance(st.get("direction"), list):
            m["direction"] = [-st["direction"][0], st["direction"][1], st["direction"][2]]
        m["derived"] = f"mirror of {src} {st.get('id')} (R_ <-> L_, x negated): {st.get('derived') or ''}"
        if not st.get("seeded"):
            m["seeded"] = True                 # a designer's order on the other side is a draft here until checked
        m.pop("confirmed", None)
        out.append(m)
    order["assemblies"][dst]["steps"] = out


def search_directions(S, order, seq, names_to_search):
    """For every seeded part step of the named assemblies whose path is blocked, keep the clearest of the seeded and six axis directions."""
    W = A.Walker(S.ctx, order)
    dflt = order.get("defaults", {})
    max_mm, step_mm = dflt.get("max_mm", 60.0), dflt.get("step_mm", 1.0)
    changed = 0
    for name in names_to_search:
        asm = order["assemblies"][name]
        assembled = []
        for m in A.prerequisites(order, name):
            assembled += [n for n in W.parts_of(m) if n not in assembled]
        assembled += [n for n in A.resolve(S.parts, asm.get("base", [])) if n in S.node_set and n not in assembled]
        for st in asm["steps"]:
            ns = [n for n in A.resolve(S.parts, st.get("part")) if n in S.node_set]
            if not ns:
                continue
            if st.get("seeded") and isinstance(st.get("direction"), list):
                obst = {n: W.world(n) for n in assembled}

                def score(d):
                    """(seat never let go, path contact (mm out, 0 = none), max overlap mm3): smaller is clearer."""
                    rs = [A.sweep_insertion(W.world(n), d, obst, max_mm, step_mm) for n in ns]
                    stuck = any((r["seat_parts_released_at_mm"] or 0.0) >= max_mm - 1e-9 for r in rs)
                    return (stuck, max((r["path_first_contact_mm"] or 0.0) for r in rs), max(r["max_overlap_mm3"] for r in rs))
                d0 = np.asarray(st["direction"], float)
                s0 = score(d0)
                if s0[0] or s0[1] > 0:
                    best = (s0, d0)
                    for d in AXES:
                        if np.allclose(d, d0):
                            continue
                        s = score(d)
                        if s < best[0]:
                            best = (s, d)
                    if not np.allclose(best[1], d0):
                        st["direction"] = [float(x) for x in best[1]]
                        st["derived"] += (f"; direction searched (six axes vs the seeded {np.round(d0, 3).tolist()}): path contact "
                                          f"{s0[1]} -> {best[0][1]} mm out, seat released {not s0[0]} -> {not best[0][0]}, max overlap {s0[2]:.1f} -> {best[0][2]:.1f} mm3")
                        changed += 1
            assembled += [n for n in ns if n not in assembled]
    return changed


def cmd_propose(a):
    """Draft steps for what no declared step names, and (--search) better directions for blocked seeded steps; write them as
    YAML snippets per owning definition file into --out (default .cache/work/bom/assembly-proposals/). Edits no definition."""
    import yaml
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    man = Path(a.manifest) if a.manifest else ROOT / json.loads((ROOT / "robot/design.json").read_text())["body_manifest"]
    D = DEF.load()
    names0 = [p["name"] for p in json.loads(Path(man).read_text())["parts"]]
    A.use_claims(names0, D)
    S = Seeder(man)
    names = list(S.raw)
    claims = A._CLAIMS
    order = D.order(names)
    seq = A.build_order(order)
    todo = a.names.split(",") if a.names else seq
    for n in todo:
        src = order["assemblies"][n].get("plan", {}).get("mirror_of")
        if src and not order["assemblies"][n]["steps"]:
            mirror_into(order, src, n, names)
            for st in order["assemblies"][n]["steps"]:
                st["id"] = None
    before = {n: [dict(st) for st in order["assemblies"][n]["steps"]] for n in seq}
    mine, left = S.assign(order, seq)
    present = set()
    for n in seq:
        asm = order["assemblies"][n]
        if n in todo:
            S.seed(order, n, mine[n], present)
        present |= set(A.resolve(S.parts, asm.get("base", []))) | {x for st in asm["steps"] for x in A.resolve(S.parts, st.get("part"))}
    unplaced = S.place_screws(order, seq)
    new_ids(order)
    hw = S.attach_hardware(order, seq)
    changed = search_directions(S, order, seq, todo) if a.search else 0
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    per_file, edits = {}, {}
    for n in seq:
        prev = None
        old = {st.get("id"): st for st in before[n]}
        for st in order["assemblies"][n]["steps"]:
            first = (A.resolve(names, st.get("part") or st.get("screw")) or [None])[0]
            f = (claims.get(first) or {}).get("file") or order["assemblies"][n].get("file")
            if st.pop("_new", False):
                row = {k: st[k] for k in ("id", "part", "screw", "with", "direction") if k in st}
                row = dict(row, subassembly=n, after=[prev] if prev else [], status="seeded", derived=st.get("derived", ""))
                per_file.setdefault(f, []).append(row)
            else:
                o = old.get(st.get("id"), {})
                ch = {k: st[k] for k in ("direction", "with") if st.get(k) != o.get(k) and k in st}
                if ch:
                    edits.setdefault(st.get("file") or f, []).append(dict(id=st["id"], change=ch))
            prev = st["id"]
    for f, rows in list(per_file.items()) + [(f, None) for f in edits if f not in per_file]:
        doc = {}
        if rows:
            doc["new_steps"] = rows
        if edits.get(f):
            doc["changed_steps"] = edits[f]
        (out / (Path(f).name.replace(".bom.yaml", "") + ".proposal.yaml")).write_text(
            f"# Proposal for {f} (robot/body/checks/assembly_seed.py propose): paste what you accept into its assembly.steps.\n"
            + yaml.safe_dump(doc, sort_keys=False, width=140))
    print(json.dumps(dict(files=sorted(set(per_file) | set(edits)), new_steps=sum(len(r) for r in per_file.values()),
                          changed_steps=sum(len(r) for r in edits.values()), directions_searched=changed, parts_no_plan_takes=left,
                          screws_not_placed=unplaced, out=str(out)), indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("propose"); p.add_argument("--names"); p.add_argument("--search", action="store_true"); p.add_argument("--manifest")
    p.add_argument("--out", default=str(ROOT / ".cache/work/bom/assembly-proposals"))
    a = ap.parse_args()
    cmd_propose(a)


if __name__ == "__main__":
    main()
