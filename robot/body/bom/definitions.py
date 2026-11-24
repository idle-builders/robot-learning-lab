"""The body definitions: one annotated `<builder>.bom.yaml` per builder, next to its script under robot/ (schema
robot/body/bom/schema.json, guide robot/body/bom/README.md). This module only reads them: it finds every file, validates
it, resolves its part globs against a manifest, and derives the generator order and the assembly order from what the files
declare. It holds no knowledge of any part. Everything it cannot resolve is returned as a finding that names the file
(and entry) to fix; nothing is filled in.

Used by robot/body/bom/build.py (BOM and build sequence), robot/body/bom/rebuild_chain.py (generator order) and the
checks (robot/body/checks/modules/assembly.py, bom.py)."""
from __future__ import annotations

import fnmatch
import heapq
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCHEMA_FILE = HERE / "schema.json"
SEARCH = ["robot"]                       # definitions live next to their builders
SKIP_DIRS = {"__pycache__", ".git", "node_modules"}
DEFAULTS = dict(max_mm=60.0, step_mm=1.0)


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def discover(root=None):
    """Every <builder>.bom.yaml under robot/ (or, with LAB_DEFINITIONS_ROOT=<dir>, every one under <dir>: an alternate set of
    definitions for the same tools, e.g. a body whose parts are renamed, or a copy with one setting changed)."""
    alt = os.environ.get("LAB_DEFINITIONS_ROOT")
    if root is None and alt:
        root, tops = Path(alt).resolve(), [""]
    else:
        root, tops = (root or ROOT), SEARCH
    out = []
    for top in tops:
        for d, dirs, files in os.walk(root / top if top else root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            out += [Path(d) / f for f in sorted(files) if f.endswith(".bom.yaml")]
    return out


def names_of(spec):
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else list(spec)


def match(names, spec):
    pats = names_of(spec)
    return [n for n in names if any(fnmatch.fnmatchcase(n, p) for p in pats)]


class Definitions:
    def __init__(self, paths=None, strip=None):
        import yaml
        import jsonschema
        schema = json.loads(SCHEMA_FILE.read_text())
        validator = jsonschema.Draft202012Validator(schema)
        self.files, self.findings = [], []
        self.generators, self.entries, self.steps, self.subassemblies = [], [], [], {}
        self.contributions = []          # `compose` entries: edits of a manifest another generator composes (compose.py)
        self.studies = {}                # study_id -> dict(study block, file, definition): dashboard metadata (publish_body.py)
        self.joints = {}                 # joint name -> dict(range_deg, doc, decision, file, definition): declared joint ranges
        self.config = {}                 # definition id -> {NAME: dict(value, unit, doc, decision, file)}: configured values
        for rank, p in enumerate(paths if paths is not None else discover()):
            f = rel(p)
            if strip and f.startswith(strip):
                f = f[len(strip):].lstrip("/")      # a snapshot copy is named by the repository path it came from
            try:
                doc = yaml.safe_load(Path(p).read_text()) or {}
            except Exception as e:  # noqa: BLE001
                self.find("unreadable", f, None, f"YAML does not parse: {e}"); continue
            errs = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
            for e in errs[:20]:
                self.find("schema", f, "/".join(str(x) for x in e.path) or None, e.message[:300])
            if errs:
                continue
            self.files.append(dict(file=f, rank=rank, definition=doc["definition"], owner=doc.get("owner"), doc=doc))
            gens = doc.get("generators", [])
            for g in gens:
                self.generators.append(dict(g, file=f))
            for c in doc.get("compose", []):
                self.contributions.append(dict(c, file=f))
            for i, e in enumerate(doc.get("parts", [])):
                if "generator" not in e and len(gens) == 1:
                    e = dict(e, generator=gens[0]["id"])
                self.entries.append(dict(e, file=f, index=i))
            asm = doc.get("assembly") or {}
            for sa in asm.get("subassemblies", []):
                if sa["name"] in self.subassemblies:
                    self.find("subassembly_declared_twice", f, sa["name"], f"also declared in {self.subassemblies[sa['name']]['file']}")
                    continue
                self.subassemblies[sa["name"]] = dict(sa, file=f)
            for i, st in enumerate(asm.get("steps", [])):
                self.steps.append(dict(st, file=f, index=i, rank=rank))
            for i, jd in enumerate(doc.get("joints", [])):
                lo, hi = jd["range_deg"]
                names = names_of(jd["match"])
                for jn in names:
                    if jn in self.joints:
                        self.find("joint_declared_twice", f, f"joints[{i}] {jn}", f"also declared in {self.joints[jn]['file']}", joint=jn)
                        continue
                    if not lo < hi:
                        self.find("joint_range_empty", f, f"joints[{i}] {jn}", f"range_deg {jd['range_deg']}: lower must be below upper", joint=jn)
                    frame = {}
                    for key in ("origin_mm", "axis"):
                        v = jd.get(key)
                        if isinstance(v, dict):
                            v = v.get(jn)
                        elif v is not None and len(names) > 1:
                            self.find("joint_frame_ambiguous", f, f"joints[{i}] {jn}", f"{key} is one vector for {len(names)} joints; give {{joint: vector}}", joint=jn)
                            v = None
                        if v is not None:
                            frame[key] = [float(x) for x in v]
                    self.joints[jn] = dict(range_deg=[float(lo), float(hi)], doc=jd.get("doc"), decision=jd.get("decision"),
                                           file=f, definition=doc["definition"], entry=f"joints[{i}]", **frame)
            if doc.get("config"):
                self.config[doc["definition"]] = {k: dict(v, file=f) for k, v in doc["config"].items()}
            if doc.get("study"):
                sid = doc["study"]["study_id"]
                if sid in self.studies:
                    self.find("study_declared_twice", f, sid, f"also declared in {self.studies[sid]['file']}")
                else:
                    self.studies[sid] = dict(doc["study"], file=f, definition=doc["definition"])
        seen = {}
        for g in self.generators:
            if g["id"] in seen:
                self.find("generator_declared_twice", g["file"], g["id"], f"also in {seen[g['id']]}")
            seen.setdefault(g["id"], g["file"])
        seen = {}
        for c in self.contributions:
            if c["id"] in seen:
                self.find("compose_declared_twice", c["file"], c["id"], f"also in {seen[c['id']]}")
            seen.setdefault(c["id"], c["file"])
        seen = {}
        for st in self.steps:
            if st["id"] in seen:
                self.find("step_declared_twice", st["file"], st["id"], f"also in {seen[st['id']]}")
            seen.setdefault(st["id"], st["file"])

    def find(self, kind, file, entry, what, **kw):
        self.findings.append(dict(kind=kind, file=file, entry=entry, what=what, **kw))

    # ------------------------------------------------------------------ joints and configured values
    def joint_findings(self, manifest):
        """Findings of the declared joint ranges against a manifest dict: manifest joints (and controls) without a
        declaration, declarations the manifest does not have, manifest ranges that differ from the declaration (a
        manifest built before the declaration changed: rebuild it). Driven joints (driven_by / coupled_to) need none."""
        out = []
        mj = {j["name"]: j for j in manifest.get("joints", [])}
        mc = {c["name"]: c for c in manifest.get("controls", []) if isinstance(c, dict) and c.get("name")}
        driven = {n for n, j in mj.items() if j.get("driven_by") or j.get("coupled_to")}
        for n in sorted(set(mj) | set(mc)):
            d = self.joints.get(n)
            if d is None:
                if n not in driven:
                    out.append(dict(kind="joint_range_undeclared", file=None, entry=None, joint=n,
                                    what=f"{n}: no definition declares its range (joints: in the owning <builder>.bom.yaml); manifest has {(mj.get(n) or mc.get(n)).get('range_deg')}"))
                continue
            for where, rec in (("joints", mj.get(n)), ("controls", mc.get(n))):
                r = (rec or {}).get("range_deg")
                if rec is not None and (r is None or [round(float(x), 6) for x in r] != d["range_deg"]):
                    out.append(dict(kind="joint_range_differs", file=d["file"], entry=f"{d['entry']} {n}", joint=n,
                                    declared=d["range_deg"], manifest=r,
                                    what=f"manifest {where}[{n}].range_deg {r} differs from the declared {d['range_deg']}: rebuild the manifest"))
        for n, d in sorted(self.joints.items()):
            if n not in mj and n not in mc:
                out.append(dict(kind="joint_not_in_manifest", file=d["file"], entry=f"{d['entry']} {n}", joint=n,
                                what=f"{n} is declared but the manifest has no such joint or control"))
        return out

    # ------------------------------------------------------------------ parts
    def claims(self, names):
        """{part: entry} for every manifest part an entry matches; findings for entries / globs that match nothing,
        parts two entries claim (first claim wins, in discovery order), parts no entry claims, quantity differences."""
        out, by_entry = {}, {}
        for e in self.entries:
            hit = match(names, e["match"])
            by_entry[(e["file"], e["index"])] = hit
            tag = f"parts[{e['index']}] line {e['line']}"
            if not hit:
                self.find("entry_matches_nothing", e["file"], tag, f"match {names_of(e['match'])[:6]} matches no part of the manifest")
            for pat in names_of(e["match"]):
                if hit and not any(fnmatch.fnmatchcase(n, pat) for n in names):
                    self.find("glob_matches_nothing", e["file"], tag, f"{pat!r} matches no part")
            if e.get("quantity") is not None and e["quantity"] != len(hit):
                self.find("quantity_differs", e["file"], tag, f"declared {e['quantity']}, manifest {len(hit)}", declared=e["quantity"], cad=len(hit))
            for n in hit:
                if n in out:
                    o = out[n]
                    self.find("claimed_twice", e["file"], tag, f"{n} is also claimed by {o['file']} parts[{o['index']}]", part=n)
                    continue
                out[n] = e
        for n in names:
            if n not in out:
                self.find("part_not_defined", None, None, f"{n}: no definition claims this manifest part", part=n)
        self._claims = out
        return out

    def generator(self, gid):
        return next((g for g in self.generators if g["id"] == gid), None)

    # ------------------------------------------------------------------ generators
    def generator_order(self):
        """Generator ids of the body rebuild in run order: an edge H -> G when G reads (or edits) a file H writes, unless G
        is (transitively) in H's `after`; plus every declared `after`. Findings: cycles, unknown `after` ids, inputs that no
        generator writes and that are not on disk."""
        gens = [g for g in self.generators if g.get("rebuild", True)]
        ids = {g["id"] for g in gens}
        decl = {g["id"]: [a for a in g.get("after", [])] for g in gens}
        for g in gens:
            for a in decl[g["id"]]:
                if a not in ids:
                    self.find("generator_after_unknown", g["file"], g["id"], f"after {a!r}: no rebuild generator has this id")

        def closure(i, seen=None):
            seen = set() if seen is None else seen
            for a in decl.get(i, []):
                if a not in seen:
                    seen.add(a); closure(a, seen)
            return seen
        clo = {i: closure(i) for i in ids}
        writes = {g["id"]: [o for o in g.get("outputs", []) + g.get("in_place_edits", [])] for g in gens}

        def covers(out, path):
            return path == out or path.startswith(out.rstrip("/") + "/")
        edges = {i: set(a for a in decl[i] if a in ids) for i in ids}          # i depends on these
        for c in self.contributions:                                            # a composing generator after what its entries need
            for g in gens:
                if g.get("composes") == c["into"]:
                    for n in c.get("needs", []):
                        if n in ids and n != g["id"] and g["id"] not in clo[n]:
                            edges[g["id"]].add(n)
                        elif n not in ids and not self.generator(n):
                            self.find("compose_needs_unknown", c["file"], c["id"], f"needs {n!r}: no generator has this id")
        for g in gens:
            for inp in g.get("inputs", []) + g.get("in_place_edits", []):
                for h in gens:
                    if h["id"] == g["id"] or g["id"] in clo[h["id"]]:
                        continue
                    if any(covers(o, inp) for o in writes[h["id"]]):
                        edges[g["id"]].add(h["id"])
        rank = {g["id"]: (k, g["id"]) for k, g in enumerate(gens)}
        indeg = {i: len(edges[i]) for i in ids}
        users = {i: [j for j in ids if i in edges[j]] for i in ids}
        pq = [rank[i] for i in ids if indeg[i] == 0]; heapq.heapify(pq)
        out = []
        while pq:
            _, i = heapq.heappop(pq)
            out.append(i)
            for j in users[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    heapq.heappush(pq, rank[j])
        for i in sorted(ids - set(out)):
            g = self.generator(i)
            self.find("generator_cycle", g["file"], i, f"in a dependency cycle with {sorted(edges[i] - set(out))}; declare `after` to break it")
        return out

    def composition(self, stage):
        """The `compose` entries into stage in the order they apply: topological by `after`, ties by id. Findings: unknown
        `after` ids, cycles, a stage no generator `composes`, a `call` whose script does not exist."""
        mine = [c for c in self.contributions if c["into"] == stage]
        ids = {c["id"] for c in mine}
        allids = {c["id"] for c in self.contributions}
        if mine and not any(g.get("composes") == stage for g in self.generators):
            self.find("compose_stage_unknown", mine[0]["file"], mine[0]["id"], f"into {stage!r}: no generator declares composes: {stage}")
        dep = {}
        for c in mine:
            dep[c["id"]] = [a for a in c.get("after", []) if a in ids]
            for a in c.get("after", []):
                if a not in allids:
                    self.find("compose_after_unknown", c["file"], c["id"], f"after {a!r}: no compose entry has this id")
            if not (ROOT / c["call"].split(":")[0]).is_file():
                self.find("compose_call_missing", c["file"], c["id"], f"call {c['call']!r}: script not found")
        by = {c["id"]: c for c in mine}
        indeg = {i: len(d) for i, d in dep.items()}
        users = {i: [j for j in dep if i in dep[j]] for i in dep}
        pq = sorted(i for i in dep if indeg[i] == 0)
        out = []
        while pq:
            i = pq.pop(0)
            out.append(by[i])
            for j in users[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    pq.append(j); pq.sort()
        for i in sorted(set(dep) - {c["id"] for c in out}):
            self.find("compose_cycle", by[i]["file"], i, "in an `after` cycle")
        return out

    @staticmethod
    def exists(rel_path):
        """On disk, or under its post-restructure name (infra/storage/path-moves.json), as the rebuild resolves it."""
        if (ROOT / rel_path).exists():
            return True
        mf = ROOT / "infra/storage/path-moves.json"
        for old, new in (json.loads(mf.read_text())["moves"] if mf.exists() else []):
            if rel_path == old or rel_path.startswith(old + "/"):
                return (ROOT / (new + rel_path[len(old):])).exists()
        return False

    def external_inputs(self):
        """Inputs no generator writes: (generator, path, exists on disk)."""
        outs = [o for g in self.generators for o in g.get("outputs", [])]
        res = []
        for g in self.generators:
            for inp in g.get("inputs", []):
                if any(inp == o or inp.startswith(o.rstrip("/") + "/") for o in outs):
                    continue
                res.append((g, inp, self.exists(inp)))
        return res

    # ------------------------------------------------------------------ assembly
    def order(self, names):
        """The assembly order in the shape robot/body/checks/modules/assembly.py walks: {defaults, assemblies: {name: {doc,
        after, base, file, steps: [{id, part | screw, with, direction, seeded, derived, file}]}}}, subassemblies by their
        `after`, steps by their `after` (ties: discovery order of the files, then position in the file). Findings: steps of an
        undeclared subassembly, `after` ids that name no step, step specs that match no part."""
        names = list(names)
        sas = dict(self.subassemblies)
        for st in self.steps:
            if st["subassembly"] not in sas:
                self.find("subassembly_undeclared", st["file"], st["id"], f"subassembly {st['subassembly']!r} is declared in no definition")
        sid = {st["id"]: st for st in self.steps}
        asm = {}
        for name in self._sa_order(sas):
            sa = sas[name]
            mine = [st for st in self.steps if st["subassembly"] == name]
            ids = {st["id"] for st in mine}
            dep = {}
            for st in mine:
                ds = []
                for a in st.get("after", []):
                    if a in ids:
                        ds.append(a)
                    elif a not in sid:
                        self.find("step_after_unknown", st["file"], st["id"], f"after {a!r}: no step has this id")
                dep[st["id"]] = ds
            key = {st["id"]: (st["rank"], st["index"]) for st in mine}
            indeg = {i: len(d) for i, d in dep.items()}
            users = {i: [j for j in dep if i in dep[j]] for i in dep}
            pq = [(key[i], i) for i in dep if indeg[i] == 0]; heapq.heapify(pq)
            seq = []
            while pq:
                _, i = heapq.heappop(pq)
                seq.append(i)
                for j in users[i]:
                    indeg[j] -= 1
                    if indeg[j] == 0:
                        heapq.heappush(pq, (key[j], j))
            for i in sorted(set(dep) - set(seq)):
                self.find("step_cycle", sid[i]["file"], i, "in an `after` cycle"); seq.append(i)
            steps = []
            for i in seq:
                st = sid[i]
                for k in ("part", "screw", "with"):
                    for pat in names_of(st.get(k)):
                        if not any(fnmatch.fnmatchcase(n, pat) for n in names):
                            self.find("step_matches_nothing", st["file"], i, f"{k} {pat!r} matches no part (stale step)")
                row = {k: st[k] for k in ("id", "part", "screw", "with", "direction", "derived", "notes") if k in st}
                row.update(seeded=st.get("status") == "seeded", designer=st.get("status") == "designer", file=st["file"])
                steps.append(row)
            asm[name] = dict(doc=sa.get("doc"), after=sa.get("after", []), base=names_of(sa.get("base")), plan=sa.get("plan", {}), file=sa["file"], steps=steps)
        return dict(defaults=DEFAULTS, assemblies=asm, source="definitions (*.bom.yaml)")

    def _sa_order(self, sas):
        out, seen = [], set()

        def visit(n, stack=()):
            if n in seen or n not in sas:
                return
            if n in stack:
                self.find("subassembly_cycle", sas[n]["file"], n, "cycle in `after`"); return
            for m in sas[n].get("after", []):
                if m not in sas:
                    self.find("subassembly_after_unknown", sas[n]["file"], n, f"after {m!r} is declared nowhere")
                visit(m, stack + (n,))
            seen.add(n); out.append(n)
        for n in sas:
            visit(n)
        return out


_LOADED: dict = {}


def load(paths=None):
    """The definitions, parsed once per process for the same files (path, size, mtime): a check run asked for them four
    times or more per process at about 1.5 s each (2026-11-22). Each caller gets its own copy."""
    import copy
    files = [Path(p) for p in (paths if paths is not None else discover())]
    key = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.exists()) + ((SCHEMA_FILE.stat().st_mtime_ns,) if SCHEMA_FILE.exists() else ())
    if key not in _LOADED:
        _LOADED.clear()
        _LOADED[key] = Definitions(files)
    return copy.deepcopy(_LOADED[key])
