"""assembly: walk the declared assembly order (the `assembly` sections of the body definitions) and measure, step by step, whether each part can be inserted
along its direction and each screw reached by its driver, given only what is already assembled at that step.

Order: the `assembly` sections of the body definitions (robot/**/<builder>.bom.yaml, robot/body/bom/definitions.py), as
`assemblies` -> {name: {after: [assemblies built before this one; all their parts are present at its start],
base: [parts present at the start], steps: [...]}}, listed in build order. A step names a `part` (name, glob or list; several
parts move together as one rigid group, e.g. a part with the steel sleeves pressed into it) or a `screw` (name, glob or list),
optionally `with` (hardware and mesh-less parts that go in with the step and are not measured: the nuts, washers and spacers on
those screws, threaded sleeves pressed into the part, parts without geometry), a `direction` (manifest-frame vector, the
direction the part travels while it goes in; the part comes from final position minus t x direction; or "screw axis" = head
toward tip of each screw), and either `seeded` (drafted by checks/assembly_seed.py, not confirmed) or `confirmed` (what
supports it: the check measurement or the designer) with `derived` (how it was drafted).
Frame: standing pose (link transforms at standing), which is the manifest frame.

Part step: the group is translated back along -direction in 1 mm samples (0 < t <= its own travel: until its box clears every
assembled part's box, capped at PATH_CAP_MM) and its overlap volume
(exact boolean of the coarse screening solids, checks/screwlib.screen, as connectivity: an unscoped run would otherwise
tessellate every supplier STEP at 0.03 mm, T-c4sy9d) is measured against every part assembled before it; a sample is only
booleaned against parts whose box it reaches. Reported: first_contact_mm (the largest sampled t with overlap, refined by
bisection to 0.01 mm: where the part first touches coming in from outside), the maximum overlap along the path (mm3) with the
part that has it, and the overlap at the final position (seat, t = 0: interference by design, not a path finding); apart from
the seat: path_first_contact_mm (first touch of a part that does not overlap at the seat) and seat_parts_released_at_mm (where
the seat-overlapping parts let go; = max_mm when they never do). Thin
obstacles thinner than the sample step can be stepped over. Seat overlaps and the first path contact apart from the seat are
confirmed by the exact OCC common of the two STEPs where both parts have one (confirm_exact; coarse values kept beside it).
A subassembly whose parts all enter another assembly in one step (a bench unit, units()) is built before that assembly.
Screw step: driver path (driver_access.measure_screw: tool blade cylinder from the head outward, data/tools.json) against the
parts assembled at that step only; `closed_in_by` lists parts named later (this assembly's later steps and every later
assembly) that would block the same path within the driver's reach (data/tools.json path min_length_mm from the head: the blade
and the hand behind it, as driver_access's first-length obstruction), with their distance from the head (`closed_in_from_mm`);
later parts further out on the path are listed apart with their distance (`closed_in_beyond_reach`); body shells going on later
over it are listed apart, `covered_later_by`: covering screws is what a cover does). Also per screw the parts it touches that are not assembled yet at its step.
Screws are the parts a definition declares as hardware with spec kind screw. Coverage, whole body (every manifest part, the raw manifest including parts without a
mesh): parts (non-hardware solids, sleeves included), screws, and other hardware (nuts, washers, spacers, inserts) that no
assembly names; per assembly the same on its own links. Measurements and counts only."""
from __future__ import annotations

import fnmatch
import json
import sys
import time
from pathlib import Path

import numpy as np

from checks import bodylib as BL
from checks import screwlib as SL
from checks.core import DATA, ROOT, tick
from checks.registry import register

EPS_MM3 = 1e-3
COVER_ROLE = "body_shell"
EXACT_MIN_MM3 = 0.01     # an exact OCC common at or below this is contact, not overlap
PATH_CAP_MM = 300.0      # the longest insertion path measured (mm): a part travels until its box clears every assembled part's box, at most this


def definitions():
    sys.path.insert(0, str(ROOT / "robot/body/bom"))
    import definitions as DEF
    return DEF


def load_order(names, defs=None):
    """The declared assembly order (the `assembly` sections of the body definitions, robot/body/bom/definitions.py) resolved
    against the manifest part names, plus the definition findings about it."""
    D = defs or definitions().load()
    order = D.order(list(names))
    order["definition_findings"] = [f for f in D.findings if f["kind"].startswith(("step_", "subassembly_"))]
    order["entered_as_unit"] = units(order, list(names))
    return order


def units(order, names):
    """Subassemblies built apart (a bench) that enter another assembly as one step: B enters A at step s when every part B
    names (base and part steps) is in s's part / with lists, and neither is built before the other through `after`. From the
    declared data only (no schema field says it). Returns {B: [{into: A, step: id}]}."""
    asm, out = order["assemblies"], {}
    for b, B in asm.items():
        pb = set(resolve(names, B.get("base", [])))
        for st in B.get("steps", []):
            pb |= set(resolve(names, st.get("part")))
        if not pb:
            continue
        for a, A in asm.items():
            if a == b or b in prerequisites(order, a) or a in prerequisites(order, b):
                continue
            for st in A.get("steps", []):
                if "part" in st and pb <= set(resolve(names, st.get("part"))) | set(resolve(names, st.get("with"))):
                    out.setdefault(b, []).append(dict(into=a, step=st.get("id")))
                    break
    return out


def names_of(spec):
    if spec is None:
        return []
    return [spec] if isinstance(spec, str) else list(spec)


def resolve(parts, spec):
    """Part names in `parts` (names in manifest order) matching a name / glob / list of them."""
    pats = names_of(spec)
    return [n for n in parts if any(fnmatch.fnmatchcase(n, p) for p in pats)]


def build_order(order):
    """Assembly names in build order: file order, moved after the assemblies each names in `after` and after the bench
    subassemblies that enter it as one step (units)."""
    asm = order["assemblies"]
    out, seen = [], set()
    before = {}
    for b, into in order.get("entered_as_unit", {}).items():
        for e in into:
            before.setdefault(e["into"], []).append(b)

    def visit(n, stack=()):
        if n in seen or n not in asm:
            return
        if n in stack:
            raise ValueError(f"assembly order: cycle in `after` through {n}")
        for m in asm[n].get("after", []) + before.get(n, []):
            visit(m, stack + (n,))
        seen.add(n); out.append(n)
    for n in asm:
        visit(n)
    return out


def prerequisites(order, name):
    """Every assembly that must be built before `name` (transitive `after`)."""
    asm, out, todo = order["assemblies"], [], list(order["assemblies"][name].get("after", []))
    while todo:
        m = todo.pop(0)
        if m in out or m not in asm:
            continue
        out.append(m); todo += asm[m].get("after", [])
    return out


def named_in(parts, a):
    """{part name: [step ids]} that assembly `a` names: base, part / screw steps and their `with` lists."""
    out = {}
    for n in resolve(parts, a.get("base", [])):
        out.setdefault(n, []).append("base")
    for st in a.get("steps", []):
        for key in ("part", "screw", "with"):
            for n in resolve(parts, st.get(key)):
                out.setdefault(n, []).append(st.get("id"))
    return out


_CLAIMS = {}


def use_claims(names, defs=None):
    """Resolve the definitions' part entries once for these manifest names (hardware_kind reads them)."""
    D = defs or definitions().load()
    _CLAIMS.clear(); _CLAIMS.update(D.claims(list(names)))
    return D


def hardware_kind(p):
    """The kind a part's definition declares: its spec kind (screw / nut / washer / spacer / ...) for hardware, else None."""
    e = _CLAIMS.get(p["name"])
    if not e or e["kind"] != "hardware":
        return None
    return (e.get("spec") or {}).get("kind") or "other_hardware"


def coverage_class(p, node_names):
    """'part' (a non-hardware solid the walk measures, sleeves included), 'screw', or 'hardware' (everything else that goes in)."""
    if p["name"] in node_names:
        return "part"
    k = hardware_kind(p)
    if k == "screw":
        return "screw"
    if k is None and not BL.is_fastener(p["name"], p) and p.get("mesh"):
        return "part"
    return "hardware"


# ------------------------------------------------------------------------------------------------ pure measurement
def _aabb(m):
    b = np.asarray(m.bounding_box() if hasattr(m, "bounding_box") else m.bounds(), float).reshape(-1)
    return b[:3], b[3:]


def _moved(m, d, t):
    return m.translate([float(x) for x in (-t * np.asarray(d))])


def box_clear_mm(lo, hi, d, olo, ohi):
    """Largest t >= 0 at which the box [lo, hi] moved by -t x d (unit) still meets the box [olo, ohi]; 0 when it never does for t > 0."""
    t0, t1 = 0.0, np.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if lo[i] > ohi[i] or hi[i] < olo[i]:
                return 0.0
            continue
        a, b = (lo[i] - ohi[i]) / d[i], (hi[i] - olo[i]) / d[i]      # lo - t d <= ohi and hi - t d >= olo
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return float(t1) if t1 >= t0 and t1 > 0 else 0.0


def sweep_insertion(moving, direction, obstacles, max_mm=PATH_CAP_MM, step_mm=1.0, refine_mm=0.01):
    """Straight insertion of `moving` (manifold, final position) along `direction` against `obstacles` {name: manifold}, same frame.
    The part sits at final position - t x direction for t = 0 .. L, L = its own travel: where its box clears the box of every
    obstacle (path_needed_mm), at most max_mm (path_capped). Returns dict(seat_overlap_mm3 (t = 0), max_overlap_mm3,
    max_overlap_at_mm, max_overlap_part, first_contact_mm (None = free path), first_part, blocked_samples, samples, by_part, path_mm)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    lo, hi = _aabb(moving)
    need = max([box_clear_mm(lo, hi, d, *_aabb(o)) for o in obstacles.values()], default=0.0)
    cap = max_mm
    max_mm = min(cap, float(np.ceil(need / step_mm) * step_mm) + step_mm) if need > 0 else step_mm
    obs, boxes = {}, {}
    for n, o in obstacles.items():
        olo, ohi = _aabb(o)
        # swept box of the mover over 0..max_mm
        s0, s1 = lo - max_mm * np.maximum(d, 0), hi - max_mm * np.minimum(d, 0)
        if np.all(olo <= s1) and np.all(ohi >= s0):
            obs[n], boxes[n] = o, (olo, ohi)

    def overlap(t, only=None):
        sh = -t * d
        mlo, mhi = lo + sh, hi + sh
        mv = None
        out = {}
        for n, o in (only or obs).items():
            olo, ohi = boxes[n]
            if np.any(olo > mhi + 1e-6) or np.any(ohi < mlo - 1e-6):
                continue                                  # boxes apart at this sample: no boolean needed
            mv = mv if mv is not None else _moved(moving, d, t)
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
    # the path apart from the seat: parts already overlapping at t = 0 (interference by design, a CAD finding of gaps /
    # static_audit) are judged by where they let go; every other part by where it is first touched
    path_hits = [(t, {n: v for n, v in ov.items() if n not in seat}) for t, ov in hits]
    path_hits = [(t, ov) for t, ov in path_hits if ov]
    path_first = None
    if path_hits:
        t_p, ov_p = path_hits[-1]
        path_first = (dict(mm=float(max_mm) if t_p >= max_mm - 1e-9 else round(t_p + 0.5 * step_mm, 2), part=max(ov_p, key=ov_p.get)))
    seat_hits = [t for t, ov in hits if set(ov) & set(seat)]
    seat_release = (max(seat_hits) + step_mm if seat_hits else 0.0) if seat else None
    return dict(seat_overlap_mm3=round(sum(seat.values()), 3), seat_overlap_by_part={k: round(v, 3) for k, v in seat.items()},
                path_first_contact_mm=path_first["mm"] if path_first else None, path_first_part=path_first["part"] if path_first else None,
                seat_parts_released_at_mm=None if seat_release is None else min(float(max_mm), seat_release),
                max_overlap_mm3=round(best[0], 3), max_overlap_at_mm=best[1], max_overlap_part=best[2], overlap_at_max_mm3=round(sum(hits[-1][1].values()), 3) if hits and hits[-1][0] >= max_mm - 1e-9 else 0.0,
                first_contact_mm=first_t, first_part=first_part, blocked_samples=len(hits), samples=len(ts),
                path_mm=float(max_mm), path_needed_mm=round(need, 2), path_capped=bool(need >= max_mm),
                by_part=sorted(({"part": n, "max_mm3": round(v, 3), "at_mm": a} for n, (v, a) in by_part.items()), key=lambda r: -r["max_mm3"]))


# ------------------------------------------------------------------------------------------------ exact confirmation
def step_frame(ctx, n):
    """(STEP path, 4x4 manifest-frame placement of its solids) of part n as screen_solid places it (native_step_offset_mm, the
    roll-flip turn), or (None, why) when the part has no usable STEP."""
    from checks.core import R
    p = ctx.body.parts[n]
    st = p.get("native_step")
    if not st or not BL.src(st).exists():
        return None, "no STEP"
    if p.get("placement_offset_mm"):
        return None, "STEP placed by placement_offset_mm (not reproduced here)"
    M = np.eye(4)
    M[:3, 3] = np.asarray(p.get("native_step_offset_mm") or (0, 0, 0), float)
    if str(p.get("roll_flip", "")).startswith("turned"):
        b = np.array(p["bounds_mm"], float)
        M = np.asarray(R.turn_T_at(0.5 * (b[0][0] + b[1][0])), float) @ M
    return str(BL.src(st)), M


EXACT_QUERY_S = 30.0     # seconds of OCC work one exact common may take (the STEP import of a 44 MB board takes about 9 s); over it: timed out


EXACT_VERSION = "v1"


def _exact_key(q):
    """Content key of an exact query: both STEPs' sha256 (header-free, core.sha) and their placements (rounded to 1e-6)."""
    import hashlib
    from checks.core import sha
    parts = [EXACT_VERSION] + [f"{sha(Path(st))}:{np.round(np.asarray(M, float), 6).tolist()}" for st, M in (q["a"], q["b"])]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def exact_commons(queries, per_query_s=EXACT_QUERY_S):
    """exact_commons_uncached through a content-addressed cache (core.CACHE/exact-common, shared with the worktrees): a value
    is reused only for the same STEP contents and placements; errors and timeouts are not stored."""
    from checks import cachefs
    from checks.core import CACHE
    f = CACHE / "exact-common" / "values.json"
    known = cachefs.read_json(f, {}) or {}
    keys = [_exact_key(q) for q in queries]
    todo = [i for i, k in enumerate(keys) if k not in known]
    got = exact_commons_uncached([queries[i] for i in todo], per_query_s)
    secs = getattr(exact_commons_uncached, "seconds", [])
    out, s_out = [known.get(k) for k in keys], [0.0] * len(keys)
    new = {}
    for j, i in enumerate(todo):
        out[i], s_out[i] = got[j], secs[j] if j < len(secs) else None
        if isinstance(got[j], float):
            new[keys[i]] = got[j]
    if new:
        cachefs.update_json(f, new)
    exact_commons.seconds, exact_commons.cached = s_out, len(keys) - len(todo)
    return out


def exact_commons_uncached(queries, per_query_s=EXACT_QUERY_S):
    """Exact OCC common volumes (mm3, checks/brep.common_volume_mm3 of the STEP solids) for [{a: [step, 4x4], b: [step, 4x4]}],
    in a child process so the STEP imports (supplier boards up to 44 MB, about 1 GB) leave the walk's memory. Each query has
    per_query_s (its STEP imports included): one over it is killed with its child, reported as timed out, and a new child takes
    the rest. A failed query is returned as its error text."""
    import os
    import select
    import subprocess
    out, secs, i = [], [], 0
    code = "import sys; sys.path.insert(0, %r); from checks.modules import assembly as A; A._exact_child()" % str(ROOT / "robot/body")
    while i < len(queries):
        cp = subprocess.Popen([sys.executable, "-B", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        cp.stdin.write(json.dumps(queries[i:]).encode()); cp.stdin.close()
        fd, buf, alive = cp.stdout.fileno(), b"", True
        while alive and i < len(queries):
            deadline = time.time() + per_query_s
            while b"\n" not in buf:
                left = deadline - time.time()
                ready = select.select([fd], [], [], max(0.0, left))[0] if left > 0 else []
                chunk = os.read(fd, 65536) if ready else b""
                if not chunk:
                    cp.kill(); cp.wait()
                    out.append(f"timed out after {per_query_s:g} s" if not ready else "exact child ended without a result"); secs.append(None)
                    i += 1; alive = False
                    break
                buf += chunk
            while alive and b"\n" in buf and i < len(queries):
                line, buf = buf.split(b"\n", 1)
                out.append(json.loads(line)[0]); secs.append(json.loads(line)[1]); i += 1
        if alive:
            cp.wait()
    exact_commons_uncached.seconds = secs
    return out


def _exact_child():
    """Reads [{a: [step, 4x4], b: [step, 4x4]}] on stdin, prints [mm3 or error text, seconds] per query. Only the solids whose
    (fast, non-optimal) box reaches the other part's box go to brep.common_volume_mm3: a supplier board holds hundreds of
    solids and their optimal boxes alone cost 48 s (ROCK4D, 624 solids)."""
    from build123d import Compound, Location, import_step
    from OCP.gp import gp_Trsf
    from checks import brep as L
    qs = json.loads(sys.stdin.read())
    cache = {}

    def solids(path):
        if path not in cache:
            if len(cache) >= 3:
                cache.pop(next(iter(cache)))
            ss = import_step(path).solids()
            boxes = []
            for x in ss:
                bb = x.bounding_box(optimal=False)
                boxes.append(np.array([[bb.min.X, bb.min.Y, bb.min.Z], [bb.max.X, bb.max.Y, bb.max.Z]]))
            cache[path] = (ss, boxes)
        return cache[path]

    def world_box(b, M):
        c = np.array([[b[i][0], b[j][1], b[k][2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)]) @ M[:3, :3].T + M[:3, 3]
        return np.array([c.min(0), c.max(0)])

    def near(path, M, other):
        ss, boxes = solids(path)
        if not ss:
            raise ValueError(f"{path}: the STEP holds no solid")
        tr = gp_Trsf(); tr.SetValues(*[float(x) for x in M[:3, :4].ravel()])
        keep = [x.moved(Location(tr)) for x, b in zip(ss, boxes) if other is None or L.aabb_gap(world_box(b, M), other) <= 0.1]
        return keep, (np.array([np.min([world_box(b, M)[0] for b in boxes], 0), np.max([world_box(b, M)[1] for b in boxes], 0)]))
    for q in qs:
        t0 = time.time()
        try:
            Ma, Mb = np.asarray(q["a"][1], float), np.asarray(q["b"][1], float)
            _, box_a = near(q["a"][0], Ma, np.array([[-1e9] * 3, [1e9] * 3]))
            sb, box_b = near(q["b"][0], Mb, box_a)
            sa, _ = near(q["a"][0], Ma, box_b)
            v = round(float(L.common_volume_mm3(Compound(children=sa), Compound(children=sb))), 4) if sa and sb else 0.0
        except Exception as e:  # noqa: BLE001
            v = f"{type(e).__name__}: {e}"
        print(json.dumps([v, round(time.time() - t0, 1)]), flush=True)


def confirm_exact(ctx, T, rows):
    """Confirm the coarse overlaps of part steps with the exact OCC common of the STEPs where both parts have one: every seat
    overlap (t = 0) and the first path contact apart from the seat (at the sample of its largest overlap). Adds per step
    seat_overlap_exact_by_part / seat_overlap_exact_mm3 (None when a part has no STEP: the coarse value stands) and
    path_first_exact_mm3; the coarse values stay as they are."""
    qs, where = [], []
    frames = {}

    def fr(n):
        if n not in frames:
            frames[n] = step_frame(ctx, n)
        return frames[n]
    for row in rows:
        if row.get("kind") != "part" or "members" not in row:
            continue
        d = np.asarray(row["direction"], float); d = d / np.linalg.norm(d)
        for n, r in row["members"].items():
            pairs = [("seat", o, 0.0) for o in r["seat_overlap_by_part"]]
            if r.get("path_first_part"):
                at = next((b["at_mm"] for b in r["by_part"] if b["part"] == r["path_first_part"]), None)
                if at is not None:
                    pairs.append(("path", r["path_first_part"], float(at)))
            for what, o, t in pairs:
                (sa, Ma), (sb, Mb) = fr(n), fr(o)
                if sa is None or sb is None:
                    where.append((row, n, what, o, f"not confirmed: {Ma if sa is None else Mb} ({n if sa is None else o})")); continue
                sh = np.eye(4); sh[:3, 3] = -t * d
                Ta, Tb = T[ctx.body.parts[n]["owning_link"]], T[ctx.body.parts[o]["owning_link"]]
                qs.append(dict(a=[sa, (sh @ Ta @ Ma).tolist()], b=[sb, (Tb @ Mb).tolist()]))
                where.append((row, n, what, o, len(qs) - 1))
    t0 = time.time()
    res = exact_commons(qs)
    secs = getattr(exact_commons, "seconds", [])
    for row, n, what, o, i in where:
        v = res[i] if isinstance(i, int) else i
        if what == "seat":
            row.setdefault("seat_overlap_exact_by_part", {})[f"{n} x {o}"] = v
        else:
            row.setdefault("path_first_exact_mm3", {})[f"{n} x {o}"] = v
    for row in rows:
        ex = row.get("seat_overlap_exact_by_part")
        if ex is not None:
            row["seat_overlap_exact_mm3"] = round(sum(v for v in ex.values() if isinstance(v, float)), 3) if all(isinstance(v, float) for v in ex.values()) else None
    slow = sorted(((secs[i] or 1e9, f"{n} x {o} ({what})") for row, n, what, o, i in where if isinstance(i, int) and i < len(secs)), reverse=True)[:5]
    return dict(queries=len(qs), cached=getattr(exact_commons, "cached", 0), seconds=round(time.time() - t0, 1), timed_out=sum(1 for v in res if isinstance(v, str) and v.startswith("timed out")),
                slowest=[dict(pair=p, seconds=None if s_ >= 1e9 else s_) for s_, p in slow])


# ------------------------------------------------------------------------------------------------ the walk
class Coarse:
    """ctx as driver_access.measure_screw reads it, but every solid is the coarse screening solid (SL.screen)."""

    def __init__(self, ctx):
        self._ctx, self.mf = ctx, {}

    def __getattr__(self, k):
        return getattr(self._ctx, k)

    def solid(self, n):
        m, how, f = SL.screen(self._ctx, n)
        self.mf[n] = f
        return (m, how)


class World:
    """Coarse screening manifolds at standing, world frame, built once per part."""

    def __init__(self, ctx, T):
        self.ctx, self.T, self.c = ctx, T, {}

    def __call__(self, n):
        if n not in self.c:
            f = SL.screen(self.ctx, n)[2]
            self.c[n] = f.transform(self.T[self.ctx.body.parts[n]["owning_link"]][:3, :4])
        return self.c[n]


def _summed(dicts):
    out = {}
    for dd in dicts:
        for k, v in dd.items():
            out[k] = round(out.get(k, 0.0) + v, 3)
    return out


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
            inside = SL.screen(ctx, n)[0].contains((pw - Tn[:3, 3]) @ Tn[:3, :3])   # a selection: coarse solid (T-c4sy9d)
        except Exception:  # noqa: BLE001
            continue
        if inside.sum() >= 2:
            touched.append(n)
    return dict(geometry=g, touched=touched, world=Ts)


class Walker:
    """Shared state for walking several assemblies of one manifest."""

    def __init__(self, ctx, order):
        self.ctx, self.order = ctx, order
        self.parts = ctx.body.parts
        self.T = SL.standing_T(ctx)
        self.nodes = SL.solid_nodes(ctx)
        self.node_set = set(self.nodes)
        self.WB = {n: SL.world_bounds(ctx, self.T, n) for n in self.nodes}
        allb = np.array(list(self.WB.values()))
        self.box = (allb[:, 0].min(0), allb[:, 1].max(0))
        self.tdata = SL.tools()
        self.world = World(ctx, self.T)
        self.coarse = Coarse(ctx)
        self.screw_names = {n for n, p in self.parts.items() if hardware_kind(p) == "screw"}
        self.seq = build_order(order)

    def resolved(self, a):
        """[(kind, measured names, with names)] per step; part names limited to solids, screw names to screws with a mesh."""
        out = []
        for st in a["steps"]:
            w = resolve(self.parts, st.get("with"))
            if "part" in st:
                out.append(("part", [n for n in resolve(self.parts, st["part"]) if n in self.node_set], w))
            else:
                out.append(("screw", [n for n in resolve(self.parts, st["screw"]) if n in self.screw_names], w))
        return out

    def parts_of(self, name):
        a = self.order["assemblies"][name]
        return [n for n in named_in(self.parts, a) if n in self.node_set]

    def walk(self, name, max_mm=None, step_mm=None):
        """Measure every step of assembly `name`; returns dict(steps [...], coverage {...}, seconds)."""
        from checks.modules import driver_access as DA
        t_start = time.time()
        order, parts = self.order, self.parts
        asm = order["assemblies"][name]
        dflt = order.get("defaults", {})
        max_mm = max_mm or PATH_CAP_MM            # a cap: each part travels until it clears what is assembled (the definitions' 60 mm default was too short)
        step_mm = step_mm or dflt.get("step_mm", 1.0)
        res = self.resolved(asm)
        pre = prerequisites(order, name)
        assembled = []
        for m in pre:
            assembled += [n for n in self.parts_of(m) if n not in assembled]
        assembled += [n for n in resolve(parts, asm.get("base", [])) if n in self.node_set and n not in assembled]
        start = len(assembled)
        later_asm = self.seq[self.seq.index(name) + 1:] if name in self.seq else []
        later_other = [n for m in later_asm for n in self.parts_of(m)]
        named_parts = [n for k, ns, _ in res if k == "part" for n in ns]
        out = []
        for i, (st, (kind, ns, w)) in enumerate(zip(asm["steps"], res)):
            t0 = time.time()
            row = dict(step=i + 1, id=st.get("id"), file=st.get("file"), kind=kind, names=ns, with_parts=w, direction=st.get("direction"),
                       seeded=bool(st.get("seeded")), confirmed=st.get("confirmed"), derived=st.get("derived"), assembled_before=len(assembled))
            if not ns:
                row["error"] = "step names nothing measurable in this manifest" + (f" (carries {len(w)} part(s) without geometry or hardware)" if w else "")
                out.append(row); continue
            if kind == "part":
                d = np.asarray(st["direction"], float)
                obst = {n: self.world(n) for n in assembled}
                per = {n: sweep_insertion(self.world(n), d, obst, max_mm, step_mm) for n in ns}
                # a group moves as one: report the worst member (largest first-contact distance, then overlap)
                worst = max(per.values(), key=lambda r: (r["first_contact_mm"] or 0.0, r["max_overlap_mm3"]))
                row.update(members=per, first_contact_mm=worst["first_contact_mm"], first_part=worst["first_part"],
                           max_overlap_mm3=max(r["max_overlap_mm3"] for r in per.values()), overlap_at_max_mm3=max(r["overlap_at_max_mm3"] for r in per.values()),
                           seat_overlap_mm3=round(sum(r["seat_overlap_mm3"] for r in per.values()), 3),
                           seat_overlap_by_part=_summed(r["seat_overlap_by_part"] for r in per.values()),
                           path_first_contact_mm=max((r["path_first_contact_mm"] for r in per.values() if r["path_first_contact_mm"] is not None), default=None),
                           path_first_part=next((r["path_first_part"] for r in sorted(per.values(), key=lambda r: -(r["path_first_contact_mm"] or 0)) if r["path_first_part"]), None),
                           seat_parts_released_at_mm=max((r["seat_parts_released_at_mm"] for r in per.values() if r["seat_parts_released_at_mm"] is not None), default=None),
                           blocked_samples=max(r["blocked_samples"] for r in per.values()), path_mm=max(r["path_mm"] for r in per.values()),
                           path_needed_mm=max(r["path_needed_mm"] for r in per.values()), path_capped=any(r["path_capped"] for r in per.values()))
                assembled += [n for n in ns if n not in assembled]
            else:
                later = [n for n in named_parts if n not in assembled] + [n for n in later_other if n not in assembled]
                screws = {}
                for k in ns:
                    try:
                        info = _screw_info(self.ctx, self.T, k, self.nodes, self.WB)
                    except Exception as e:  # noqa: BLE001
                        screws[k] = dict(error=f"screw geometry: {e}", touches_not_yet_assembled=[]); continue
                    r = DA.measure_screw(self.coarse, k, info, self.T, assembled, self.WB, self.box, self.tdata)    # with the ball-end tilt where blocked
                    rl = DA.measure_screw(self.coarse, k, info, self.T, assembled + later, self.WB, self.box, self.tdata, tilt=False) if "error" not in r else None
                    if "error" not in r:
                        new = {b["part"] for b in rl.get("by_part", [])} - {b["part"] for b in r.get("by_part", [])}
                        # a cover going on over its screws is the order working; measure_screw reports body shells apart (shell_by_part)
                        cover = ({b["part"] for b in rl.get("shell_by_part", [])} - {b["part"] for b in r.get("shell_by_part", [])}) \
                            | {n for n in new if parts[n].get("role") == COVER_ROLE}
                        dist = {b["part"]: b["from_mm"] for b in rl.get("by_part", []) if b["part"] in new - cover}
                        reach = self.tdata["path"]["min_length_mm"]                       # a part further out is passed by tilting the driver
                        r["closed_in_by"] = sorted(n for n, d in dist.items() if d <= reach)
                        r["closed_in_from_mm"] = {n: dist[n] for n in r["closed_in_by"]}
                        r["closed_in_beyond_reach"] = {n: d for n, d in sorted(dist.items(), key=lambda t: t[1]) if d > reach}
                        r["covered_later_by"] = sorted(cover)
                        r["closed_in_mm3"] = round(sum(b["mm3"] for b in rl.get("by_part", []) if b["part"] in r["closed_in_by"]), 3)
                    r["touches_not_yet_assembled"] = [n for n in info["touched"] if n not in assembled]
                    screws[k] = r
                ok = [r for r in screws.values() if "error" not in r]
                row.update(screws=screws, obstruction_mm3=max([r["obstruction_mm3"] for r in ok], default=0.0),
                           first_distance_mm=min([r["first_distance_mm"] for r in ok if r["first_distance_mm"] is not None], default=None),
                           first_part=next((r["first_part"] for r in ok if r["first_part"]), None),
                           obstruction_within_min_length_mm3=max([r["obstruction_within_min_length_mm3"] for r in ok], default=0.0),
                           closed_in_by=sorted({p for r in ok for p in r.get("closed_in_by", [])}),
                           closed_in_beyond_reach=sorted({p for r in ok for p in r.get("closed_in_beyond_reach", {})}),
                           covered_later_by=sorted({p for r in ok for p in r.get("covered_later_by", [])}),
                           touches_not_yet_assembled=sorted({p for r in screws.values() for p in r["touches_not_yet_assembled"]}),
                           screw_errors={k: r["error"] for k, r in screws.items() if "error" in r})
            row["seconds"] = round(time.time() - t0, 2)
            out.append(row)
        links = sorted({parts[n]["owning_link"] for n in named_parts})
        exact = confirm_exact(self.ctx, self.T, out)
        return dict(steps=out, after=pre, assembled_at_start=start, links=links, seconds=round(time.time() - t_start, 1), exact_confirmation=exact,
                    seeded_steps=sum(1 for r in out if r["seeded"]), confirmed_steps=sum(1 for r in out if r["confirmed"]))


def coverage(ctx, order):
    """Whole body: every manifest part (raw manifest, so parts without a mesh too) against what the assemblies name.
    Returns dict(total {class: n}, not_named {class: [names]}, named_twice [...], per_assembly {name: {class: n}},
    matching_nothing [step ids whose spec matches no part])."""
    from checks.modules import connectivity as CN
    raw = [p for p in CN.json_manifest(ctx)["parts"]]
    names = [p["name"] for p in raw]
    nodes = set(SL.solid_nodes(ctx))
    cls = {p["name"]: coverage_class(p, nodes) for p in raw}
    where, per, nothing = {}, {}, []
    for an, a in order["assemblies"].items():
        hit = named_in(names, a)
        for n, ids in hit.items():
            where.setdefault(n, []).append(an)
        c = per.setdefault(an, dict(parts=0, screws=0, hardware=0, steps=len(a.get("steps", [])),
                                    seeded=sum(1 for s in a.get("steps", []) if s.get("seeded")),
                                    confirmed=sum(1 for s in a.get("steps", []) if s.get("confirmed"))))
        for n in hit:
            c[{"part": "parts", "screw": "screws", "hardware": "hardware"}[cls[n]]] += 1
        for st in a.get("steps", []):
            for key in ("part", "screw", "with"):
                for pat in names_of(st.get(key)):
                    if not any(fnmatch.fnmatchcase(n, pat) for n in names):
                        nothing.append(f"{an}:{st.get('id')}:{pat}")
    total, missing = {}, {}
    for n, c in cls.items():
        total[c] = total.get(c, 0) + 1
        if n not in where:
            missing.setdefault(c, []).append(n)
    # a part of a bench subassembly is named again where the bench enters as one step: that is the unit, not a second home
    into = {b: {e["into"] for e in v} for b, v in order.get("entered_as_unit", {}).items()}

    def receivers(b, seen=()):
        r = set(into.get(b, ()))
        for a in list(r):
            if a not in seen:
                r |= receivers(a, seen + (b,))
        return r
    twice, as_unit = [], {}
    for n, w in where.items():
        w = set(w)
        if len(w) < 2:
            continue
        home = {b for b in w if not any(b in receivers(o) for o in w if o != b)}
        if len(home) == 1 and w - home <= receivers(next(iter(home))):
            as_unit[n] = sorted(w)
        else:
            twice.append(n)
    return dict(total=total, not_named={k: sorted(missing.get(k, [])) for k in ("part", "screw", "hardware")},
                named_in_two_assemblies=sorted(twice), named_again_where_its_bench_enters_as_unit=as_unit,
                per_assembly=per, matching_nothing=nothing)


def _selected(ctx, order, scope):
    parts = ctx.body.parts
    sel = []
    for name in build_order(order):
        if set(named_in(parts, order["assemblies"][name])) & set(scope):
            sel.append(name)
    return sel


@register("assembly", lambda ctx: bool(ctx.body.parts))
def run(ctx):
    """Assembly order walk (the definitions' `assembly` sections): per step the insertion path of a part (first contact mm, overlap mm3 vs what is assembled) or the driver path of a screw at that step, plus whole-body coverage (parts, screws, hardware no step names)."""
    t0 = time.time()
    from checks.modules import connectivity as CN
    names = [p["name"] for p in CN.json_manifest(ctx)["parts"]]
    D = use_claims(names)
    order = load_order(names, D)
    sel = _selected(ctx, order, ctx.scope)
    W = Walker(ctx, order)
    per, cases, evidence_rows = {n: dict(steps=[]) for n in ctx.scope}, [], {}
    for name in sel:
        w = W.walk(name)
        evidence_rows[name] = w
        for r in w["steps"]:
            touched = set(r["names"]) | {p for s in r.get("screws", {}).values() for p in s.get("touches", [])}
            for n in ctx.scope:
                if n in touched:
                    per[n]["steps"].append(dict(assembly=name, **{k: v for k, v in r.items() if k not in ("members", "screws")}))
            if r.get("first_contact_mm"):
                cases.append(dict(part=(r["names"] or [None])[0], mating_part=r["first_part"], pose="standing", location_mm=None,
                                  value=r["max_overlap_mm3"], unit="mm3 max overlap along the insertion path",
                                  what=f"{name} step {r['step']} ({r['id']}, {r.get('file')}): first contact {r['first_contact_mm']} mm from the final position along {r['direction']}"))
            elif r.get("obstruction_mm3", 0) > 0:
                cases.append(dict(part=(r["names"] or [None])[0], mating_part=r["first_part"], pose="standing", location_mm=None,
                                  value=r["obstruction_within_min_length_mm3"], unit="mm3 in the first driver-path length",
                                  what=f"{name} step {r['step']} ({r['id']}, {r.get('file')}): driver path blocked {r['obstruction_mm3']} mm3, first {r['first_distance_mm']} mm from the head; closed in later by {r['closed_in_by']}"))
    cases.sort(key=lambda c: -c["value"])
    steps = [r for w in evidence_rows.values() for r in w["steps"]]
    cov = coverage(ctx, order)
    in_any = {n for name in order["assemblies"] for n in named_in(ctx.body.parts, order["assemblies"][name])}
    for n in ctx.scope:
        per[n]["named_in_assembly"] = n in in_any
    per_asm = {name: dict(cov["per_assembly"].get(name, {}), seconds=w["seconds"],
                          part_steps_with_contact_on_path=sum(1 for r in w["steps"] if r.get("first_contact_mm")),
                          part_steps_touching_a_part_not_in_the_seat=sum(1 for r in w["steps"] if r.get("path_first_contact_mm")),
                          screw_steps_with_obstruction=sum(1 for r in w["steps"] if r.get("obstruction_mm3", 0) > 0))
               for name, w in evidence_rows.items()}
    tick("assembly", t0)
    return dict(per_part=per, worst_cases=cases[:30],
                bad_events=dict(assemblies=len(sel), steps=len(steps), part_steps_with_contact_on_path=sum(1 for r in steps if r.get("first_contact_mm")),
                                part_steps_touching_a_part_not_in_the_seat=sum(1 for r in steps if r.get("path_first_contact_mm")),
                                part_steps_seat_not_released_within_path=sum(1 for r in steps if (r.get("seat_parts_released_at_mm") or 0) >= (r.get("path_mm") or 60) - 1e-9),
                                part_steps_with_seat_overlap=sum(1 for r in steps if r.get("seat_overlap_mm3", 0) > 0),
                                part_steps_with_seat_overlap_after_exact=sum(1 for r in steps if r.get("seat_overlap_mm3", 0) > 0
                                                                            and (r.get("seat_overlap_exact_mm3") is None or r["seat_overlap_exact_mm3"] > EXACT_MIN_MM3)),
                                part_steps_touching_a_part_not_in_the_seat_after_exact=sum(1 for r in steps if r.get("path_first_contact_mm") and any(
                                    not isinstance(v, float) or v > EXACT_MIN_MM3 for v in (r.get("path_first_exact_mm3") or {"": None}).values())),
                                subassemblies_entered_as_unit=len(order.get("entered_as_unit", {})),
                                part_steps_path_capped=sum(1 for r in steps if r.get("path_capped")),
                                parts_named_in_two_assemblies=len(cov["named_in_two_assemblies"]),
                                screw_steps_with_obstruction=sum(1 for r in steps if r.get("obstruction_mm3", 0) > 0),
                                screw_steps_closed_in_by_later_part=sum(1 for r in steps if r.get("closed_in_by")),
                                screw_steps_covered_later_by_shell=sum(1 for r in steps if r.get("covered_later_by")),
                                steps_naming_nothing_measurable=sum(1 for r in steps if r.get("error")),
                                seeded_steps=sum(1 for r in steps if r.get("seeded")), confirmed_steps=sum(1 for r in steps if r.get("confirmed")),
                                parts_not_named=len(cov["not_named"]["part"]), screws_not_named=len(cov["not_named"]["screw"]),
                                hardware_not_named=len(cov["not_named"]["hardware"]), specs_matching_nothing=len(cov["matching_nothing"]),
                                definition_findings=len(order["definition_findings"]),
                                scoped_parts_in_no_assembly=sum(1 for n in ctx.scope if not per[n]["named_in_assembly"])),
                evidence=[], assemblies=evidence_rows, per_assembly=per_asm, coverage=cov, definition_findings=order["definition_findings"],
                entered_as_unit=order.get("entered_as_unit", {}), build_order=W.seq,
                notes=f"straight insertion along each step's direction in 1 mm samples until the part's box clears every assembled part's box (capped at {PATH_CAP_MM:g} mm: path_capped), exact booleans of the coarse screening solids "
                      "against the parts assembled earlier (this assembly's base and steps, plus every assembly in its `after`); screw steps = driver_access path against the "
                      "assembled parts only (with a ball-end key tilt sweep where blocked, driver_access tilt_sweep); standing pose; coverage over every manifest part; seeded steps are drafts, confirmed steps name what supports them; "
                      "seat overlaps and the first path contact apart from the seat confirmed by the exact OCC common of the STEPs where both parts have one (seat_overlap_exact_mm3, path_first_exact_mm3; the coarse values kept); "
                      "a subassembly whose parts all enter another assembly in one step is a bench unit: built before it (build_order), its parts not counted as named twice (entered_as_unit)")
