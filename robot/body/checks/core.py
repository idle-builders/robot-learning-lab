"""Shared body model for the checks: manifest + one forward kinematics (motor-refit fit_check.Body.link_T, from joints[]),
sha256-keyed caches, true-shape solids (native STEP tessellation, else closed mesh), pose sets as data.
Moved from leg/fast_check.py without changing the maths; fast_check imports it from here."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body/leg"))
import roll_flip_checks as R  # noqa: E402  (sets up kit, leg_motion, motor-refit paths)
from checks import bodylib as BL
from checks import cachefs
from checks import resultcache
from checks.relations import ROLES as _ROLES   # data/roles.json (or LAB_CHECKS_DATA's): kinds and role classes

kit = R.kit
WORK = ROOT / ".cache/work/cad-speed"          # reports: this checkout's own
CACHE = cachefs.work() / "cad-speed/cache"     # content-addressed caches: shared with the worktrees (checks/cachefs.py)
DATA = Path(__import__("os").environ.get("LAB_CHECKS_DATA") or HERE / "data")   # LAB_CHECKS_DATA: another set of check tables
T = {}


def tick(name, t0):
    T[name] = round(T.get(name, 0.0) + time.time() - t0, 2)


def load_body(manifest: Path):
    """Body (parts, joints, link_T) of one manifest, loaded once; every joint's range_deg is the one the body definitions declare
    (declare_ranges)."""
    return declare_ranges(kit.base(manifest))


# ------------------------------------------------------------------------------------------------ joint ranges
def declared_joints():
    """({joint: dict(range_deg, file, entry, ...)}, error or None): the joint ranges the body definitions declare (`joints:` of
    robot/**/<builder>.bom.yaml, robot/body/bom/bom_config.joints; LAB_DEFINITIONS_ROOT gives another set). The single source of
    every range the checks use (owner 2026-11-23)."""
    try:
        sys.path.insert(0, str(ROOT / "robot/body/bom"))
        import bom_config
        return bom_config.joints(), None
    except Exception as exc:  # noqa: BLE001  (unreadable definitions: every joint reads 'manifest only', and the error is reported)
        return {}, f"{type(exc).__name__}: {exc}"


def declare_ranges(body, declared=None):
    """Set each joint's range_deg to its declared range. A joint no definition declares keeps the manifest's range_deg, marked
    'manifest only'; a joint with neither has range_deg None and is not swept (no invented range). body.range_source[joint] =
    {range_deg, manifest_range_deg, from: definition | manifest only | none, definition: 'file entry' | None}; body.range_findings
    lists the declared ranges that differ from the manifest's, the manifest-only joints and the joints without a range."""
    if getattr(body, "range_source", None) is not None and declared is None:
        return body
    err = None
    if declared is None:
        declared, err = declared_joints()
    src, diff = {}, []
    for jn, j in body.joints.items():
        man = [float(x) for x in j["range_deg"]] if j.get("range_deg") else None
        d = declared.get(jn)
        if d is not None:
            r, frm, where = [float(x) for x in d["range_deg"]], "definition", f"{d['file']} {d['entry']}"
            if man is not None and man != r:
                diff.append(dict(joint=jn, declared_range_deg=r, manifest_range_deg=man, definition=where))
        elif man is not None:
            r, frm, where = man, "manifest only", None
        else:
            r, frm, where = None, "none", None
        j["range_deg"] = r
        src[jn] = dict(range_deg=r, manifest_range_deg=man, **{"from": frm}, definition=where)
    body.range_source = src
    body.range_findings = dict(declared_differs_from_manifest=diff,
                               manifest_only=sorted(n for n, s in src.items() if s["from"] == "manifest only"),
                               without_range=sorted(n for n, s in src.items() if s["from"] == "none"),
                               definitions_error=err)
    body.unranged_in_poses = set()     # joints a pose list names that have no range (pose left out), filled by the pose builders
    return body


def joint_range(body, jn):
    """[lo, hi] of joint jn as declared (declare_ranges), None when the joint has no range."""
    declare_ranges(body)
    j = body.joints.get(jn)
    return list(j["range_deg"]) if j and j.get("range_deg") else None


BEYOND_TOL_DEG = 0.01


def beyond_declared(body, q):
    """[{joint, declared_range_deg, value_deg}] of the joints pose q sets outside their declared range (by more than 0.01 deg); []
    for a pose inside every declared range. Such a pose is a stress pose: measured, reported apart (beyond_declared_range), never
    part of the design-pose results; joints without a range are not judged."""
    out = []
    for jn, a in q.items():
        r = joint_range(body, jn)
        if r and not (min(r) - BEYOND_TOL_DEG <= a <= max(r) + BEYOND_TOL_DEG):
            out.append(dict(joint=jn, declared_range_deg=r, value_deg=round(float(a), 3)))
    return out


def range_report(body, joints):
    """(joint_ranges {joint: range_source} of `joints`, counts for bad_events, findings) for a check's result: where each range came
    from, the declared ranges that differ from the manifest's, manifest-only joints and joints without a range (left out)."""
    declare_ranges(body)
    js = sorted({j for j in joints if j in body.joints} | set(body.unranged_in_poses))
    jr = {j: body.range_source[j] for j in js if j in body.range_source}
    f = body.range_findings
    find = dict(declared_differs_from_manifest=[d for d in f["declared_differs_from_manifest"] if d["joint"] in jr],
                manifest_only=[j for j in f["manifest_only"] if j in jr],
                without_range=sorted({j for j in f["without_range"] if j in jr} | set(body.unranged_in_poses)),
                definitions_error=f["definitions_error"])
    counts = dict(joints_range_differs_from_manifest=len(find["declared_differs_from_manifest"]),
                  joints_range_manifest_only=len(find["manifest_only"]), joints_without_range=len(find["without_range"]))
    return jr, counts, find


# ------------------------------------------------------------------------------------------------ hashing / caches
_SHA_FILE = CACHE / "sha256.json"
_SHA: dict = {}
_SHA_NEW: dict = {}


def _content(path: Path, raw: bytes) -> bytes:
    """What a cache is keyed by: a STEP without its HEADER section. Its FILE_NAME carries the export time, so two builds
    of the same shape differ only there (measured 2026-11-23: build123d export_step twice, data sections identical), and
    a raw sha256 would make every rebuild a cold cache."""
    if path.suffix.lower() in (".step", ".stp") and raw[:13] == b"ISO-10303-21;":
        i = raw.find(b"ENDSEC;")
        if i > 0:
            return raw[i:]
    return raw


def sha(path: Path, raw: bool = False) -> str:
    """sha256 of a file's content, remembered by (absolute path, size, mtime_ns) across runs: the cache key of everything
    made from it. A STEP is hashed without its header (export timestamp, see _content). raw=True is the sha256 of the
    file's bytes, for comparing with a sha256 a generator recorded."""
    global _SHA
    resultcache.note(path)                  # an input of the running check (checks/resultcache.py)
    if not _SHA:
        _SHA = cachefs.read_json(_SHA_FILE, {}) or {}
    st = path.stat()
    k = ("raw:" if raw else "content:") + str(path)
    if k in _SHA and _SHA[k][0] == st.st_size and _SHA[k][1] == st.st_mtime_ns:
        return _SHA[k][2]
    b = path.read_bytes()
    h = hashlib.sha256(b if raw else _content(path, b)).hexdigest()
    _SHA[k] = _SHA_NEW[k] = (st.st_size, st.st_mtime_ns, h)
    return h


def save_sha():
    """Add this process's new hashes to the shared index (merged under a lock: concurrent runs keep each other's)."""
    if _SHA_NEW:
        cachefs.update_json(_SHA_FILE, _SHA_NEW)
        _SHA_NEW.clear()


def step_mesh(step: Path, offset=None, tol=0.03, ang=0.2, largest=False, min_solid_mm3=0.0):
    """Closed triangle mesh of a native STEP, cached by the STEP's sha256: all solids (or the largest, or those of at
    least min_solid_mm3, e.g. a servo's housing without its internal gears), each tessellated and unioned so touching
    solids give one manifold. OCC leaves a few small faces of some mirrored STEPs (left ST3025 cases) untriangulated at
    0.03 mm, and a shape whose meshing failed fails again at any tolerance: each retry re-imports the STEP and meshes it
    coarser (0.1, then 0.5 mm). The tolerance actually used is in m.metadata["tol"]."""
    import trimesh
    key = (f"{sha(step)[:16]}-{tol}-{ang}-{int(largest)}-{min_solid_mm3:g}-"
           f"{'_'.join(f'{v:g}' for v in (offset or (0, 0, 0)))}")
    f = CACHE / "step-mesh" / f"{key}.npz"
    d = cachefs.load_npz(f) if f.exists() else None
    if d is not None and "f" in d:
        if len(d["f"]):                       # an empty entry was written by the old fallback: rebuild
            m = trimesh.Trimesh(d["v"], d["f"], process=False)
            m.metadata["tol"] = float(d["t"]) if "t" in d else tol
            return m
    import manifold3d as M
    from build123d import Location, import_step
    err = None
    for t_try, a_try in dict.fromkeys(((tol, ang), (min(tol, 0.03), min(ang, 0.2)), (max(tol, 0.1), max(ang, 0.5)),
                                       (max(tol, 0.5), max(ang, 0.5)))):
        sh = import_step(str(step)).moved(Location(tuple(float(v) for v in (offset or (0.0, 0.0, 0.0)))))
        sols = list(sh.solids())
        if largest:
            sols = [max(sols, key=lambda s: s.volume)]
        sols = [s for s in sols if s.volume >= min_solid_mm3] or sols
        try:
            parts, pieces = [], []
            for so in sols:
                v, fc = so.tessellate(t_try, a_try)
                t = trimesh.Trimesh(np.array([(p.X, p.Y, p.Z) for p in v]), np.array(fc), process=True)
                if t.volume < 0:
                    t.invert()
                pieces.append(t)
                parts.append(kit.mf(t))
        except Exception as exc:  # noqa: BLE001
            err = exc
            continue
        if any(x.status().name != "NoError" or x.is_empty() for x in parts):
            # a tessellation that is not a manifold reads as EMPTY to manifold3d (the chassis at 0.1 mm: 9123 faces,
            # NotManifold; fine at 0.03): meshing again, finer or coarser, on a fresh import gives another triangulation
            err = ValueError(f"tessellation at {t_try} mm is not a manifold")
            continue
        u = parts[0] if len(parts) == 1 else M.Manifold.batch_boolean(parts, M.OpType.Add)
        m = kit.tm(u) if u.status().name == "NoError" and not u.is_empty() else trimesh.util.concatenate(pieces)
        f.parent.mkdir(parents=True, exist_ok=True)
        cachefs.savez(f, v=np.asarray(m.vertices), f=np.asarray(m.faces), t=t_try)
        m.metadata["tol"] = t_try
        return m
    raise err


_FAILED_FILE = CACHE / "step-mesh" / "not-usable.json"


def _failed(key=None, why=None):
    """STEPs that did not give a usable screening solid (keyed by part, STEP sha, tolerance): not retried."""
    if key:
        return cachefs.update_json(_FAILED_FILE, {key: why})
    return cachefs.read_json(_FAILED_FILE, {}) or {}


# Which parts are screened with their STEP when not in scope, and the smallest STEP solid kept: pose_sets.json screening_solids,
# over the role table's kinds and classes (data/roles.json; a part's body definition wins on kind and exact_fit), never names.
SCREENING = json.loads((DATA / "pose_sets.json").read_text())["screening_solids"]


def _role_classes(role):
    return {c for c, v in _ROLES["classes"].items() if role in v["roles"]}


def _takes_step(kind, role):
    cls = _role_classes(role)
    return (kind in SCREENING["step_kinds"] or bool(cls & set(SCREENING["step_classes"]))) and not cls & set(SCREENING["except_classes"])


def takes_step(n, p):
    """Is part n screened with its native STEP when not in scope? Its kind (body definition, else role table) or role class."""
    rel = BL.RELATIONS
    kind = rel.kind(n) if rel is not None and n in rel.raw else _ROLES["kind"].get(p.get("role"))
    return _takes_step(kind, p.get("role"))


def min_solid_mm3(n, p):
    """Smallest STEP solid kept in part n's screening solid (mm3): its body definition's exact_fit.min_solid_mm3, else its role
    class's class_min_solid_mm3 (a servo case: the housing without its internal gears), else 0."""
    rel = BL.RELATIONS
    ef = ((rel.claims.get(n) or {}).get("exact_fit") if rel is not None else None) or {}
    if isinstance(ef, dict) and ef.get("min_solid_mm3") is not None:
        return float(ef["min_solid_mm3"])
    return max([float(SCREENING["class_min_solid_mm3"].get(c, 0.0)) for c in _role_classes(p.get("role"))] or [0.0])


# roles whose parts take the STEP when not in scope (by the role table alone), with their class's min solid: run.py reads it
STEP_ROLES = {r: max([float(SCREENING["class_min_solid_mm3"].get(c, 0.0)) for c in _role_classes(r)] or [0.0])
              for r in set(_ROLES["kind"]) | {x for v in _ROLES["classes"].values() for x in v["roles"]}
              if _takes_step(_ROLES["kind"].get(r), r)}


SOLID_VERSION = "v4"   # 2026-11-23: a rejection is keyed by what the screening solid is made from (STEP sha, tolerance, offset, the
                       # roll-flip turn, the manifest box, the min solid), not the part's name: a renamed body read the rejection a
                       # wrongly turned case had left under its opaque id, and a moved part kept its old one. v3 entries are ignored.
                       # v3 2026-11-22: rejections re-evaluated once (a translated STEP now reports its offset; a tool exception
                       # had been remembered as an unusable STEP); v2 entries are ignored.
                       # v2 (2026-11-19): the part's own closed mesh before any voxel fill; STEP retried with all solids and
                       # coarser on a fresh import. Older not-usable entries were written by v1 and are ignored.


def _step_fits(m, b):
    """Same frame as the manifest mesh (a housing without internals may sit slightly inside it)."""
    if m is None or len(m.faces) == 0 or m.bounds is None:
        return False
    return kit.mf(m).status().name == "NoError" and np.all(m.bounds[0] >= b[0] - 0.5) and np.all(m.bounds[1] <= b[1] + 0.5) \
        and np.abs(m.bounds - b).max() < 3.0


FINE_STEP_MAX_MB = 5.0


def _mb(step):
    try:
        return BL.src(step).stat().st_size / 1e6
    except OSError:
        return 0.0


def screen_solid(body, n, changed):
    """Closed screening solid of part n in the manifest frame and how it was made. Order: native STEP tessellation;
    else the part's own mesh when it is already a closed manifold (decimated only within 1.5 % of its volume); only an
    open mesh gets the voxel fill / hull of roll_flip_checks.closed(), which can be far off (+-50 % on open covers)."""
    p = body.parts[n]
    step = p.get("native_step")
    if step and not BL.src(step).exists():          # a STEP reference that resolves nowhere (R_toe_lateral_sensor): the mesh
        step = None
    stepped = takes_step(n, p)
    # a scoped part gets the fine STEP tessellation, except a bought part (not screened by its STEP when unscoped) with a big
    # STEP: supplier boards hold thousands of component solids (power supervisor 78 MB, ROCK4D 44, B0196 26, B0410 22) and
    # an unscoped run scopes every part, so it never finished (T-c4sy9d, 2026-11-20); those take the coarse path
    fine = n in changed and not (step and not stepped and _mb(step) > FINE_STEP_MAX_MB)
    tol = 0.03 if fine else 0.1                     # unchanged parts: coarser screening mesh, faster sweep
    step_why = None                                 # why the STEP is not the screening solid: named in the solid's `how`
    b = np.array(p["bounds_mm"], float)
    turn = R.turn_T_at(0.5 * (b[0][0] + b[1][0])) if str(p.get("roll_flip", "")).startswith("turned") else None  # the roll-flip case:
    if step and not p.get("placement_offset_mm") and (fine or stepped):                           # turned about the roll axis, on its side
        made_of = json.dumps(dict(offset=p.get("native_step_offset_mm"), turn=None if turn is None else np.round(turn, 6).tolist(),
                                  box=np.round(b, 3).tolist(), min_solid=min_solid_mm3(n, p)))
        fail_key = f"{SOLID_VERSION}:{sha(BL.src(step))[:16]}:{tol}:{hashlib.sha256(made_of.encode()).hexdigest()[:16]}"
        failed = _failed()
        if fail_key in failed and n not in changed:
            step, step_why = None, failed[fail_key]
    if step and not p.get("placement_offset_mm") and (fine or stepped):
        why = None
        # a servo STEP may hold its housing as several small solids (ST3025: 11 pieces, one over 4000 mm3): when the
        # housing filter leaves a solid that does not span the manifest bounds, take every solid
        for mins in dict.fromkeys((min_solid_mm3(n, p), 0.0)):
            try:
                m = step_mesh(BL.src(step), p.get("native_step_offset_mm"), tol=tol, ang=0.2 if tol < 0.05 else 0.5,
                              min_solid_mm3=mins)
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
                break
            if turn is not None:
                m = m.copy(); m.apply_transform(turn)
            if m is None or len(m.faces) == 0:
                why = "empty tessellation"
                break
            if _step_fits(m, b):
                t_used = m.metadata.get("tol", tol)
                return m, f"native STEP tessellation ({t_used} mm)" + (" housing solids" if mins else "")
            why = f"frame or manifold mismatch (solids >= {mins:g} mm3)"
            if m is not None and len(m.faces) and m.bounds is not None:
                d = np.asarray(m.bounds, float) - b
                if np.abs(d[0] - d[1]).max() < 0.05 and np.abs(d[0]).max() > 0.5:   # every face of the box moved alike
                    why = (f"the STEP is the manifest mesh's shape moved by {np.round(d[0], 2).tolist()} mm (all six box faces alike): "
                           "a placement or native_step_offset_mm to fix in its builder")
        if not str(why).startswith(("TypeError", "AttributeError", "NameError", "KeyError", "ImportError", "ModuleNotFoundError")):
            _failed(fail_key, why)          # a property of the STEP: remembered; a tool exception is retried next run
        step_why = why
        print(f"  {n}: STEP solid not used ({why}); mesh solid instead (remembered)", flush=True)
    mesh_file = BL.src(p["mesh"])
    if not mesh_file.is_file():                     # nothing left to screen it with: say which part and why, not a bare stat error
        raise FileNotFoundError(f"{n}: no screening solid: native STEP " + (f"not used ({step_why})" if step_why else "absent")
                                + f" and its mesh {p['mesh']} resolves nowhere")
    p["mesh_sha256"] = sha(mesh_file)               # key roll_flip_checks' solid cache by the actual file
    full = body.mesh(n)
    if full.volume < 0:
        full = full.copy(); full.invert()
    if len(full.faces) and kit.mf(full).status().name == "NoError" and not kit.mf(full).is_empty():
        s = R.slim(full, 40000)
        return s, ("closed CAD mesh" + (" (decimated within 1.5 % of its volume)" if len(s.faces) < len(full.faces) else "")
                   + (f"; STEP not used: {step_why}" if step_why else ""))
    s = R.closed(body, n)
    if s.volume < 0:                                # inside-out cached solid (center_front_shell): manifold3d would see it empty
        s = s.copy(); s.invert()
    hull = s.convex_hull
    how = "convex hull (open mesh)" if abs(s.volume - hull.volume) < 0.005 * hull.volume else "voxel fill (open mesh)"
    if kit.mf(s).status().name != "NoError":
        # manifold3d treats a non-manifold mesh as EMPTY (every overlap 0, every distance = search length): the cached
        # hull of the turned roll case loads that way. Use a freshly built hull and say so.
        s, how = hull, f"convex hull (rebuilt: cached solid was not a manifold, {how})"
    return s, how


def full_closed_mesh(body, n):
    """Part n's own mesh, not decimated, when it is a closed manifold; else None. screen_solid decimates such a mesh to 40k
    faces for neighbour screening (R.slim); the print checks read surface detail (walls, bed, small features) the
    decimation moves, so they measure a mesh-only part on this instead. Additive: screen_solid is unchanged."""
    p = body.parts[n]
    if not p.get("mesh"):
        return None
    mesh_file = BL.src(p["mesh"])
    if not mesh_file.is_file():
        return None
    sha(mesh_file)                                  # an input of the running check (result cache)
    full = body.mesh(n)
    if full.volume < 0:
        full = full.copy(); full.invert()
    if len(full.faces) and kit.mf(full).status().name == "NoError" and not kit.mf(full).is_empty():
        return full
    return None


# ------------------------------------------------------------------------------------------------ poses
def _ranges(body=None):
    """{joint: [lo, hi]}: the body's declared ranges (declare_ranges), or without a body the definitions' own."""
    if body is not None:
        declare_ranges(body)
        return {n: list(j["range_deg"]) for n, j in body.joints.items() if j.get("range_deg")}
    return {n: list(d["range_deg"]) for n, d in declared_joints()[0].items()}


def _end(r, which):
    """'lo' / 'hi' of a declared range."""
    return float(min(r)) if which == "lo" else float(max(r))


def _region_list(body, v, which):
    """[(name, {joint: deg})] of one region (pose_sets.json region_poses.regions[k]), every angle a declared range end (declare_ranges):
    `ends` = each joint alone at both ends; `combinations` = {joint: lo | hi} together (optional `name`); with library / all, `corners`
    = every combination of ends of the listed groups (the joints of a group take the same end). A joint the body lacks is left out of
    a pose; a pose naming a joint without a range is left out and the joint listed (body.unranged_in_poses)."""
    rng = _ranges(body)
    unranged = body.__dict__.setdefault("unranged_in_poses", set())
    P = []

    def pose(ends):
        q = {}
        for j, e in ends.items():
            if j not in body.joints:
                continue
            if j not in rng:
                unranged.add(j)
                return None
            q[j] = _end(rng[j], e)
        return q or None
    for j in v.get("ends", []):
        for e in ("lo", "hi"):
            q = pose({j: e})
            if q:
                P.append((f"{j} {q[j]:+g}", q))
    for c in v.get("combinations", []):
        q = pose(c["ends"])
        if q:
            P.append((c.get("name") or " + ".join(f"{j} {a:+g}" for j, a in q.items()), q))
    if which in ("library", "all"):
        import itertools
        for groups in v.get("corners", []):
            for es in itertools.product(("lo", "hi"), repeat=len(groups)):
                q = pose({j: e for g, e in zip(groups, es) for j in g})
                if q:
                    P.append(("corner " + " ".join(f"{'/'.join(g)} {q[g[0]]:+g}" for g in groups if g[0] in q), q))
    return P


def region_poses(body, which, links):
    """[(name, family, {joint: deg})] of the arm / neck / torso regions (pose_sets.json region_poses) that hold one of `links`, the
    angles taken from the declared joint ranges (_region_list). [] when no region holds a link."""
    regs = [(k, v) for k, v in json.loads((DATA / "pose_sets.json").read_text())["region_poses"]["regions"].items() if set(v["links"]) & set(links)]
    P = []
    for k, v in regs:
        fam = k.split("_")[0]
        P += [(f"{fam}: {nm}", fam, {**body.standing, **q}) for nm, q in _region_list(body, v, which)]
    return P


def fit_poses(body=None):
    """[(name, {joint: deg})] of the exact-fit list, pose_sets.json fit_poses: standing; each sweep's joint alone at the multiples of
    its step_deg inside its declared range (0 left out) and at both declared ends, then at its stress angles (beyond_deg: deliberately
    outside the declared range, reported apart by the checks); then the combined poses. The ranges are the body's (declare_ranges),
    or the definitions' without a body; a sweep joint without a range keeps its stress angles only (listed in body.unranged_in_poses)."""
    fp = json.loads((DATA / "pose_sets.json").read_text())["fit_poses"]
    rng = _ranges(body)
    P = [("standing", {})]
    for sw in fp["sweeps"]:
        r, st = rng.get(sw["joint"]), float(sw["step_deg"])
        angles = []
        if r:
            lo, hi = float(min(r)), float(max(r))
            k0, k1 = int(np.ceil(lo / st - 1e-9)), int(np.floor(hi / st + 1e-9))
            angles = sorted(({lo, hi} | {round(k * st, 6) for k in range(k0, k1 + 1)}) - {0.0})
        elif body is not None:
            body.__dict__.setdefault("unranged_in_poses", set()).add(sw["joint"])
        angles += [float(a) for a in sw.get("beyond_deg", []) if not (r and min(r) <= a <= max(r))]
        P += [(sw["name"].format(deg=a), {sw["joint"]: float(a)}) for a in sorted(angles)]
    return P + [(nm, {j: float(a) for j, a in q.items()}) for nm, q in fp["combined"]]


def _leg_sources(body, spec):
    """[(name, family, {joint: deg})] of a set's leg-motion sources (pose_sets.json: leg_motion KEY / library / GRID on
    leg_motion.joints, fit_poses, range_ends = both declared ends of each listed joint, stress_ends = deliberate angles beyond the
    declared range), each the joints it sets, before standing is filled in. Poses beyond a declared range stay in the set: the
    checks measure them and report them apart (beyond_declared)."""
    from leg_motion import GRID, KEY, library_poses
    J = json.loads((DATA / "pose_sets.json").read_text())["leg_motion"]["joints"]

    def q4(p):
        return dict(zip(J, map(float, p)))
    P = []
    if spec.get("leg_motion_key"):
        P += [(f"key: {k}", "key", q4(v)) for k, v in KEY.items()]
    if spec.get("fit_poses"):       # the exact-fit list (single joints over wide ranges + combined crouches)
        P += [(f"fit: {nm}", "fit poses", cmd) for nm, cmd in fit_poses(body)]
    if spec.get("fit_extremes"):
        P += [(f"fit: {nm}", "fit extremes", cmd) for nm, cmd in fit_poses(body)[-spec["fit_extremes"]:]]
    rng = _ranges(body)
    for j in spec.get("range_ends", {}).get("joints", []):
        if j not in body.joints:
            continue
        if j not in rng:
            body.__dict__.setdefault("unranged_in_poses", set()).add(j)
            continue
        P += [(f"range end: {j} {a:+g}", "range ends", {j: a}) for a in (_end(rng[j], "lo"), _end(rng[j], "hi"))]
    P += [(f"stress end: {j} {a:+g}", "stress ends", {j: float(a)}) for j, a in spec.get("stress_ends", []) if j in body.joints]
    if spec.get("leg_motion_library"):
        P += [(f"{fam} {np.round(p, 1).tolist()}", fam, q4(p)) for fam, p, _ in library_poses()]
    if spec.get("leg_motion_grid"):
        P += [(f"all combinations {list(g)}", "all combinations", q4(g)) for g in GRID]
    return P


def joint_mirrors(body):
    """{joint: (mirror joint, angle sign)}: the joint table's (checks.relations.joint_mirrors: origin and axis mirrored), plus the
    mirrors pose_sets.json joint_mirrors declares, both ways. Cached on the body."""
    if getattr(body, "_joint_mirrors", None) is None:
        from checks.relations import joint_mirrors as derived
        jm = derived(body.joints)
        for a, (b, s) in ((k, v) for k, v in json.loads((DATA / "pose_sets.json").read_text())["joint_mirrors"].items() if not k.startswith("_")):
            if a in body.joints and b in body.joints:
                jm[a], jm[b] = (b, float(s)), (a, float(s))
        body._joint_mirrors = jm
    return body._joint_mirrors


def leg_sides(body, joints, links):
    """(as named, mirrored): which side(s) the leg-motion sources, which set `joints`, are given for scoped parts on `links`.
    Limbs and mirror joints come from the joint table (checks.relations.limbs / joint_mirrors), not names: mirrored when a
    scoped link is on the mirror limb of a limb the sources move; as named when a scoped link is on a limb they move, on no
    limb (torso: it meets both legs), or when nothing is scoped or no mirror applies."""
    from checks.relations import limbs
    lim = limbs(body.joints, {p["owning_link"] for p in body.parts.values()})
    jm = joint_mirrors(body)
    src = {lim.get(body.joints[j]["child"]) for j in joints if j in body.joints} - {None}
    mirror = {jm[r][0] for r in src if r in jm}
    scoped = {lim.get(lk) for lk in links}
    mirrored = bool(scoped & mirror)
    return (not mirrored or bool(scoped & src) or None in scoped), mirrored


def mirror_cmd(body, cmd):
    """cmd's angles moved onto the mirror joints (angle x sign, joint_mirrors); a joint without a mirror keeps its own."""
    jm = joint_mirrors(body)
    return {jm.get(j, (j, 1.0))[0]: jm.get(j, (j, 1.0))[1] * a for j, a in cmd.items()}


def pose_set(body, which, links=()):
    """[(name, family, {joint: deg})] of a named set in data/pose_sets.json. links: owning links of the scoped parts; the parts on an arm,
    neck or torso link get that region's poses (region_poses) in place of the leg-motion poses; leg parts and mixed scopes keep the leg set
    (mixed: plus the region poses). The leg set names one leg's joints; scoped parts on the other leg get it mirrored onto that leg
    (leg_sides, mirror_cmd: '(mirror)' after the pose name, a range or stress end named by its own joint), a scope on both legs gets
    both. Range ends and region poses are the declared ranges' ends (declare_ranges); poses beyond a declared range are kept
    (beyond_declared tells them apart)."""
    spec = json.loads((DATA / "pose_sets.json").read_text())[which]
    st = body.standing
    reg = region_poses(body, which, links) if links else []
    in_region = {lk for v in json.loads((DATA / "pose_sets.json").read_text())["region_poses"]["regions"].values() for lk in v["links"]}
    if reg and set(links) <= in_region:
        return [("standing", "standing", dict(st))] + reg
    src = _leg_sources(body, spec)
    as_named, mirrored = leg_sides(body, {j for _, _, c in src for j in c}, links)
    P = [("standing", "standing", dict(st))] if spec.get("standing") else []
    if as_named:
        P += [(nm, fam, {**st, **cmd}) for nm, fam, cmd in src]
    if mirrored:
        for nm, fam, cmd in src:
            m = mirror_cmd(body, cmd)
            if fam in ("range ends", "stress ends"):
                (j, a), = m.items()
                nm = f"{fam[:-1]}: {j} {a:+g}"
            else:
                nm = f"{nm} (mirror)"
            P.append((nm, fam, {**st, **m}))
    return P + reg
