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

kit = R.kit
WORK = ROOT / ".cache/work/cad-speed"          # reports: this checkout's own
CACHE = cachefs.work() / "cad-speed/cache"     # content-addressed caches: shared with the worktrees (checks/cachefs.py)
DATA = Path(__import__("os").environ.get("LAB_CHECKS_DATA") or HERE / "data")   # LAB_CHECKS_DATA: another set of check tables
T = {}


def tick(name, t0):
    T[name] = round(T.get(name, 0.0) + time.time() - t0, 2)


def load_body(manifest: Path):
    """Body (parts, joints, link_T) of one manifest, loaded once."""
    return kit.base(manifest)


# ------------------------------------------------------------------------------------------------ hashing / caches
_SHA_FILE = CACHE / "sha256.json"
_SHA: dict = {}
_SHA_NEW: dict = {}


def sha(path: Path) -> str:
    """sha256 of a file, remembered by (absolute path, size, mtime_ns) across runs."""
    global _SHA
    resultcache.note(path)                  # an input of the running check (checks/resultcache.py)
    if not _SHA:
        _SHA = cachefs.read_json(_SHA_FILE, {}) or {}
    st = path.stat()
    k = str(path)
    if k in _SHA and _SHA[k][0] == st.st_size and _SHA[k][1] == st.st_mtime_ns:
        return _SHA[k][2]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
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


STEP_ROLES = {"printed_structural_connector": 0.0, "printed_structural_carrier": 0.0,
              "purchased_servo_case_family_reference": 4000.0, "motor": 4000.0}   # min solid volume kept (mm3)
# (ST3025 "motor" STEPs hold the housing as 11 pieces, only one over 4000 mm3: screen_solid then retries with every solid)


SOLID_VERSION = "v3"   # 2026-11-22: rejections re-evaluated once (a translated STEP now reports its offset; a tool exception
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
    role = p.get("role")
    # a scoped part gets the fine STEP tessellation, except a bought part (role outside STEP_ROLES) with a big STEP:
    # supplier boards hold thousands of component solids (power supervisor 78 MB, ROCK4D 44, B0196 26, B0410 22) and
    # an unscoped run scopes every part, so it never finished (T-c4sy9d, 2026-11-20); those take the coarse path
    fine = n in changed and not (step and role not in STEP_ROLES and _mb(step) > FINE_STEP_MAX_MB)
    tol = 0.03 if fine else 0.1                     # unchanged parts: coarser screening mesh, faster sweep
    step_why = None                                 # why the STEP is not the screening solid: named in the solid's `how`
    if step and not p.get("placement_offset_mm") and (fine or role in STEP_ROLES):
        fail_key = f"{SOLID_VERSION}:{n}:{sha(BL.src(step))[:16]}:{tol}"
        failed = _failed()
        if fail_key in failed and n not in changed:
            step, step_why = None, failed[fail_key]
    if step and not p.get("placement_offset_mm") and (fine or role in STEP_ROLES):
        b = np.array(p["bounds_mm"], float)
        why = None
        # a servo STEP may hold its housing as several small solids (ST3025: 11 pieces, one over 4000 mm3): when the
        # housing filter leaves a solid that does not span the manifest bounds, take every solid
        for mins in dict.fromkeys((STEP_ROLES.get(role, 0.0), 0.0)):
            try:
                m = step_mesh(BL.src(step), p.get("native_step_offset_mm"), tol=tol, ang=0.2 if tol < 0.05 else 0.5,
                              min_solid_mm3=mins)
            except Exception as exc:  # noqa: BLE001
                why = f"{type(exc).__name__}: {exc}"
                break
            if str(p.get("roll_flip", "")).startswith("turned"):   # the roll-flip case: turned about the roll axis
                m = m.copy(); m.apply_transform(R.turn_T(n[0]))
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


# ------------------------------------------------------------------------------------------------ poses
def region_poses(body, which, links):
    """[(name, family, {joint: deg})] of the arm / neck / torso regions (pose_sets.json region_poses) that hold one of `links`; standing
    first. Joints the body lacks are left out of a pose. [] when no region holds a link."""
    regs = [(k, v) for k, v in json.loads((DATA / "pose_sets.json").read_text())["region_poses"]["regions"].items() if set(v["links"]) & set(links)]
    P = []
    for k, v in regs:
        fam = k.split("_")[0]
        lists = [v["key"]] + ([v["library_extra"]] if which in ("library", "all") else [])
        for L in lists:
            P += [(f"{fam}: {nm}", fam, {**body.standing, **{j: float(a) for j, a in q.items() if j in body.joints}}) for nm, q in L.items()]
    return P


def pose_set(body, which, links=()):
    """[(name, family, {joint: deg})] of a named set in data/pose_sets.json. links: owning links of the scoped parts; the parts on an arm,
    neck or torso link get that region's poses (region_poses) in place of the leg-motion poses; leg parts and mixed scopes keep the leg set
    (mixed: plus the region poses)."""
    from leg_motion import GRID, J, KEY, library_poses
    spec = json.loads((DATA / "pose_sets.json").read_text())[which]
    st = body.standing
    reg = region_poses(body, which, links) if links else []
    in_region = {lk for v in json.loads((DATA / "pose_sets.json").read_text())["region_poses"]["regions"].values() for lk in v["links"]}
    if reg and set(links) <= in_region:
        return [("standing", "standing", dict(st))] + reg

    def q4(p):
        return {**st, **dict(zip(J, map(float, p)))}
    P = []
    if spec.get("standing"):
        P.append(("standing", "standing", dict(st)))
    if spec.get("leg_motion_key"):
        P += [(f"key: {k}", "key", q4(v)) for k, v in KEY.items()]
    if spec.get("fit_poses"):       # roll_flip_checks.fit_poses(): the exact-fit list (single joints over wide ranges + combined crouches)
        P += [(f"fit: {nm}", "fit poses", {**st, **cmd}) for nm, cmd in R.fit_poses()]
    if spec.get("fit_extremes"):
        P += [(f"fit: {nm}", "fit extremes", {**st, **cmd}) for nm, cmd in R.fit_poses()[-spec["fit_extremes"]:]]
    P += [(f"range end: {j} {a:+g}", "range ends", {**st, j: float(a)}) for j, a in spec.get("range_ends", [])]
    if spec.get("leg_motion_library"):
        P += [(f"{fam} {np.round(p, 1).tolist()}", fam, q4(p)) for fam, p, _ in library_poses()]
    if spec.get("leg_motion_grid"):
        P += [(f"all combinations {list(g)}", "all combinations", q4(g)) for g in GRID]
    return P + reg
