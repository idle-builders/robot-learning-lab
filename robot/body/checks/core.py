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

kit = R.kit
WORK = ROOT / ".cache/work/cad-speed"
CACHE = WORK / "cache"
DATA = HERE / "data"
T = {}


def tick(name, t0):
    T[name] = round(T.get(name, 0.0) + time.time() - t0, 2)


def load_body(manifest: Path):
    """Body (parts, joints, link_T) of one manifest, loaded once."""
    return kit.base(manifest)


# ------------------------------------------------------------------------------------------------ hashing / caches
_SHA_FILE = CACHE / "sha256.json"
_SHA: dict = {}


def sha(path: Path) -> str:
    """sha256 of a file, remembered by (size, mtime_ns) across runs."""
    global _SHA
    if not _SHA and _SHA_FILE.exists():
        _SHA = json.loads(_SHA_FILE.read_text())
    st = path.stat()
    k = str(path)
    if k in _SHA and _SHA[k][0] == st.st_size and _SHA[k][1] == st.st_mtime_ns:
        return _SHA[k][2]
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    _SHA[k] = (st.st_size, st.st_mtime_ns, h)
    return h


def save_sha():
    CACHE.mkdir(parents=True, exist_ok=True)
    _SHA_FILE.write_text(json.dumps(_SHA))


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
    if f.exists():
        d = np.load(f)
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
        np.savez_compressed(f, v=np.asarray(m.vertices), f=np.asarray(m.faces), t=t_try)
        m.metadata["tol"] = t_try
        return m
    raise err


_FAILED_FILE = CACHE / "step-mesh" / "not-usable.json"


def _failed(key=None, why=None):
    """STEPs that did not give a usable screening solid (keyed by part, STEP sha, tolerance): not retried."""
    d = json.loads(_FAILED_FILE.read_text()) if _FAILED_FILE.exists() else {}
    if key:
        d[key] = why
        _FAILED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _FAILED_FILE.write_text(json.dumps(d, indent=0))
    return d


STEP_ROLES = {"printed_structural_connector": 0.0, "printed_structural_carrier": 0.0,
              "purchased_servo_case_family_reference": 4000.0, "motor": 4000.0}   # min solid volume kept (mm3)
# (ST3025 "motor" STEPs hold the housing as 11 pieces, only one over 4000 mm3: screen_solid then retries with every solid)


SOLID_VERSION = "v2"   # 2026-11-19: the part's own closed mesh before any voxel fill; STEP retried with all solids and
                       # coarser on a fresh import. Older not-usable entries were written by v1 and are ignored.


def _step_fits(m, b):
    """Same frame as the manifest mesh (a housing without internals may sit slightly inside it)."""
    if m is None or len(m.faces) == 0 or m.bounds is None:
        return False
    return kit.mf(m).status().name == "NoError" and np.all(m.bounds[0] >= b[0] - 0.5) and np.all(m.bounds[1] <= b[1] + 0.5) \
        and np.abs(m.bounds - b).max() < 3.0


def screen_solid(body, n, changed):
    """Closed screening solid of part n in the manifest frame and how it was made. Order: native STEP tessellation;
    else the part's own mesh when it is already a closed manifold (decimated only within 1.5 % of its volume); only an
    open mesh gets the voxel fill / hull of roll_flip_checks.closed(), which can be far off (+-50 % on open covers)."""
    p = body.parts[n]
    step = p.get("native_step")
    role = p.get("role")
    tol = 0.03 if n in changed else 0.1             # unchanged parts: coarser screening mesh, faster sweep
    if step and not p.get("placement_offset_mm") and (n in changed or role in STEP_ROLES):
        fail_key = f"{SOLID_VERSION}:{n}:{sha(BL.src(step))[:16]}:{tol}"
        failed = _failed()
        if fail_key in failed and n not in changed:
            step = None
    if step and not p.get("placement_offset_mm") and (n in changed or role in STEP_ROLES):
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
        _failed(fail_key, why)
        print(f"  {n}: STEP solid not used ({why}); mesh solid instead (remembered)", flush=True)
    mesh_file = BL.src(p["mesh"])
    p["mesh_sha256"] = sha(mesh_file)               # key roll_flip_checks' solid cache by the actual file
    full = body.mesh(n)
    if full.volume < 0:
        full = full.copy(); full.invert()
    if len(full.faces) and kit.mf(full).status().name == "NoError" and not kit.mf(full).is_empty():
        s = R.slim(full, 40000)
        return s, "closed CAD mesh" + (" (decimated within 1.5 % of its volume)" if len(s.faces) < len(full.faces) else "")
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
