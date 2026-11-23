"""Roll-flip hip variant (owner D-12 / D-13, hip-shell F-03 variant B): the manifest builder, the helpers that the hip-shell
and thigh-shell scripts import (kit, closed(), library_poses(), merge(), save_npz(), STEEL, VARIANT, OUT, F, fit_poses(), brep()),
the mass / COM report, and thin wrappers that run the shared checks (robot/body/checks/run.py) and write their results into
artifacts/robot/body/leg/roll-flip/report.json in the sections the roll-flip publication reads.

  manifest  assembly-manifest variant next to the shell-pass one (assembly-parts-roll-flip.json): case turned and
            moved to the roll link, horn / idler to the pitch link, new Stage 4 / 5 in place of the selected ones;
            it starts from the shell-pass manifest, so run `thigh_shell.py build` and then `thigh_shell.py shin-cover`
            after it (they put the blue thigh shells and the trimmed shin back cover back into the variant)
  mass      masses of new vs old parts; mass, COM and inertia of R_pitch_link and R_roll_link before and after
  fit       wrapper: checks `fit` (exact B-rep, --poses fit) of the roll-flip Stage 4 / 5 -> report.fit_brep.worst_by_pair
            (--stage5-only: Stage 5 only, Stage 4 rows kept from the previous report)
  press     wrapper: `press roll-flip` / `press selected` -> report.press_roll_flip / press_selected (checks `press`)
  fem KEY   wrapper: checks `fem` of the part of KEY (stage4_roll_flip, stage5_roll_flip, stage4_selected, stage5_selected)
            -> report.fem[KEY]

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/leg/roll_flip_checks.py STEP
Every result is a continuous measurement; the checks accept or reject nothing."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
HIP = ROOT / "robot/body/hip-shell"
sys.path.insert(0, str(HIP)); sys.path.insert(0, str(HIP / "sections"))
import kit  # noqa: E402  (puts robot/body/motor-refit first on sys.path: fit_check there is the motor-refit one)
from leg_motion import GRID, J, library_poses, solid  # noqa: E402

sys.path.append(str(HERE))
import build_upper_hip_connectors as B  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(ROOT / "robot/body"))
from checks import brep as LFIT  # noqa: E402                    # B-rep helpers (common volume, distance, transform, boxes)
F = kit.F                                                       # motor-refit fit_check shim of checks/bodylib (Body, load_mesh, src)

OUT = B.RF_ART
REPORT = OUT / "report.json"
SHELL_PASS = B.RF_SHELL_PASS
VARIANT = SHELL_PASS.with_name("assembly-parts-roll-flip.json")
MOVED = OUT / "parts" / "moved"
NEW4, NEW5 = "thigh_roll_flip", "hip_pitch_to_roll_roll_flip"
OLD4, OLD5 = "thigh_combined", "hip_pitch_to_roll_combined"
CASE = "roll_servo_case_reference"
COVERS = ("thigh_front_hard_cover", "thigh_back_hard_cover")


def merge(section, data):
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    rep[section] = data
    rep["source_code"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(B.__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")


def turn_T(side):
    """Case turn about its own shaft (= the roll axis): Ry(RF_TURN_DEG) about (56, 190) on the right, mirrored left."""
    a = np.radians(B.RF_TURN_DEG)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    p = np.array([B.RF_AX, 0.0, B.RF_AZ])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - R @ p
    if side == "L":
        M = np.diag([-1.0, 1.0, 1.0, 1.0]); T = M @ T @ M
    return T


def turn_T_at(x_mm):
    """turn_T for a case whose box centre is at x_mm: the right-side turn on RF_AX's side of the mirror plane (geometry
    mirror_plane_x_mm, checks/data/roles.json), its mirror on the other side. The side comes from where the part is, not its name."""
    from checks.relations import GEO
    x0 = GEO["mirror_plane_x_mm"]
    return turn_T("R" if (x_mm - x0) * (B.RF_AX - x0) > 0 else "L")


# ------------------------------------------------------------------------------------------------ manifest variant
def save_npz(name, mesh):
    MOVED.mkdir(parents=True, exist_ok=True)
    f = MOVED / f"{name}.npz"
    np.savez_compressed(f, vertices=np.asarray(mesh.vertices, float), faces=np.asarray(mesh.faces, np.int64))
    return dict(mesh=str(f.relative_to(ROOT)), mesh_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), mesh_units="mm",
                bounds_mm=[np.round(mesh.bounds[0], 3).tolist(), np.round(mesh.bounds[1], 3).tolist()])


OLD_KNEE_SCREWS = "A9_BPA2x3_proposal_"     # the old C047 knee's 4 PA2x3 per leg (shell-pass manifest)
STEEL = 7.85e-3                             # g/mm3


def _generated(name):
    """Is this manifest part one of the screws hip_screws() generates? (The names stay as they were: part settings and the
    assembly order name them; the generated thread and length are in the part's screw_plan.)"""
    import re
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import screw_plan as SP
    for conn in SP.connectors().values():
        for g in conn["groups"].values():
            if re.fullmatch(re.escape(g["name"]).replace(r"\{S\}", "[RL]_").replace(r"\{i\}", r"\d+"), name):
                return True
    return False


def _mirrored_holes(spec, iface, conn, r_open, tol=0.2):
    """The connector spec for the left side with each group's hole labels chosen as the mirror (x -> -x) of the right side's
    screws: the mirrored standard parts number their holes differently, so the right labels (STS3250.idler-y.1 / .2 / .4,
    the .3 dropped for the stub slot) put a left screw in the dropped site and left a sleeve without one. A right opening
    with no left hole within tol keeps its label."""
    import copy
    spec = copy.deepcopy(spec)
    for gname, g in spec["groups"].items():
        ro = r_open.get((conn, gname))
        if not ro:
            continue
        sp = g["standard_part"].replace("{S}", "L_")
        cand = {h["hole_id"].split(":", 1)[1]: np.asarray(h["opening"], float) for h in iface["holes"]
                if h["hole_id"].startswith(sp + ":")}
        labels = []
        for k, o in enumerate(ro):
            t = o * [-1.0, 1.0, 1.0]
            lb, d = min(((lb, float(np.linalg.norm(c - t))) for lb, c in cand.items()), key=lambda q: q[1], default=(None, np.inf))
            labels.append(lb if d <= tol else g["holes"][k])
        g["holes"] = labels
    return spec


def hip_screws(parts, man, only=None):
    """The hip / thigh screws of both sides, generated from the defined tapped holes they land in (robot/body/fastening/
    screw_plan.py, connector_screws.json): axis = the registered standard part's hole axis, head on the clamped parts'
    outer face, family by terminal material, standard length (data/hardware.json screw_standards). `parts` is the
    manifest part list without these screws; `only` = connector names to generate (default all). Returns (rows, plans)."""
    sys.path.insert(0, str(ROOT / "robot/body/fastening"))
    import interfaces as I
    import screw_plan as SP
    body = F.Body(dict(man, parts=parts))
    iface = I.instances(body, man)
    rows, plans = [], []
    r_open = {}                                    # (connector, group) -> R openings in plan order
    for side in ("R_", "L_"):
        for conn in (only or SP.connectors()):
            spec = SP.connectors()[conn]
            if side == "L_":
                spec = _mirrored_holes(spec, iface, conn, r_open)
            for pl in SP.plan_connector(body, iface, conn, side, spec=spec):
                if side == "R_":
                    r_open.setdefault((conn, pl["group"]), []).append(np.asarray(pl["opening"], float))
                m = SP.screw_mesh(pl)
                nm = pl["name"]
                lbl = f"{pl['thread_label']}x{pl['length_mm']:g}"
                rows.append(dict(name=nm, physical_part_id=nm, bom_group=pl["bom_group"], owning_link=pl["owning_link"],
                                 role="fastener_proxy", **save_npz(nm, m), volume_mm3=round(float(m.volume), 3),
                                 mass_kg=round(float(m.volume) * STEEL / 1000.0, 7),
                                 material=f"{pl['head_standard']} {lbl} into {pl['hole_id'].split(':')[1]} ({pl['terminal_material']}); "
                                          f"head dia {pl['head_dia_mm']} x {pl['head_height_mm']}, shank dia {pl['thread'].lstrip('M')}, "
                                          f"{pl['point_mm']} mm point (envelope)",
                                 color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                                 overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly",
                                 inspection_reference=False, mass_basis="mesh volume x 7.85 g/cm3",
                                 roll_flip="generated from its defined hole by robot/body/fastening/screw_plan.py",
                                 screw_plan={k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in pl.items()
                                             if k in ("hole_id", "thread_label", "head_standard", "length_mm", "stack_mm", "head_gap_mm",
                                                      "engagement_mm", "reference_engagement_mm", "hole_depth_mm", "tip_margin_mm",
                                                      "shortfall_mm", "overrun_mm", "stack_change_to_fit_mm")}))
                plans.append(pl)
    return rows, plans


def _keep_shell_step_rows(man):
    """Carry the outputs of `thigh_shell.py build` / `shin-cover` over from the previous variant manifest, so a manifest
    rebuild without those two steps no longer drops the blue thigh shells and their post hardware or restores the
    untrimmed shin back cover (it did on 2026-11-21). A row is carried only while it still matches its source in the
    new manifest (the trimmed cover's trim_source_mesh, the drilled shin frame's pilot_source_step); otherwise the new
    row stays and a warning says to rerun the shell steps. Rerun them whenever Stage 4 or a covered part changes."""
    if not VARIANT.exists():
        return ["WARNING no previous variant manifest: run thigh_shell.py build and shin-cover"]
    prev = {p["name"]: p for p in json.loads(VARIANT.read_text())["parts"]}
    cur = {p["name"]: p for p in man["parts"]}
    log = []

    def shell_row(n):
        b = n[2:]
        return (b in ("thigh_shell_front", "thigh_shell_back", "thigh_shell_front_steel_sleeves", "thigh_shell_back_steel_sleeves")
                or b.startswith(("Thigh_carrier_shell_thigh_front_", "Thigh_rear_carrier_shell_thigh_back_", "shin_back_cover_PA2")))
    carried = [p for n, p in prev.items() if shell_row(n) and n not in cur]
    if not any(p["name"][2:] in ("thigh_shell_front", "thigh_shell_back") for p in carried + list(cur.values())):
        log.append("WARNING the previous manifest has no thigh shells: run thigh_shell.py build and shin-cover")
    parts = list(man["parts"])
    if any(p["name"][2:] == "thigh_shell_front" for p in carried):
        parts = [p for p in parts if p["name"][2:] not in ("thigh_front_hard_cover", "thigh_back_hard_cover")]
    if any(p["name"][2:].startswith("shin_back_cover_PA2") for p in carried):
        parts = [p for p in parts if not p["name"][2:].startswith("Shin_carrier_shell_shin_back")]
    for i, p in enumerate(parts):
        q = prev.get(p["name"])
        if not q:
            continue
        if q.get("trim_source_mesh") and q["trim_source_mesh"] == p.get("mesh"):
            parts[i] = copy.deepcopy(q); log.append(f"{p['name']}: trimmed shin back cover carried over (shin-cover output)")
        elif q.get("trim_source_mesh"):
            log.append(f"WARNING {p['name']}: its source changed, untrimmed row kept: run thigh_shell.py shin-cover")
        if q.get("pilot_source_step") and q["pilot_source_step"] == p.get("native_step"):
            parts[i] = copy.deepcopy(q); log.append(f"{p['name']}: drilled shin frame carried over (shin-cover output)")
        elif q.get("pilot_source_step"):
            log.append(f"WARNING {p['name']}: its source changed, undrilled frame kept: run thigh_shell.py shin-cover")
    man["parts"] = parts + [copy.deepcopy(p) for p in carried]
    if carried:
        log.append(f"{len(carried)} thigh-shell / shin-cover rows carried over from the previous manifest "
                   "(rerun thigh_shell.py build and shin-cover when Stage 4 or a covered part changed)")
    return log


def build_manifest():
    man = json.loads(SHELL_PASS.read_text())
    cad = json.loads((OUT / "cad-manifest.json").read_text())
    new = {r["name"]: r for r in cad["parts"]}
    body = F.Body(man)
    parts, log = [], []
    for p in man["parts"]:
        n = p["name"]; s = n[:2]
        base = n[2:]
        if base in (OLD4, OLD4 + "_steel_sleeves", OLD5, OLD5 + "_steel_sleeves"):
            log.append(f"{n}: replaced"); continue
        if base.startswith(OLD_KNEE_SCREWS) or _generated(n):
            log.append(f"{n}: legacy screw, replaced by the one generated from its defined hole"); continue
        q = copy.deepcopy(p)
        if base == CASE or base.startswith("A13_roll_face"):
            m = body.mesh(n).copy(); m.apply_transform(turn_T(s[0]))
            q.update(save_npz(n, m)); q["owning_link"] = f"{s}roll_link"
            q["roll_flip"] = f"turned {B.RF_TURN_DEG:+.0f} deg about the roll axis (Ry, URDF sign), rides with the thigh"
            q.pop("placement_offset_mm", None)
            log.append(f"{n}: turned, now on {q['owning_link']}")
        elif base in ("roll_upper_horn_reference", "roll_passive_idler_reference") or base.startswith("A9_hip_M3x4"):
            q["owning_link"] = f"{s}pitch_link"; q["roll_flip"] = "unchanged place, bolted to the new Stage 5 (pitch link)"
            log.append(f"{n}: now on {q['owning_link']}")
        elif "Thigh_carrier_shell" in n or "Thigh_rear_carrier_shell" in n:
            # the old cover screws / nuts: thigh_shell.py build generates them at the posts (post_screws), with sleeves
            log.append(f"{n}: legacy thigh-cover hardware, replaced by thigh_shell.py build"); continue
        parts.append(q)
    for r in cad["parts"]:
        for nm, src, role, mass in ((r["name"], r, "printed_structural_connector", r["polymer_mass_g"] / 1000.0),
                                    (r["sleeves"]["name"], r["sleeves"], "compression_limit_sleeves", r["sleeves"]["mass_g"] / 1000.0)):
            parts.append(dict(name=nm, owning_link=r["owning_link"], role=role, native_step=src["native_step"],
                              native_step_sha256=src["native_step_sha256"], mesh=src["mesh"], mesh_sha256=src["mesh_sha256"],
                              mesh_units="mm", bounds_mm=src["bounds_mm"], mass_kg=round(mass, 6), volume_mm3=src["volume_mm3"],
                              color=[228, 190, 40, 255] if "pitch_to_roll" in nm and role != "compression_limit_sleeves"
                              else [64, 156, 212, 255] if role != "compression_limit_sleeves" else [215, 222, 230, 255],
                              mass_basis=f"native STEP volume x {r['polymer_density_g_cm3']} g/cm3 ({r['polymer_material']}, "
                                         "part setting material, checks/data/materials.json)" if role != "compression_limit_sleeves"
                              else "304 SS sleeves, native STEP volume", roll_flip="new part (build_upper_hip_connectors.py --roll-flip)"))
    sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
    import ankle_coupler_screws as ACS      # run its `build` first (T-9wpegg): edited ankle coupler + sleeves
    parts = ACS.swap_parts(parts)
    sys.path.insert(0, str(ROOT / "robot/body/torso"))
    import torso_screws as TS               # T-nh6ytf: stage 3 carrier rebuilt, stale torso_mate hardware dropped (warns when not built)
    parts = TS.swap_parts(parts)
    import foot_screws as FS                # run its `build` first (T-rptay5): edited foot monocoque + sleeves
    parts = FS.swap_parts(parts)
    import hip_yaw_screws as HYS            # run its `build` first (T-v3kzs4): edited hip yaw-to-pitch connector + sleeves
    parts = HYS.swap_parts(parts)
    sys.path.insert(0, str(ROOT / "robot/body/assembly"))
    import held_parts as HP                 # run its `build` first (T-3npyjq): fan bridge plate, chest indicator, R roll case, stale screws
    parts = HP.swap_parts(parts)
    with FS.left_labels():
        srows, plans = hip_screws(parts, man)
    parts += srows
    log += [f"{r['name']}: {r['screw_plan']['thread_label']}x{r['screw_plan']['length_mm']:g} ({r['screw_plan']['head_standard']}) into {r['screw_plan']['hole_id']}" for r in srows]
    man["parts"] = parts
    sys.path.insert(0, str(ROOT / "robot/body/neck-head"))
    import neck_parts as NP                 # T-6dvwjq: rebuilt XM335 neck connectors + their screws, when built (warns if not)
    log += NP.apply(man)
    import neck_collar as NC                # run its `build` first (T-a2mt33): head_neck_collar screwed to the yaw-to-pitch link (after NP: edits the link)
    man["parts"] = NC.swap_parts(man["parts"])
    log += _keep_shell_step_rows(man)
    parts = man["parts"]
    man["study_id"] = man["result_id"] = "cove50-hip-roll-flip"
    man["roll_flip"] = dict(idle_task="T-fs5v6d", owner=["D-12", "D-13"], source_manifest=str(SHELL_PASS.relative_to(ROOT)),
                            source_manifest_sha256=hashlib.sha256(SHELL_PASS.read_bytes()).hexdigest(),
                            cad_manifest=str((OUT / "cad-manifest.json").relative_to(ROOT)), changes=log,
                            note="Kinematics (joints, ranges, URDF) unchanged; only part placement and ownership change. "
                                 "Thigh hard covers are the selected ones and are being redesigned.")
    prev_rf = json.loads(VARIANT.read_text()).get("roll_flip", {}) if VARIANT.exists() else {}
    if "thigh_shell" in prev_rf and any(p["name"][2:] == "thigh_shell_front" for p in parts):
        man["roll_flip"]["thigh_shell"] = prev_rf["thigh_shell"]     # written by thigh_shell.py build; kept with its rows
    VARIANT.write_text(json.dumps(man, indent=1) + "\n")
    print(f"wrote {VARIANT.relative_to(ROOT)} ({len(parts)} parts)")
    for line in log:
        print(" ", line)


def brep(name, man_parts, turned=False, min_solid_mm3=0.0):
    """The part's native STEP solids in the manifest frame; min_solid_mm3: only solids of at least this volume (a servo's
    external housing without its internal gears: the part's exact_fit.min_solid_mm3 setting); turned: the roll-flip turn of
    the case's side (turn_T_at its box centre)."""
    from build123d import Compound, import_step, Location
    p = man_parts[name]
    path = p.get("native_step") or p.get("native_step_unmodified")
    sh = import_step(str(F.src(path)))
    off = p.get("native_step_offset_mm") or [0.0, 0.0, 0.0]
    sh = sh.moved(Location(tuple(float(v) for v in off)))
    sols = sh.solids()
    if min_solid_mm3:
        sols = [s for s in sols if s.volume >= min_solid_mm3]
    c = Compound(children=list(sols))
    return LFIT.transform_shape(c, turn_T_at(0.5 * sum(float(b[0]) for b in p["bounds_mm"]))) if turned else c


def fit_poses():
    """The exact-fit pose list [(name, {joint: deg})]: checks/data/pose_sets.json fit_poses (fit_check_upper_hip.py's list on the
    current joint names, plus the hip-shell target extremes), read by checks.core."""
    from checks import core
    return core.fit_poses()


# ------------------------------------------------------------------------------------------------ closed solids
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import cachefs  # noqa: E402
from checks.relations import ROLES as _ROLES  # noqa: E402   (role classes: which roles are a servo case)
SOLIDS = cachefs.work() / "roll-flip/solids"    # keyed by the part name and its mesh sha256: shared with the worktrees
VOX = 0.4


def closed(body, n):
    """Closed mesh of a part for volume overlaps. leg_motion.solid() falls back to the convex hull when its voxel fill
    does not come out watertight; that happens for the shin, the battery cradle, the selected Stage 5 and the new
    parts (hull / true volume up to 7x). Here the voxel field is smoothed before marching cubes so the surface comes
    out manifold; genuinely convex parts (motor boxes, cells) keep leg_motion's solid."""
    import trimesh
    p = body.parts[n]
    tag = (p.get("mesh_sha256") or p.get("native_step_sha256") or "nohash")[:12]
    f = SOLIDS / f"{n}-{tag}.npz"
    d = cachefs.load_npz(f) if f.exists() else None
    if d is not None:
        return trimesh.Trimesh(d["v"], d["f"], process=False)
    s = solid(body, n)
    if p.get("role") in _ROLES["classes"]["servo_case"]["roles"]:   # a servo case (role class, checks/data/roles.json) is box-like:
        return s                                                      # leg_motion's solid (its hull), as in F-03
    hull = s.convex_hull
    m = s
    if s.is_watertight and s.volume < 0.98 * hull.volume:   # leg_motion's solid is already a real (non-hull) closed mesh
        return s
    step = p.get("native_step")
    g = None
    if p.get("role") == "printed_structural_connector" and step:   # gmsh conforming surface of the native STEP,
        import subprocess                                           # in a subprocess under a hard 300 s limit
        tmp = SOLIDS / f"{n}-gmsh-{os.getpid()}.npz"; SOLIDS.mkdir(parents=True, exist_ok=True)
        code = ("import sys, numpy as np; sys.path.insert(0, %r); from surface_mesh import step_to_closed_mesh; "
                "from build123d import import_step; "
                "g = step_to_closed_mesh(%r, 1.0); vs = sum(s.volume for s in import_step(%r).solids()); "
                "np.savez(%r, v=g.vertices, f=g.faces, vs=vs)"
                % (str(ROOT / "robot/body/motor-refit"), str(F.src(step)), str(F.src(step)), str(tmp)))
        try:
            subprocess.run([sys.executable, "-B", "-c", code], timeout=300, check=True, capture_output=True)
            d = np.load(tmp); g = trimesh.Trimesh(d["v"], d["f"], process=True); v_step = float(d["vs"]); tmp.unlink()
        except Exception as exc:  # noqa: BLE001
            print(f"  {n}: gmsh surface not used ({type(exc).__name__}); voxel fill instead", flush=True)
    if g is not None:
        g.apply_translation(p.get("native_step_offset_mm") or [0.0, 0.0, 0.0])
        v_ref = p.get("volume_mm3") or v_step      # the manifest's volume, else the STEP's own (sum of its solids)
        if g.volume < 0:
            g.invert()
        ok = kit.mf(g).status().name == "NoError"   # manifold3d accepts the few 4-face edges where faces touch
        if ok and abs(g.volume - v_ref) < 0.02 * v_ref:
            g = slim(g)
            cachefs.savez(f, v=np.asarray(g.vertices), f=np.asarray(g.faces))
            return g
    if s.volume > 0.98 * hull.volume or not s.is_watertight:
        from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
        from skimage.measure import marching_cubes
        full = body.mesh(n)
        lo = full.bounds[0] - 3 * VOX; shape = np.ceil((full.bounds[1] - lo) / VOX).astype(int) + 4
        pts = np.vstack([full.sample(int(min(4e6, max(5e4, full.area / (0.3 * VOX) ** 2)))), full.vertices])
        idx = np.clip(np.round((pts - lo) / VOX).astype(int), 0, shape - 1)
        g = np.zeros(shape, bool); g[tuple(idx.T)] = True
        g = binary_fill_holes(binary_closing(g, iterations=1))
        v, fc, _, _ = marching_cubes(gaussian_filter(np.pad(g, 2).astype(np.float32), 0.7), 0.5)
        mv = trimesh.Trimesh((v - 2) * VOX + lo, fc, process=True)
        if mv.is_watertight and mv.volume < 0:
            mv.invert()
        if mv.is_watertight and mv.volume < 0.97 * hull.volume:
            if len(mv.faces) > 30000:
                md = mv.simplify_quadric_decimation(face_count=30000)
                mv = md if md.is_watertight else mv
            m = mv
        else:
            m = hull
    m = slim(m)
    cachefs.savez(f, v=np.asarray(m.vertices), f=np.asarray(m.faces))
    return m


def slim(m, faces=40000):
    """Decimate a closed mesh to at most `faces` while it stays a valid manifold within 1.5 % of its volume."""
    if len(m.faces) <= faces:
        return m
    v0 = abs(m.volume)
    for target in (faces, 2 * faces):
        try:
            d = m.simplify_quadric_decimation(face_count=target)
        except Exception:  # noqa: BLE001
            continue
        if kit.mf(d).status().name == "NoError" and abs(abs(d.volume) - v0) < 0.015 * v0:
            return d
    return m


# ------------------------------------------------------------------------------------------------ mass / COM
def link_props(man_path, link):
    body = kit.base(man_path)
    rows, missing = [], []
    for n, p in body.parts.items():
        if p["owning_link"] != link:
            continue
        m = p.get("mass_kg")
        if not m:
            missing.append(n); continue
        try:
            s = closed(body, n)
        except Exception:  # noqa: BLE001
            s = body.mesh(n).convex_hull
        c = s.center_mass if s.is_watertight and s.volume > 1e-6 else s.bounds.mean(0)
        I = s.moment_inertia * (m * 1000.0 / s.volume) if s.is_watertight and s.volume > 1e-6 else np.zeros((3, 3))  # g mm2 about centroid
        rows.append((n, m * 1000.0, np.asarray(c, float), np.asarray(I, float)))
    M = sum(r[1] for r in rows)
    com = sum(r[1] * r[2] for r in rows) / M
    I = np.zeros((3, 3))
    for _, m, c, Ic in rows:
        d = c - com
        I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    j = {jj["child"]: jj for jj in json.loads(Path(man_path).read_text())["joints"]}[link]
    return dict(mass_g=round(M, 2), com_world_neutral_mm=np.round(com, 2).tolist(),
                com_from_joint_origin_mm=np.round(com - np.array(j["origin_world_mm"]), 2).tolist(),
                joint=j["name"], joint_origin_world_mm=j["origin_world_mm"],
                inertia_about_com_kg_m2=np.round(I * 1e-9, 9).tolist(), parts_counted=len(rows),
                parts_without_mass=missing,
                parts=[dict(name=n, mass_g=round(m, 3), centroid_mm=np.round(c, 2).tolist()) for n, m, c, _ in rows])


def run_mass():
    cad = {r["name"]: r for r in json.loads((OUT / "cad-manifest.json").read_text())["parts"]}
    sp = {p["name"]: p for p in json.loads(SHELL_PASS.read_text())["parts"]}
    parts = {}
    for side in "RL":
        old4 = sp[f"{side}_{OLD4}"]["mass_kg"] * 1000 + sp[f"{side}_{OLD4}_steel_sleeves"]["mass_kg"] * 1000
        old5 = sp[f"{side}_{OLD5}"]["mass_kg"] * 1000 + sp[f"{side}_{OLD5}_steel_sleeves"]["mass_kg"] * 1000
        n4, n5 = cad[f"{side}_{NEW4}"], cad[f"{side}_{NEW5}"]
        parts[side] = {
            "stage4": dict(selected_polymer_g=round(sp[f"{side}_{OLD4}"]["mass_kg"] * 1000, 2), selected_with_sleeves_g=round(old4, 2),
                           roll_flip_polymer_g=n4["polymer_mass_g"], roll_flip_with_sleeves_g=n4["total_stage_mass_g"],
                           roll_flip_material=n4["polymer_material"],
                           roll_flip_polymer_g_PETG_CF=n4["polymer_mass_g_PETG_CF"], sleeves=n4["sleeves"]["count"]),
            "stage5": dict(selected_polymer_g=round(sp[f"{side}_{OLD5}"]["mass_kg"] * 1000, 2), selected_with_sleeves_g=round(old5, 2),
                           roll_flip_polymer_g=n5["polymer_mass_g"], roll_flip_with_sleeves_g=n5["total_stage_mass_g"],
                           roll_flip_material=n5["polymer_material"],
                           roll_flip_polymer_g_PETG_CF=n5["polymer_mass_g_PETG_CF"], sleeves=n5["sleeves"]["count"]),
        }
    links = {}
    for link in ("R_pitch_link", "R_roll_link", "L_pitch_link", "L_roll_link"):
        a, b = link_props(SHELL_PASS, link), link_props(VARIANT, link)
        links[link] = dict(selected=a, roll_flip=b, delta_mass_g=round(b["mass_g"] - a["mass_g"], 2),
                           delta_com_mm=np.round(np.array(b["com_world_neutral_mm"]) - np.array(a["com_world_neutral_mm"]), 2).tolist())
        print(f"{link}: {a['mass_g']} g COM {a['com_world_neutral_mm']} -> {b['mass_g']} g COM {b['com_world_neutral_mm']}"
              f"  (missing masses: {b['parts_without_mass']})")
    merge("mass", dict(parts=parts, links=links,
                       basis="manifest masses (motors: catalogue; printed: native STEP volume x 1.22 g/cm3 PAHT-CF; sleeves 304 SS); "
                             "part COM / inertia from the closed (voxel-filled where open) mesh at uniform density; frame = "
                             "world at zero joint angles (mm); the URDF is not changed here",
                       note="Roll horn and idler have no mass in the manifest (null) and are not counted on either link."))
    for s, r in parts.items():
        print(s, r)


# ------------------------------------------------------------------------------------------------ wrappers on robot/body/checks
REFIT_MAN = ROOT / "artifacts/robot/body/motor-refit/assembly-parts.json"
FEM_KEYS = {"stage4_roll_flip": (f"R_{NEW4}", VARIANT), "stage5_roll_flip": (f"R_{NEW5}", VARIANT),      # report key -> (part with the FEM setup, manifest)
            "stage4_selected": ("R_thigh_combined", REFIT_MAN), "stage5_selected": ("R_hip_pitch_to_roll_combined", REFIT_MAN)}  # selected = motor-refit parts
PRINTED_ROLES = "printed_structural_connector,printed_structural_carrier"


def _tool(label, manifest, checks, *selectors):
    """Run robot/body/checks/run.py in this process; return its report.json (written under .cache/work/cad-speed/checks/LABEL)."""
    from checks import run as RUN
    RUN.main(["--manifest", str(manifest), "--checks", checks, "--label", label, *selectors])
    return json.loads((ROOT / ".cache/work/cad-speed/checks" / label / "report.json").read_text())


def subsystem(name):
    n = name.lower()
    if any(k in n for k in ("thigh", "hip_", "shin", "ankle", "foot")):
        return "lower body"
    if any(k in n for k in ("shoulder", "upper_arm", "forearm")):
        return "arm"
    if any(k in n for k in ("neck", "head")):
        return "neck / head"
    return "torso / pelvis"


def run_fit():
    """Exact B-rep fit of the roll-flip Stage 4 / 5 against their neighbours over the fit pose list (--poses fit).
    With --stage5-only only Stage 5 is run; Stage 4 rows not involving Stage 5 are kept from the previous report."""
    s5_only = "--stage5-only" in sys.argv
    parts = [f"R_{NEW5}"] if s5_only else [f"R_{NEW4}", f"R_{NEW5}"]
    rep = _tool("roll-flip-fit", VARIANT, "fit", "--parts", ",".join(parts), "--poses", "fit")
    if rep["checks"]["fit"].get("failed"):         # never merge the previous rows back in as if fresh: the section says it failed
        merge("fit_brep", dict(failed=rep["checks"]["fit"]["failed"], source="robot/body/checks fit (exact B-rep), pose set fit"))
        print(f"  fit FAILED: {rep['checks']['fit']['failed']}")
        return
    prev = json.loads(REPORT.read_text()).get("fit_brep", {}) if REPORT.exists() else {}
    mp = {p["name"]: p for p in json.loads(VARIANT.read_text())["parts"]}
    stage = {f"R_{NEW4}": "stage4 (R_roll_link)", f"R_{NEW5}": "stage5 (R_pitch_link)"}
    worst = dict(prev.get("worst_by_pair", {}))
    for part, mates in rep["checks"]["fit"]["per_part"].items():
        for mate, w in mates.items():
            if not s5_only and part == f"R_{NEW5}" and mate == f"R_{NEW4}":
                continue                                   # a Stage 4 x Stage 5 pair is reported once, as stage4 x stage5
            label = "stage4 (R_roll_link) x stage5" if part in stage and mate in stage else \
                f"{stage[part]} x {mate}" + (" (turned)" if str(mp.get(mate, {}).get("roll_flip", "")).startswith("turned") else "")
            worst[label] = dict(max_overlap_mm3=w["max_overlap_mm3"], min_distance_mm=w["min_distance_mm"], pose_of_max=w["pose_of_max"],
                                pose_of_min=w["pose_of_min"], same_link=w["same_link"])
    merge("fit_brep", dict(prev, frame="current assembly frame, standing splay from the manifest; links posed with the manifest joints",
                           poses_checked=rep["checks"]["fit"]["notes"].split(" poses")[0], worst_by_pair=worst,
                           source="robot/body/checks fit (exact B-rep), pose set fit" + (", Stage 5 only" if s5_only else "")))
    for k, w in sorted(worst.items()):
        print(f"  {k:70s} max {w['max_overlap_mm3']:8.3f} mm3, min dist {w['min_distance_mm']} mm")


def run_press(which):
    """Sleeve pressability (checks `press`): roll-flip Stage 4 / 5 of both legs on the variant manifest, or the printed connectors of the
    motor-refit ('selected') manifest. Writes report.press_roll_flip / press_selected."""
    from checks import press_lib as PC
    jobs = []
    if which in ("selected", "all"):
        jobs.append(("selected", REFIT_MAN, ("--role", PRINTED_ROLES)))
    if which in ("roll-flip", "all"):
        jobs.append(("roll_flip", VARIANT, ("--parts", ",".join(f"{s}_{b}" for s in "RL" for b in (NEW4, NEW5)))))
    for grp, man, sel in jobs:
        rep = _tool(f"roll-flip-press-{grp}", man, "press", *sel)
        if rep["checks"].get("press", {}).get("failed"):
            merge(f"press_{grp}", dict(failed=rep["checks"]["press"]["failed"], manifest=str(man.relative_to(ROOT)),
                                       source="robot/body/checks press"))
            print(f"{grp} press FAILED: {rep['checks']['press']['failed']}")
            continue
        parts, by = {}, {}
        for n, r in rep["checks"].get("press", {}).get("per_part", {}).items():
            parts[n] = dict(part=n, rows=r["sleeves"], summary=r["summary"], method=r["method"], subsystem=subsystem(n))
            g = by.setdefault(subsystem(n), dict(parts=0, sleeves=0, with_tool_obstruction=0, with_anvil_obstruction=0,
                                                 counter_face_coverage_below_90pct=0, any_issue=0))
            g["parts"] += 1
            for k in ("sleeves", "with_tool_obstruction", "with_anvil_obstruction", "counter_face_coverage_below_90pct"):
                g[k] += r["summary"][k]
            g["any_issue"] += sum(1 for x in r["sleeves"] if "error" not in x and (
                x["tool_tip_obstruction_mm3"] + x["tool_body_obstruction_mm3"] > 0.01 or x["anvil_path_obstruction_mm3"] > 0.01
                or x["counter_face"]["coverage_pct"] < 90.0))
            print(f"{grp:9s} {n:50s} {r['summary']}", flush=True)
        merge(f"press_{grp}", dict(assumptions=PC.ASSUMPTIONS, by_subsystem=by, parts=parts, manifest=str(man.relative_to(ROOT)),
                                   source="robot/body/checks press"))


def run_fem(only=None):
    """Static FEM of the parts of the given report keys (checks `fem`, per-part setups); report.fem[KEY] in the shape the
    publication reads: by_layer_normal -> material (PAHT_CF_dry, PAHT_CF_wet, PETG_CF) -> load case -> measurements."""
    rep = json.loads(REPORT.read_text()).get("fem", {}) if REPORT.exists() else {}
    for key, (part, man) in FEM_KEYS.items():
        if only and key not in only:
            continue
        r = _tool(f"roll-flip-fem-{key}", man, "fem", "--parts", part)
        fr = r["checks"].get("fem", {})
        if fr.get("failed") or part in (fr.get("failed_parts") or {}):
            why = fr.get("failed") or fr["failed_parts"][part]
            rep[key] = dict(failed=why, manifest=str(man.relative_to(ROOT)), source="robot/body/checks fem")
            merge("fem", rep)
            print(f"{key}: FEM FAILED: {why}", flush=True)
            continue
        p = fr.get("per_part", {}).get(part)
        if p is None:
            print(f"{key}: part {part} has no FEM setup or no native STEP in {man.name}", flush=True)
            continue
        bl = {ln: {m.replace("legacy_fdm_", ""): rows for m, rows in by_m.items() if m.startswith("legacy_fdm_")}
              for ln, by_m in p["by_layer_normal"].items()}
        rep[key] = dict(nodes=p["nodes"], tets=p["tets"], origin_mm=p["origin_mm"], wrenches_N_Nmm=p["wrenches_N_Nmm"], by_layer_normal=bl,
                        patches=p["patches"], seconds=p["seconds"], manifest=str(man.relative_to(ROOT)), source="robot/body/checks fem")
        merge("fem", dict(rep, method="robot/body/checks fem: gmsh P1 tets, orthotropic FDM polymer with bonded sleeves, per-part setup "
                                      "(data/part_settings.json), unit-wrench basis at the load patch",
                          patches="Supports and load patches come from the shared per-part FEM setups (robot/body/checks/data/part_settings.json): "
                                  "surfaces in contact with the named mating parts / the part's own bolt holes. Roll-flip keys run on "
                                  "assembly-parts-roll-flip.json, selected keys on the motor-refit assembly-parts.json (manifest per key in each entry)."))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    t_start = time.time()
    steps = {"manifest": build_manifest, "mass": run_mass, "fit": run_fit,
             "press": lambda: run_press(sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "all"),
             "fem": lambda: run_fem(sys.argv[2].split(",") if len(sys.argv) > 2 else None)}
    if step in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    if step not in steps:
        sys.exit(f"usage: roll_flip_checks.py {{{','.join(steps)}}} [ARG]   (the other checks: robot/body/checks/run.py)")
    steps[step]()
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    wt = rep.get("wall_time_s", {}); wt[" ".join(sys.argv[1:])] = round(time.time() - t_start, 1)
    merge("wall_time_s", wt)
