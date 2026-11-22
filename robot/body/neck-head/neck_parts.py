"""Neck / head printed connectors and their screws in the body manifest (Idle T-6dvwjq).

  connectors CONNECTOR...   rebuild the selected-family solid and the sleeves of the connectors (names of
                            connector_screws.json), write the native STEP + NPZ and update their manifest entries
  screws [CONNECTOR...]     generate the screws of the connectors from the defined holes they land in
                            (robot/body/fastening/screw_plan.py, connector_screws.json here) and put them into the
                            manifest in place of the legacy neck screws and washers (names stay)

  python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/neck-head/neck_parts.py connectors neck_yaw_to_pitch_link
The manifest is robot/design.json body_manifest (an artifact: run this after the manifest is rebuilt). Every number is a
measurement in mm; nothing here accepts or rejects."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _own(name: str, alias: str):
    """This folder's module by file path under a unique name: other body folders also have a `common.py`, and when
    roll_flip_checks.build_manifest imports this file the leg's `common` is already in sys.modules."""
    import importlib.util
    if alias in sys.modules:
        return sys.modules[alias]
    spec = importlib.util.spec_from_file_location(alias, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


C = _own("common", "neck_head_common")
DH = _own("defined_holes", "neck_head_defined_holes")

ROOT = C.ROOT
STAGE = {"neck_root_cradle": "stage1_neck_root_cradle", "neck_yaw_to_pitch_link": "stage2_neck_yaw_to_pitch_link",
         "neck_pitch_to_head_cradle": "stage3_neck_pitch_to_head_cradle",
         "head_sensor_compute_carrier": "stage4_head_sensor_compute_carrier"}
STEEL_G_MM3 = 7.85e-3
LEGACY = {"neck_root_cradle": ("neck_root_M2x6_", "neck_root_M2_washer_"),
          "neck_yaw_to_pitch_link": ("neck_yaw_horn_M2x5_", "neck_pitch_horn_M2x5_"),
          "neck_pitch_to_head_cradle": ("neck_head_M2x5_", "neck_head_M2_washer_"),
          "head_sensor_compute_carrier": ()}


def _sha(f: Path) -> str:
    return hashlib.sha256(f.read_bytes()).hexdigest()


def _rel(f: Path) -> str:
    return str(f.relative_to(ROOT))


def _entry(man: dict, name: str) -> dict:
    return next(p for p in man["parts"] if p["name"] == name)


def _npz(m, f: Path):
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, vertices=np.asarray(m.vertices, np.float32), faces=np.asarray(m.faces, np.int32))


def rebuild(connector: str, man: dict) -> None:
    import trimesh
    from build123d import Compound, export_step
    import build_neck_head_connectors as B
    stage = STAGE[connector]
    bname = C.STAGE_PART_BASENAMES[stage]
    fam = json.loads((C.ART / "combined/manifest.json").read_text())["stages"][stage]["selected_family"]
    sol = B.BUILDERS[stage](fam)
    name = f"combined__{bname}"
    step, npz = C.ART / "parts" / f"{name}.step", C.ART / "parts" / f"{name}.npz"
    export_step(sol, str(step))
    C.save_solid_npz(sol, npz, tol=0.05)
    z = np.load(npz); mesh = trimesh.Trimesh(z["vertices"], z["faces"], process=False)
    rho = C.MATERIALS["PAHT_CF_dry"]["density_g_cm3"]
    e = _entry(man, name)
    e.update(native_step=_rel(step), native_step_sha256=_sha(step), mesh=_rel(npz), mesh_sha256=_sha(npz),
             bounds_mm=[np.round(mesh.bounds[0], 3).tolist(), np.round(mesh.bounds[1], 3).tolist()],
             volume_mm3=round(float(sol.volume), 2), mass_kg=round(float(sol.volume) * rho * 1e-6, 6))
    slv = C.build_stage_sleeves(stage)
    sstep, snpz = C.ART / "combined" / f"{bname}_sleeves.step", C.ART / "combined" / f"{bname}_sleeves.npz"
    export_step(Compound(children=[s["solid"] for s in slv]), str(sstep))
    parts = []
    for s in slv:
        v, f = C.solid_to_watertight_arrays(s["solid"], tol=0.04)
        parts.append(trimesh.Trimesh(v, f, process=False))
    allm = trimesh.util.concatenate(parts)
    _npz(allm, snpz)
    es = _entry(man, f"{name}_steel_sleeves")
    es.update(native_step=_rel(sstep), native_step_sha256=_sha(sstep), mesh=_rel(snpz), mesh_sha256=_sha(snpz),
              bounds_mm=[np.round(allm.bounds[0], 3).tolist(), np.round(allm.bounds[1], 3).tolist()],
              native_solid_count=len(slv), native_validity=[True] * len(slv),
              mass_kg=round(sum(s["mass_g"] for s in slv) / 1000.0, 6))
    print(f"{connector}: {fam} volume {sol.volume:.1f} mm3, {len(slv)} sleeves of {slv[0]['length_mm']} .. {slv[-1]['length_mm']} mm")


def apply(man: dict) -> list[str]:
    """Point the manifest's neck connector + sleeve entries at the files `connectors` built (artifacts/robot/body/neck-head)
    and generate their screws; called by robot/body/leg/roll_flip_checks.py build_manifest so a rebuild keeps them.
    Connectors whose files are not built here are left as they are, with a warning (run `neck_parts.py connectors ...`)."""
    import trimesh
    log, built = [], []
    rho = C.MATERIALS["PAHT_CF_dry"]["density_g_cm3"]
    for conn, stage in STAGE.items():
        bname = C.STAGE_PART_BASENAMES[stage]
        name = f"combined__{bname}"
        files = (C.ART / "parts" / f"{name}.step", C.ART / "parts" / f"{name}.npz",
                 C.ART / "combined" / f"{bname}_sleeves.step", C.ART / "combined" / f"{bname}_sleeves.npz")
        if not all(f.exists() for f in files):
            print(f"WARNING neck: {conn} not built in {_rel(C.ART)}, the unedited connector and its legacy screws stay; "
                  f"run `.venv/bin/python -B robot/body/neck-head/neck_parts.py connectors {conn}` (T-6dvwjq) first")
            continue
        for (step, npz), ename, dens in (((files[0], files[1]), name, rho), ((files[2], files[3]), f"{name}_steel_sleeves", STEEL_G_MM3 * 1e3)):
            z = np.load(npz); m = trimesh.Trimesh(z["vertices"], z["faces"], process=False)
            _entry(man, ename).update(native_step=_rel(step), native_step_sha256=_sha(step), mesh=_rel(npz), mesh_sha256=_sha(npz),
                                      bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()],
                                      mass_kg=round(abs(float(m.volume)) * dens * 1e-6, 6),
                                      neck_parts="built by robot/body/neck-head/neck_parts.py connectors (T-6dvwjq)")
        built.append(conn)
        log.append(f"{name}: rebuilt neck connector + sleeves (neck_parts.py)")
    if built:
        log += [f"{p['name']}: {p['thread_label']}x{p['length_mm']:g} into {p['hole_id']}" for p in screws(built, man, None)]
    return log


def screws(connectors: list[str], man: dict, path: Path) -> list[dict]:
    """Replace the legacy neck screws and washers by the screws generated from the defined holes."""
    from checks import bodylib
    import interfaces as I
    import screw_plan as SP
    drop = {p["name"] for p in man["parts"] if any(p["name"].endswith(k + str(i)) for c in connectors for k in LEGACY[c] for i in range(8))
            and p["name"].startswith("upper__neck_")}
    man["parts"] = [p for p in man["parts"] if p["name"] not in drop]
    body = bodylib.Body(man)
    iface = I.instances(body, man)
    for h in iface["holes"]:     # a hole is tapped as deep as the drawing says (the modelled bore of some case holes runs on to 18.6 mm)
        if h.get("drawing_max_tap_depth_mm"):
            h["depth_mm"] = min(h["depth_mm"], h["drawing_max_tap_depth_mm"])
    plans = []
    for conn in connectors:
        if conn == "head_sensor_compute_carrier":
            plans += head_screws(man)
            continue
        for pl in SP.plan_connector(body, iface, conn, "", spec=DH.connectors()[conn]):
            plans.append(_add_screw(man, pl))
    return plans


def head_screws(man: dict) -> list[dict]:
    """The head carrier's 4 compute, 8 camera and 2 mic screws generated from the supplier spacers / nuts and the catalogue inserts
    (head_holes.py); the legacy camera / plate / mic screws they replace are dropped, the camera standoffs and mic shims go (printed
    columns / skin bosses take their place) and the skin gets its two mic bosses with the inserts."""
    import screw_plan as SP
    import head_holes as HH
    plans = HH.screw_plans()                 # measures (and caches) the standoff / shim bores before they are dropped
    import re
    skin_mic_bosses(man)
    drop = {n for g in HH.POSTS.values() if g is not HH.POSTS["compute"] for n in g["spacers"]}
    names = {p["name"] for p in plans} | {n for g in HH.POSTS.values() for n in g.get("legacy_screws", [])} | drop
    # an earlier generated length of the same post: B0410_camera_M2_0 is stored as B0410_camera_M2x<L>_0
    stale = [re.compile(re.escape(b).replace("_M2_", r"_M2(?:x[0-9p.]+)?_") + "$") for g in HH.POSTS.values() if "legacy_screws" in g for b in g["screws"]]
    man["parts"] = [p for p in man["parts"] if p["name"] not in names and not any(r.match(p["name"]) for r in stale)]
    for pl in plans:
        pl["bom_group"] = f"{pl['group']}_{pl['thread_label']}x{pl['length_mm']:g}"
        _add_screw(man, pl)
    return plans


def skin_mic_bosses(man: dict) -> None:
    """Two printed bosses on the head skin's inner side carry the microphone's two catalogue M2 heat-set inserts (T-5bqc3m, replaces
    the carrier's mic wing, which left 2.4 mm between board and skin at the upper screw for a 3 mm insert). Each boss is a cylinder
    (head_carrier_posts.json mic.boss.ro) from the board's back face to a plane inside the skin wall (least-squares plane through the wall's
    mid surface over the boss footprint), so the skin's outer surface is untouched; the insert hole (OD of the insert, its length) and a
    narrow hole for the screw's point are cut from the board face. Writes the skin mesh into this folder's parts/, the inserts as the
    skin's `_steel_sleeves` part, and points the manifest at both. Idempotent (a second run unions / cuts the same solids)."""
    import copy
    import manifold3d as md
    import trimesh
    from build123d import Compound, export_step
    import head_holes as HH
    from checks import bodylib as BL
    body, dman = HH._body()
    spec = HH.POSTS["mic"]
    bs, sl = spec["boss"], HH.sleeve("mic")
    name = bs["skin_part"]
    npz = C.ART / "parts" / f"{name}.npz"
    # the skin as it was before this function touched it: the manifests' own entry unless that is already our output (then the entry's
    # recorded source), so a second run (rebuild chain, default manifest already bossed) cuts the same solids from the same skin
    srcs = []
    for m_ in (man, dman):
        pe = next(p for p in m_["parts"] if p["name"] == name)
        srcs += [pe.get("mic_boss_source_mesh"), pe["mesh"]]
    src = next(x for x in srcs if x and (ROOT / x).resolve() != npz.resolve())
    skin = BL.load_mesh(src)
    posts = HH.threaded_posts()["mic"]
    ro, L = float(bs["ro"]), float(sl["length"])
    rays = skin.ray

    def wall(y, z):        # (inner, outer) x of the skin at (y, z): the first two hits of a ray from the inside
        loc, _, _ = rays.intersects_location([[30.0, y, z]], [[1.0, 0.0, 0.0]])
        xs = np.sort(loc[:, 0])
        return float(xs[0]), float(xs[-1])

    def to_mf(m):
        return md.Manifold(md.Mesh(np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.uint32)))

    def cyl_x(x0, x1, r, y, z):
        c = md.Manifold.cylinder(float(x1 - x0), float(r), float(r), 64)          # along z from 0
        return c.rotate([0.0, 90.0, 0.0]).translate([float(x0), float(y), float(z)])

    res = to_mf(skin)
    holes = []
    for r in posts:
        pl = r["plan"]
        c = np.asarray(pl["opening"], float)           # on the board's back face
        x0 = float(c[0])
        ang = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        pts = [(c[1] + ro * k * np.cos(t), c[2] + ro * k * np.sin(t)) for k in (0.0, 0.5, 1.0) for t in (ang if k else [0.0])]
        w = np.array([wall(y, z) for y, z in pts])
        mid = (w[:, 0] + w[:, 1]) / 2
        A = np.c_[np.ones(len(pts)), np.array(pts) - c[1:]]
        coef = np.linalg.lstsq(A, mid, rcond=None)[0]            # x = a + b dy + c dz through the wall's mid surface
        nrm = np.array([-1.0, coef[1], coef[2]]); nl = float(np.linalg.norm(nrm))
        # boss = the cylinder trimmed to the side of the plane that holds the board (x <= plane)
        cy = cyl_x(x0, float(mid.max() + 2.0), ro, c[1], c[2])
        # plane in world coordinates: x = a + b (y - cy) + c (z - cz); it passes through (a, cy, cz)
        n_w = nrm / nl
        p_on = np.array([coef[0], c[1], c[2]])                   # a point of the plane
        boss = cy.trim_by_plane(tuple(n_w), float(np.dot(n_w, p_on)))
        res = res + boss
        resid = w[:, 1] - (coef[0] + A[:, 1:] @ coef[1:])       # outer surface minus the plane: the wall left behind the boss end
        print(f"{pl['name']}: boss R{ro} x {x0:.2f} .. wall mid plane x {coef[0]:.2f} (slope y {coef[1]:+.3f} z {coef[2]:+.3f}); "
              f"skin inner {w[:, 0].min():.2f}..{w[:, 0].max():.2f}, outer {w[:, 1].min():.2f}..{w[:, 1].max():.2f}; end plane to outer {resid.min():.2f} mm")
        holes.append(cyl_x(x0 - 0.5, x0 + L, float(sl["ro"]), c[1], c[2]))
        holes.append(cyl_x(x0 + L - 0.1, x0 + L + float(bs["tip_hole_extra_mm"]), float(bs["tip_hole_ro"]), c[1], c[2]))
    for h in holes:
        res = res - h
    m = res.to_mesh()
    v, f = np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)
    _npz(trimesh.Trimesh(v, f, process=False), npz)
    mesh = trimesh.Trimesh(v, f, process=False)
    e = _entry(man, name)
    e.update(mic_boss_source_mesh=src, mesh=_rel(npz), mesh_sha256=_sha(npz), bounds_mm=[np.round(mesh.bounds[0], 3).tolist(), np.round(mesh.bounds[1], 3).tolist()],
             mesh_note="range notch applied to the mesh (T-jrnxfn); two microphone insert bosses on the inner side added (T-5bqc3m, neck_parts.py skin_mic_bosses); "
                       "the native STEP predates both and is dropped")
    for k in ("native_step", "native_step_sha256"):
        e.pop(k, None)
    # the inserts: tubes OD 2 ro, bore = the thread's minor diameter, from the board's back face into the boss
    slv = []
    for r in posts:
        c = np.asarray(r["plan"]["opening"], float)
        outer = C.cyl_along(c, np.array([1.0, 0.0, 0.0]), float(sl["ro"]), L)
        inner = C.cyl_along(c - 0.05 * np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), float(sl["ri"]), L + 0.1)
        from build123d import Solid
        slv.append(Solid((outer - inner).wrapped))
    sname = f"{name}_steel_sleeves"
    sstep, snpz = C.ART / "parts" / f"{sname}.step", C.ART / "parts" / f"{sname}.npz"
    export_step(Compound(children=slv), str(sstep))
    parts = []
    for s_ in slv:
        vv, ff = C.solid_to_watertight_arrays(s_, tol=0.04)
        parts.append(trimesh.Trimesh(vv, ff, process=False))
    allm = trimesh.util.concatenate(parts)
    _npz(allm, snpz)
    tmpl = copy.deepcopy(_entry(man, "combined__head_sensor_compute_carrier_steel_sleeves"))
    tmpl.update(name=sname, native_step=_rel(sstep), native_step_sha256=_sha(sstep), mesh=_rel(snpz), mesh_sha256=_sha(snpz),
                bounds_mm=[np.round(allm.bounds[0], 3).tolist(), np.round(allm.bounds[1], 3).tolist()], native_solid_count=len(slv),
                native_validity=[True] * len(slv), mass_kg=round(sum(float(s_.volume) for s_ in slv) * 8.5e-6, 6),
                neck_parts="2 M2 x 3 mm brass heat-set inserts in the head skin's mic bosses (neck_parts.py skin_mic_bosses, T-5bqc3m)")
    man["parts"] = [p for p in man["parts"] if p["name"] != sname] + [tmpl]
    print(f"{name}: 2 mic bosses + inserts, {abs(float(mesh.volume)):.1f} mm3 (was {abs(float(skin.volume)):.1f})")


def _add_screw(man: dict, pl: dict) -> dict:
    import screw_plan as SP
    m = SP.screw_mesh(pl)
    nm = pl["name"]
    f = C.ART / "parts" / f"{nm}.npz"
    _npz(m, f)
    lbl = f"{pl['thread_label']}x{pl['length_mm']:g}"
    man["parts"].append(dict(
        name=nm, physical_part_id=nm, bom_group=pl["bom_group"], owning_link=pl["owning_link"], role="fastener_proxy",
        mesh=_rel(f), mesh_sha256=_sha(f), mesh_units="mm",
        bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()],
        volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * STEEL_G_MM3 / 1000.0, 7),
        material=f"{pl['head_standard']} {lbl} into {pl['hole_id'].split(':')[1]} ({pl['terminal_material']}); head dia "
                 f"{pl['head_dia_mm']} x {pl['head_height_mm']}, shank dia {pl['thread'].lstrip('M')}, {pl['point_mm']} mm point (envelope)",
        color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
        overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly", inspection_reference=False,
        mass_basis="mesh volume x 7.85 g/cm3",
        screw_plan={k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in pl.items()
                    if k in ("hole_id", "thread_label", "head_standard", "length_mm", "stack_mm", "head_gap_mm", "engagement_mm",
                             "reference_engagement_mm", "hole_depth_mm", "tip_margin_mm", "shortfall_mm", "overrun_mm",
                             "stack_change_to_fit_mm")}))
    print(f"{nm}: {lbl} into {pl['hole_id']}: stack {pl['stack_mm']:.3f}, engagement {pl['engagement_mm']} of {pl['reference_engagement_mm']}, tip margin {pl['tip_margin_mm']}")
    return pl


def main() -> None:
    cmd, names = sys.argv[1], sys.argv[2:]
    path = DH.manifest_path()
    man = json.loads(path.read_text())
    if cmd == "connectors":
        for n in names:
            rebuild(n, man)
    elif cmd == "screws":
        screws(names or list(DH.connectors()), man, path)
    path.write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", _rel(path))


if __name__ == "__main__":
    main()
