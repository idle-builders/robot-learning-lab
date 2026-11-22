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
    if connector == "head_sensor_compute_carrier":
        carrier_sleeve_entry(es, allm)
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
            if ename == name:            # the recorded volume follows the rebuilt part (it kept the volume of an earlier build)
                _entry(man, ename)["volume_mm3"] = round(abs(float(m.volume)), 2)
            if conn == "head_sensor_compute_carrier" and ename.endswith("_steel_sleeves"):
                carrier_sleeve_entry(_entry(man, ename), m)
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
    """The head carrier's 4 compute, 8 camera and 1 mic screws generated from the supplier spacers / nuts / shims and the catalogue inserts
    (head_holes.py, measured on the shell-pass manifest); the legacy camera / plate / mic screws they replace are dropped, and so are the
    camera standoffs and washers and the mic shims (printed columns and the carrier's mic wing take their place). A skin or mic board an
    earlier iteration had edited (mic bosses on the skin, board moved 0.5 mm) is pointed back at its source mesh."""
    import re
    import head_holes as HH
    plans = HH.screw_plans()
    _undo_skin_mount(man)
    drop = {n for g in HH.POSTS.values() if g is not HH.POSTS["compute"] for n in g["spacers"] + g.get("washers", []) + g.get("drop", [])}
    names = {p["name"] for p in plans} | {n for g in HH.POSTS.values() for n in g.get("legacy_screws", [])} | drop
    # an earlier generated length of the same post: B0410_camera_M2_0 is stored as B0410_camera_M2x<L>_0
    stale = [re.compile(re.escape(b).replace("_M2_", r"_M2(?:x[0-9p.]+)?_") + "$") for g in HH.POSTS.values() if "legacy_screws" in g for b in g["screws"]]
    man["parts"] = [p for p in man["parts"] if p["name"] not in names and not any(r.match(p["name"]) for r in stale)]
    for pl in plans:
        pl["bom_group"] = f"{pl['group']}_{pl['thread_label']}x{pl['length_mm']:g}"
        _add_screw(man, pl)
    return plans


def _undo_skin_mount(man: dict) -> None:
    """Point the head skin and the mic board back at the meshes they had before the skin-boss iteration of T-5bqc3m (recorded on their
    entries) and drop the skin's insert part; nothing to do on a manifest that never had them."""
    for p in man["parts"]:
        src = p.pop("mic_boss_source_mesh", None) or p.pop("mic_board_source_mesh", None)
        p.pop("mic_board_shift_mm", None)
        if src:
            m = _load_mesh(src)
            p.update(mesh=src, mesh_sha256=_sha(ROOT / src), bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()])
            if p.get("mesh_note", "").find("microphone insert bosses") >= 0:
                p["mesh_note"] = "range notch applied to the mesh (T-jrnxfn); the native STEP predates it and is dropped"
    man["parts"] = [p for p in man["parts"] if p["name"] != "original_profile_head_skin_v23_steel_sleeves"]


def _load_mesh(path: str):
    from checks import bodylib as BL
    return BL.load_mesh(path)


def carrier_sleeve_entry(e: dict, mesh) -> None:
    """Material, mass and note of the carrier's sleeve part from what it holds (head_holes.py): the 304 neck-root and compute sleeves and the
    brass camera / mic heat-set inserts; a component whose extents are the insert's length and twice its modelled radius (0.15 mm) is an insert."""
    import head_holes as HH
    ins = HH.sleeve("camera")
    n_cam, n_mic = len(HH.POSTS["camera"]["screws"]), len(HH.POSTS["mic"]["screws"])
    mass, n_brass, n_steel = 0.0, 0, 0
    for c in mesh.split(only_watertight=False):
        brass = bool(np.allclose(np.sort(c.extents), np.sort([float(ins["length"]), 2 * float(ins["ro"]), 2 * float(ins["ro"])]), atol=0.15))
        mass += abs(float(c.volume)) * (8.5e-6 if brass else STEEL_G_MM3 * 1e-3)
        n_brass += brass; n_steel += not brass
    e.update(material=f"{n_steel} x 304 stainless sleeves (neck root, compute posts; 7.85 g/cm3) + {n_brass} x brass M2 x {ins['length']:g} knurled heat-set "
                      f"inserts ({n_cam} camera, {n_mic} mic; 8.5 g/cm3; modelled at the {2 * ins['ro']:g} mm hole, catalogue knurl OD {ins['catalogue_od']:g})",
             mass_kg=round(mass, 6), mass_basis="per component: inserts x 8.5 g/cm3 (brass), sleeves x 7.85 g/cm3",
             neck_parts=f"built by robot/body/neck-head/neck_parts.py connectors (T-6dvwjq); {n_brass} brass heat-set inserts (T-5bqc3m)")


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
