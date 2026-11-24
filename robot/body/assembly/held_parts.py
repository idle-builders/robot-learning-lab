"""Parts of the selected body that nothing held (Idle T-3npyjq, connectivity triage 2026-11-20): rebuilt or removed here,
manifest patched like the other screw edits (torso_screws.py, foot_screws.py).

Per part, what was missing and what this script does (measurements: docs/research/design-checks.md, connectivity):

  fan stack (SameSky 4010 fan,     the 4 M3 fan screws clamped only the fan (head on its front flange at y 10.6, washer and
  lead guide, leads, cap)          nut behind it), and the two compute M2.5x30 screws through fan_bridge_standoff_0/1 ended in
                                   nuts that floated 2.0 mm off the standoff ends (standoff face y 8.6, nut face y 10.6): the
                                   2 mm fan bridge plate between them was never modelled. Added: fan_bridge_plate (printed,
                                   y 8.6..10.6, flush steel sleeves), clamped by the two compute screws between standoffs and
                                   bridge nuts; the fan's two upper holes are screwed to it with M3x16 (heads on the plate's
                                   front face, the old M3 screws, washers and nuts of those holes kept, only the screws
                                   re-seated). The lower right hole (x 10.55, z 434.5) cannot take a head on the plate: the
                                   Radian heatsink fins stand 0.6 mm in front of the plate there; its screw, washer and nut
                                   held nothing but the fan and are removed. The lead-guide screw (lower left) is unchanged.
                                   The fan model's corner holes are counterbored 3.7 mm from the front (face y 10.6, floor
                                   14.3, r about 2.9): the plate sleeve faces that pocket and the plate bears on the fan face
                                   around it (mount_check reads +3.7 mm sleeve-to-fan; the old screws stood 3.7 mm proud of
                                   the same floor). Fan-to-plate is a bench subassembly: at standing the ROCK4D blocks 10 %
                                   of the driver path 7.6-11.8 mm from the heads.
  chest indicator relief           a separate lozenge in an ellipsoid pocket of the front shell, 0.24 mm clear all round and
                                   free inward (+y). There is no indicator light in the electronics; it is a colour accent of
                                   the selected look. Printed with the front shell in one job on the second nozzle (Bambu
                                   X2D, dual nozzle), so it is bonded where it meets the shell: rebuilt as the insert grown
                                   0.4 mm and clipped to the pocket ellipsoid (the tool assemble_complete_fdm_body.py cuts),
                                   so it fills the pocket and meets the shell wall.
  center_front_low_L/R_M2x10       the two lower front-shell screws of the r1 shell (x +-16, z 224): motor_refit.py
                                   edit_front_shell removed the shelves they screwed into (the hip pitch ST3025 sweeps there)
                                   and re-attached the shell with the FRONT_POSTS; no part around them, tips 0.495 mm3 in the
                                   hip pitch case. Modelling leftovers: removed.
  R_roll_servo_case_reference      purchased case reference (vendor STEP): solid 2 (rear housing, 11128 mm3) holds 7 tiny
                                   void shells, two of zero volume, and fails BRepCheck. Rebuilt from its outer shell (voids
                                   filled, +17 mm3); the other 5 solids unchanged. The L case is valid (motor_refit mirror).

Not changed here (need other builders; see the Idle task): pelvis cells 08 / 09 (the narrower-hip sweep removed the
cradle floor under the lower cell and both cell ends are open: retention belongs in the cradle's B-rep pass, T-dgz6qv),
head_neck_collar (needs screw lands on the neck yaw-to-pitch link). B0410 already reaches the core by screw through its
F-F standoffs (5e3e81fa0).

  recreate:  .venv/bin/python -B robot/body/assembly/held_parts.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/assembly/held_parts.py apply   (patch the roll-flip variant manifest)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("held-parts")   # configured values: robot/body/assembly/held_parts.bom.yaml config:

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(ROOT / "robot/body/fastening"))

OUT = ROOT / "artifacts/robot/body/assembly/held-parts"
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"   # unedited sources: build repeats after apply
VARIANT = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
TASK = "T-3npyjq"

# fan bridge plate (manifest frame, head link at standing = neutral)
PLATE_Y = CFG["PLATE_Y"]
FAN_C, FAN_HALF = CFG["FAN_C"], CFG["FAN_HALF"]
FAN_OPEN_R = CFG["FAN_OPEN_R"]
FAN_HOLES = CFG["FAN_HOLES"]
FAN_DROP = CFG["FAN_DROP"]
LEAD_GUIDE_KEEPOUT = CFG["LEAD_GUIDE_KEEPOUT"]
GUIDE_SCREW = CFG["GUIDE_SCREW"]
BOSS_R = CFG["BOSS_R"]
SLEEVE_M3 = CFG["SLEEVE_M3"]
SLEEVE_M2P5 = CFG["SLEEVE_M2P5"]
COMPUTE = CFG["COMPUTE"]
PLATE, PLATE_SLEEVES = CFG["PLATE"], CFG["PLATE_SLEEVES"]

# chest indicator: the pocket tool of assemble_complete_fdm_body.py (sphere scaled by these semi-axes x scale_t)
SCALE_T = CFG["SCALE_T"]
POCKET_AXES = CFG["POCKET_AXES"]
POCKET_Z = CFG["POCKET_Z"]
INDICATOR = CFG["INDICATOR"]
INDICATOR_GROW = CFG["INDICATOR_GROW"]

CASE = CFG["CASE"]
REMOVE = ("center_front_low_L_M2x10", "center_front_low_R_M2x10") + FAN_DROP


def _src():
    return {p["name"]: p for p in json.loads(SHELL_PASS.read_text())["parts"]}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def repair_case(MR, by):
    """Each invalid solid of the case STEP rebuilt from its outer shell (largest enclosed volume)."""
    from build123d import Solid, import_step
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepCheck import BRepCheck_Analyzer
    import cad_ops as C
    p = by[CASE]
    sols, notes = [], []
    for i, so in enumerate(import_step(str(MR.F.src(p["native_step"]))).solids()):
        if BRepCheck_Analyzer(so.wrapped).IsValid():
            sols.append(so.wrapped); continue
        shells = sorted(so.shells(), key=lambda sh: -abs(Solid(BRepBuilderAPI_MakeSolid(sh.wrapped).Solid()).volume))
        new = BRepBuilderAPI_MakeSolid(shells[0].wrapped).Solid()
        assert BRepCheck_Analyzer(new).IsValid(), f"{CASE} solid {i}: outer shell is not a valid solid either"
        notes.append(f"solid {i}: {len(shells) - 1} void shells dropped, volume {so.volume:.1f} -> {C.volume(new):.1f} mm3")
        sols.append(new)
    rec = MR.save(CASE, MR.compound(sols), "purchased")
    keep = {k: rec[k] for k in ("native_step", "native_step_sha256", "native_solid_count", "native_validity")}
    keep["held_parts"] = f"invalid vendor solid rebuilt from its outer shell ({TASK}): " + "; ".join(notes)
    (OUT / "parts" / f"{CASE}.npz").unlink(missing_ok=True)      # the manifest keeps its own (turned) mesh
    print(CASE, keep["held_parts"], flush=True)
    return keep


def _screw_axis(by, name):
    from checks import bodylib as BL
    from mount_check import screw_geometry
    g = screw_geometry(BL.load_mesh(by[name]["mesh"]))
    return np.asarray(g["_head"], float), np.asarray(g["_d"], float)


def build_plate(MR, by):
    import cad_ops as C
    y0, y1 = PLATE_Y
    cx, cz = FAN_C
    s = C.box((cx - FAN_HALF, y0, cz - FAN_HALF), (cx + FAN_HALF, y1, cz + FAN_HALF))
    bosses = {}
    for scr in COMPUTE:
        h, d = _screw_axis(by, scr)
        assert abs(d[1]) > 0.999, f"{scr}: axis not along y"
        bosses[scr] = (float(h[0]), float(h[2]))
    arms = []
    x_in = cx + FAN_HALF - 4.5                   # arms start inside the fan frame's right edge
    for bx, bz in bosses.values():
        lo = bz < cz                             # lower boss below the fan, upper boss above it
        z_in = cz - FAN_HALF + 1.5 if lo else cz + FAN_HALF - 1.5
        arms.append(C.cylinder((bx, y0, bz), (bx, y1, bz), BOSS_R))
        arms.append(C.box((x_in, y0, bz - BOSS_R if lo else z_in), (bx, y1, z_in if lo else bz + BOSS_R)))
        arms.append(C.box((bx - 0.01, y0, bz if lo else z_in), (bx + BOSS_R, y1, z_in if lo else bz)))
    s = C.fuse(s, *arms)
    (lx0, lx1), (lz0, lz1) = LEAD_GUIDE_KEEPOUT
    tools = [C.cylinder((cx, y0 - 1, cz), (cx, y1 + 1, cz), FAN_OPEN_R), C.box((lx0, y0 - 1, lz0), (lx1, y1 + 1, lz1))]
    holes = [((x, z), SLEEVE_M3) for x, z in FAN_HOLES.values()] + [(b, SLEEVE_M2P5) for b in bosses.values()]
    tools += [C.cylinder((x, y0 - 1, z), (x, y1 + 1, z), od / 2) for (x, z), (od, _) in holes]
    s = C.largest_solid(C.cut_many(s, tools, min_keep=0.1))   # the air opening takes most of the square
    assert C.valid(s) and C.n_solids(s) == 1, "fan bridge plate is not one valid solid"
    sleeves = [C.cut(C.cylinder((x, y0, z), (x, y1, z), od / 2), C.cylinder((x, y0 - 0.5, z), (x, y1 + 0.5, z), idd / 2))
               for (x, z), (od, idd) in holes]
    rec = MR.save(PLATE, s, "printed")
    srec = MR.save(PLATE_SLEEVES, MR.compound(sleeves), "purchased")
    srec.update(BOM_CONFIG.mass_record(PLATE_SLEEVES, srec["volume_mm3"], "held-parts", fallback="HARDWARE_MATERIAL", basis="native STEP volume"))
    print(PLATE, rec["volume_mm3"], "mm3;", len(sleeves), "sleeves", flush=True)
    return rec, srec, dict(holes=[dict(at_xz=list(c), sleeve_od_id_mm=list(sz)) for c, sz in holes], bosses=bosses)


def plate_rows(by, rec, srec):
    """Manifest rows: the plate, its sleeves, and the two upper fan screws re-seated as M3x16 (the guide screw's mesh moved)."""
    from checks import bodylib as BL
    stand = by["fan_bridge_standoff_0"]
    common = dict(owning_link=stand["owning_link"], parent_assembly=stand.get("parent_assembly"), display_group="mechanism",
                  display_groups=["mechanism"], overlay_id=stand.get("overlay_id"), inspection_reference=False,
                  held_parts=f"{TASK}: fan bridge plate between the bridge standoffs and their nuts")
    rows = [dict(name=PLATE, role="printed_structural_connector", color=[64, 156, 212, 255],
                 material="Bambu PAHT-CF (1.22 g/cm3) FDM", **common, **rec),
            dict(name=PLATE_SLEEVES, role="compression_limit_sleeves", color=[215, 222, 230, 255],
                 material="304 Stainless Steel compression sleeves (7.93 g/cm3)", **common, **srec)]
    g = by[GUIDE_SCREW]
    gm = BL.load_mesh(g["mesh"])
    gh, _ = _screw_axis(by, GUIDE_SCREW)
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    for old, (x, z) in FAN_HOLES.items():
        m = gm.copy(); m.apply_translation([x - gh[0], 0.0, z - gh[2]])
        name = old.replace("fan_M3_", "fan_plate_M3x16_")
        npz = OUT / "parts" / f"{name}.npz"
        np.savez_compressed(npz, vertices=np.asarray(m.vertices), faces=np.asarray(m.faces))
        q = copy.deepcopy(g)
        for k in ("native_step", "native_step_sha256", "source", "source_transform"):
            q.pop(k, None)
        q.update(name=name, mesh=str(npz.relative_to(ROOT)), mesh_sha256=_sha(npz),
                 bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()],
                 replaces=old, held_parts=f"{TASK}: M3x16, head on the fan bridge plate (mesh of {GUIDE_SCREW} moved to the hole)")
        rows.append(q)
    return rows


def build_indicator(MR, by):
    """Insert grown about INDICATOR_GROW mm sideways (x, z scaled about its box centre; an OCC offset of this B-rep fails),
    clipped to the pocket ellipsoid: fills the pocket and meets the shell wall."""
    from build123d import Solid, import_step
    from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
    from OCP.gp import gp_GTrsf, gp_Mat, gp_XYZ
    import cad_ops as C

    def scaled(shape, k, centre):
        gt = gp_GTrsf(); gt.SetVectorialPart(gp_Mat(k[0], 0, 0, 0, k[1], 0, 0, 0, k[2]))
        gt.SetTranslationPart(gp_XYZ(*[c - kk * c for c, kk in zip(centre, k)]))
        return BRepBuilderAPI_GTransform(shape, gt, True).Shape()
    p = by[INDICATOR]
    ins = import_step(str(MR.F.src(p["native_step"]))).solids()
    ins = C.fuse(*[s.wrapped for s in ins]) if len(ins) > 1 else ins[0].wrapped
    b = np.array(p["bounds_mm"], float)
    ctr, ext = b.mean(0), b[1] - b[0]
    grown = scaled(ins, ((ext[0] + 2 * INDICATOR_GROW) / ext[0], 1.0, (ext[2] + 2 * INDICATOR_GROW) / ext[2]), ctr)
    a = [v * SCALE_T for v in POCKET_AXES]
    ell = scaled(Solid.make_sphere(1.0).wrapped, a, (0.0, 0.0, 0.0))
    ell = C.translate(ell, (0.0, float(ctr[1]), POCKET_Z * SCALE_T))
    plug = C.largest_solid(C.common(grown, ell))
    assert C.valid(plug), "chest indicator plug is not a valid solid"
    rec = MR.save(INDICATOR, plug, "printed")
    rec["held_parts"] = (f"{TASK}: printed with center_front_shell in one job (second nozzle, colour accent), bonded at the pocket "
                         f"wall; insert grown {INDICATOR_GROW} mm and clipped to the pocket ellipsoid")
    print(INDICATOR, "volume", round(C.volume(ins), 1), "->", rec["volume_mm3"], flush=True)
    return rec


def seat_nuts(by):
    """The two plate screws' nuts sat 1.0 mm off their washers (washer y 21.1..21.6, nut from y 22.6; mount_check after
    T-vza9m4): each nut is moved along the screw axis (+y) onto its washer's back face. Records keyed by nut name."""
    from checks import bodylib as BL
    recs = {}
    for scr in FAN_HOLES:
        tag = scr[len("fan_M3_"):]                         # "11_462", "-21_462"
        nut, washer = f"fan_nut_{tag}", f"fan_washer_{tag}"
        nm, wm = BL.load_mesh(by[nut]["mesh"]), BL.load_mesh(by[washer]["mesh"])
        dy = float(wm.bounds[1][1] - nm.bounds[0][1])        # < 0: towards the washer
        m = nm.copy(); m.apply_translation([0.0, dy, 0.0])
        npz = OUT / "parts" / f"{nut}.npz"
        np.savez_compressed(npz, vertices=np.asarray(m.vertices), faces=np.asarray(m.faces))
        recs[nut] = dict(mesh=str(npz.relative_to(ROOT)), mesh_sha256=_sha(npz), native_step=None, native_step_sha256=None,
                         bounds_mm=[m.bounds[0].round(3).tolist(), m.bounds[1].round(3).tolist()],
                         held_parts=f"{TASK}: seated on {washer} (moved {dy:+.3f} mm along y)")
        print(nut, f"moved {dy:+.3f} mm onto {washer}", flush=True)
    return recs


def build():
    import motor_refit as MR
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    MR.PARTS = OUT / "parts"
    by = _src()
    plate, psleeves, plan = build_plate(MR, by)
    recs = {CASE: repair_case(MR, by), INDICATOR: build_indicator(MR, by), **seat_nuts(by)}
    new_rows = plate_rows(by, plate, psleeves)
    (OUT / "records.json").write_text(json.dumps(dict(records=recs, new_rows=new_rows, remove=list(REMOVE),
                                                      replace=list(FAN_HOLES), plate_plan=plan), indent=1) + "\n")
    print("wrote", (OUT / "records.json").relative_to(ROOT), flush=True)


def swap_parts(parts):
    """The part list with the held-parts edits applied (idempotent). Without the build output it stays as it was and says so."""
    if not (OUT / "records.json").exists():
        print(f"WARNING held parts: {(OUT / 'records.json').relative_to(ROOT)} missing, the fan stack, chest indicator, R roll case and "
              f"the stale front-shell screws stay as they were; run `.venv/bin/python -B robot/body/assembly/held_parts.py build` ({TASK})")
        return parts
    d = json.loads((OUT / "records.json").read_text())
    new_names = {r["name"] for r in d["new_rows"]}
    drop = set(d["remove"]) | set(d["replace"]) | new_names
    res = []
    for p in parts:
        if p["name"] in drop:
            continue
        if p["name"] in d["records"]:
            q = dict(p); q.update(d["records"][p["name"]]); q.pop("placement_offset_mm", None)
            if p["name"] != CASE:
                q.pop("native_step_offset_mm", None)
            res.append(q)
        else:
            res.append(p)
    return res + [copy.deepcopy(r) for r in d["new_rows"]]


def apply(variant: Path = VARIANT, out: Path | None = None):
    man = json.loads(variant.read_text())
    n0 = len(man["parts"])
    man["parts"] = swap_parts(man["parts"])
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, n0, "->", len(man["parts"]), "parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
