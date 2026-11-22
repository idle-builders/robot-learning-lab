"""Torso and pelvis screw connections (Idle T-nh6ytf): parts rebuilt for the mount check, manifest patched.

What the mount check measured on the selected body (2026-11-20, robot/body/checks run.py --section Torso):

  stage 3 power carrier   the board bosses stopped 0.51 mm (power supervisor) / 1.74 mm (main precharge) short of the PCB, and the
                          supervisor sleeves (4.0 mm) covered 4.0 of the 6.45 mm boss + frame, so the clamped stack had layer gaps
                          of 0.52 / 1.74 mm (board to boss) and 2.29 mm (sleeve to the rear washer); the precharge sleeves had a
                          2.0 mm bore for a 2.0 mm M2 shank.
  torso_mate x4           M3x16 screws, washers and nuts from the earlier waist-pelvis mate (artifact mate-r7, no script): the nut
                          sits in the honeycomb web, 1.5-2.0 mm of shank in solid chassis, heads inside the stage 4 plate. They
                          clamp nothing of the present stages (those use the stage bores of common.stage_interface_specs).

This script rebuilds the stage 3 carrier and its sleeves from build_torso_carriers.py (the same builder as the selected STEP:
17364.17 mm3 before the edit) with bosses that reach the PCB faces and full-thickness flush M2 sleeves (bore 2.2 mm,
standard-parts.json sleeve.bore_for; head on the outer face), and drops the stale torso_mate parts. Positions come from
common.stage_interface_specs and common.STAGE3_*; nothing is typed here.

Stage 4 IMU/ADC carrier (Idle T-htfaaz): outer face on the boards' back face (layer gaps 0.10 -> 0.00 mm), the 6 rail screws (PA2,
heads on the outer face outside the board footprints, full-thickness flush sleeves) land in printed pilots added to Stage 1's rails
(Stage 1's 6 dorsal sleeves dropped: the receiving side carries no sleeve; the unscripted Stage 1 STEP is edited in place here, the
chain script of T-dgz6qv should take the pilots over), the 8 board screws end in printed pilots of bosses on the plate's rear face
(sleeves dropped: the clamped part is the board). Screw lengths from screw_plan.py, board sites from the boards' own STEP.

  recreate:  .venv/bin/python -B robot/body/torso/torso_screws.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/torso/torso_screws.py screws  (only the stage 4 screws, against the last build)
             .venv/bin/python -B robot/body/torso/torso_screws.py apply   (patch the roll-flip variant manifest)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(ROOT / "robot/body"))
sys.path.insert(0, str(ROOT / "robot/body/fastening"))

OUT = ROOT / "artifacts/robot/body/torso/torso-screws"
VARIANT = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json"
STAGE3 = "stage3_front_power_electronics_carrier"
PART, SLEEVES = f"combined__{STAGE3}", f"combined__{STAGE3}_steel_sleeves"
STAGE4, STAGE1 = "stage4_rear_imu_adc_carrier", "stage1_center_torso_spine_chassis"
PART4, SLEEVES4 = f"combined__{STAGE4}", f"combined__{STAGE4}_steel_sleeves"
PART1, SLEEVES1 = f"combined__{STAGE1}", f"combined__{STAGE1}_steel_sleeves"
PA2_PILOT_DIA, RAIL_PILOT_DEPTH = 1.6, 5.6    # PA2 thread-forming pilot: 80 % of the 2.0 major diameter; depth below the tip of the PA2x12
LEGACY_BOARD_SCREWS = ("IMU_M2x6_", "ADC_M2x6_")
STALE = ("torso_mate_",)      # torso_mate_<n>_M3x16_socket_screw / _rear_M3_washer / _front_M3_washer / _M3_hex_nut


def build():
    import build_torso_carriers as B
    import common
    import motor_refit as MR
    (OUT / "parts").mkdir(parents=True, exist_ok=True)
    MR.PARTS = OUT / "parts"
    body, _ = B.BUILDERS[STAGE3]("truss_box")        # COMBINED_WINNERS: stage 3 is the truss_box family
    sleeves, _, _ = common.build_stage_sleeves(STAGE3)
    recs = {PART: MR.save(PART, body.wrapped, "printed"),
            SLEEVES: MR.save(SLEEVES, sleeves.wrapped, "purchased")}
    recs[SLEEVES]["mass_kg"] = round(recs[SLEEVES]["volume_mm3"] * 7.93e-6, 6)
    recs[SLEEVES]["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    body4, _ = B.BUILDERS[STAGE4]("topopt")          # COMBINED_WINNERS: stage 4 is the topopt family
    sleeves4, _, _ = common.build_stage_sleeves(STAGE4)
    recs[PART4] = MR.save(PART4, body4.wrapped, "printed")
    recs[SLEEVES4] = MR.save(SLEEVES4, sleeves4.wrapped, "purchased")
    recs[SLEEVES4]["mass_kg"] = round(recs[SLEEVES4]["volume_mm3"] * 7.93e-6, 6)
    recs[SLEEVES4]["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    src1 = stage1_source()
    body1, sleeves1 = stage1_pilots(src1)
    recs[PART1] = MR.save(PART1, body1.wrapped, "printed")
    recs[SLEEVES1] = MR.save(SLEEVES1, sleeves1.wrapped, "purchased")
    recs[SLEEVES1]["mass_kg"] = round(recs[SLEEVES1]["volume_mm3"] * 7.93e-6, 6)
    recs[SLEEVES1]["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    screws = stage4_screws(recs)
    (OUT / "records.json").write_text(json.dumps(dict(records=recs, screws=screws, stage1_source=src1), indent=1) + "\n")
    print("wrote", OUT / "records.json", {k: v["volume_mm3"] for k, v in recs.items()}, len(screws), "screws", flush=True)


def stage1_source() -> dict:
    """The unedited Stage 1 part + sleeves of the roll-flip variant manifest (paths); after `apply` the manifest points at OUT, then
    the paths recorded by the earlier build are used."""
    man = json.loads(VARIANT.read_text())
    ent = {p["name"]: p for p in man["parts"]}
    res = {k: ent[k]["native_step"] for k in (PART1, SLEEVES1)}
    if any(v.startswith(str(OUT.relative_to(ROOT))) for v in res.values()):
        res = json.loads((OUT / "records.json").read_text())["stage1_source"]
    return res


def stage1_pilots(src: dict):
    """Stage 1 with printed pilots at the 6 rail screw sites and without its 6 dorsal rail sleeves (decision M: the receiving side
    is a printed pilot, no sleeve). The old bores (stage_interface_specs aux_bores on Y = 33.6) are filled inside the rail
    (the battery pockets stay open) and the pilots (PA2, dia 1.6, RAIL_PILOT_DEPTH deep from the rail face) drilled at the new sites."""
    from build123d import Compound, Cylinder, Pos, import_step
    import common
    from checks import bodylib
    body = max(import_step(str(bodylib.src(src[PART1]))).solids(), key=lambda x: x.volume)
    sl = import_step(str(bodylib.src(src[SLEEVES1]))).solids()
    spec1 = common.stage_interface_specs()[STAGE1]
    old = [b["center"] for b in spec1["support_bores"] + spec1["load_bores"] + spec1.get("aux_bores", [])
           if abs(b["center"][1] - 33.6) < 1e-6 and abs(b["normal"][1]) > 0.9]
    y0 = common.STAGE4_RAIL_Y0
    assert len(old) == 6, old
    fills = [common.aligned_cylinder((c[0], y0 - 2.6, c[2]), (0.0, 1.0, 0.0), 1.99, 5.2) for c in old]
    pockets = [Pos(c["center"][0], c["center"][1], 301.8) * Cylinder(10.95, 76.0)
               for c in common.HONEYCOMB_9CELL_BATTERY_LAYOUT if c["stage"] == STAGE1]
    body = body.fuse(*fills).cut(*pockets)
    pilots = [common.aligned_cylinder((x, y0 - RAIL_PILOT_DEPTH / 2 + 0.5, z), (0.0, 1.0, 0.0), PA2_PILOT_DIA / 2, RAIL_PILOT_DEPTH + 1.0)
              for x, z in common.stage4_rail_sites()]
    body = max(body.cut(*pilots).solids(), key=lambda x: x.volume)
    keep = [q for q in sl if not any(abs(q.center().X - c[0]) < 0.3 and abs(q.center().Z - c[2]) < 0.3 and abs(q.center().Y - c[1]) < 0.3
                                     for c in old)]
    assert len(keep) == len(sl) - 6, (len(keep), len(sl))
    return body, Compound(children=keep)


def stage4_screws(recs: dict) -> list[dict]:
    """The 6 rail + 8 board screws as manifest entries (screw_plan.py plans, envelope meshes written next to the parts)."""
    import numpy as np
    import trimesh
    import common
    import screw_plan as SP
    rho = SP.standards()["density_g_cm3"]
    meshes = {n: trimesh.Trimesh(*[np.load(ROOT / recs[n]["mesh"])[k] for k in ("vertices", "faces")], process=False) for n in (PART4, SLEEVES4)}
    sites = common.stage4_board_sites()
    y0, y1 = common.STAGE4_RAIL_Y0, common.STAGE4_FRONT_Y
    holes, boards = [], {}
    for i, (x, z) in enumerate(common.stage4_rail_sites()):      # Stage 1 pilot openings on the rail face
        holes.append(dict(hole_id=f"stage1_rail:rail.{i}", opening=[x, y0, z], axis_out=[0.0, 1.0, 0.0], thread="M2", terminal_material="plastic",
                          depth_mm=RAIL_PILOT_DEPTH, reference_engagement_mm=2.0 * 2.0))
    for part in common.STAGE4_BOARDS:                              # PCB slab: back face to the far end of the mount hole
        hs = [h for h in sites if h["board"] == part]
        (xa, xb), (za, zb), top = hs[0]["board_x"], hs[0]["board_z"], max(h["y_top"] for h in hs)
        boards[part] = trimesh.creation.box(extents=[xb - xa, top - hs[0]["y_seat"], zb - za],
                                            transform=trimesh.transformations.translation_matrix([(xa + xb) / 2, (top + hs[0]["y_seat"]) / 2, (za + zb) / 2]))
        for i, h in enumerate(hs):
            holes.append(dict(hole_id=f"{part}:pcb.{i}", opening=[h["x"], h["y_seat"], h["z"]], axis_out=[0.0, 1.0, 0.0], thread="M2",
                              terminal_material="plastic", depth_mm=y1 - common.STAGE4_BOSS_REAR_Y, reference_engagement_mm=2.0 * 2.0))
    iface = dict(holes=holes)
    mesh = lambda n: {**meshes, **boards}[n]
    out = []
    for conn in ("torso_stage4_rail", "torso_stage4_imu_board", "torso_stage4_adc_board"):
        for pl in SP.plan_connector(None, iface, conn, "", mesh=mesh):
            # thread-forming screws into the printed pilots: the standards table has no PA2 head series, the M2 series is used
            pl.update(thread_label="PA2", family="PA")
            pl["name"] = pl["name"].replace("{L}", f"{pl['length_mm']:g}")
            pl["bom_group"] = f"{pl['group']}_PA2x{pl['length_mm']:g}"
            m = SP.screw_mesh(pl)
            f = OUT / "parts" / f"{pl['name']}.npz"
            np.savez_compressed(f, vertices=np.asarray(m.vertices, np.float32), faces=np.asarray(m.faces, np.int32))
            lbl = f"PA2x{pl['length_mm']:g}"
            out.append(dict(
                name=pl["name"], physical_part_id=pl["name"], bom_group=pl["bom_group"], owning_link=pl["owning_link"], role="fastener_proxy",
                mesh=str(f.relative_to(ROOT)), mesh_sha256=hashlib.sha256(f.read_bytes()).hexdigest(), mesh_units="mm",
                bounds_mm=[np.round(m.bounds[0], 3).tolist(), np.round(m.bounds[1], 3).tolist()],
                volume_mm3=round(float(m.volume), 3), mass_kg=round(float(m.volume) * rho / 1000.0, 7),
                material=f"{pl['head_standard']} {lbl} into {pl['hole_id']} (printed pilot, dia {PA2_PILOT_DIA}); head dia "
                         f"{pl['head_dia_mm']} x {pl['head_height_mm']}, shank dia 2, {pl['point_mm']} mm point (envelope)",
                color=[195, 204, 214, 255], display_group="mechanism", display_groups=["mechanism"],
                overlay_id="actuators_and_hardware", parent_assembly="complete_fdm_ape_body_assembly", inspection_reference=False,
                mass_basis="mesh volume x 7.85 g/cm3", torso_screws="generated by robot/body/torso/torso_screws.py (T-htfaaz)",
                screw_plan={k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in pl.items()
                            if k in ("hole_id", "thread_label", "head_standard", "length_mm", "stack_mm", "head_gap_mm", "engagement_mm",
                                     "reference_engagement_mm", "hole_depth_mm", "tip_margin_mm", "shortfall_mm", "overrun_mm",
                                     "stack_change_to_fit_mm")}))
            print(f"{pl['name']}: {lbl} into {pl['hole_id']}: stack {pl['stack_mm']}, engagement {pl['engagement_mm']} of "
                  f"{pl['reference_engagement_mm']}, tip margin {pl['tip_margin_mm']}", flush=True)
    return out


def screws_only():
    """Regenerate the stage 4 screws against the parts of the last build (records.json)."""
    data = json.loads((OUT / "records.json").read_text())
    data["screws"] = stage4_screws(data["records"])
    (OUT / "records.json").write_text(json.dumps(data, indent=1) + "\n")


def swap_parts(parts):
    """The manifest part list with the stage 1, 3 and 4 carriers + sleeves replaced by the rebuilt ones, the stale torso_mate
    hardware dropped and the stage 4 screws generated (the 8 legacy IMU / ADC screws are replaced in place by the generated ones
    of the same name, the 6 rail screws are appended). Without the build output the manifest stays buildable (unedited parts, stale
    hardware kept) and says so."""
    if not (OUT / "records.json").exists():
        print(f"WARNING torso screws: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED stage 1 / 3 / 4 carriers and the "
              "stale torso_mate parts stay in the manifest; run `.venv/bin/python -B robot/body/torso/torso_screws.py build` "
              "(T-nh6ytf, T-htfaaz) first")
        return parts
    data = json.loads((OUT / "records.json").read_text())
    recs = data["records"]
    new = {q["name"]: q for q in data.get("screws", [])}
    res = []
    for p in parts:
        if p["name"].startswith(STALE):
            continue
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("native_step_offset_mm", None); q.pop("placement_offset_mm", None)
            q["torso_screws"] = "rebuilt by robot/body/torso/torso_screws.py (T-nh6ytf, T-htfaaz)"
            res.append(q)
        elif p["name"] in new:
            res.append(new.pop(p["name"]))
        elif p["name"].startswith(LEGACY_BOARD_SCREWS):
            continue
        else:
            res.append(p)
    return res + list(new.values())


def apply(variant: Path = VARIANT, out: Path | None = None):
    """Patch a roll-flip variant manifest in place (or into `out`)."""
    man = json.loads(variant.read_text())
    n0 = len(man["parts"])
    man["parts"] = swap_parts(man["parts"])
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, n0, "->", len(man["parts"]), "parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "screws", "apply"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.phase == "apply":
        apply(Path(a.variant), Path(a.out) if a.out else None)
    else:
        build() if a.phase == "build" else screws_only()
