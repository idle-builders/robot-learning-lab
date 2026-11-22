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

Stage interface joints (T-bv7bwe, owner 2026-11-22: threaded steel sleeves): each stage-1 aux bore that is coaxial with a
cradle / stage 3 / stage 4 bore gets an internally threaded steel sleeve in place of its clearance sleeve (same outer diameter
and length) and an ISO 4762 screw from the carrier side, no nut; the screw is the shortest stocked length that fully engages
the threaded sleeve with its tip MARGIN_MM clear of the cells, fins and boards of the torso section. A pair is skipped (and
recorded with the reason in records.json `interface_joints`) when the stage-1 sleeve itself runs into a cell or the head meets
a board. `plan` prints those plans without writing CAD.

Stage 4 IMU/ADC carrier (Idle T-htfaaz): the outer face sits on the boards' back face (layer gaps 0.10 -> 0.00 mm). The 6 rail
pairs are stage interface joints as above (carrier side: full-thickness flush clearance sleeves from Stage 1's rail face to the
outer face, heads on it); the two on x -11.2 sat under the IMU / ADC boards, so common.stage4_rail_sites moves them to the strip
between the boards and Stage 1 moves with them: the old bores are filled, the pockets re-cut, new bores drilled and the old sleeve
solids dropped (the unscripted Stage 1 STEP is edited here; the chain script of T-dgz6qv should take this over). The 8 board
screws end in printed pilots in bosses on the plate's rear face (PA2, lengths from screw_plan.py, sites from the boards' STEP).

  recreate:  .venv/bin/python -B robot/body/torso/torso_screws.py build   (parts + records.json in OUT)
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
STAGE4 = "stage4_rear_imu_adc_carrier"
PART4, SLEEVES4 = f"combined__{STAGE4}", f"combined__{STAGE4}_steel_sleeves"
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
    recs[PART1] = MR.save(PART1, stage1_moved_rails(common).wrapped, "printed")
    new_parts, joints = build_interface_joints(MR, common, recs[PART4])
    recs.update({k: v for k, v in new_parts.items() if k == STAGE1_SLEEVES})
    added = {k: v for k, v in new_parts.items() if k != STAGE1_SLEEVES}
    screws = stage4_board_screws(recs)
    (OUT / "records.json").write_text(json.dumps(dict(records=recs, added=added, screws=screws, interface_joints=joints), indent=1) + "\n")
    print("wrote", OUT / "records.json", {k: v["volume_mm3"] for k, v in recs.items()}, flush=True)
    for j in joints:
        print(" ", j["joint"], j["status"], j.get("screw"), j.get("engagement_mm"), j.get("tip_margin_mm"), j.get("why", ""), flush=True)


# ------------------------------------------------------------------------------------------- stage interface joints
# Owner 2026-11-22 (T-bv7bwe): the stage-1 side sleeve of each stage-1 / carrier interface pair becomes an internally
# threaded steel sleeve (same outer diameter and length, so the press fit in the chassis hole is unchanged); the clearance
# sleeve stays on the carrier side; the screw goes in from the carrier side, no nut.
STAGE1 = "stage1_center_torso_spine_chassis"
STAGE1_SLEEVES = f"combined__{STAGE1}_steel_sleeves"
PART1 = f"combined__{STAGE1}"
MATES = ("stage2_lower_pelvis_battery_cradle", STAGE3, STAGE4)
SECTION = ROOT / "artifacts/robot/body/torso/combined/assembly-parts.json"    # torso section manifest (cells, fins, boards)
SLEEVE_SOURCE = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"   # chain step before this one
# thread by sleeve bore (clearance r_id): major / minor (ISO 724, the threaded sleeve's modelled bore), ISO 4762 socket
# head diameter / height and the stocked lengths (ISO 4762 preferred lengths)
THREADS = {1.05: dict(thread="M2", major=2.0, minor=1.567, head_d=3.8, head_h=2.0, lengths=(3, 4, 5, 6, 8, 10, 12, 16, 20)),
           1.30: dict(thread="M2.5", major=2.5, minor=2.013, head_d=4.5, head_h=2.5, lengths=(4, 5, 6, 8, 10, 12, 16, 20))}
MARGIN_MM = 0.3          # clearance kept from a cell can, fin, board or other part (screw tip, head, threaded sleeve)


def interface_pairs(common):
    """Coaxial stage-1 / carrier bore pairs from common.stage_interface_specs: the stage-1 aux bore and the mating carrier
    bore on the same y axis. Each with the stage-1 span, the carrier span and the direction from the head to the tip."""
    import numpy as np
    spec = common.stage_interface_specs()
    s1 = [b for b in spec[STAGE1]["aux_bores"] if tuple(b["normal"]) == (0.0, 1.0, 0.0)]
    out = []
    for st in MATES:
        k = 0
        for b in spec[st]["support_bores"] + spec[st]["load_bores"] + spec[st].get("aux_bores", []):
            if tuple(b["normal"]) != (0.0, 1.0, 0.0):
                continue
            a = next((a for a in s1 if abs(a["center"][0] - b["center"][0]) < 0.05 and abs(a["center"][2] - b["center"][2]) < 0.05), None)
            if a is None or abs(a["r_id"] - b["r_id"]) > 1e-6:
                continue
            ya, yb = a["center"][1], b["center"][1]
            d = 1.0 if ya > yb else -1.0                                   # head on the carrier, tip toward stage 1
            span = lambda q: (q["center"][1] - q["length"] / 2, q["center"][1] + q["length"] / 2)
            sa, sb = span(a), span(b)
            out.append(dict(joint=f"{st.split('_')[0]}_{k}", mate=st, s1=a, mate_bore=b, d=d,
                            seat=sb[0] if d > 0 else sb[1],                 # carrier sleeve outer end: the head seat
                            s1_entry=sa[0] if d > 0 else sa[1], s1_exit=sa[1] if d > 0 else sa[0],
                            x=a["center"][0], z=a["center"][2], th=THREADS[round(a["r_id"], 2)]))
            k += 1
    return out


def _ring_hits(meshes, x, z, ys, r):
    """Names of the meshes that contain any point of rings of radius r (16 points) around the y axis at (x, z), y in ys."""
    import numpy as np
    a = np.linspace(0, 2 * np.pi, 17)[:-1]
    pts = np.array([[x + r * np.cos(t), y, z + r * np.sin(t)] for y in ys for t in a])
    hit = set()
    for n, (me, lo, hi) in meshes.items():
        if np.any(pts.max(0) < lo) or np.any(pts.min(0) > hi):
            continue
        if me.contains(pts).any():
            hit.add(n)
    return hit


def _section_meshes():
    """Torso section parts that a screw tip, head or threaded sleeve must keep MARGIN_MM from: cells, insulators, fins,
    boards and modules (the carriers, chassis and sleeves are the clamped stack and are handled by the spans)."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "robot/body/checks"))
    from checks import bodylib as BL
    man = json.loads(SECTION.read_text())
    out = {}
    for p in (man["parts"] if isinstance(man, dict) else man):
        if p.get("owning_link") != "torso" or not p.get("mesh") or BL.is_fastener(p["name"], p):
            continue
        # body shells: outside the stack (heads and tips stay > 4 mm from them); an open shell's hull would cover the torso
        if p["role"] in ("printed_structural_carrier", "compression_limit_sleeves", "body_shell") or p["name"].startswith("combined__"):
            continue
        try:
            me = BL.load_mesh(p["mesh"])
        except Exception:  # noqa: BLE001
            print(f"WARNING torso screws: {p['name']} mesh not loaded, not in the clearance test", flush=True)
            continue
        if not me.is_watertight:          # open mesh (boards, servo cases): holes filled, else its convex hull (conservative)
            m2 = me.copy(); m2.fill_holes()
            me = m2 if m2.is_watertight else me.convex_hull
        out[p["name"]] = (me, me.bounds[0] - 1, me.bounds[1] + 1)
    return out


def _carrier_meshes(rebuilt: dict | None = None):
    """The mating carriers as the chain carries them (shell-pass manifest), the rebuilt ones (name -> record) in place of
    theirs: the head seats on their outer face."""
    sys.path.insert(0, str(ROOT / "robot/body/checks"))
    from checks import bodylib as BL
    ps = {p["name"]: dict(p, **(rebuilt or {}).get(p["name"], {})) for p in json.loads(SLEEVE_SOURCE.read_text())["parts"]}
    return {st: BL.load_mesh(ps[f"combined__{st}"]["mesh"]) for st in MATES}


def head_seat(j, carriers):
    """Outermost surface under the head annulus (between the clearance sleeve and the head rim): the sleeve end, or the
    carrier face where it stands proud of the sleeve."""
    import numpy as np
    d, x, z = j["d"], j["x"], j["z"]
    r = 0.5 * (j["mate_bore"]["r_od"] + j["th"]["head_d"] / 2)
    a = np.linspace(0, 2 * np.pi, 9)[:-1]
    o = np.array([[x + r * np.cos(t), j["seat"] - d * 20.0, z + r * np.sin(t)] for t in a])
    loc, ri, _ = carriers[j["mate"]].ray.intersects_location(o, np.tile([0.0, d, 0.0], (len(o), 1)))
    first = [min(loc[ri == i][:, 1] * d) * d for i in range(len(o)) if np.any(ri == i)]
    out = min([j["seat"] * d] + [f * d for f in first if abs(f - j["seat"]) < 3.0]) * d
    return float(out)


def plan_joint(j, meshes, carriers=None):
    """Threaded sleeve length, screw length, engagement and margins of one pair; status 'placed' or 'skipped' with why."""
    import numpy as np
    if carriers is not None:
        j = dict(j, sleeve_end=j["seat"], seat=head_seat(j, carriers))
    th, d, x, z = j["th"], j["d"], j["x"], j["z"]
    r_od, L_s1 = j["s1"]["r_od"], j["s1"]["length"]
    step = 0.1
    # threaded sleeve: the stage-1 sleeve's own span; its outer wall must keep MARGIN_MM from every part in `meshes`
    ys = np.arange(0, L_s1 + 1e-9, step) * d + j["s1_entry"]
    first_hit = next((i for i, y in enumerate(ys) if _ring_hits(meshes, x, z, [y], r_od + MARGIN_MM)), None)
    free_len = L_s1 if first_hit is None else step * first_hit
    res = dict(joint=j["joint"], mate=j["mate"], axis_mm=[round(x, 2), round(z, 2)], thread=th["thread"],
               threaded_sleeve=dict(od_mm=round(2 * r_od, 2), length_mm=L_s1, bore_mm=th["minor"],
                                    span_y_mm=sorted([round(j["s1_entry"], 2), round(j["s1_exit"], 2)])))
    hit_sleeve = sorted(_ring_hits(meshes, x, z, ys, r_od))
    if hit_sleeve or free_len < L_s1 - 1e-6:
        res.update(status="skipped", free_sleeve_length_mm=round(float(free_len), 2),
                   why=f"stage-1 sleeve span reaches {', '.join(hit_sleeve) or 'a part'} (free {free_len:.2f} of {L_s1} mm at "
                       f"{MARGIN_MM} mm margin): the bore must move off it before a threaded sleeve can go in")
        return res
    # head: socket head cylinder from the seat outward must be clear
    hy = j["seat"] - d * np.arange(0.1, th["head_h"] + 1e-9, step)
    hit_head = sorted(_ring_hits(meshes, x, z, hy, th["head_d"] / 2 + MARGIN_MM) | _ring_hits(meshes, x, z, hy, 0.5))
    if hit_head:
        res.update(status="skipped", why=f"screw head ({th['head_d']} x {th['head_h']} mm) on the carrier face meets {', '.join(hit_head)}")
        return res
    # tip limit: the shank (major radius + margin) beyond the threaded sleeve exit must keep clear of the parts
    beyond = np.arange(step, 12.0, step)
    lim = next((t for t in beyond if _ring_hits(meshes, x, z, [j["s1_exit"] + d * t], th["major"] / 2 + MARGIN_MM)), 12.0) - step
    need = d * (j["s1_exit"] - j["seat"])                                   # screw length that reaches the sleeve exit
    ok = [L for L in th["lengths"] if L <= need + lim + 1e-9]
    full = [L for L in ok if L >= need - 1e-9]
    if not ok:
        res.update(status="skipped", why="no stocked length fits")
        return res
    L = full[0] if full else ok[-1]
    tip = j["seat"] + d * L
    eng = min(L_s1, max(0.0, d * (tip - j["s1_entry"])))
    res.update(status="placed", screw_length_mm=L, screw=f"torso_iface_{j['joint']}_{th['thread'].replace('.', 'p')}x{L}",
               engagement_mm=round(eng, 2), engagement_of_sleeve_pct=round(100 * eng / L_s1, 1),
               tip_beyond_sleeve_mm=round(d * (tip - j["s1_exit"]), 2),
               tip_margin_mm=round(lim - d * (tip - j["s1_exit"]) + MARGIN_MM, 2), head_seat_y_mm=round(j["seat"], 2),
               carrier_face_proud_of_sleeve_mm=round(d * (j.get("sleeve_end", j["seat"]) - j["seat"]), 2))
    return res


def build_interface_joints(MR, common, rec4=None):
    """Threaded stage-1 sleeves + screws for every interface pair that has room; the stage-1 clearance sleeve part
    without the threaded positions (and without the sleeves of the stage 4 rail sites that moved). rec4: record of the rebuilt
    stage 4 carrier. Returns ({part: record}, [joint plan])."""
    from build123d import Compound
    meshes, carriers = _section_meshes(), _carrier_meshes({PART4: rec4} if rec4 else None)
    pairs = interface_pairs(common)
    plans = [plan_joint(j, meshes, carriers) for j in pairs]
    pairs = [dict(j, seat=p["head_seat_y_mm"]) if p["status"] == "placed" else j for j, p in zip(pairs, plans)]
    import numpy as np
    from build123d import import_step
    from checks import bodylib as BL
    placed = [np.array(j["s1"]["center"], float) for j, p in zip(pairs, plans) if p["status"] == "placed"]
    # the stage-1 clearance sleeves the chain carries (shell-pass manifest; the shoulder pitch sleeves there sit at x +-42.4,
    # not at common.stage_interface_specs' +-30.4), minus the sleeves that become threaded
    src = next(p for p in json.loads(SLEEVE_SOURCE.read_text())["parts"] if p["name"] == STAGE1_SLEEVES)["native_step"]
    sols = list(import_step(str(BL.src(src))).solids())
    def centre(s):
        bb = s.bounding_box(); return np.array([(bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2, (bb.min.Z + bb.max.Z) / 2])
    legacy = [np.array([x, 33.6, z]) for x, z in common.legacy_stage4_rail_sites()]
    keep = [s for s in sols if min((np.linalg.norm(centre(s) - c) for c in placed + legacy), default=9.0) > 0.2]
    moved = [c for c in placed if any(abs(c[0] - x) < 0.05 and abs(c[2] - z) < 0.05 for x, z in common.stage4_rail_sites() if z not in common.STAGE4_RAIL_Z)]
    if len(sols) - len(keep) != len(placed) - len(moved) + len(legacy):     # the moved sites have no solid in the STEP yet
        raise SystemExit(f"stage-1 sleeve STEP {src}: {len(sols) - len(keep)} solids dropped at {len(placed)} threaded positions "
                         f"({len(moved)} of them moved rail sites) + {len(legacy)} old rail sites")
    out = {STAGE1_SLEEVES: MR.save(STAGE1_SLEEVES, Compound(children=keep).wrapped, "purchased")}
    out[STAGE1_SLEEVES]["sleeve_source"] = (f"{src} minus the {len(placed)} threaded interface positions (T-bv7bwe) and the {len(legacy)} "
                                            "old stage 4 rail sites (T-htfaaz)")
    out[STAGE1_SLEEVES]["mass_kg"] = round(out[STAGE1_SLEEVES]["volume_mm3"] * 7.93e-6, 6)
    out[STAGE1_SLEEVES]["mass_basis"] = "native STEP volume x 7.93 g/cm3 (304 SS)"
    for j, p in zip(pairs, plans):
        if p["status"] != "placed":
            continue
        th, d, x, z = j["th"], j["d"], j["x"], j["z"]
        b = j["s1"]
        sl = common.aligned_cylinder(b["center"], b["normal"], b["r_od"], b["length"]).cut(
            common.aligned_cylinder(b["center"], b["normal"], th["minor"] / 2, b["length"] + 0.4))
        sn = f"stage1_iface_threaded_sleeve_{j['joint']}_{th['thread'].replace('.', 'p')}"
        out[sn] = dict(MR.save(sn, sl.wrapped, "purchased"), role="threaded_sleeve", owning_link="torso",
                       hardware_spec=dict(kind="threaded_sleeve", thread=th["thread"], outer_dia_mm=round(2 * b["r_od"], 2),
                                          length_mm=b["length"], material="304 stainless steel (as the clearance sleeves)",
                                          press_fit="same outer diameter as the clearance sleeve it replaces"))
        out[sn]["mass_kg"] = round(out[sn]["volume_mm3"] * 7.93e-6, 6)
        L = p["screw_length_mm"]
        head_c = (x, j["seat"] - d * th["head_h"] / 2, z)
        shank_c = (x, j["seat"] + d * L / 2, z)
        scr = common.aligned_cylinder(head_c, (0.0, 1.0, 0.0), th["head_d"] / 2, th["head_h"]).fuse(
            common.aligned_cylinder(shank_c, (0.0, 1.0, 0.0), th["major"] / 2, L))
        out[p["screw"]] = dict(MR.save(p["screw"], scr.wrapped, "purchased"), role="fastener", owning_link="torso",
                               hardware_spec=dict(kind="screw", standard="ISO 4762 socket head cap screw", thread=th["thread"],
                                                  length_mm=L, threaded_into=sn, clamps=[j["mate"], STAGE1]))
        p["threaded_sleeve_part"] = sn
    return out, plans


def stage1_moved_rails(common):
    """Stage 1 (the chain's unedited STEP) with the bores of the moved stage 4 rail sites filled and the bores of their new sites
    drilled: the old bore (dia 3.98, from its bottom y 25.6 to the rail face y 35.4) is filled outside the battery pockets (the bore ran
    through the web between two pockets); the new bore is the old one's size from 0.2 mm beyond the sleeve's inner end to the rail face
    (the threaded sleeve sits in it, press fit unchanged), then a r 1.5 tip clearance bore 0.5 mm deeper (floor y 31.4). Prints the volume added and removed."""
    from build123d import Cylinder, Pos, import_step
    from checks import bodylib as BL
    src = next(p for p in json.loads(SLEEVE_SOURCE.read_text())["parts"] if p["name"] == PART1)["native_step"]
    body = max(import_step(str(BL.src(src))).solids(), key=lambda x: x.volume)
    y0 = common.STAGE4_RAIL_Y0
    spec = common.stage_interface_specs()[STAGE1]
    length = next(b["length"] for b in spec["aux_bores"] if abs(b["center"][1] - 33.6) < 1e-6 and b["r_id"] > 1.2)
    # stepped bore: the sleeve bore (r 1.99, the STEP's old rail bore, dia 3.98) ends 0.2 mm beyond the sleeve end (y 31.9, as before), a
    # tip clearance bore r 1.5 (M2.5 thread r 1.25 + 0.25) goes on to y 31.4: the M2.5x10 tip (y 32.0) stays 0.6 mm off the floor (ISO 4762
    # length tolerance +-0.29). The battery pockets (r 10.95, 1.99 mm off the bore axis at the rim) reach y 31.52 at r 1.99 but only
    # y 30.68 at r 1.5, so the deeper bore stays r <= 1.5: floor 0.72 mm to the pocket at r 1.5 (a full-radius bore to 31.4 broke into the pockets)
    # @lore the first build left 0.2 mm (floor 0.1 mm under the tip): a long screw would bottom before the head seated
    r_bore, r_tip, y_in = 1.99, 1.5, 33.6 - length / 2 - 0.2
    y_tip = y_in - 0.5
    y_old = 25.6                                          # the old bores' bottom in the STEP: the fill starts there (25.9 left a 0.25 mm slot)
    old = common.legacy_stage4_rail_sites()
    new = [s for s in common.stage4_rail_sites() if s[1] not in common.STAGE4_RAIL_Z]
    assert len(old) == len(new) == 2, (old, new)
    pockets = [Pos(c["center"][0], c["center"][1], 301.8) * Cylinder(10.95, 76.0)
               for c in common.HONEYCOMB_9CELL_BATTERY_LAYOUT if c["stage"] == STAGE1]
    fills = [common.aligned_cylinder((x, (y_old + y0) / 2, z), (0.0, 1.0, 0.0), r_bore + 0.05, y0 - y_old) for x, z in old]
    holes = [common.aligned_cylinder((x, (y_in + y0 + 0.5) / 2, z), (0.0, 1.0, 0.0), r_bore, y0 + 0.5 - y_in) for x, z in new]
    holes += [common.aligned_cylinder((x, (y_tip + y_in) / 2 + 0.05, z), (0.0, 1.0, 0.0), r_tip, y_in - y_tip + 0.1) for x, z in new]
    fills = [f.cut(*pockets) for f in fills]                # only the part of the old bore outside the battery pockets is filled
    out = max(body.fuse(*fills).cut(*holes).solids(), key=lambda x: x.volume)
    print(f"stage 1 moved rail bores: {out.cut(body).volume:.2f} mm3 added (old bores filled), {body.cut(out).volume:.2f} mm3 removed "
          f"(new bores: at most {sum(h.volume for h in holes):.2f} mm3)", flush=True)
    return out


def stage4_board_screws(recs: dict) -> list[dict]:
    """The 8 IMU / ADC board screws as manifest entries (screw_plan.py plans, envelope meshes written next to the parts): M2 PA2
    thread-forming screws from the boards' outer side into printed pilots of the carrier bosses."""
    import numpy as np
    import trimesh
    import common
    import screw_plan as SP
    rho = SP.standards()["density_g_cm3"]
    carrier = np.load(ROOT / recs[PART4]["mesh"])
    meshes = {PART4: trimesh.Trimesh(carrier["vertices"], carrier["faces"], process=False)}
    sites = common.stage4_board_sites()
    y1 = common.STAGE4_FRONT_Y
    holes = []
    for part in common.STAGE4_BOARDS:                              # PCB slab: back face to the far end of the mount hole
        hs = [h for h in sites if h["board"] == part]
        (xa, xb), (za, zb), top = hs[0]["board_x"], hs[0]["board_z"], max(h["y_top"] for h in hs)
        meshes[part] = trimesh.creation.box(extents=[xb - xa, top - hs[0]["y_seat"], zb - za],
                                            transform=trimesh.transformations.translation_matrix([(xa + xb) / 2, (top + hs[0]["y_seat"]) / 2, (za + zb) / 2]))
        for i, h in enumerate(hs):
            holes.append(dict(hole_id=f"{part}:pcb.{i}", opening=[h["x"], h["y_seat"], h["z"]], axis_out=[0.0, 1.0, 0.0], thread="M2",
                              terminal_material="plastic", depth_mm=y1 - common.STAGE4_BOSS_REAR_Y, reference_engagement_mm=2.0 * 2.0))
    iface = dict(holes=holes)
    out = []
    for conn in ("torso_stage4_imu_board", "torso_stage4_adc_board"):
        for pl in SP.plan_connector(None, iface, conn, "", mesh=lambda n: meshes[n]):
            pl.update(thread_label="PA2", family="PA")            # the standards table has no PA2 head series, the M2 series is used
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
                material=f"{pl['head_standard']} {lbl} into {pl['hole_id']} (printed pilot, dia {common.PA2_PILOT_DIA}); head dia "
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


def swap_parts(parts):
    """The manifest part list with the stage 1 (rail bores), 3 and 4 carriers + sleeves replaced by the rebuilt ones, the stale
    torso_mate hardware dropped and the 8 stage 4 board screws replaced by the generated ones. Without the build output the manifest stays buildable (unedited stage 3, stale parts kept) and says so."""
    if not (OUT / "records.json").exists():
        print(f"WARNING torso screws: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED stage 3 carrier and the "
              "stale torso_mate parts stay in the manifest; run `.venv/bin/python -B robot/body/torso/torso_screws.py build` "
              "(T-nh6ytf) first")
        return parts
    doc = json.loads((OUT / "records.json").read_text())
    recs, added = doc["records"], doc.get("added", {})
    new = {q["name"]: q for q in doc.get("screws", [])}           # stage 4 board screws replace the legacy ones of the same name
    res = []
    for p in parts:
        if p["name"].startswith(STALE) or p["name"] in added:     # added: re-applied below (idempotent)
            continue
        if p["name"] in new:
            res.append(new.pop(p["name"]))
            continue
        if p["name"].startswith(LEGACY_BOARD_SCREWS):
            continue
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("native_step_offset_mm", None); q.pop("placement_offset_mm", None)
            q["torso_screws"] = "rebuilt by robot/body/torso/torso_screws.py (T-nh6ytf)"
            res.append(q)
        else:
            res.append(p)
    tmpl = next((p for p in res if p["name"] == STAGE1_SLEEVES), None)
    for n, rec in added.items():                 # threaded stage-1 sleeves and their screws (T-bv7bwe)
        q = {k: v for k, v in (tmpl or {}).items() if k in ("color", "display_group", "display_groups", "parent_assembly",
                                                              "inspection_reference", "overlay_id")}
        q.update(name=n, **rec)
        q["torso_screws"] = "added by robot/body/torso/torso_screws.py (T-bv7bwe: threaded stage-1 interface sleeves)"
        res.append(q)
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
    ap.add_argument("phase", choices=["build", "apply", "plan"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.phase == "plan":                        # the interface joint plans only (no CAD written)
        import common
        ms = _section_meshes()
        cs = _carrier_meshes()
        print(json.dumps([plan_joint(j, ms, cs) for j in interface_pairs(common)], indent=1))
    else:
        build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
