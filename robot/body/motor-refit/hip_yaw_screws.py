"""Hip yaw-to-pitch connector (R/L_hip_yaw_to_pitch_combined): mountable A14 screws and pressable sleeves.

The part comes from the motor refit (motor_refit.py, truss-box stage 6 edited for the ST3025 hip pitch case). Its screws
were drawn for the old C047 case and the horn reference of r1: 4 M3x14 reached 1.5 of the horn flange's 2.5 mm, the head
seat of the sleeves sat 12.0 mm from the horn, and the
4 front PA2x4 stood at C047 hole positions that the ST3025 case does not have (shank in solid print). This script edits the
refit STEP with the same B-rep operations (cad_ops) and takes every hole position from the standard-part definitions
(robot/actuators/interfaces/standard-parts.json, registered instances) through screw_plan.hole_positions:

  yaw horn   4 x ST3215.horn+y.n (aluminium, tapped through the 2.5 mm flange, reference engagement 2.5 mm): the head seat
             sits at opening - (L - point - reference engagement) of the standard length L (screw_plan.choose_length), the
             sleeve runs the full printed thickness down to the seat. Under the deck one 1 mm layer carries the four sleeves: an
             R 9.0 disc united with an R 4.9 collar per screw (separate collars, 9.9 mm apart on the diagonals, left 0.1 mm
             slots) and a tab from the two window-side collars to the window wall (90 deg corners, 54 before), minus the hub
             bore cut from the seat up (the hub pocket floor moves from 262.0 to the seat 261.75: a collar cut only from 262.0
             left a 0.25 mm lip with a 0.4 mm gap under it). The -x collar stands 1.35 mm past the deck disc: it rises 1.0 mm
             above the deck bottom (a 2.0 mm wall; a 1.0 mm fin at the top of the -x-up print before, 8 mm2; trimmed to the disc
             instead, that sleeve's press support fell 88.1 -> 76.7 %). Where the window-side collars cross the walls' -x
             corner the layer takes the deck's own outline (0.22 mm crescent and 0.04 mm2 overhang before). Driving: the
             sleeve of the front screws z.1 / z.2 (y -6 / 10, Z 247) ends 0.09 mm short of the yaw axes (59,-5) / (59,9), whose hole
             positions are fixed, so no straight tool passes (obstruction 52 mm3 part + 4.5 mm3 sleeve, r 1.45 key cylinder probe of review rv2-24r9hj). The key is tilted
             12-25 deg toward the free +x side (ISO 2936 ball-end hex key, 2.5 mm AF): nothing of the part or the sleeves is in that cone.
             The window edges (y -7.85 / 11.85) stood 0.1 mm off the heads (R 2.75): a R 3.25 pocket in the wall gives 0.5 mm.
  lattice    the front lattice's flat top (refit z 250.0012) is raised to 250.056, the crown of the outer front pads: one plane,
             1.25 mm of print over the sleeves; the head pockets of those two screws are open to the top (no skin, no 0.05 mm breakout).
  trim       the refit's plate ends left 0.0-0.65 mm tongues and a 0.4 mm ramp above the lattice top at x 57.2 (51 deg tips), a 0.4 mm proud rim
             (x 58.91-59.3) round the front pockets and crescents at the bottom bar's old hole. Ramp and tongues are filled to a 1.6 mm step (x 57.2-58.8, up to
             z 251.5, the fill carries the load the thin ramp did: FEM heavy (Bambu PAHT-CF) 1.25, 1.19 with the fill top at 250.55) and cut above it,
             the rim is cut flush, the crescents are cut at the old hole's centre plane (-y) / the refit's own inner face y 12.152 (+y).
             The 0.002-0.03 mm2 faces at the fill top (y -7.85 / -1.6 / 5.6 / 11.85) were split pieces of the window walls' and the
             middle plate's side planes, whose front rounds out past x 58.8 there: the part is unified after the cuts. Filling or padding the
             front to the seat plane 58.91 instead left 0.0002 mm edges or 0.006 mm walls along the padded fillets.
  ear        a 0.6 mm chamfer along the ear's outer vertical edge (55.8, 19.0) gives 0.5 mm to the spine chassis tab at standing (0.30 before).
  front      ST3025.case+z holes (aluminium M2, depth 4.5, reference engagement 3.0) listed in connector_screws.json: pad from the case's
             main face (exact contact), M2 sleeve through the plate, seat plane at the standard length's window.  The refit's old C047 holes (y -8.25 / 12.25) stay as they are: filling them left slivers.
  recreate:  .venv/bin/python -B robot/body/motor-refit/hip_yaw_screws.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/motor-refit/hip_yaw_screws.py apply   (patch the roll-flip variant manifest)
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
sys.path.insert(0, str(ROOT / "robot/body/fastening"))
sys.path.insert(0, str(ROOT / "robot/body"))
import cad_ops as C  # noqa: E402
import motor_refit as MR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("hip-yaw-screws")   # configured values: robot/body/motor-refit/hip_yaw_screws.bom.yaml config:

OUT = ROOT / "artifacts/robot/body/leg/hip-yaw-screws"
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
VARIANT = SHELL_PASS.with_name("assembly-parts-roll-flip.json")
PART, SLEEVES = CFG["PART"], CFG["SLEEVES"]

R_SLEEVE_YAW = CFG["R_SLEEVE_YAW"]
R_SLEEVE_FRONT = CFG["R_SLEEVE_FRONT"]
COLLAR_R = CFG["COLLAR_R"]
COLLAR_DISC_R = CFG["COLLAR_DISC_R"]
DECK_BOTTOM = CFG["DECK_BOTTOM"]
FRONT_PAD_R = CFG["FRONT_PAD_R"]
LATTICE_TOP, FRONT_TOP = CFG["LATTICE_TOP"], CFG["FRONT_TOP"]
YAW_NOTCH_R, YAW_NOTCH_Z = CFG["YAW_NOTCH_R"], CFG["YAW_NOTCH_Z"]
WINDOW_WALL_Y = CFG["WINDOW_WALL_Y"]
REFIT_WALL_X = CFG["REFIT_WALL_X"]
COLLAR_BOSS_H = CFG["COLLAR_BOSS_H"]
HUB_R, DECK_R = CFG["HUB_R"], CFG["DECK_R"]
HUB = CFG["HUB"]
YAW_TOP = CFG["YAW_TOP"]
FILL_TOP = CFG["FILL_TOP"]
# the refit's bottom bar (Z 224.2-230) is cut by its own old R 1.7 driver channel and R 1.8 holes at the yaw axes: crescent fins 0.02-0.3 mm thick
# stay; a box per yaw axis takes them off in clean planes: -y to the old hole's centre plane (-8.25), +y to the refit's own inner face beside
# the hole (y 12.1519, x 55.8-57.55; a cut at 11.6 left a 0.55 mm web, one at the centre 12.25 a 0.1 mm step against that face)
RIB_LEN, RIB_Z, RIB_X = CFG["RIB_LEN"], CFG["RIB_Z"], CFG["RIB_X"]
FOOT_CUTS = CFG["FOOT_CUTS"]

EAR_CORNER, EAR_CHAMFER = CFG["EAR_CORNER"], CFG["EAR_CHAMFER"]


def ear_chamfer_tool():
    import numpy as np
    a = EAR_CHAMFER * 2 ** 0.5
    T = np.eye(4); c, s_ = np.cos(np.pi / 4), np.sin(np.pi / 4)
    T[:2, :2] = [[c, -s_], [s_, c]]; T[0, 3], T[1, 3] = EAR_CORNER
    return C.transform(C.box((-a / 2, -a / 2, 224.0), (a / 2, a / 2, 267.0)), T)


def window_stack(length, point, ref):
    """Clamped stack that gives exactly the reference engagement for a standard length (screw_plan.choose_length)."""
    return length - point - ref


def build_right():
    import interfaces as I
    import screw_plan as SP
    from checks import bodylib as F
    man = json.loads(SHELL_PASS.read_text())
    body = F.Body(man)
    iface = I.instances(body, man)
    st = SP.standards()
    pt = st["point_mm"]
    groups = SP.connectors()["hip_yaw_to_pitch_combined"]["groups"]      # which defined holes: connector_screws.json
    yaw = SP.hole_positions(iface, "R_yaw_upper_horn_reference", groups["A14_yaw"]["holes"])
    axes = [tuple(round(float(v), 2) for v in h["opening"][:2]) for h in yaw]   # the refit bore sits on the 0.01 grid
    window = [((x, y), WINDOW_WALL_Y[int(y > HUB[1])]) for x, y in axes if abs(y - HUB[1]) > 1.0]   # the two screws beside the key windows, with their wall
    outer_yaw = [(x, y) for x, y in axes if x < HUB[0] - 1.0]                                       # the -x screw: its collar overhangs the deck disc
    front = SP.hole_positions(iface, "R_hip_pitch_ST3025_case", groups["A14_front"]["holes"])
    by = {p["name"]: p for p in man["parts"]}
    P = MR.load_step(by["R_" + PART]["native_step"])
    S = MR.load_step(by["R_" + SLEEVES]["native_step"])
    out = {}

    # ---- yaw horn: M3 ISO 4762, length from the hole definition ----
    h0 = yaw[0]
    ref, depth = h0["reference_engagement_mm"], h0["depth_mm"]
    lens = st["head_types"]["ISO 4762"]["M3"]["lengths"]
    L = min(l for l in lens if window_stack(l, pt, ref) >= h0["opening"][2] - DECK_BOTTOM)   # seat at or below the deck: no deck cut
    stack = window_stack(L, pt, ref)
    seat = round(h0["opening"][2] - stack, 2)
    print(f"yaw: M3x{L:g}, ref engagement {ref}, hole depth {depth}, stack {stack:.2f}, head seat z {seat:.2f}", flush=True)
    add, cut, new_sleeves, collars = [], [], [], []
    hub_bore = C.cylinder((HUB[0], HUB[1], seat), (HUB[0], HUB[1], YAW_TOP + 1.0), HUB_R)   # from the seat: the pocket floor (refit 262.0) moves down to the collar layer's underside
    for h in yaw:
        x, y, z0 = (round(v, 2) for v in h["opening"])   # the refit bore sits on the 0.01 grid; 0.002 off it the fill leaves crescents and splits the wall
        if (x, y) in outer_yaw:   # the collar's -x side stands past the deck disc (R 10.55): a 1.0 mm lip there unless it rises above the deck bottom
            collars.append(C.cylinder((x, y, DECK_BOTTOM - 0.5), (x, y, DECK_BOTTOM + COLLAR_BOSS_H), COLLAR_R))
        collars.append(C.cylinder((x, y, seat), (x, y, DECK_BOTTOM), COLLAR_R))
        add.append(C.cylinder((x, y, DECK_BOTTOM - 0.5), (x, y, z0), R_SLEEVE_YAW[0]))   # refills the refit's own bore: the new bore is then cut once, so its wall is one face
        cut.append(C.cylinder((x, y, seat - 1.0), (x, y, z0 + 1.0), R_SLEEVE_YAW[0]))
        new_sleeves.append(C.cut(C.cylinder((x, y, seat), (x, y, z0), R_SLEEVE_YAW[0]),
                                 C.cylinder((x, y, seat - 0.5), (x, y, z0 + 0.5), R_SLEEVE_YAW[1])))

    # ---- front: M2 ISO 7380-1 into the defined case holes ----
    f0 = front[0]
    ref_f, depth_f = f0["reference_engagement_mm"], f0["depth_mm"]
    lens_f = st["head_types"]["ISO 7380-1"]["M2"]["lengths"]
    Lf = min(l for l in lens_f if window_stack(l, pt, ref_f) >= 2.2)   # keeps the 2.5 mm plate at least 1 mm behind the seat
    stack_f = window_stack(Lf, pt, ref_f) - 0.25          # 0.25 mm engagement margin (engagement = L - point - stack)
    x_face = f0["opening"][0] + 0.02                       # the case face reads 56.64-56.65: start 0.02 off it, no overlap
    seat_f = x_face + stack_f
    print(f"front: M2x{Lf:g}, ref engagement {ref_f}, depth {depth_f}, stack {stack_f:.2f}, seat x {seat_f:.2f}", flush=True)
    fpads, fcuts = [], []
    for h in front:
        _, y, z = h["opening"]
        outer = z + FRONT_PAD_R > FRONT_TOP - 0.1          # the two outer bosses reach the lattice top
        pr = FRONT_TOP - z if outer else FRONT_PAD_R       # crown exactly on the raised lattice top; the pocket takes the same radius (a 1 um shell of the pad would stay otherwise)
        fpads.append(C.cylinder((x_face, y, z), (seat_f + 0.5, y, z), pr))
        fcuts += [C.cylinder((x_face - 1.0, y, z), (x_face + 8, y, z), R_SLEEVE_FRONT[0]),
                  C.cylinder((seat_f, y, z), (seat_f + 6, y, z), pr)]
        if outer:   # head pocket open to the top: a pocket crown 0.00-0.06 mm off the lattice top would leave a skin or a 0.05 mm breakout
            fcuts.append(C.box((seat_f, y - pr, z), (seat_f + 6, y + pr, FRONT_TOP + 0.3)))
        new_sleeves.append(C.cut(C.cylinder((x_face, y, z), (seat_f, y, z), R_SLEEVE_FRONT[0]),
                                 C.cylinder((x_face - 0.5, y, z), (seat_f + 0.5, y, z), R_SLEEVE_FRONT[1])))

    # deck walls: the refit's hub pocket (R 4.2) and disc (R 10.0) leave 0.55 / 0.75 mm round the yaw bores; this disc makes both 1.30
    add.append(C.cut(C.cylinder((HUB[0], HUB[1], DECK_BOTTOM), (HUB[0], HUB[1], YAW_TOP), DECK_R), hub_bore))
    # one collar layer under the deck (seat .. DECK_BOTTOM): a disc fills the 0.1 mm slots between the diagonal collars
    # the collars at the window walls met them at 54 deg on the +x side: a tab from the collar's +x tangent to the wall makes it 90
    tabs = [C.box((x, min(y, w), seat), (x + COLLAR_R, max(y, w), DECK_BOTTOM)) for (x, y), w in window]
    add.append(C.cut(C.fuse(C.cylinder((HUB[0], HUB[1], seat), (HUB[0], HUB[1], DECK_BOTTOM), COLLAR_DISC_R), *collars, *tabs), hub_bore))
    P = C.cut_many(P, [C.box(lo, hi) for lo, hi in FOOT_CUTS])
    for h in front:
        _, y, z = h["opening"]
        if abs(y - HUB[1]) > 4.0:
            sg = 1.0 if y > HUB[1] else -1.0
            P = C.fuse(P, C.box((RIB_X[0], min(y, y + sg * RIB_LEN), RIB_Z[0]), (RIB_X[1], max(y, y + sg * RIB_LEN), RIB_Z[1])))
    P, v_top = C.pad(P, "z", LATTICE_TOP, FRONT_TOP, ((56.6, -9.6, 0.0), (59.4, 12.0, 0.0)))   # region z is replaced by the slab; solids are counted below
    print(f"lattice top raised to {FRONT_TOP}: {v_top:.2f} mm3, solids {C.n_solids(P)}", flush=True)
    P = C.fuse(P, *add, *fpads)
    # the window-side collars' -x side, where the circle crosses the wall corner: a 0.22 mm crescent stood past the deck's outline
    # (deck disc + the refit's R 3.19 corner round) and the deck overhung the collar between the circle and the wall face (0.04 mm2).
    # The collar layer there takes the deck's own outline: the deck section just above DECK_BOTTOM (vertical faces) moved down one layer.
    t = DECK_BOTTOM - seat
    for (x, y), w in window:
        s = 1.0 if w > HUB[1] else -1.0          # outward, into the wall
        y_in, y_out = w - s * 0.35, w + s * 1.9   # the crescent runs from the deck-circle cusp (0.12 inside the wall plane) to 0.86 into the wall
        lo, hi = (x - COLLAR_R - 0.1, min(y_in, y_out)), (REFIT_WALL_X + 0.1, max(y_in, y_out))
        deck = C.translate(C.common(P, C.box((*lo, DECK_BOTTOM), (*hi, DECK_BOTTOM + t))), (0.0, 0.0, -t))
        P = C.cut(P, C.cut(C.box((*lo, seat), (*hi, DECK_BOTTOM)), deck))                       # the crescent outside the deck outline
        y_f0, y_f1 = w + s * 0.55, y_out                                                         # collar circle outside the deck there: nothing to add
        P = C.fuse(P, C.common(deck, C.box((REFIT_WALL_X - 0.5, min(y_f0, y_f1), seat), (hi[0], max(y_f0, y_f1), DECK_BOTTOM))))   # under the deck's corner round
    # @lore no driver channel through the front bosses: the screw planner reads the stack on rays at r 1.9-2.7 around the sleeve, so a trimmed pad picks M2x5 with the head 1.3 mm inside the sleeve; and the sleeve end sits 0.09 mm off the yaw axis, so a channel never makes a straight path. A tilted ball-end key does.
    notch = [C.cylinder((x, y, YAW_NOTCH_Z[0]), (x, y, YAW_NOTCH_Z[1]), YAW_NOTCH_R) for (x, y), _ in window]
    # @lore thin ramp/tongues of the refit wall (0.0-0.65 mm, 51 deg tips): cutting them alone lowers FEM heavy (PAHT-CF 1.25 with the fill, 1.19 with the fill top at 250.55; they carry load in the model), so fill to a 1.6 mm step and cut above; the 0.4 mm rim round the front pockets is cut flush
    nz0, nz1 = YAW_NOTCH_Z
    y_lo, y_hi = HUB[1] - 3.6, HUB[1] + 3.6                  # the plate between the outer bosses, y -1.6 .. 5.6
    wy0, wy1 = WINDOW_WALL_Y
    fill = [C.box((57.2, wy0, FRONT_TOP - 0.1), (58.8, y_lo, FILL_TOP)), C.box((57.2, y_hi, FRONT_TOP - 0.1), (58.8, wy1, FILL_TOP)),
            C.box((58.6, y_lo, nz1 - 0.2), (61.4, y_hi, nz1)),   # the plate's old top rounds (x 58.8-61.2) left a 0.03 mm step 0.15 under the seat: square to the seat
            C.box((RIB_X[0], 11.5, 230.0), (57.2, wy1, RIB_Z[0]))]   # the refit's 0.40 x 0.35 mm slot under the +y rib (sub-line gap 9.5 -> 0.8 mm2, its 0.11 mm2 ceiling face)
    # @lore filling the +y rib slot reads FEM heavy PAHT-CF 1.24 instead of 1.25 only because the load patch took the slot's R 0.4 round for a case hole face (528 -> 477 mm2); the hot spot (58.8, -1.5, 251.6) is 10 mm away and its peak drops 353 -> 281 MPa (rv3-24r9hj-slotfill)
    P = C.fuse(P, *fill)
    trim = [C.box((56.6, wy0, FILL_TOP), (61.0, y_lo, nz0)), C.box((56.6, y_hi, FILL_TOP), (61.0, wy1, nz0)),   # tongues above the fill, below the key windows
            C.box((56.6, wy0, nz0), (61.0, y_lo, nz1)), C.box((56.6, y_hi, nz0), (61.0, wy1, nz1)),             # tips of the plate ends beside the notches, to the wall (a 0.19 mm column stood in the wall's corner round)
            C.box((seat_f, -10.5, 224.0), (59.6, 12.5, FRONT_TOP))]
    P = C.cut_many(P, cut + fcuts + notch + trim + [ear_chamfer_tool(), hub_bore])   # the hub bore from the seat: the refit's pocket floor (262.0) goes, no 0.25 mm step against the collar layer
    P = C.largest_solid(C.unify(P))   # one face per plane / cylinder: the cuts left coplanar faces split (0.002-0.03 mm2 pieces, 0.007-0.25 mm edges)
    S = MR.compound(new_sleeves)
    out["part"], out["sleeves"] = P, S
    out["plan"] = dict(yaw_length=L, yaw_stack=stack, yaw_seat_z=seat, front_length=Lf, front_stack=stack_f, front_seat_x=seat_f)
    return out


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parts").mkdir(exist_ok=True)
    MR.PARTS = OUT / "parts"
    r = build_right()
    recs = {}
    for side in ("R", "L"):
        for base, key, kind in ((PART, "part", "printed"), (SLEEVES, "sleeves", "purchased")):
            sh = r[key] if side == "R" else C.mirror_x(r[key])
            nm = f"{side}_{base}"
            rec = MR.save(nm, sh, kind)
            if kind == "printed":   # the refit writes 1.22 g/cm3; the printed material's density is in the checks' materials table (PAHT-CF 1.06)
                rho = json.loads((ROOT / "robot/body/checks/data/materials.json").read_text())["Bambu PAHT-CF"]["density_g_cm3"]
                rec["mass_kg"] = round(C.volume(sh) * rho * 1e-6, 6)
                rec["mass_basis"] = f"native STEP volume x {rho} g/cm3 (Bambu PAHT-CF, checks/data/materials.json)"
            if kind == "purchased":
                rec["mass_kg"] = round(C.volume(sh) * 7.85e-6, 6)
                rec["mass_basis"] = "native STEP volume x 7.85 g/cm3 (304 SS)"
            recs[nm] = rec
    (OUT / "records.json").write_text(json.dumps(dict(plan=r["plan"], records=recs), indent=1) + "\n")
    print("wrote", OUT / "records.json", flush=True)


def apply(variant: Path = VARIANT, out: Path | None = None):
    """Patch a roll-flip variant manifest in place (or into `out`): the hip yaw-to-pitch part + sleeves of both sides from
    records.json, the legacy A14 screws replaced by the ones generated from their defined holes."""
    sys.path.insert(0, str(ROOT / "robot/body/leg"))
    import roll_flip_checks as RF
    man = json.loads(variant.read_text())
    man["parts"] = patch_parts(man["parts"], man)
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, len(man["parts"]), "parts")


def swap_parts(parts):
    """The manifest part list with the yaw-to-pitch connector + sleeves of both sides replaced by the edited ones (offset 0)."""
    if not (OUT / "records.json").exists():   # derived output not built here: keep the manifest buildable, say so loudly
        print(f"WARNING hip yaw-to-pitch: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED connector stays in the "
              "manifest; run `.venv/bin/python -B robot/body/motor-refit/hip_yaw_screws.py build` first")
        return parts
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("placement_offset_mm", None)
            q["hip_yaw_screws"] = "edited by robot/body/motor-refit/hip_yaw_screws.py"
            res.append(q)
        else:
            res.append(p)
    return res


def patch_parts(parts, man):
    import roll_flip_checks as RF
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        n = p["name"]
        if n in recs:
            q = dict(p); q.update({k: v for k, v in recs[n].items()}); q.pop("placement_offset_mm", None)
            q["hip_yaw_screws"] = "edited by robot/body/motor-refit/hip_yaw_screws.py"
            res.append(q)
        elif n[2:].startswith(("A14_yaw_M3x14_", "A14_front_PA2x4_")):
            continue
        else:
            res.append(p)
    rows, _ = RF.hip_screws(res, man, only=("hip_yaw_to_pitch_combined",))
    return res + rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
