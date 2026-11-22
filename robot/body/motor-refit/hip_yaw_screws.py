"""Hip yaw-to-pitch connector (R/L_hip_yaw_to_pitch_combined): mountable A14 screws and pressable sleeves.

The part comes from the motor refit (motor_refit.py, truss-box stage 6 edited for the ST3025 hip pitch case). Its screws
were drawn for the old C047 case and the horn reference of r1: 4 M3x14 reached 1.5 of the horn flange's 2.5 mm, the head
seat of the sleeves sat 12.0 mm from the horn, and the
4 front PA2x4 stood at C047 hole positions that the ST3025 case does not have (shank in solid print). This script edits the
refit STEP with the same B-rep operations (cad_ops) and takes every hole position from the standard-part definitions
(robot/actuators/interfaces/standard-parts.json, registered instances) through screw_plan.hole_positions:

  yaw horn   4 x ST3215.horn+y.n (aluminium, tapped through the 2.5 mm flange, reference engagement 2.5 mm): the head seat
             sits at opening - (L - point - reference engagement) of the standard length L (screw_plan.choose_length), the
             sleeve runs the full printed thickness down to the seat (a 1 mm collar under the deck carries it). Driving: the
             sleeve of the front screws z.1 / z.2 (y -6 / 10, Z 247) ends 0.09 mm short of the yaw axes (59,-5) / (59,9), whose hole
             positions are fixed, so no straight tool passes (obstruction 62 mm3 part + 4.5 mm3 sleeve at r 1.45). The key is tilted
             12-25 deg toward the free +x side (ISO 2936 ball-end hex key, 2.5 mm AF): nothing of the part or the sleeves is in that cone.
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

OUT = ROOT / "artifacts/robot/body/leg/hip-yaw-screws"
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
VARIANT = SHELL_PASS.with_name("assembly-parts-roll-flip.json")
PART, SLEEVES = "hip_yaw_to_pitch_combined", "hip_yaw_to_pitch_combined_steel_sleeves"

R_SLEEVE_YAW = (4.5 / 2, 3.2 / 2)        # OD / ID of the 304 tube for M3 (standard-parts sleeve.bore_for)
R_SLEEVE_FRONT = (3.6 / 2, 2.2 / 2)      # M2
COLLAR_R = 4.9                           # printed collar under the deck: carries the lower end of a yaw sleeve, flat counter face to R 5 for the press
DECK_BOTTOM = 262.75
FRONT_PAD_R = FRONT_POCKET_R = 3.05       # pad wall round the sleeve: 3.05 - 1.8 = 1.25 mm >= 2 lines (0.62)
HUB_R, DECK_R = 3.45, 10.55              # deck: hub pocket (refit 4.2) and outer disc (refit 10.0) leave 1.30 mm walls (>= 2 lines) round the yaw bores, which sit 7.0 from the hub axis
HUB = (59.0, 2.0)
HUB_Z0 = 262.0
HUB_TOOL = C.cylinder((59.0, 2.0, HUB_Z0), (59.0, 2.0, 264.0), HUB_R)
YAW_TOP = 274.75
# the refit's bottom bar (Z 224.2-230) is cut by its own old R 1.7 driver channel and R 1.8 holes at the yaw axes: crescent fins 0.02-0.3 mm thick
# stay; a box per yaw axis takes them off in clean planes (y ranges keep 0.45 mm off the refit's own y steps at -7.45 / 11.85, so no new micro-edges)
RIB_LEN, RIB_Z, RIB_X = 7.0, (244.0, 250.0), (56.8, 58.8)                # flat tab beside each outer front boss (z.1 / z.2), away from the hub axis: more flat counter face for the press anvil (dia 10) where the slab ends
FOOT_CUTS = (((56.0, -7.0, 223.0), (60.5, -3.0, 230.0)), ((56.0, 7.0, 223.0), (60.5, 11.6, 230.0)))


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
    add, cut, new_sleeves = [], [], []
    for h in yaw:
        x, y, z0 = (round(v, 2) for v in h["opening"])   # the refit bore sits on the 0.01 grid; 0.002 off it the fill leaves crescents and splits the wall
        add.append(C.cut(C.cylinder((x, y, seat), (x, y, DECK_BOTTOM), COLLAR_R), HUB_TOOL))   # not into the central pocket: its floor (262.0) and wall stay as the refit has them
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
        fpads.append(C.cylinder((x_face, y, z), (seat_f + 0.5, y, z), FRONT_PAD_R))
        fcuts += [C.cylinder((x_face - 1.0, y, z), (x_face + 8, y, z), R_SLEEVE_FRONT[0]),
                  C.cylinder((seat_f, y, z), (seat_f + 6, y, z), FRONT_POCKET_R)]
        new_sleeves.append(C.cut(C.cylinder((x_face, y, z), (seat_f, y, z), R_SLEEVE_FRONT[0]),
                                 C.cylinder((x_face - 0.5, y, z), (seat_f + 0.5, y, z), R_SLEEVE_FRONT[1])))

    # deck walls: the refit's hub pocket (R 4.2) and disc (R 10.0) leave 0.55 / 0.75 mm round the yaw bores; this disc makes both 1.30
    add.append(C.cut(C.cylinder((HUB[0], HUB[1], DECK_BOTTOM), (HUB[0], HUB[1], YAW_TOP), DECK_R), C.cylinder((HUB[0], HUB[1], DECK_BOTTOM - 1.0), (HUB[0], HUB[1], YAW_TOP + 1.0), HUB_R)))
    P = C.cut_many(P, [C.box(lo, hi) for lo, hi in FOOT_CUTS])
    for h in front:
        _, y, z = h["opening"]
        if abs(y - HUB[1]) > 4.0 and RIB_LEN > 0:
            sg = 1.0 if y > HUB[1] else -1.0
            P = C.fuse(P, C.box((RIB_X[0], min(y, y + sg * RIB_LEN), RIB_Z[0]), (RIB_X[1], max(y, y + sg * RIB_LEN), RIB_Z[1])))
    P = C.fuse(P, *add, *fpads)
    # @lore no driver channel through the front bosses: the screw planner reads the stack on rays at r 1.9-2.7 around the sleeve, so a trimmed pad picks M2x5 with the head 1.3 mm inside the sleeve; and the sleeve end sits 0.09 mm off the yaw axis, so a channel never makes a straight path. A tilted ball-end key does.
    P = C.cut_many(P, cut + fcuts)
    P = C.largest_solid(P)
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
