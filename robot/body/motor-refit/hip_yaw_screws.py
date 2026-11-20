"""Hip yaw-to-pitch connector (R/L_hip_yaw_to_pitch_combined): mountable A14 screws and pressable sleeves (Idle T-v3kzs4).

The part comes from the motor refit (motor_refit.py, truss-box stage 6 edited for the ST3025 hip pitch case). Its screws
were drawn for the old C047 case and the horn reference of r1: 4 M3x14 reached 1.5 of the horn flange's 2.5 mm, the head
seat of the sleeves sat 12.0 mm from the horn, the driver channel (dia 3.4) was narrower than the hex-key blade, and the
4 front PA2x4 stood at C047 hole positions that the ST3025 case does not have (shank in solid print). This script edits the
refit STEP with the same B-rep operations (cad_ops) and takes every hole position from the standard-part definitions
(robot/actuators/interfaces/standard-parts.json, registered instances) through screw_plan.hole_positions:

  yaw horn   4 x ST3215.horn+y.n (aluminium, tapped through the 2.5 mm flange, reference engagement 2.5 mm): the head seat
             sits at opening - (L - point - reference engagement) of the standard length L (screw_plan.choose_length), the
             sleeve runs the full printed thickness down to the seat (a 1 mm collar under the deck carries it), the head
             cavity ends at the seat, the driver channel is as wide as the driver blade plus clearance.
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
COLLAR_R = 5.0                           # printed collar under the deck: carries the lower end of a yaw sleeve, flat counter face to R 5 for the press
CBORE_R = 2.85                           # head cavity (M3 cap head dk 5.5)
CBORE_Z0 = 251.0                         # head cavity floor (the refit had 251.5)
DRIVER_R = 2.4                           # driver channel: hex-key blade 4.0 (data/tools.json) + 0.8 clearance; 2.2 grazed the wall face at x 56.79
CHANNEL_Z0 = 215.0
DECK_BOTTOM = 262.75
FRONT_PAD_R, FRONT_POCKET_R = 3.0, 3.0
BLADE_R = 2.0                            # driver blade 4.0 (data/tools.json): the front bosses of z.1 / z.2 are trimmed out of the yaw screws' blade path (measured 17.5 -> 0.001 mm3 per screw)
POCKET = C.cylinder((59.0, 2.0, 262.0), (59.0, 2.0, 264.0), 4.2)   # the refit's central pocket under the deck (Z 262.0-274.75, R 4.2)


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
    chan, add, cut, new_sleeves = [], [], [], []
    for h in yaw:
        x, y, z0 = (round(v, 2) for v in h["opening"])   # the refit bore sits on the 0.01 grid; 0.002 off it the fill leaves crescents and splits the wall
        add.append(C.cut(C.cylinder((x, y, seat), (x, y, DECK_BOTTOM), COLLAR_R), POCKET))   # not into the central pocket: its floor (262.0) and wall stay as the refit has them
        add.append(C.cylinder((x, y, DECK_BOTTOM - 0.5), (x, y, z0), R_SLEEVE_YAW[0]))   # refills the refit's own bore: the new bore is then cut once, so its wall is one face
        chan += [C.cylinder((x, y, CBORE_Z0), (x, y, seat), CBORE_R), C.cylinder((x, y, CHANNEL_Z0), (x, y, 252.5), DRIVER_R)]
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

    P = C.cut_many(P, chan)                       # driver channels first: a front pad inside one stays (it carries its sleeve)
    P = C.fuse(P, *add, *fpads)
    P = C.cut_many(P, cut + fcuts)
    def reaches(h):   # the blade disc overlaps a front pad's x-y footprint (x_face..seat_f, y +- pad radius)
        x, y, _ = h["opening"]
        dx = max(0.0, x_face - x, x - seat_f)
        return min(max(0.0, abs(y - f["opening"][1]) - FRONT_PAD_R) for f in front) ** 2 + dx ** 2 < BLADE_R ** 2
    keep = [h["opening"] for h in yaw if reaches(h)]
    P = C.cut_many(P, [C.cylinder((round(x, 2), round(y, 2), seat - 40.0), (round(x, 2), round(y, 2), seat), BLADE_R) for x, y, _ in keep])
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
              "manifest; run `.venv/bin/python -B robot/body/motor-refit/hip_yaw_screws.py build` (T-v3kzs4) first")
        return parts
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("placement_offset_mm", None)
            q["hip_yaw_screws"] = "edited by robot/body/motor-refit/hip_yaw_screws.py (T-v3kzs4)"
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
            q["hip_yaw_screws"] = "edited by robot/body/motor-refit/hip_yaw_screws.py (T-v3kzs4)"
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
