"""Foot monocoque (R/L_foot_monocoque_combined): the four ST3215 foot-roll case screws mountable, the sleeves pressable (Idle T-rptay5).

The part comes from the TopOpt chain (build_topopt.py, not touched here). Its four PA2x6 screws (two on the +y case face,
two on the -y face) were modelled with 0.55 mm engagement in the case's blind tapped holes, the sleeves 0.2 mm off the
case face, and the screw not a standard part. This script edits the foot STEP with exact B-rep operations (cad_ops) and
takes every hole position from the standard-part definition of the registered case
(robot/actuators/interfaces/standard-parts.json, ST3215.case+y.2/.4 and case-y.2/.4, through screw_plan.hole_positions):

  screws    4 x M2 machine screw (the ST3215 case is plastic: a tapped hole of a standard part takes a machine screw, not a
            thread-forming PA screw), ISO 7045 pan head (sourced). Modelled tap depth: 1.4 mm (+y) / 2.8 mm (-y); the
            true depth is unknown (T-2yg465). The 4.0 mm reference (2 x 2.0) cannot be reached by a blind hole this deep, so
            the screw is the longest standard length whose clamped stack does not exceed the plate's present floor
            (L - point - depth <= floor): the tip stops at the modelled hole bottom, engagement = modelled depth.
  sleeves   full thickness of that stack, from the case face (0.2 mm off the hole opening, 0.00 contact) to the head seat, head on the outer face.
  floor     the stack plane is the head seat: a pocket around each head removes the printed wall above the plane (driver path).
  ears      the part's back face under the sleeves was partly recessed from the case face: a printed ear (press_lib ANVIL_D annulus)
            reaches the case face (0.00 contact), the flat counter face the anvil rests on.
The boot shell (a body shell) is not edited.

  recreate:  .venv/bin/python -B robot/body/leg/foot_screws.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/leg/foot_screws.py apply   (patch the roll-flip variant manifest)
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "robot/body/motor-refit"))
sys.path.insert(0, str(ROOT / "robot/body/fastening"))
sys.path.insert(0, str(ROOT / "robot/body"))
import cad_ops as C  # noqa: E402
import motor_refit as MR  # noqa: E402

OUT = ROOT / "artifacts/robot/body/leg/foot-screws"
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
VARIANT = SHELL_PASS.with_name("assembly-parts-roll-flip.json")
CONNECTOR = "foot_monocoque_combined"
PART, SLEEVES = "foot_monocoque_combined", "foot_monocoque_combined_steel_sleeves"
CASE = "B13_C047_foot_roll_case"
OLD_SCREWS = ("B13_front_PA2x6_", "B13_rear_PA2x6_")
POCKET_R = 8.0 / 2 + 0.05      # pocket radius around each head: the heated press tip (checks/press_lib.py TIP_D / 2) + clearance, so the tip reaches the sleeve
EAR_R = 10.0 / 2 + 0.1          # printed ear from the case face to the part's back: the flat counter face the anvil annulus rests on
EAR_H = 0.5                     # ear height above the case face (the case face stands 0.2 mm off the hole opening); ends inside the part
POCKET_H = 16.0                 # pocket cut height above the head seat: the heated tip length (press_lib TIP_L 15) + 1


def bbox(s):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box(); BRepBndLib.Add_s(s, b)
    return np.array(b.Get(), float)


def build_right():
    import interfaces as I
    import screw_plan as SP
    from checks import bodylib as F
    man = json.loads(SHELL_PASS.read_text())      # the declared input: the unedited sources (build is repeatable after apply)
    iface = I.instances(F.Body(man), man)         # foot servo holes are identical in the roll-flip variant (written later in the chain)
    st = SP.standards()
    pt = st["point_mm"]
    by = {p["name"]: p for p in man["parts"]}

    def load(name):
        s = MR.load_step(by[name]["native_step"])
        return C.translate(s, by[name].get("native_step_offset_mm") or (0, 0, 0))
    P, S = load("R_" + PART), load("R_" + SLEEVES)
    old = C.solids(S)
    assert len(old) == 4, len(old)
    ob = [bbox(s) for s in old]
    od = float(np.mean([b[3] - b[0] for b in ob]))                    # sleeve OD of the pressed-in tube
    idia = json.loads((ROOT / "robot/actuators/interfaces/standard-parts.json").read_text())["sleeve"]["bore_for"]["M2"][0]
    lens = st["head_types"]["ISO 7045"]["M2"]["lengths"]
    sleeves, pockets, ears, plan = [], [], [], []
    for gname, g in SP.connectors()[CONNECTOR]["groups"].items():
        for h in SP.hole_positions(iface, g["standard_part"].replace("{S}", "R_"), g["holes"]):
            x, y0, z = h["opening"]; a = float(np.sign(h["axis_out"][1])); depth = h["depth_mm"]
            b = next(b for b in ob if abs((b[0] + b[3]) / 2 - x) < 0.5 and abs((b[2] + b[5]) / 2 - z) < 0.5 and (b[1] + b[4]) / 2 * a > 0)
            x, z = (b[0] + b[3]) / 2, (b[2] + b[5]) / 2              # the printed bore's own centre (the registered hole is within 0.003 mm of it)
            floor = a * (b[4] if a > 0 else b[1]) - a * y0          # present stack, from the hole opening
            y_face = b[1] if a > 0 else b[4]                          # where the old sleeve met the case: its face stands 0.2 mm off the opening
            L = max(l for l in lens if l - pt - depth <= floor + 1e-6)   # tip at the modelled hole bottom, stack not thicker than now
            stack = L - pt - depth
            y1 = y0 + a * stack
            sleeves.append(C.cut(C.cylinder((x, y_face, z), (x, y1, z), od / 2), C.cylinder((x, y_face - a * 0.5, z), (x, y1 + a * 0.5, z), idia / 2)))
            pockets.append(C.cylinder((x, y1, z), (x, y1 + a * POCKET_H, z), POCKET_R))
            ears.append(C.cut(C.cylinder((x, y_face, z), (x, y_face + a * EAR_H, z), EAR_R),       # a tube: the bore is its own, no cut through the fused part
                              C.cylinder((x, y_face - a * 0.5, z), (x, y_face + a * (EAR_H + 0.5), z), round(od / 2, 2))))   # = the printed bore radius
            plan.append(dict(hole=h["hole_id"], length_mm=L, stack_mm=round(stack, 3), sleeve_len_mm=round(a * (y1 - y_face), 3), floor_before_mm=round(floor, 3), depth_mm=depth,
                             head_seat_y=round(y1, 3)))
            print(plan[-1], flush=True)
    v0 = C.volume(P)
    keep_out = [load("R_" + n) for n in (CASE, "B13_C047_foot_roll_rear_idler")]   # the ears stop at the servo (standing contact, no overlap)
    ears = [C.cut(e, *keep_out) for e in ears]
    P = C.largest_solid(C.cut_many(C.fuse(P, *ears), pockets))
    assert C.valid(P), "edited foot is not a valid solid"
    print(f"printed volume {v0:.0f} -> {C.volume(P):.0f} mm3, solids {C.n_solids(P)}", flush=True)
    return dict(part=P, sleeves=MR.compound(sleeves), plan=dict(holes=plan, pocket_r_mm=POCKET_R, ear_r_mm=EAR_R, sleeve_od_mm=od, sleeve_id_mm=idia,
                                                              volume_before_mm3=v0, volume_after_mm3=C.volume(P)))


def build():
    OUT.mkdir(parents=True, exist_ok=True)
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


def swap_parts(parts):
    """The manifest part list with the foot monocoque + sleeves of both sides replaced by the edited ones (offset 0)."""
    if not (OUT / "records.json").exists():   # derived output not built here: keep the manifest buildable, say so loudly
        print(f"WARNING foot: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED foot monocoque stays in the "
              "manifest; run `.venv/bin/python -B robot/body/leg/foot_screws.py build` (T-rptay5) first")
        return parts
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("native_step_offset_mm", None); q.pop("placement_offset_mm", None)
            q["foot_screws"] = "edited by robot/body/leg/foot_screws.py (T-rptay5)"
            res.append(q)
        else:
            res.append(p)
    return res


@contextlib.contextmanager
def left_labels():
    """screw_plan.hole_positions reading `holes_L` of this connector's groups on the left side: the left case instance numbers
    its holes mirrored in z, so the same physical holes carry other labels (connector_screws.json)."""
    import screw_plan as SP
    full, spec = SP.hole_positions, SP.connectors()[CONNECTOR]["groups"]
    side = {tuple(g["holes"]): g["holes_L"] for g in spec.values()}
    SP.hole_positions = lambda iface, sp, labels: full(iface, sp, side.get(tuple(labels), labels) if sp.startswith("L_") else labels)
    try:
        yield
    finally:
        SP.hole_positions = full


@contextlib.contextmanager
def only_connector():
    """screw_plan.connectors() restricted to this connector (roll_flip_checks.hip_screws walks all of them)."""
    import screw_plan as SP
    full = SP.connectors
    SP.connectors = lambda: {CONNECTOR: full()[CONNECTOR]}
    try:
        yield
    finally:
        SP.connectors = full


def patch_parts(parts, man):
    import roll_flip_checks as RF
    res = [p for p in swap_parts(parts) if not (p["name"][2:].startswith(OLD_SCREWS) and p["role"] == "fastener_proxy")]
    with only_connector(), left_labels():
        rows, _ = RF.hip_screws(res, man)
    return res + rows


def apply(variant: Path = VARIANT, out: Path | None = None):
    """Patch a roll-flip variant manifest in place (or into `out`)."""
    sys.path.insert(0, str(ROOT / "robot/body/leg"))
    man = json.loads(variant.read_text())
    man["parts"] = patch_parts(man["parts"], man)
    (out or variant).write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out or variant, len(man["parts"]), "parts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["build", "apply"])
    ap.add_argument("--variant", default=str(VARIANT))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build() if a.phase == "build" else apply(Path(a.variant), Path(a.out) if a.out else None)
