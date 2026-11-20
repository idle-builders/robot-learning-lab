"""Ankle coupler (R/L_ankle_coupler_combined): the four foot-roll horn screws mountable, the sleeves pressable (Idle T-9wpegg).

The part comes from the motor refit (motor_refit.py: r1 coupler, roll-horn plate padded 3 mm at the back). Its four
NBK M3x4 screws were modelled with the head underside 0.8 mm inside the steel sleeve (head top on the sleeve end), reached
1.5 of the horn flange's 2.5 mm reference engagement, and three of the four heads sat inside the 3 mm pad; the horn
centre pocket (r 4.5) left 0.25 mm of wall to the sleeves on the 7 mm bolt circle. This script edits the refit STEP
with the same exact B-rep operations (cad_ops) and takes every hole position from the standard-part definition of the
registered horn (robot/actuators/interfaces/standard-parts.json, ST3215.horn+y.1-4, through screw_plan.hole_positions):

  screws    4 x ISO 7380-1 M3 (the NBK slim head is not a standard part; ISO 7380-1 starts at M3) into the horn flange
            (tapped through its 2.5 mm: reference engagement 2.5, hole depth 2.5). The standard length is the smallest one
            whose clamped stack (= L - point - reference engagement) is at least the plate's present floor under the
            sleeves (screw_plan.choose_length window), so the plate under the heads is not thinned: M3x6, stack 3.0.
  sleeves   full thickness of that stack (3.0, flush with the printed floor around the head), head on the outer face;
            the sleeve ends on the horn face (0.00 contact).
  floor     the stack plane is the head seat: a pocket of the heated tip's radius (press_lib TIP_D / 2) around each
            head removes the 3 mm back pad down to the floor (hex-key access and press-tool path in one cut); the hole
            on the thin side of the plate (no pad) gets a boss up to the plane.
  walls     the horn centre pocket is closed from r 4.5 to CENTRE_R between the floor and the horn face, so the printed
            wall between pocket and sleeve is 7 - CENTRE_R - 2.25 (two line widths = 0.84 at the 0.4 nozzle).
The eight ankle-pitch (ST3025 wheel) sleeves of the part and their cavity are not touched.

  recreate:  .venv/bin/python -B robot/body/motor-refit/ankle_coupler_screws.py build   (parts + records.json in OUT)
             .venv/bin/python -B robot/body/motor-refit/ankle_coupler_screws.py apply   (patch the roll-flip variant manifest)
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "robot/body/fastening"))
sys.path.insert(0, str(ROOT / "robot/body"))
import cad_ops as C  # noqa: E402
import motor_refit as MR  # noqa: E402

OUT = ROOT / "artifacts/robot/body/leg/ankle-coupler-screws"
SHELL_PASS = ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
VARIANT = SHELL_PASS.with_name("assembly-parts-roll-flip.json")
CONNECTOR = "ankle_coupler_combined"
PART, SLEEVES = "ankle_coupler_combined", "ankle_coupler_combined_steel_sleeves"
HORN = "B13_C047_foot_roll_main_horn"

R_SLEEVE = (4.5 / 2, 3.2 / 2)       # OD / ID of the 304 tube for M3 (standard-parts sleeve.bore_for)
EAR_LIP = 0.5                       # ear top above the floor plane (no coplanar fuse: the floor is made by the pocket cut alone)
FILLET_R = 0.6                      # concave edges where a head pocket wall meets its floor (stress concentration, poor to print)
TIP_R = 8.0 / 2 + 0.05              # heated press tip (checks/press_lib.py TIP_D) + clearance: pocket radius around each head
EAR_R = 10.0 / 2 + 0.1              # printed ear around each sleeve from the horn face to the floor: the flat counter face the anvil
                                    # annulus (press_lib ANVIL_D) rests on, where the 19.8 mm horn disc outline ends before it
CENTRE_R = 3.2                      # horn centre pocket after closing (was 4.5; centre of the bolt circle 7 mm from each hole)
CENTRE_OLD_R = 4.5
POCKET_TOP = 3.0                    # pocket cut runs this far above the pad's outer face
BACK_PAD_TOP = 23.8                 # outer face of the motor-refit pad (motor_refit.py: pad 20.8 -> 23.8)


def bbox(s):
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box(); BRepBndLib.Add_s(s, b)
    return np.array(b.Get(), float)


def window_stack(length, point, ref):
    """Clamped stack that gives exactly the reference engagement for a standard length (screw_plan.choose_length)."""
    return length - point - ref


def rounded_pocket(x, y0, z, y1, r, fr):
    """Pocket cutter: cylinder r from y0 to y1 with its bottom edge rounded by fr, so the cut leaves a filleted floor edge."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    cyl = C.cylinder((x, y0, z), (x, y1, z), r)
    ex = TopExp_Explorer(cyl, TopAbs_EDGE)
    mk = BRepFilletAPI_MakeFillet(cyl)
    while ex.More():
        e = TopoDS.Edge_s(ex.Current())
        if abs(BRepAdaptor_Curve(e).Value(0.0).Y() - y0) < 1e-6 and abs(BRepAdaptor_Curve(e).Value(0.5 * (BRepAdaptor_Curve(e).FirstParameter() + BRepAdaptor_Curve(e).LastParameter())).Y() - y0) < 1e-6:
            mk.Add(fr, e)
        ex.Next()
    mk.Build()
    assert mk.IsDone()
    return mk.Shape()


def fillet_ear_junctions(P, holes, y_face, y_seat, centre):
    """Fillet the vertical concave edges where an ear's wall meets the horn-disc outline (peak stress sat there)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
    from OCP.GeomAbs import GeomAbs_Line
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    def pick(S):
        out = []
        ex = TopExp_Explorer(S, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge_s(ex.Current()); c = BRepAdaptor_Curve(e)
            if c.GetType() == GeomAbs_Line:
                a, b = c.Value(c.FirstParameter()), c.Value(c.LastParameter())
                if abs(a.X() - b.X()) < 1e-4 and abs(a.Z() - b.Z()) < 1e-4 and abs(a.Y() - b.Y()) > 1.0 \
                        and y_face - 0.01 <= min(a.Y(), b.Y()) and max(a.Y(), b.Y()) <= y_seat + EAR_LIP + 0.01:
                    d = min(((a.X() - h["opening"][0]) ** 2 + (a.Z() - h["opening"][2]) ** 2) ** 0.5 for h in holes)
                    if abs(d - EAR_R) < 0.05:
                        out.append(e)
            ex.Next()
        return out
    done, skip = 0, set()
    for _ in range(60):          # one edge at a time on the current shape; edges the B-rep refuses are skipped
        cand = [e for e in pick(P) if (round(BRepAdaptor_Curve(e).Value(0).X(), 2), round(BRepAdaptor_Curve(e).Value(0).Z(), 2)) not in skip]
        if not cand:
            break
        e = cand[0]; key = (round(BRepAdaptor_Curve(e).Value(0).X(), 2), round(BRepAdaptor_Curve(e).Value(0).Z(), 2))
        try:
            mk = BRepFilletAPI_MakeFillet(P); mk.Add(FILLET_R, e); mk.Build()
            if mk.IsDone() and C.valid(mk.Shape()) and C.n_solids(mk.Shape()) == 1:
                P = mk.Shape(); done += 1; continue
        except Exception:
            pass
        skip.add(key)
    print(f"filleted {done} ear junction edges at r {FILLET_R}, skipped {len(skip)}", flush=True)
    return P


def build_right():
    import interfaces as I
    import screw_plan as SP
    from checks import bodylib as F
    man = json.loads(VARIANT.read_text())
    body = F.Body(man)
    iface = I.instances(body, man)
    st = SP.standards()
    pt = st["point_mm"]
    g = SP.connectors()[CONNECTOR]["groups"]["B13_horn"]
    holes = SP.hole_positions(iface, "R_" + HORN, g["holes"])
    by = {p["name"]: p for p in man["parts"]}

    def load(name):                       # body frame: the manifest's native offset applied once, saved with offset 0
        s = MR.load_step(by[name]["native_step"])
        return C.translate(s, by[name].get("native_step_offset_mm") or (0, 0, 0))
    P, S = load("R_" + PART), load("R_" + SLEEVES)

    h0 = holes[0]
    ref, depth = h0["reference_engagement_mm"], h0["depth_mm"]
    y_face = h0["opening"][1]
    old = []                              # the four sleeves on the horn: axis along y (axis_out is +y)
    keep = []
    for sol in C.solids(S):
        b = bbox(sol)
        c = ((b[0] + b[3]) / 2, (b[2] + b[5]) / 2)
        hit = any(abs(c[0] - h["opening"][0]) < 0.5 and abs(c[1] - h["opening"][2]) < 0.5 for h in holes)
        (old if hit else keep).append(sol)
    assert len(old) == 4 and len(keep) == len(C.solids(S)) - 4, (len(old), len(keep))
    floor_now = max(bbox(s)[4] for s in old) - y_face                       # present plate floor under the sleeves (2.8)
    lens = st["head_types"]["ISO 7380-1"]["M3"]["lengths"]
    L = min(l for l in lens if window_stack(l, pt, ref) >= floor_now - 1e-6)
    stack = window_stack(L, pt, ref)                                       # tip margin = depth - ref
    assert stack + depth - (L - pt) >= -1e-6, "tip passes the hole bottom"
    y_seat = y_face + stack
    print(f"M3x{L:g}: ref engagement {ref}, hole depth {depth}, floor now {floor_now:.2f} -> stack {stack:.2f}, "
          f"head seat y {y_seat:.2f}", flush=True)

    cx = float(np.mean([h["opening"][0] for h in holes])); cz = float(np.mean([h["opening"][2] for h in holes]))
    # the floor plane y_seat is made by the pocket cut alone (no coplanar fuse): ring and bosses stand proud of it first
    ring = C.cut(C.cylinder((cx, y_face, cz), (cx, BACK_PAD_TOP, cz), CENTRE_OLD_R + 0.3),
                 C.cylinder((cx, y_face - 1, cz), (cx, BACK_PAD_TOP + 1, cz), CENTRE_R))
    add, pockets, bores, new_sleeves = [ring], [C.cylinder((cx, y_face - 1, cz), (cx, BACK_PAD_TOP + 1, cz), CENTRE_R)], [], []
    for h in holes:
        x, _, z = h["opening"]
        add.append(C.cylinder((x, y_face, z), (x, y_seat + EAR_LIP, z), EAR_R))      # the lip above the floor is interior where the pad stands
        pockets.append(rounded_pocket(x, y_seat, z, BACK_PAD_TOP + POCKET_TOP, TIP_R, FILLET_R))
        bores.append(C.cylinder((x, y_face - 0.5, z), (x, y_seat + 0.5, z), R_SLEEVE[0]))
        new_sleeves.append(C.cut(C.cylinder((x, y_face, z), (x, y_seat, z), R_SLEEVE[0]),
                                 C.cylinder((x, y_face - 0.5, z), (x, y_seat + 0.5, z), R_SLEEVE[1])))
    v0 = C.volume(P)
    # old bores out of the way first: the boss and ring fuse into solid, then the pockets and the new bores are cut
    P = C.fuse(P, *add)
    P = C.cut_many(P, pockets + bores)
    P = C.largest_solid(P)
    P = fillet_ear_junctions(P, holes, y_face, y_seat, (cx, cz))
    assert C.valid(P), "edited coupler is not a valid solid"
    print(f"printed volume {v0:.0f} -> {C.volume(P):.0f} mm3, solids {C.n_solids(P)}", flush=True)
    return dict(part=P, sleeves=MR.compound(keep + new_sleeves),
                plan=dict(length_mm=L, stack_mm=stack, head_seat_y=y_seat, plate_floor_before_mm=floor_now,
                          centre_pocket_r_mm=CENTRE_R, pocket_r_mm=TIP_R, volume_before_mm3=v0, volume_after_mm3=C.volume(P)))


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


def swap_parts(parts):
    """The manifest part list with the ankle coupler + sleeves of both sides replaced by the edited ones (offset 0)."""
    if not (OUT / "records.json").exists():   # derived output not built here: keep the manifest buildable, say so loudly
        print(f"WARNING ankle coupler: {(OUT / 'records.json').relative_to(ROOT)} missing, the UNEDITED coupler stays in the "
              "manifest; run `.venv/bin/python -B robot/body/motor-refit/ankle_coupler_screws.py build` (T-9wpegg) first")
        return parts
    recs = json.loads((OUT / "records.json").read_text())["records"]
    res = []
    for p in parts:
        if p["name"] in recs:
            q = dict(p); q.update(recs[p["name"]]); q.pop("native_step_offset_mm", None); q.pop("placement_offset_mm", None)
            q["ankle_coupler_screws"] = "edited by robot/body/motor-refit/ankle_coupler_screws.py (T-9wpegg)"
            res.append(q)
        else:
            res.append(p)
    return res


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
    sys.path.insert(0, str(ROOT / "robot/body/leg"))
    import roll_flip_checks as RF
    res = [p for p in swap_parts(parts) if not (p["name"][2:].startswith("B13_output_NBK_M3x4_H0p8_") and p["role"] == "fastener_proxy")]
    with only_connector():
        rows, _ = RF.hip_screws(res, man)
    return res + rows


def apply(variant: Path = VARIANT, out: Path | None = None):
    """Patch a roll-flip variant manifest in place (or into `out`)."""
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
