"""S3 (P-04 exploration, D-08): every way to arrange the roll and pitch servos so that all three hip axes meet at C,
scored on the size of the hip globe they need and on contacts over the whole-leg poses (Idle T-czv2d8).

Rules from A-18: one servo body sits on C (its shaft on C, body centred on C along its own axis); the other stands
off along its own axis, output face pointing away from C (horn into a U-bracket arm), near face 1 mm clear of the
central body. A servo body can point any way at right angles to its shaft except up (the yaw neck and U-bracket
bridge come down onto the block from the torso). The yaw servo stays in the torso.

Scores per arrangement (no pass/fail, A-17 counting: servos, horns, battery keel/cells, cradle plate, other leg):
  globe_r     radius of the smallest sphere about C holding both hip servos and horns
  lower_r     same, counting only what lies below C (the part of the globe that shows under the pelvis)
  outboard_x  hip outboard extent; inboard_x the smallest x of the moving servos over the poses
  contacts    library poses (A-11 end poses + paths) and range combinations (A-12) with servo contact > 20 mm3

Run: python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/sketch2d/block_sweep.py
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import hip_layouts as H  # noqa: E402

C = H.C
DIRS = {"down": (0, 0, -1), "front": (0, -1, 0), "back": (0, 1, 0), "inboard": (-1, 0, 0), "outboard": (1, 0, 0)}
SERVO = {"pitch": H.ST3025, "roll": H.STS3250}
AXIS = {"pitch": np.array([1.0, 0, 0]), "roll": np.array([0, 1.0, 0])}


def central(joint, body_dir, horn_sign):
    """Central servo: shaft on C along its axis, body centred on C along the axis, horn on the face at horn_sign."""
    sv = SERVO[joint]; s = AXIS[joint] * horn_sign
    face = C + s * sv["H"] / 2
    return H.brick(sv, face, s, DIRS[body_dir])


def standoff(joint, side, body_dir, near):
    """Stand-off servo on its axis at `side` (+1/-1), near face `near` mm from C, output facing away from C."""
    sv = SERVO[joint]; s = AXIS[joint] * side
    face = C + s * (near + sv["H"])
    return H.brick(sv, face, s, DIRS[body_dir])


def extent(meshes, axis, side):
    v = np.vstack([m.vertices for m in meshes])
    return float(((v - C) @ (axis * side)).max())


def arrangements():
    out = []
    for cj, sj in (("pitch", "roll"), ("roll", "pitch")):
        for cdir in DIRS:
            if np.allclose(np.abs(DIRS[cdir]), np.abs(AXIS[cj])):
                continue
            for hs in (1, -1):
                cm = central(cj, cdir, hs)
                for side in (1, -1):
                    near = extent(cm[:1], AXIS[sj], side) + 1.0
                    for sdir in DIRS:
                        if np.allclose(np.abs(DIRS[sdir]), np.abs(AXIS[sj])):
                            continue
                        sm = standoff(sj, side, sdir, near)
                        name = (f"{cj} central (body {cdir}, horn {'+' if hs > 0 else '-'}{'x' if cj == 'pitch' else 'y'}); "
                                f"{sj} {'behind' if (sj == 'roll' and side > 0) else 'in front' if sj == 'roll' else 'outboard' if side > 0 else 'inboard'}"
                                f" (body {sdir})")
                        out.append(dict(name=name, central=cj, cdir=cdir, horn=hs, standoff=sj, side=side, sdir=sdir,
                                        near=near, meshes={cj: cm, sj: sm}))
    return out


def layout(a):
    """Order yaw -> roll -> pitch (P-04): roll servo body on J2 if it is the stand-off (it turns with yaw, roll is its
    own output... ) -- see note. Link assignment for a concurrent yaw->roll->pitch chain:
      J1 (yaw link): U-bracket, roll horn when roll is the stand-off servo
      J2 (roll link): the roll-pitch block (both servo bodies)
      J3 (thigh): pitch horn, fork.
    When roll is central it is still the roll joint: its body is on J1 (fixed to the U) and its horn drives J2? No: we
    keep one rule for both: the servo bodies form the J2 block, the roll horn is on J1, the pitch horn on J3."""
    parts = [("yaw servo", H.bx((24, -10, 276), (69, 14, 313)), "torso")]
    for j in ("roll", "pitch"):
        body, horn = a["meshes"][j]
        parts.append((f"{j} servo", body, "J2"))
        parts.append((f"{j} horn", horn, "J1" if j == "roll" else "J3"))
    parts += [("knee servo", H.bx((32, -6, 113), (76, 14, 153)), "thigh"),
              ("shin", H.bx((36, -12, 60), (70, 18, 112)), "shin")]
    return H.Layout(a["name"], ["yaw", "roll", "pitch"],
                    {"yaw": ((59, 2, 0), "z"), "roll": ((59, 0, 219), "y"), "pitch": ((0, 2, 219), "x")},
                    parts, hip_servos=("roll servo", "pitch servo", "roll horn", "pitch horn"))


def score(a):
    lay = layout(a)
    hs = [m for n, m, l in lay.parts if n in lay.hip_servos]
    v = np.vstack([m.vertices for m in hs]) - C
    r = np.linalg.norm(v, axis=1)
    res = H.evaluate(lay)
    return dict(name=a["name"], central=a["central"], cdir=a["cdir"], horn=a["horn"], standoff=a["standoff"],
                side=a["side"], sdir=a["sdir"], near_mm=round(a["near"], 1),
                globe_r=round(float(r.max()), 1), lower_r=round(float(r[v[:, 2] < 0].max()), 1),
                outboard_x=round(float((v[:, 0] + C[0]).max()), 1), inboard_x_standing=round(float((v[:, 0] + C[0]).min()), 1),
                inboard_x_poses=res["closest_to_centreline_x_mm"],
                lib_contact=res["library_poses_with_contact"], lib_n=res["library_poses"],
                grid_contact=res["combinations_with_contact"],
                worst=[(w["family"], w["pose"], w["contact_mm3"], w["pairs"]) for w in res["library_worst"][:3]])


def main():
    arr = arrangements()
    # quick geometric screen: only arrangements whose globe is within 12 mm of the best get the pose check
    rs = []
    for a in arr:
        v = np.vstack([m.vertices for m in a["meshes"]["roll"] + a["meshes"]["pitch"]]) - C
        rs.append(float(np.linalg.norm(v, axis=1).max()))
    best = min(rs)
    print(f"{len(arr)} arrangements; smallest servo globe r {best:.1f}")
    rows = []
    for a, r in sorted(zip(arr, rs), key=lambda t: t[1]):
        if r > best + 12:
            rows.append(dict(name=a["name"], globe_r=round(r, 1), screened_out=True)); continue
        s = score(a); rows.append(s)
        print(f"r {s['globe_r']:5.1f} lower {s['lower_r']:5.1f} out {s['outboard_x']:5.1f} in {s['inboard_x_poses']:6.1f} "
              f"lib {s['lib_contact']:3d}/{s['lib_n']} grid {s['grid_contact']:2d}  {s['name']}")
    (HERE / "block-sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
