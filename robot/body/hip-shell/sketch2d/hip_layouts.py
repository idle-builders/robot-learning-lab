"""S3/S4: hip actuator layouts as block sketches, checked over the whole-leg pose library (Idle T-czv2d8).

A layout is an ordered hip chain (yaw / roll / pitch in any order, each axis a line through a chosen point) and boxes
for the servo bodies, horns, thigh and knee servo, each on a link. The check poses the right leg through the A-11
library (with paths) and the A-12 combinations, and measures:
  - contact volume between boxes on different links (mm3), standing contact (assembly) left out;
  - the radius of the smallest sphere about the hip centre that holds every hip servo (a hip globe would need this);
  - axis distances (0 = concurrent), hip outboard extent, closest approach of the moving leg to the centreline;
  - the gap between the last hip servo and the knee servo along the thigh.
Boxes only: brackets are designable later and are not modelled. Torso keep-out = battery cells and keel, the cradle
plate, and the other leg (mirrored, standing).

Run: python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/sketch2d/hip_layouts.py
"""
import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "sections"))
import kit  # noqa: E402
from leg_motion import GRID, LIBRARY, STEPS  # noqa: E402

C = np.array([59.0, 2.0, 219.0])
AX = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
# brick servos (mm): length L with the shaft 10 mm from one end, width W, height H along the shaft (+5 horn)
ST3025 = dict(L=40.0, W=20.0, H=43.0)   # hip pitch, knee (fast leg tier)
STS3250 = dict(L=45.2, W=24.7, H=35.0)  # hip roll (strong tier), C047 case
ST3215 = dict(L=45.2, W=24.7, H=35.0)   # hip yaw (light tier), C047 case
HORN = 5.0
TORSO_KEEP = [("battery cells + keel", (-13, -31, 205), (13, 43, 257)), ("cradle plate", (-55, -10, 258), (55, 38, 276))]


def brick(servo, face, shaft, length_dir, horn=True):
    """Box of a brick servo: output face centre on the shaft at `face`, shaft pointing out of the face along
    `shaft`, body behind the face; the shaft sits 10 mm from the end of the length L that points away along
    -length_dir... i.e. the body spans -10..L-10 along length_dir from the shaft."""
    s = np.asarray(shaft, float); s /= np.linalg.norm(s)
    l = np.asarray(length_dir, float); l /= np.linalg.norm(l)
    w = np.cross(s, l)
    corners = []
    for a in (0.0, -servo["H"]):
        for b in (-10.0, servo["L"] - 10.0):
            for c in (-servo["W"] / 2, servo["W"] / 2):
                corners.append(np.asarray(face, float) + a * s + b * l + c * w)
    body = trimesh.convex.convex_hull(np.array(corners))
    out = [body]
    if horn:
        h = trimesh.creation.cylinder(radius=9.0, height=HORN, sections=24)
        T = trimesh.geometry.align_vectors([0, 0, 1], s); h.apply_transform(T)
        h.apply_translation(np.asarray(face, float) + s * (HORN / 2 + 0.5))
        out.append(h)
    return out


def bx(lo, hi):
    return trimesh.creation.box(bounds=[lo, hi])


@dataclass
class Layout:
    name: str
    order: list                      # e.g. ["yaw", "pitch", "roll"]: torso -> link1 -> link2 -> thigh
    axes: dict                       # joint -> (point, 'x'|'y'|'z' or vector)
    parts: list = field(default_factory=list)   # (label, mesh, link) ; link in torso, J1, J2, thigh, shin
    hip_servos: tuple = ()           # labels counted for the globe radius
    note: str = ""


def rot_about(p, a, deg):
    a = np.asarray(a, float) / np.linalg.norm(a)
    R = trimesh.transformations.rotation_matrix(np.radians(deg), a, point=p)
    return R


def link_T(lay, q):
    """Neutral-frame axes composed parent first (as checks/bodylib.Body.link_T). q: pitch, roll, yaw, knee."""
    Ts = {"torso": np.eye(4)}
    T = np.eye(4)
    for k, j in enumerate(lay.order):
        p, a = lay.axes[j]
        a = AX[a] if isinstance(a, str) else np.asarray(a, float)
        T = T @ rot_about(p, a, q[j])
        Ts[f"J{k + 1}"] = T.copy()
    Ts["thigh"] = Ts[f"J{len(lay.order)}"]
    Ts["shin"] = Ts["thigh"] @ rot_about([52.0, 4.0, 123.0], [1, 0, 0], q["knee"])
    return Ts


def poses():
    out = [("standing", (0, 0, 0, 0))]
    for fam, ps in LIBRARY.items():
        for p in ps:
            for k in range(1, STEPS + 1):
                out.append((fam, tuple(np.array(p, float) * k / STEPS)))
    return out


import os
COUNT_BRACKETS = os.environ.get("HIP_COUNT_BRACKETS") == "1"


def counted(name):
    if COUNT_BRACKETS:
        return name not in ("shin", "thigh")
    """A-17: the layout check counts servo bodies, horns, the torso keep-out and the other leg's servos; brackets,
    necks, thigh and shin boxes are drawn but not counted (they are designed after the layout)."""
    return ("servo" in name or "horn" in name or name in [n for n, _, _ in TORSO_KEEP]) and "bracket" not in name


def evaluate(lay):
    import manifold3d as M
    parts = list(lay.parts) + [(n, bx(lo, hi), "torso") for n, lo, hi in TORSO_KEEP]
    # the other leg, mirrored, standing (torso-fixed for this check)
    for n, m, l in lay.parts:
        if l != "torso":
            mm = m.copy(); mm.vertices[:, 0] *= -1; mm.faces = mm.faces[:, ::-1]
            parts.append(("left " + n, mm, "torso"))
    mf = [(n, kit.mf(m), l) for n, m, l in parts]

    def overlaps(p4):
        q = dict(zip(("pitch", "roll", "yaw", "knee"), p4))
        Ts = link_T(lay, q)
        posed = [(n, g.transform(Ts[l][:3, :4]), l) for n, g, l in mf]
        res = {}
        for (a, ga, la), (b, gb, lb) in itertools.combinations(posed, 2):
            if la == lb or not (counted(a) and counted(b)):
                continue
            v = (ga ^ gb).volume()
            if v > 0.5:
                res[(a, b)] = v
        return res, Ts

    base, _ = overlaps((0, 0, 0, 0))
    rows = []
    min_x = 1e9
    for fam, p in poses() + [("all combinations", g) for g in GRID]:
        o, Ts = overlaps(p)
        new = {k: v for k, v in o.items() if v > base.get(k, 0.0) + 2.0}
        rows.append(dict(family=fam, pose=[round(float(x), 1) for x in p], contact_mm3=round(sum(new.values()), 1),
                         pairs={f"{a} x {b}": round(v, 1) for (a, b), v in sorted(new.items(), key=lambda t: -t[1])[:3]}))
        if fam != "all combinations":
            for n, m, l in lay.parts:
                if l != "torso":
                    min_x = min(min_x, float((m.vertices @ Ts[l][:3, :3].T + Ts[l][:3, 3])[:, 0].min()))
    hs = [m for n, m, l in lay.parts if n in lay.hip_servos]
    r_globe = max(float(np.linalg.norm(m.vertices - C, axis=1).max()) for m in hs) if hs else 0.0
    ax = {}
    for j1, j2 in itertools.combinations(lay.order, 2):
        (p1, a1), (p2, a2) = lay.axes[j1], lay.axes[j2]
        a1 = AX[a1] if isinstance(a1, str) else np.asarray(a1, float); a2 = AX[a2] if isinstance(a2, str) else np.asarray(a2, float)
        n = np.cross(a1, a2)
        ax[f"{j1}-{j2}"] = round(float(abs((np.asarray(p2) - np.asarray(p1)) @ n) / np.linalg.norm(n)), 1)
    lib = [r for r in rows if r["family"] != "all combinations"]
    return dict(name=lay.name, order=lay.order, note=lay.note, axis_distances_mm=ax, globe_radius_mm=round(r_globe, 1),
                outboard_x_mm=round(max(float(m.vertices[:, 0].max()) for n, m, l in lay.parts), 1),
                closest_to_centreline_x_mm=round(min_x, 1),
                library_poses=len(lib), library_poses_with_contact=sum(r["contact_mm3"] > 20 for r in lib),
                library_worst=sorted(lib, key=lambda r: -r["contact_mm3"])[:6],
                combinations_with_contact=sum(r["contact_mm3"] > 20 for r in rows if r["family"] == "all combinations"),
                standing_assembly_overlap={f"{a} x {b}": round(v, 1) for (a, b), v in base.items()})


# ---------------------------------------------------------------- layouts
def as_built():
    """Today's hip as boxes from the CAD bounds (calibration against the full-mesh analysis)."""
    return Layout(
        "as built (yaw -> pitch -> roll)", ["yaw", "pitch", "roll"],
        {"yaw": ((59, 2, 0), "z"), "pitch": ((0, 2, 219), "x"), "roll": ((56, 0, 190), "y")},
        [("yaw servo", bx((24, -10, 276), (69, 14, 313)), "torso"),
         ("yaw neck", bx((49, -15, 249), (72, 19, 275)), "J1"),
         ("pitch servo", bx((14, -8, 209), (57, 12, 249)), "J1"),
         ("roll servo", bx((21, -10.5, 178), (66, 25.8, 202)), "J2"),
         ("pitch-roll bracket", bx((57, -14, 177), (71, 25, 229)), "J2"),
         ("thigh", bx((31, -20, 128), (68, 32, 175)), "thigh"),
         ("knee servo", bx((32, -6, 113), (76, 14, 153)), "thigh"),
         ("shin", bx((36, -12, 60), (70, 18, 112)), "shin")],
        hip_servos=("yaw servo", "pitch servo", "roll servo"))


def pitch_first():
    """P-03: G1 / GR-2 order with our servos; pitch and roll stage as built, yaw servo in the thigh, knee servo turned."""
    return Layout(
        "P-03 pitch first (pitch -> roll -> yaw), roll 29 mm below", ["pitch", "roll", "yaw"],
        {"pitch": ((0, 2, 219), "x"), "roll": ((56, 0, 190), "y"), "yaw": ((56, 2, 0), "z")},
        [("pitch servo", bx((14, -8, 209), (57, 12, 249)), "torso"),
         ("roll servo", bx((21, -10.5, 178), (66, 25.8, 202)), "J1"),
         ("pitch-roll bracket", bx((57, -14, 177), (71, 25, 229)), "J1"),
         ("roll bracket", bx((18, -16, 170), (70, 30, 178)), "J2"),
         # long side front-to-back: 24.7 mm across the roll swing instead of 45 (cross step +25 hit the roll servo)
         ("yaw servo", bx((43.6, -8, 133.5), (68.4, 37, 170.5)), "thigh"),
         ("knee servo", bx((32, -6, 113), (76, 34, 133)), "thigh"),
         ("shin", bx((36, -12, 60), (70, 18, 112)), "shin")],
        hip_servos=("pitch servo", "roll servo", "yaw servo"),
        note="yaw servo body rides on the lower thigh, long side front-to-back, horn bolted into the roll bracket")


def neck():
    m = trimesh.creation.cylinder(radius=10.0, height=21.0, sections=32)
    m.apply_translation([59.0, 2.0, 265.5])
    return m


def op3_style():
    """P-04: OP3's concurrent arrangement with our servos (yaw -> roll -> pitch, all three axes through C).
    Roll (STS3250) and pitch (ST3025) servos stand upright fore and aft as one bolted block with flush tops, so both
    shafts sit 10 mm below the top at z 219. The pitch servo straddles C along its own axis (thigh fork around it:
    horn outboard, idler inboard). The roll servo stands behind it, output facing back into the rear arm of the yaw
    U-bracket; the front arm runs on an idler on the pitch servo's front face. The yaw servo stays where it is in the
    torso, and the U-bracket hangs from its horn (as today's neck)."""
    top = 229.0
    return Layout(
        "P-04 OP3-style concurrent (yaw -> roll -> pitch, all axes through C)", ["yaw", "roll", "pitch"],
        {"yaw": ((59, 2, 0), "z"), "roll": ((59, 0, 219), "y"), "pitch": ((0, 2, 219), "x")},
        [("yaw servo", bx((24, -10, 276), (69, 14, 313)), "torso"),
         # bridge under the cradle plate (z 258); a round neck on the yaw axis carries it up to the yaw horn
         ("yaw neck (round, on the yaw axis)", neck(), "J1"),
         ("yaw U-bracket bridge", bx((46, -15, 250), (72, 59, 255)), "J1"),
         ("yaw U-bracket front arm", bx((46, -15, 210), (72, -10, 250)), "J1"),
         ("yaw U-bracket rear arm", bx((46, 54, 210), (72, 59, 250)), "J1"),
         ("roll horn", bx((50, 48.5, 210), (68, 53.5, 228)), "J1"),
         ("roll servo", bx((46.65, 13, top - 45.2), (71.35, 48, top)), "J2"),
         ("pitch servo", bx((37.5, -8, top - 40), (80.5, 12, top)), "J2"),
         ("pitch horn", bx((81, -7, 210), (86, 11, 228)), "J3"),
         ("thigh fork outboard arm", bx((86, -12, 158), (91, 16, 229)), "J3"),
         ("thigh fork inboard arm", bx((31, -12, 158), (37, 16, 229)), "J3"),
         ("thigh fork crossbar", bx((31, -12, 153), (91, 16, 158)), "J3"),
         ("knee servo", bx((32, -6, 113), (76, 14, 153)), "thigh"),
         ("shin", bx((36, -12, 60), (70, 18, 112)), "shin")],
        hip_servos=("roll servo", "pitch servo"),
        note="globe radius counts the roll and pitch servos; the yaw servo stays in the torso above the globe")


LAYOUTS = {"as-built": as_built, "p03-pitch-first": pitch_first, "p04-op3-style": op3_style}


def main():
    if "--figure" in sys.argv:
        print(figure(json.loads((HERE / "hip-layouts.json").read_text()))); return
    names = sys.argv[1:] or list(LAYOUTS)
    out = {}
    for n in names:
        r = evaluate(LAYOUTS[n]())
        out[n] = r
        print(json.dumps({k: v for k, v in r.items() if k not in ("library_worst", "standing_assembly_overlap")}))
        for w in r["library_worst"][:4]:
            print("   ", w["family"], w["pose"], w["contact_mm3"], w["pairs"])
    f = HERE / "hip-layouts.json"
    old = json.loads(f.read_text()) if f.exists() else {}
    old.update(out); f.write_text(json.dumps(old, indent=1))



# ---------------------------------------------------------------- figure
VIEW_R = {"front": np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], float),   # from the front, robot right on the left
          "side": np.array([[0, -1, 0], [0, 0, 1], [1, 0, 0]], float),    # from the right, forward to the right
          "top": np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]], float)}     # from above, forward up
LCOL = {"torso": (0.55, 0.55, 0.55), "J1": (0.2, 0.65, 0.45), "J2": (0.95, 0.55, 0.15), "J3": (0.25, 0.45, 0.85),
        "thigh": (0.25, 0.45, 0.85), "shin": (0.55, 0.35, 0.75)}
AXCOL = {"yaw": "#1a9850", "roll": "#2166ac", "pitch": "#d73027"}


def draw_layout(ax, lay, q, view, lim, globe=None):
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Circle
    R = VIEW_R[view]
    Ts = link_T(lay, q)
    L = np.array([0.3, -0.6, 0.7]); L /= np.linalg.norm(L)
    P, Cc, D = [], [], []
    for n, m, l in list(lay.parts) + [(n, bx(lo, hi), "torso") for n, lo, hi in TORSO_KEEP]:
        T = Ts[l]
        v = (m.vertices @ T[:3, :3].T + T[:3, 3]) @ R.T
        nn = m.face_normals @ T[:3, :3].T
        tri = v[m.faces]; vis = (nn @ R.T)[:, 2] > -0.05
        sh = 0.5 + 0.5 * np.clip(np.abs(nn @ L), 0, 1)
        base = np.array(LCOL.get(l, (0.7, 0.7, 0.7))) if not n.startswith(("battery", "cradle")) else np.array((0.85, 0.85, 0.8))
        a = 0.35 if ("bracket" in n or "fork" in n or "neck" in n) else 1.0
        P.extend(tri[vis][:, :, :2]); D.extend(tri[vis][:, :, 2].mean(1))
        Cc.extend([(*np.clip(base * s, 0, 1), a) for s in sh[vis]])
    o = np.argsort(np.array(D))
    ax.add_collection(PolyCollection([P[i] for i in o], facecolors=[Cc[i] for i in o], edgecolors="none"))
    # axes, posed: an axis of joint k moves with the link before it
    for k, j in enumerate(lay.order):
        p, a = lay.axes[j]
        a = AX[a] if isinstance(a, str) else np.asarray(a, float)
        Tp = Ts["torso"] if k == 0 else Ts[f"J{k}"]
        p0 = Tp[:3, :3] @ np.asarray(p, float) + Tp[:3, 3]; a0 = Tp[:3, :3] @ a
        if j != "yaw":   # anchor the drawn segment near the hip
            p0 = p0 + a0 * ((np.array([59, 2, 219]) - p0) @ a0)
        else:
            p0 = p0 + a0 * ((219 - p0[2]) / a0[2] if abs(a0[2]) > 1e-6 else 0)
        e = np.array([p0 - 120 * a0, p0 + 120 * a0]) @ R.T
        ax.plot(e[:, 0], e[:, 1], color=AXCOL[j], lw=2.2, alpha=0.9)
    c = R @ np.array([59.0, 2.0, 219.0])
    ax.plot(c[0], c[1], "k+", ms=14, mew=2.5)
    if globe:
        ax.add_patch(Circle(c[:2], globe, fill=False, ls="--", ec="k", lw=1.5))
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.grid(alpha=0.2)


def figure(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lays = [as_built(), pitch_first(), op3_style()]
    poses_shown = [("standing", dict(pitch=0, roll=0, yaw=0, knee=0), ("front", "side", "top")),
                   ("squat -85, knee 95", dict(pitch=-85, roll=0, yaw=0, knee=95), ("side",)),
                   ("leg out -45", dict(pitch=0, roll=-45, yaw=0, knee=0), ("front",)),
                   ("climb, toes in", dict(pitch=-85, roll=-30, yaw=-30, knee=95), ("front",))]
    lims = {"front": ((-130, 40), (90, 330)), "side": ((-110, 110), (90, 330)), "top": ((-30, 140), (-80, 100))}
    ncol = sum(len(v) for _, _, v in poses_shown)
    fig, axs = plt.subplots(len(lays), ncol, figsize=(4.6 * ncol, 5.8 * len(lays)))
    keys = ["as-built", "p03-pitch-first", "p04-op3-style"]
    for i, lay in enumerate(lays):
        r = results.get(keys[i], {})
        k = 0
        for lab, q, views in poses_shown:
            for v in views:
                ax = axs[i, k]; k += 1
                draw_layout(ax, lay, q, v, lims[v], globe=r.get("globe_radius_mm") if keys[i] == "p04-op3-style" else None)
                ax.set_title(f"{lab} - {v}", fontsize=11)
        ad = r.get("axis_distances_mm", {})
        axs[i, 0].set_ylabel(f"{lay.name}\naxis distances {ad}\nservo globe r {r.get('globe_radius_mm', 0):.0f} mm, "
                             f"outboard x {r.get('outboard_x_mm', 0):.0f}\nrealistic poses with servo contact: "
                             f"{r.get('library_poses_with_contact', '?')} / {r.get('library_poses', '?')}", fontsize=11)
    fig.suptitle("Hip actuator layouts with our servos, block sketches (servos solid, brackets translucent). Green = yaw "
                 "axis, blue = roll, red = pitch; black + = hip centre C; dashed = sphere holding the hip servos (P-04). "
                 "Grey-white = battery keel and cradle plate.", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out = HERE / "hip-layouts.png"; plt.savefig(out, dpi=55); plt.close(fig)
    return out


if __name__ == "__main__":
    main()
