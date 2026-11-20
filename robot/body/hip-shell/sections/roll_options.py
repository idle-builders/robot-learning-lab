"""Drawings for two open questions on the owner's hip shell idea (D-12), 2026-11-16.

1. roll-motor-options.png: the roll motor case flipped into the thigh, three ways, front X-ray of the hip parts at
   standing, roll +25 (leg inward) and the climb step (-85 / -30 / toes in 30 / knee 95). Red = overlap.
   A  flipped as is (case horizontal, 35 mm inboard of the roll axis at z 190)
   B  case turned 90 deg about its own shaft so it hangs down into the thigh (axis stays at z 190)
   C  case horizontal, roll axis lowered 18 mm to z 172
   Overlap counts are over the 151 realistic poses (A-11), case against the hip parts that do not roll with it,
   except the pitch-to-roll connector, which wraps today's roll case and is redrawn for the flip (drawn, not counted).
2. green-inboard-plan.png: plan cut at z 230 through the hip-pitch motor at hip yaw -40 / 0 / +40, with the static
   battery cradle and keel, to show where an inboard green wall would have to go.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sections/roll_options.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import kit  # noqa: E402
import drawing_sheets as DS  # noqa: E402
from leg_motion import library_poses, q_of, solid  # noqa: E402

OUT = HERE / "drawing-sheets"
CASE = "R_roll_servo_case_reference"
AXIS_X, AXIS_Z = 56.0, 190.0
VARIANTS = {"A": ("A  flipped as is (case lies inboard of the roll axis)", 0.0, 0.0),
            "B": ("B  case turned 90 deg to hang down (axis stays at z 190)", -90.0, 0.0),
            "C": ("C  roll axis lowered 18 mm to z 172 (case lies inboard)", 0.0, -18.0)}
SHOW = [("standing", (0, 0, 0, 0)), ("leg rolled inward +25", (0, 25, 0, 0)),
        ("climb step -85 / -30 / toes in 30 / knee 95", (-85, -30, -30, 95))]
OTHERS_LINKS = ("torso", "R_yaw_link", "R_pitch_link")
# the pitch-to-roll connector wraps today's roll case and gets redrawn for the flip: drawn, not counted
SKIP_COUNT = ("R_hip_pitch_to_roll_combined",)


def Ry(deg, px, pz):
    a = np.radians(deg)
    R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = np.array([px, 0, pz]) - R @ np.array([px, 0, pz])
    return T


def roll_T(body, q, dz):
    """Thigh transform with the roll axis at z 190 + dz (same sign convention as the URDF: rotation about +y)."""
    return body.link_T("R_pitch_link", q) @ Ry(q.get("R_hip_roll", 0.0), AXIS_X, AXIS_Z + dz)


def case_mesh(body, turn, dz):
    m = solid(body, CASE).copy()
    T = Ry(turn, AXIS_X, AXIS_Z)            # turn about its own shaft (the roll axis) so the body hangs down
    m.apply_transform(T)
    m.apply_translation([0, 0, dz])
    return m


def others(body):
    out = {}
    for n, p in kit._hip_parts(body).items():
        if p["owning_link"] in OTHERS_LINKS and p.get("role") != "body_shell" and n != CASE \
                and not n.startswith("R_roll_") and "Fin" not in n and "insulator" not in n:
            b = np.array(p["bounds_mm"])
            if np.linalg.norm(np.clip(kit.HIP_C, b[0], b[1]) - kit.HIP_C) < 90:
                out[n] = p
    return out


def overlaps(body, O, mfO, case_mf, q, dz):
    Tc = roll_T(body, q, dz)
    c = case_mf.transform(Tc[:3, :4]); bc = np.array(c.bounding_box()).reshape(2, 3)
    res, inter = {}, []
    for n, m in mfO.items():
        if n in SKIP_COUNT:
            continue
        o = m.transform(body.link_T(O[n]["owning_link"], q)[:3, :4]); bo = np.array(o.bounding_box()).reshape(2, 3)
        if np.any(bc[1] < bo[0]) or np.any(bo[1] < bc[0]):
            continue
        x = c ^ o
        v = x.volume()
        if v > 0.5:
            res[n] = v; inter.append(kit.tm(x))
    return res, inter


def proj(ax, m, T, fc, ec, alpha, lw, z, outline=True):
    """Front X-ray projection of a mesh (robot right on the left)."""
    from matplotlib.collections import PolyCollection
    v = m.vertices @ T[:3, :3].T + T[:3, 3]
    P = np.stack([-v[:, 0], v[:, 2]], 1)[m.faces]
    ax.add_collection(PolyCollection(P, facecolors=fc, edgecolors="none", alpha=alpha, zorder=z))
    if not outline:
        return
    # outline from the 2D union
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    try:
        u = unary_union([Polygon(t).buffer(0.05) for t in P[::1] if Polygon(t).area > 1e-3])
        for g in getattr(u, "geoms", [u]):
            if g.geom_type == "Polygon":
                ax.plot(*g.exterior.xy, color=ec, lw=lw, zorder=z + 0.5)
    except Exception:  # noqa: BLE001
        pass


def roll_sheet(body):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    O = others(body)
    mfO = {n: kit.mf(solid(body, n)) for n in O}
    thigh = {n: p for n, p in kit._hip_parts(body).items() if p["owning_link"] == "R_roll_link"
             and p.get("role") in ("motor", "printed_structural_connector")}
    stats = {}
    fig, axs = plt.subplots(3, 3, figsize=(21, 22))
    for i, (key, (lab, turn, dz)) in enumerate(VARIANTS.items()):
        cm = case_mesh(body, turn, dz); cmf = kit.mf(cm)
        base, _ = overlaps(body, O, mfO, cmf, q_of(body, (0, 0, 0, 0)), dz)
        n_hit, worst, by_part = 0, (0, None), {}
        for fam, p, end in library_poses():
            o, _ = overlaps(body, O, mfO, cmf, q_of(body, p), dz)
            new = {k: v - base.get(k, 0) for k, v in o.items() if v > base.get(k, 0) + 2}
            tot = sum(new.values())
            if tot > 20:
                n_hit += 1
                for k, v in new.items():
                    by_part[k] = max(by_part.get(k, 0), v)
            if tot > worst[0]:
                worst = (tot, f"{fam} {[round(float(x)) for x in p]}")
        stats[key] = dict(label=lab, poses_with_contact=n_hit, of=151, worst_mm3=round(worst[0], 1), worst_pose=worst[1],
                          max_by_part={k: round(v, 1) for k, v in by_part.items()},
                          standing_overlap_mm3={k: round(v, 1) for k, v in base.items()})
        for j, (plab, p4) in enumerate(SHOW):
            ax = axs[i, j]
            q = q_of(body, p4)
            for n, p in O.items():
                col = "#2e9e5b" if p["owning_link"] == "R_yaw_link" else "#e0a400" if p["owning_link"] == "R_pitch_link" else "#9a9a9a"
                proj(ax, solid(body, n), body.link_T(p["owning_link"], q), col, "#333333", 0.22, 0.6, 1, outline=False)
            Tth = roll_T(body, q, dz)
            for n in thigh:
                m = solid(body, n).copy()
                # variant C: thigh parts keep their place below the lowered axis (the thigh connector shortens)
                proj(ax, m, Tth, "#2f6fd0", "#1f4f9f", 0.18, 0.6, 2, outline=False)
            proj(ax, cm, Tth, "#4a4a4a", "#111111", 0.65, 1.2, 3)
            o, inter = overlaps(body, O, mfO, cmf, q, dz)
            new = {k: v - base.get(k, 0) for k, v in o.items() if v > base.get(k, 0) + 2}
            for x in inter:
                proj(ax, x, np.eye(4), "#e41a1c", "#b00000", 0.85, 1.0, 5)
            C = kit.HIP_C
            ax.plot(-C[0], C[2], "+", color="#d62728", ms=14, mew=2.5, zorder=8)
            ra = (Tth @ np.r_[AXIS_X, 0, AXIS_Z + dz, 1])[[0, 2]]
            ax.plot(-ra[0], ra[1], "x", color="#1f5fd6", ms=11, mew=2.2, zorder=8)
            ax.set_xlim(-130, 20); ax.set_ylim(100, 290); ax.set_aspect("equal"); ax.grid(alpha=0.25)
            tot = sum(new.values())
            ax.set_title(f"{plab}\n{'overlap ' + str(round(tot)) + ' mm3' if tot > 20 else 'no new overlap'}", fontsize=12,
                         color="#b00000" if tot > 20 else "#222")
            if j == 0:
                ax.set_ylabel(f"{lab}\n{n_hit} of 151 realistic poses touch\nworst {worst[0]:.0f} mm3", fontsize=12)
    fig.suptitle("Roll motor flipped into the thigh: three ways (front X-ray, hip parts only, robot right on the left).\n"
                 "Dark = roll motor case (rides with the thigh), blue = thigh connector and knee motor, orange = "
                 "pitch-to-roll connector, green = yaw link and pitch motor, grey = battery cradle, keel and core parts. "
                 "Red = overlap.\nRed + = hip centre, blue x = roll axis. The standing overlap of the case with the "
                 "orange connector is today's mount; it is redrawn once the case is flipped.", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(OUT / "roll-motor-options.png", dpi=75, facecolor="white"); plt.close(fig)
    (OUT / "roll-motor-options.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


def plan_sheet(body):
    """Plan cut at z 230: pitch motor at yaw -40/0/+40 and the static cradle/keel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MP
    z = 230.0
    fig, ax = plt.subplots(figsize=(13, 10))
    stat = {n: p for n, p in kit._hip_parts(body).items() if p["owning_link"] == "torso" and p.get("role") != "body_shell"}
    statics = []
    for n, p in stat.items():
        b = np.array(p["bounds_mm"])
        if not (b[0][2] <= z <= b[1][2]) or b[1][0] < -40 or b[0][0] > 110 or b[0][1] > 70 or b[1][1] < -60:
            continue
        s = kit._mesh(body, n).section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])   # real mesh: keeps the pockets
        if s is None:
            continue
        for e in s.entities:
            pts = s.vertices[e.points][:, :2]
            ax.plot(pts[:, 0], -pts[:, 1], color="#555", lw=1.0, zorder=1)
        statics.append(kit._mesh(body, n))
    gaps = []
    for yaw, col, ls in ((-40, "#2e9e5b", "--"), (0, "#1b6e3e", "-"), (40, "#2e9e5b", ":")):
        q = q_of(body, (0, 0, yaw, 0))
        for n in ("R_hip_pitch_ST3025_case", "R_hip_yaw_to_pitch_combined"):
            m = solid(body, n).copy(); m.apply_transform(body.link_T("R_yaw_link", q))
            s = m.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
            if s is None:
                continue
            for e in s.entities:
                pts = s.vertices[e.points][:, :2]
                ax.plot(pts[:, 0], -pts[:, 1], color=col, lw=2.2 if yaw == 0 else 1.6, ls=ls, zorder=3)
            if n == "R_hip_pitch_ST3025_case":
                mm = kit._mesh(body, n).copy(); mm.apply_transform(body.link_T("R_yaw_link", q))
                P = mm.sample(20000)
                dmin = min(float(trimesh.proximity.closest_point(st, P)[1].min()) for st in statics) if statics else -1
                gaps.append((yaw, dmin))
    ax.plot(59, -2, "+", color="#d62728", ms=16, mew=2.5, zorder=5)
    ax.text(61, -6, "yaw axis", color="#d62728", fontsize=11)
    txt = "\n".join(f"yaw {y:+d}: closest gap pitch motor to cradle/keel/cells {g:.1f} mm (3D)" for y, g in gaps)
    ax.text(-38, 58, txt, fontsize=11, color="#b00000", va="top")
    ax.set_xlim(-40, 110); ax.set_ylim(-60, 65); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_xlabel("x (mm), robot's right ->"); ax.set_ylabel("forward (mm) ->")
    ax.set_title("Plan cut at z 230 (through the hip-pitch motor), seen from above, forward up. Grey = static battery "
                 "cradle, keel and cells (real mesh, lines).\nGreen = pitch motor and its yaw connector at hip yaw -40 (dashed), 0 (solid), +40 "
                 "(dotted). An inboard green wall would have to fit between them.", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT / "green-inboard-plan.png", dpi=90, facecolor="white"); plt.close(fig)
    print("gaps", gaps)


def main():
    body = kit.base(DS.SHELL_PASS)
    plan_sheet(body)
    if "--plan" not in sys.argv:
        roll_sheet(body)


if __name__ == "__main__":
    main()
