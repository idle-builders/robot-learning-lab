"""Cross-sections exactly through the right hip (T-czv2d8): side cut x = 59 (through the hip centre C and the yaw
axis) and front cut y = 2 (through the pitch axis and C), standing and at -90 hip flexion.

Run: python3 infra/tools/memguard.py --limit-gb 8 -- .venv/bin/python -B robot/body/hip-shell/sections/hip_sections.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import kit  # noqa: E402

FACES = 20000
def side(x):
    return np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, x], [0, 0, 0, 1]], float)


CUTS = {  # name: (to_2D world -> section frame; section plane is local z = 0), axis labels, limits
    "side cut x = 59 (through hip centre C)": (side(59), "forward (-y) mm", (-110, 110), (80, 330)),
    "side cut x = 36 (through the hip-pitch servo)": (side(36), "forward (-y) mm", (-110, 110), (80, 330)),
    "front cut y = 2 (through hip pitch axis)": (np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, -2], [0, 0, 0, 1]], float),
                                                "robot right (+x) <-  mm", (-150, 150), (80, 330)),
}
POSES = [("standing", {}), ("hip -90 flexion, knee 90", {"R_hip_pitch": -90.0, "R_knee": 90.0})]
SWEEP = np.arange(-90.0, 40.1, 2.5)  # hip pitch, for the swept-region panel (side cuts: pitch keeps x, so it is planar)
SHORT = [("center_front_shell", "belly shell"), ("center_back_shell", "back shell"), ("thigh_front", "thigh cover F"),
         ("thigh_back", "thigh cover B"), ("hip_pitch", "pitch servo"), ("hip_yaw", "yaw servo"), ("hip_roll", "roll servo"),
         ("cradle", "battery cradle"), ("cell", "cell"), ("connector", "connector"), ("spine", "spine"), ("carrier", "carrier")]


def short(n):
    for k, s in SHORT:
        if k in n.lower():
            return s
    return n.replace("R_", "").replace("L_", "")[:18]


def sections(body, q, T):
    out = []
    for n, p in kit._hip_parts(body).items():
        m = kit._mesh(body, n, FACES).copy()
        m.apply_transform(body.link_T(p["owning_link"], q))
        m.apply_transform(T)
        if m.bounds[0][2] > 0 or m.bounds[1][2] < 0:
            continue
        s = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
        if s is None:
            continue
        col = np.array(p.get("color") or [180, 180, 180, 255], float)[:3] / 255
        out.append((n, [c[:, :2] for c in s.discrete], col, p["owning_link"]))
    return out


def swept_panel(ax, body, T, xlim, zlim, cut, joint, angles, what):
    """Static parts (do not move with this joint) drawn solid; the region the leg sweeps over the joint range shaded.
    Each angle is posed in 3D and cut, so the region is exact for the cut plane (others joints at standing).
    Clipped to r <= 100 mm about C, the hip region."""
    from shapely.geometry import Polygon as SP, Point
    from shapely.ops import unary_union
    from matplotlib.patches import Polygon
    q0 = dict(body.standing)
    q1 = {**q0, joint: 30.0}
    Cp = (T @ np.r_[kit.HIP_C, 1.0])[:2]
    clip = Point(*Cp).buffer(100)
    parts = kit._hip_parts(body)
    mov = {n: p for n, p in parts.items() if not np.allclose(body.link_T(p["owning_link"], q0), body.link_T(p["owning_link"], q1))}
    stat = [x for x in sections(body, q0, T) if x[0] not in mov]
    polys = []
    for a in angles:
        q = {**q0, joint: float(a)}
        for n, p in mov.items():
            m = kit._mesh(body, n, 5000).copy()
            m.apply_transform(body.link_T(p["owning_link"], q)); m.apply_transform(T)
            if m.bounds[0][2] > 0 or m.bounds[1][2] < 0:
                continue
            sec = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
            if sec is None:
                continue
            for c in sec.discrete:
                if len(c) > 3:
                    g = SP(c[:, :2]).buffer(0)
                    polys.append(g if g.area > 1 else SP(c[:, :2]).buffer(0.8))
    env = unary_union(polys).buffer(0.6).buffer(-0.6).intersection(clip)
    for g in getattr(env, "geoms", [env]):
        if g.is_empty or g.geom_type != "Polygon":
            continue
        ax.add_patch(Polygon(np.array(g.exterior.coords), closed=True, fc="#f4a6a6", ec="#c0392b", lw=1.0, alpha=0.6))
        for h in g.interiors:
            ax.add_patch(Polygon(np.array(h.coords), closed=True, fc="white", ec="#c0392b", lw=0.6))
    for n, curves, col, _ in stat:
        for pts in curves:
            if len(pts) > 3:
                ax.add_patch(Polygon(pts, closed=True, fc=col, ec="k", lw=0.6, alpha=0.9))
            else:
                ax.plot(pts[:, 0], pts[:, 1], color="k", lw=0.6)
    ax.plot(*Cp, "r+", ms=16, mew=2)
    ax.add_patch(plt_circle(Cp, 100))
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.grid(alpha=0.25)
    ax.set_title(f"{cut} — static parts (solid), region the leg sweeps, {what} (red)", fontsize=12)


def plt_circle(c, r):
    from matplotlib.patches import Circle
    return Circle(c, r, fill=False, ls="--", ec="#999", lw=0.8)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Circle
    body = kit.base()
    fig, axs = plt.subplots(3, 3, figsize=(31, 30))
    for r, (cut, (T, xl, xlim, zlim)) in enumerate(CUTS.items()):
        for c, (pl, dq) in enumerate(POSES):
            ax = axs[r, c]
            q = {**body.standing, **dq}
            labelled = set()
            for n, curves, col, _ in sections(body, q, T):
                for pts in curves:
                    closed = len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 1e-3
                    if closed:
                        ax.add_patch(Polygon(pts, closed=True, fc=col, ec="k", lw=0.6, alpha=0.85))
                    else:
                        ax.plot(pts[:, 0], pts[:, 1], color="k", lw=0.6)
                big = max(curves, key=lambda p: np.ptp(p, 0).prod(), default=None)
                if big is not None and np.ptp(big, 0).prod() > 120:
                    lab = short(n)
                    side = "L " if n.startswith("L_") else ""
                    key = side + lab
                    cx, cy = big.mean(0)
                    if xlim[0] < cx < xlim[1] and zlim[0] < cy < zlim[1] and key not in labelled:
                        labelled.add(key)
                        ax.annotate(key, (cx, cy), fontsize=8, ha="center", color="#222",
                                    bbox=dict(fc="white", ec="none", alpha=0.6, pad=0.5))
            Cw = np.r_[kit.HIP_C, 1.0]; Cp = (T @ Cw)[:2]
            ax.plot(*Cp, "r+", ms=16, mew=2)
            for rad in (30, 45, 60):
                ax.add_patch(Circle(Cp, rad, fill=False, ls=":", ec="r", lw=0.8))
                ax.text(Cp[0] + rad * 0.71, Cp[1] + rad * 0.71, f"r{rad}", color="r", fontsize=8)
            if r == 1:
                Rw = np.array([56.0, 2.0, 190.0, 1.0]); Rp = (T @ Rw)[:2]
                ax.plot(*Rp, "bx", ms=12, mew=2); ax.text(Rp[0] + 3, Rp[1] - 8, "roll axis", color="b", fontsize=9)
            ax.text(Cp[0] + 3, Cp[1] + 4, "C (yaw x pitch)", color="r", fontsize=9)
            ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal"); ax.grid(alpha=0.25)
            ax.set_xlabel(xl); ax.set_ylabel("z mm")
            ax.set_title(f"{cut} — {pl}", fontsize=13)
        ax = axs[r, 2]
        if cut.startswith("side"):
            swept_panel(ax, body, T, xlim, zlim, cut, "R_hip_pitch", np.arange(-90.0, 40.1, 2.5), "hip pitch -90..+40")
        else:
            swept_panel(ax, body, T, xlim, zlim, cut, "R_hip_roll", np.arange(-45.0, 25.1, 2.5), "hip roll -45..+25")
    fig.suptitle("Ape body, right hip: what fills the space around the hip centre C. Dotted rings r 30/45/60 mm about C.",
                 fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "hip-sections.png"
    plt.savefig(out, dpi=70)
    print(out)


if __name__ == "__main__":
    main()
