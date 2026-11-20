"""Explainer for A-06 (DECISIONS.md): a globe over a joint stays put only if every axis it covers passes through its
centre. Today hip yaw and pitch meet at C; hip roll turns 29 mm below C, so a hip globe centred on C slides when the
leg rolls. Drawn on the real skeleton cuts (no shells).

Run: python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/explainers/globe_principle.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "sections"))
import kit  # noqa: E402
from joint_cuts import cut_polys, draw, front, side, skeleton_parts, union  # noqa: E402

R_GLOBE = 45.0
C = np.array([59.0, 2.0, 219.0]); ROLL = np.array([56.0, 2.0, 190.0])


def layer(body, ax, T, q, moving_links, faces=6000):
    parts = skeleton_parts(body)
    for n, p in parts.items():
        g = union(cut_polys(kit._mesh(body, n, faces), T @ body.link_T(p["owning_link"], q)))
        if g is None:
            continue
        if p["owning_link"] in moving_links:
            draw(ax, g, fc=(0.35, 0.45, 0.8, 0.25), ec=(0.2, 0.3, 0.6), lw=1.0)
        else:
            draw(ax, g, fc="#a8a8a8", ec="k", lw=0.4)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    body = kit.base()
    leg = {"R_pitch_link", "R_roll_link", "R_shin", "R_ankle_pitch", "R_foot"}
    fig, axs = plt.subplots(1, 4, figsize=(34, 9.5))
    # A: pitch, side cut through C: the globe about C does not move
    ax = axs[0]; T = side(59.0)
    for a, al in ((0.0, 1.0), (-85.0, 0.5), (40.0, 0.5)):
        layer(body, ax, T, {**body.standing, "R_hip_pitch": a}, leg | {"R_pitch_link"})
    c = (T @ np.r_[C, 1])[:2]
    ax.add_patch(Circle(c, R_GLOBE, fill=False, ec="#d07000", lw=4))
    ax.plot(*c, "+", color="#d00", ms=20, mew=3)
    ax.set_title("A  Pitch (side cut through C): a globe centred on C\nis the same circle at -85, 0 and +40: it never moves",
                 fontsize=14)
    # B: yaw, plan cut through C
    ax = axs[1]; from joint_cuts import plan; T = plan(219.0)
    for a in (0.0, -40.0, 40.0):
        layer(body, ax, T, {**body.standing, "R_hip_yaw": a}, leg | {"R_yaw_link"})
    c = (T @ np.r_[C, 1])[:2]
    ax.add_patch(Circle(c, R_GLOBE, fill=False, ec="#d07000", lw=4)); ax.plot(*c, "+", color="#d00", ms=20, mew=3)
    ax.set_title("B  Yaw (plan cut through C): the yaw axis also passes\nthrough C, so the same globe stays put at -40, 0, +40",
                 fontsize=14)
    # C: roll, front cut through the pitch axis: the globe rides on the leg and its centre swings about the roll axis
    ax = axs[2]; T = front(2.0)
    rp = (T @ np.r_[ROLL, 1])[:2]
    for a, col in ((0.0, "#d07000"), (-45.0, "#7a1fa2"), (25.0, "#1f77b4")):
        q = {**body.standing, "R_hip_roll": a}
        layer(body, ax, T, q, leg - {"R_pitch_link"})
        cw = body.link_T("R_roll_link", q) @ np.r_[C, 1]
        cc = (T @ cw)[:2]
        ax.add_patch(Circle(cc, R_GLOBE, fill=False, ec=col, lw=3.5, ls="-" if a == 0 else "--"))
        ax.plot(*cc, "o", color=col, ms=8)
        d = np.linalg.norm(cw[:3] - C)
        if a:
            ax.annotate(f"roll {a:+.0f}: globe centre moves {d:.0f} mm", cc, xytext=(cc[0] - 10, cc[1] + 55 + (a > 0) * 18),
                        fontsize=12, color=col, arrowprops=dict(arrowstyle="->", color=col))
    ax.plot(*rp, "x", color="b", ms=16, mew=3); ax.text(rp[0] + 4, rp[1] - 10, "roll axis (29 mm below C)", color="b", fontsize=12)
    ax.set_title("C  Roll (front cut): the roll axis misses C by 29 mm, so a globe\non the leg centred on C slides sideways and up",
                 fontsize=14)
    # D: displacement vs roll angle for offsets
    ax = axs[3]
    ang = np.linspace(-45, 25, 141)
    for h, ls in ((29, "-"), (15, "--"), (5, ":"), (0, "-.")):
        ax.plot(ang, 2 * h * np.abs(np.sin(np.radians(ang) / 2)), ls, lw=3, label=f"roll axis {h} mm below the globe centre")
    ax.set_xlabel("hip roll (deg)", fontsize=12); ax.set_ylabel("globe centre displacement (mm)", fontsize=12)
    ax.grid(alpha=0.3); ax.legend(fontsize=12)
    ax.set_title("D  How far a hip globe slides under roll, by how far the\nroll axis sits below the point where yaw and pitch meet",
                 fontsize=14)
    for ax, lim in zip(axs[:3], (((-110, 110), (95, 335)), ((-60, 160), (-90, 110)), ((-160, 40), (95, 335)))):
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.suptitle("A-06: a globe over a joint only stays put if every axis it covers runs through its centre "
                 f"(globe r {R_GLOBE:.0f} mm, bare skeleton; blue = moving parts)", fontsize=17)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "globe-principle.png"
    plt.savefig(out, dpi=55); print(out)


if __name__ == "__main__":
    main()
