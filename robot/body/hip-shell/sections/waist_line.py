"""How high can a leg enclosure go before it hits the core shell? (Idle T-czv2d8, 2026-11-15)

Hip pitch turns every leg point about the pitch axis (along x through y 2, z 219). A leg point at radius r and angle
phi0 from straight up (positive towards the front) passes through phi0 - 85 .. phi0 + 40 over hip pitch -85..+40, so
the highest it reaches is 219 + r if that interval crosses the top, else 219 + r * cos(nearest end). The core shell
over that leg must end above that height. The map shades, for every point of the side plane, that highest height:
put the leg enclosure's surface at a point and the colour tells you where the core shell's edge has to be.

Run: python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/sections/waist_line.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import kit  # noqa: E402
from hip_sections import sections, side  # noqa: E402

FLEX, EXT = -85.0, 40.0
CY, CZ = 2.0, 219.0
HEM_90 = dict(front=248.0, back_x59=232.0, back_x36=220.0)  # measured on the shell carved for -90 (shell-pass)
LEVELS = [235, 248, 262, 276, 290]


def max_height(fwd, z):
    """fwd = -y (forward). Highest z reached over hip pitch FLEX..EXT by the leg point at standing (fwd, z)."""
    dy, dz = fwd + CY, z - CZ                       # forward offset from the axis (y_axis = 2 -> fwd_axis = -2)
    r = np.hypot(dy, dz)
    phi = np.degrees(np.arctan2(dy, dz))           # 0 = up, +90 = front, +-180 = down
    lo, hi = phi + FLEX, phi + EXT                 # flexion turns front points up (phi decreases)
    top = (lo <= 0) & (hi >= 0)
    best = np.maximum(np.cos(np.radians(lo)), np.cos(np.radians(hi)))
    return CZ + r * np.where(top, 1.0, best)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    body = kit.base()
    fig, axs = plt.subplots(1, 3, figsize=(30, 11.5))
    F, Z = np.meshgrid(np.linspace(-110, 110, 441), np.linspace(80, 330, 501))
    H = max_height(F, Z)
    for ax, x, pose, lab in ((axs[0], 59, {}, "standing"), (axs[1], 36, {}, "standing"),
                             (axs[2], 59, {"R_hip_pitch": FLEX, "R_knee": 90.0}, f"hip {FLEX:.0f}, knee 90")):
        q = {**body.standing, **pose}
        T = side(x)
        cs = ax.contourf(F, Z, H, levels=[150] + LEVELS + [340], cmap="RdYlGn_r", alpha=0.35)
        cl = ax.contour(F, Z, H, levels=LEVELS, colors="k", linewidths=1.0, linestyles="--")
        ax.clabel(cl, fmt=lambda v: f"reaches z {v:.0f}", fontsize=10)
        for n, curves, col, link in sections(body, q, T):
            shell = "center_" in n
            for pts in curves:
                if len(pts) > 3 and not shell:
                    ax.add_patch(Polygon(pts, closed=True, fc=col, ec="k", lw=0.6, alpha=0.95))
                else:
                    ax.plot(pts[:, 0], pts[:, 1], color="#8a7f6a" if shell else "k", lw=2.0 if shell else 0.6)
        back = HEM_90["back_x59"] if x == 59 else HEM_90["back_x36"]
        ax.axhline(HEM_90["front"], xmin=0.5, xmax=1.0, color="#1f5fbf", lw=3)
        ax.axhline(back, xmin=0.0, xmax=0.5, color="#1f5fbf", lw=3)
        ax.text(105, HEM_90["front"] + 3, f"belly edge of the 90-deg design, z {HEM_90['front']:.0f}", ha="right",
                color="#1f5fbf", fontsize=11)
        ax.text(-105, back + 3, f"back edge, z {back:.0f}", color="#1f5fbf", fontsize=11)
        ax.plot(-CY, CZ, "r+", ms=18, mew=2)
        ax.set_xlim(-110, 110); ax.set_ylim(80, 330); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlabel("forward (-y) mm"); ax.set_ylabel("z mm")
        ax.set_title(f"side cut x = {x}, {lab}", fontsize=14)
    fig.colorbar(cs, ax=axs, shrink=0.7, label="highest z this leg point reaches over hip pitch -85..+40")
    fig.suptitle("Leg enclosure vs core shell. A leg surface placed at a point rises to the dashed height when the hip flexes "
                 "to -85 (front) or extends to +40 (back);\nthe core shell over that leg must end above it. Blue: the edges "
                 "of the shell carved for -90 (the '90-deg design'). Green = low, red = high.", fontsize=15, y=0.98)
    out = HERE / "waist-line.png"
    plt.savefig(out, dpi=60, bbox_inches="tight")
    print(out)
    for fwd, z, what in ((25, 207, "current thigh cover, top front corner"), (45, 230, "belly-flush thigh front at z 230"),
                         (45, 245, "belly-flush thigh front at z 245"), (20, 172, "pitch-to-roll connector front-bottom"),
                         (-23, 177, "pitch-to-roll connector back-bottom"), (-50, 200, "seat-flush thigh back at z 200")):
        print(f"{what:42s} standing (fwd {fwd}, z {z}) -> reaches z {max_height(np.array(fwd), np.array(z)):.0f}")


if __name__ == "__main__":
    main()
