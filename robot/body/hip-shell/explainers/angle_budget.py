"""Explainer for A-08 (DECISIONS.md): the angle budget of a hip globe in the side view (pitch plane), from the real
skeleton cuts. A globe of radius R around C is crossed by the torso-side internals at the top (the "torso neck",
static under pitch) and by the leg-side internals at the bottom (the "thigh neck", turning with pitch). The rest
of the circle is the globe's skin: a front span and a back span. Hip pitch changes those spans by the pitch angle:
flexion -85 shortens the front span by 85 deg and lengthens the back one; extension +40 does the reverse. A rigid
skin piece cannot change length, so each span has to be split into nested bands that slide over each other; the
number of bands is at least the ratio of the longest to the shortest span (plus overlap).

Run: python3 infra/tools/memguard.py --limit-gb 4 -- .venv/bin/python -B robot/body/hip-shell/explainers/angle_budget.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "sections"))
import kit  # noqa: E402
from joint_cuts import JOINTS, analyse, draw, side  # noqa: E402

FLEX, EXT = -85.0, 40.0
CLEAR = 2.0          # mm clearance around internals where they cross the globe
OVERLAP = 0.15       # each nested band overlaps its neighbour by 15 % of its length
RADII = np.arange(36.0, 72.1, 4.0)
CUTS = [("x = 59 (great circle through C)", 59.0), ("x = 44", 44.0), ("x = 74", 74.0)]


def sectors(mask, phis):
    """Contiguous angle runs where mask is True (phi in degrees, 0 = up, + = front)."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if m else i - 1
            runs.append((phis[start], phis[end]))
            start = None
    return runs


def budget(stat, mov, r, ap):
    """Torso-neck and thigh-neck sectors on the circle of radius r about ap, and the front/back skin spans."""
    from shapely.geometry import Point
    phis = np.arange(-180.0, 180.0, 0.5)
    pts = [Point(ap[0] + r * np.sin(np.radians(p)), ap[1] + r * np.cos(np.radians(p))) for p in phis]
    S = stat.buffer(CLEAR) if stat is not None else None
    M = mov.buffer(CLEAR) if mov is not None else None
    ms = np.array([S is not None and S.contains(p) for p in pts])
    mm = np.array([M is not None and M.contains(p) for p in pts])
    top = phis[ms & (np.abs(phis) < 90)]          # torso neck: static internals crossing in the upper half
    bot = phis[mm & (np.abs(phis) > 60)]          # thigh neck: leg internals crossing in the lower part
    tn = (top.min(), top.max()) if len(top) else (0.0, 0.0)
    if len(bot):
        b = np.where(bot < 0, bot + 360, bot)       # unwrap around 180
        bn = (b.min(), b.max())
    else:
        bn = (180.0, 180.0)
    front0 = bn[0] - tn[1]                         # from the torso neck's front edge down to the thigh neck's front edge
    back0 = (tn[0] + 360) - bn[1]                  # from the thigh neck's back edge up to the torso neck's back edge
    spans = dict(front={0: front0, FLEX: front0 + FLEX, EXT: front0 + EXT},
                 back={0: back0, FLEX: back0 - FLEX, EXT: back0 - EXT})
    need = {}
    for k, s in spans.items():
        lo, hi = min(s.values()), max(s.values())
        need[k] = float("inf") if lo <= 0 else int(np.ceil(hi / lo * (1 + OVERLAP)))
    return dict(r=float(r), torso_neck_deg=[float(tn[0]), float(tn[1])], thigh_neck_deg=[float(bn[0]), float(bn[1])],
                front_span_deg={str(k): float(v) for k, v in spans["front"].items()},
                back_span_deg={str(k): float(v) for k, v in spans["back"].items()},
                bands_needed=need)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    body = kit.base()
    spec = dict(JOINTS["pitch"]); spec["cuts"] = [(t, side(x)) for t, x in CUTS]
    _, _, res = analyse(body, spec)
    out = {}
    fig, axs = plt.subplots(2, 3, figsize=(30, 19))
    for k, r in enumerate(res):
        ap = r["ap"]; title = r["title"]
        dx = abs(CUTS[k][1] - 59.0)
        rows = []
        for R in RADII:
            if R <= dx + 5:
                continue
            rc = float(np.sqrt(R ** 2 - dx ** 2))   # the globe's circle in this cut
            b = budget(r["S"], r["M0"], rc, ap); b["globe_R"] = float(R); rows.append(b)
        out[title] = rows
        ax = axs[0, k]
        draw(ax, r["S"], fc="#9a9a9a", ec="k", lw=0.4)
        draw(ax, r["M0"], fc=(0.35, 0.45, 0.8, 0.35), ec=(0.2, 0.3, 0.6), lw=1.0)
        show = next((x for x in rows if x["globe_R"] == 48.0), rows[len(rows) // 2])
        rc = show["r"]
        tn, bn = show["torso_neck_deg"], show["thigh_neck_deg"]

        def wedge(a0, a1, col, lab):
            # matplotlib wedge angles: from +x counter-clockwise; ours: 0 = up, + = towards +u (front)
            th0, th1 = 90 - a1, 90 - a0
            ax.add_patch(Wedge(ap, rc + 6, th0, th1, width=12, fc=col, ec="none", alpha=0.8, label=lab))
        wedge(tn[0], tn[1], "#7f7f7f", f"torso neck {tn[1] - tn[0]:.0f} deg")
        wedge(bn[0], bn[1], "#3355aa", f"thigh neck {bn[1] - bn[0]:.0f} deg (turns with pitch)")
        wedge(tn[1], bn[0], "#2ca02c", f"front skin {show['front_span_deg']['0']:.0f} deg standing")
        wedge(bn[1] - 360, tn[0], "#ff7f0e", f"back skin {show['back_span_deg']['0']:.0f} deg standing")
        ax.add_patch(Circle(ap, rc, fill=False, ec="k", lw=1.5, ls="--"))
        ax.plot(*ap, "r+", ms=18, mew=3)
        f, bk = show["front_span_deg"], show["back_span_deg"]
        ax.text(ap[0] - 105, ap[1] - 118,
                f"globe R {show['globe_R']:.0f} (circle r {rc:.0f} in this cut)\n"
                f"front skin: {f['0']:.0f} deg standing, {f[str(FLEX)]:.0f} at {FLEX:.0f}, {f[str(EXT)]:.0f} at {EXT:+.0f}"
                f"  -> {show['bands_needed']['front']} nested bands\n"
                f"back skin: {bk['0']:.0f} deg standing, {bk[str(FLEX)]:.0f} at {FLEX:.0f}, {bk[str(EXT)]:.0f} at {EXT:+.0f}"
                f"  -> {show['bands_needed']['back']} nested bands", fontsize=13, va="top",
                bbox=dict(fc="white", ec="#999"))
        ax.legend(loc="upper right", fontsize=11)
        ax.set_xlim(-115, 115); ax.set_ylim(70, 335); ax.set_aspect("equal"); ax.grid(alpha=0.2)
        ax.set_xlabel("forward (-y) mm"); ax.set_ylabel("z mm")
        ax.set_title(f"{title}: where the skeleton crosses a hip globe", fontsize=14)
        ax = axs[1, k]
        Rs = [x["globe_R"] for x in rows]
        for key, col in (("front", "#2ca02c"), ("back", "#ff7f0e")):
            ax.plot(Rs, [min(x["bands_needed"][key], 12) for x in rows], "o-", color=col, lw=3, label=f"{key} skin")
        ax.set_xlabel("globe radius R (mm)", fontsize=12); ax.set_ylabel("nested bands needed (capped at 12)", fontsize=12)
        ax.grid(alpha=0.3); ax.legend(fontsize=12)
        ax2 = ax.twinx()
        ax2.plot(Rs, [x["torso_neck_deg"][1] - x["torso_neck_deg"][0] for x in rows], "--", color="#7f7f7f", label="torso neck")
        ax2.plot(Rs, [x["thigh_neck_deg"][1] - x["thigh_neck_deg"][0] for x in rows], "--", color="#3355aa", label="thigh neck")
        ax2.set_ylabel("neck width (deg)", fontsize=12); ax2.legend(loc="center right", fontsize=11)
        ax.set_title(f"{title}: bands needed and neck widths vs globe radius", fontsize=14)
    fig.suptitle("A-08: angle budget of a hip globe in the side view (hip pitch -85..+40). The two necks are where the "
                 "skeleton crosses the globe; the skin between them must stretch and shrink by the pitch angle,\n"
                 f"so a hard skin needs nested bands. Bands needed = ceil(longest / shortest span x {1 + OVERLAP:.2f}). "
                 "Bare skeleton, today's actuators, 2 mm clearance.", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    png = HERE / "angle-budget.png"
    plt.savefig(png, dpi=55)
    (HERE / "angle-budget.json").write_text(json.dumps(out, indent=1, default=str))
    print(png)
    for t, rows in out.items():
        for x in rows:
            print(f"{t:32s} R {x['globe_R']:4.0f}  torso neck {x['torso_neck_deg']}  thigh neck {x['thigh_neck_deg']}  "
                  f"front {x['front_span_deg']}  back {x['back_span_deg']}  bands {x['bands_needed']}")


if __name__ == "__main__":
    main()
