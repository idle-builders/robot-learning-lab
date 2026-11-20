"""The owner's hip case (D-12) from all six sides, kept simple (owner 2026-11-16).

Standing, zero joint angles. Static core shell cut at z 271 (black outline); battery cradle, keel and cells (grey);
yaw link: yaw-to-pitch connector (green) and pitch motor (dark grey); pitch-to-roll connector and pitch output
(yellow); roll motor case (red) turned to hang down inside the thigh, its axis unchanged at z 190, horn and idler on
the yellow connector; thigh connector (blue) and knee motor (dark grey). The left leg mirrors the right.
X-ray: every part is drawn through the others (light fill, solid outline).

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sections/owner_case_views.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import kit  # noqa: E402
import drawing_sheets as DS  # noqa: E402
import roll_options as RO  # noqa: E402

OUT = DS.OUT
PX = 0.3
GREEN, YELLOW, RED, BLUE, MOTOR, STATIC = "#2e9e5b", "#e0a400", "#d62728", "#2f6fd0", "#555555", "#b5b5b5"
# view: (name, title, h(world xyz), v(world xyz), left/right/top/bottom labels, lims)
VIEWS = [
    ("front", "FRONT view - you stand in front of the robot", lambda p: -p[:, 0], lambda p: p[:, 2],
     ("robot's RIGHT", "robot's LEFT", "UP", "DOWN"), (-140, 140), (100, 300)),
    ("back", "BACK view - you stand behind the robot", lambda p: p[:, 0], lambda p: p[:, 2],
     ("robot's LEFT", "robot's RIGHT", "UP", "DOWN"), (-140, 140), (100, 300)),
    ("right", "RIGHT side view - you stand at the robot's right (right leg only)", lambda p: p[:, 1], lambda p: p[:, 2],
     ("FRONT", "BACK", "UP", "DOWN"), (-100, 100), (100, 300)),
    ("left", "LEFT side view of the right leg - you stand at the robot's left, left leg removed", lambda p: -p[:, 1],
     lambda p: p[:, 2], ("BACK", "FRONT", "UP", "DOWN"), (-100, 100), (100, 300)),
    ("top", "TOP view - you look down from above (core shell removed)", lambda p: -p[:, 0], lambda p: p[:, 1],
     ("robot's RIGHT", "robot's LEFT", "BACK", "FRONT"), (-140, 140), (-80, 80)),
    ("bottom", "BOTTOM view - you look up from below", lambda p: p[:, 0], lambda p: p[:, 1],
     ("robot's LEFT", "robot's RIGHT", "BACK", "FRONT"), (-140, 140), (-80, 80)),
]


def parts(body):
    """(name, mesh in world at standing, colour, is_motor) for the right side; mirrored copies for the left."""
    P = []
    for n, p in kit._hip_parts(body).items():
        link = p["owning_link"]
        role = p.get("role")
        b = np.array(p["bounds_mm"])
        if link.startswith("L_") or any(k in link for k in ("shoulder", "elbow", "arm", "wrist", "hand", "head", "neck")):
            continue
        if role == "body_shell" and "thigh_shell" in n:     # blue thigh shell (roll-flip manifest): outline only
            P.append((n, kit._mesh(body, n).copy(), "SHELL", False, "R")); continue
        if role == "body_shell" and n not in DS.TORSO_SHELLS:
            continue
        if role in ("fastener", "nominal_fastener", "fastener_proxy", "nominal_hardware", "standard_fastener_reference",
                    "compression_limit_sleeves", "spacer", "battery_heatsink_fin", "battery_insulator", "standard_nominal"):
            continue
        if link == "torso" and n not in DS.TORSO_SHELLS:
            if b[1][2] < 195 or b[0][2] > 300 or not ("cradle" in n or "Cell_0" in n or "yaw" in n.lower()):
                continue
        if link in ("R_shin", "R_foot", "R_ankle_link") or b[1][2] < 100:
            continue
        if n in DS.TORSO_SHELLS:
            import trimesh
            m = trimesh.intersections.slice_mesh_plane(kit._mesh(body, n), [0, 0, 1], [0, 0, DS.CORE_CUT_Z])
            P.append((n, m, None, False, "core")); continue
        m = kit._mesh(body, n).copy()
        if n == RO.CASE:
            if link.endswith("pitch_link"):              # the roll-flip manifest (HIP_MANIFEST) already holds it turned
                m = RO.case_mesh(body, -90.0, 0.0)        # roll motor case hanging down in the thigh (owner's red box)
            P.append((n, m, RED, False, "R")); continue
        if link == "torso":
            col, motor = (MOTOR, True) if "yaw" in n.lower() and "ST" in n or "servo" in n.lower() else (STATIC, False)
        elif link == "R_yaw_link":
            col, motor = (MOTOR, True) if role == "motor" else (GREEN, False)
        elif link == "R_pitch_link" or n.startswith("R_roll_upper_horn") or n.startswith("R_roll_passive"):
            col, motor = YELLOW, False
        elif link == "R_roll_link":
            col, motor = (MOTOR, True) if role == "motor" else (BLUE, False)
        else:
            continue
        P.append((n, m, col, motor, "R"))
    out = list(P)
    for n, m, col, motor, side in P:
        if side == "R":
            mm = m.copy(); mm.vertices[:, 0] *= -1; mm.faces = mm.faces[:, ::-1]
            out.append(("L" + n[1:], mm, col, motor, "L"))
    return out


def mask(m, fh, fv, hl, vl):
    from PIL import Image, ImageDraw
    W = int((hl[1] - hl[0]) / PX); H = int((vl[1] - vl[0]) / PX)
    u = (fh(m.vertices) - hl[0]) / PX; w = (vl[1] - fv(m.vertices)) / PX
    img = Image.new("1", (W, H), 0); d = ImageDraw.Draw(img)
    for f in m.faces:
        d.polygon([(u[f[0]], w[f[0]]), (u[f[1]], w[f[1]]), (u[f[2]], w[f[2]])], fill=1)
    return np.array(img, bool)


_MASKS = {}


def draw_view(ax, P, view):
    from matplotlib.patches import Polygon
    from skimage.measure import find_contours
    name, title, fh, fv, labs, hl, vl = view
    for n, m, col, motor, side in P:
        if name in ("right", "left") and side == "L":
            continue
        if name in ("top", "bottom") and col is None:
            continue
        key = (id(m), name)          # each view is drawn twice (own PNG + six-side sheet): rasterise once
        if key not in _MASKS:
            _MASKS[key] = mask(m, fh, fv, hl, vl)
        mk = _MASKS[key]
        if not mk.any():
            continue
        for c in find_contours(np.pad(mk.astype(float), 1), 0.5):
            if len(c) < 5:
                continue
            pts = np.c_[hl[0] + (c[:, 1] - 1) * PX, vl[1] - (c[:, 0] - 1) * PX]
            if col is None:                                   # core shell: bold outline only
                ax.plot(pts[:, 0], pts[:, 1], color="k", lw=2.4, zorder=6)
            elif col == "SHELL":                              # blue thigh shell: bold blue outline only
                ax.plot(pts[:, 0], pts[:, 1], color="#1f4fd6", lw=2.2, zorder=7)
            else:
                ax.add_patch(Polygon(pts, closed=True, fc=col, alpha=0.30 if not motor else 0.45, ec="none", zorder=2))
                ax.add_patch(Polygon(pts, closed=True, fill=False, ec=col, lw=1.6 if col != STATIC else 0.8, zorder=4))
    ax.set_xlim(*hl); ax.set_ylim(*vl); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#999")
    ax.set_title(title, fontsize=12)
    kw = dict(transform=ax.transAxes, fontsize=13, fontweight="bold", color="#444")
    ax.text(0.01, 0.5, labs[0], ha="left", va="center", rotation=90, **kw)
    ax.text(0.99, 0.5, labs[1], ha="right", va="center", rotation=90, **kw)
    ax.text(0.5, 0.99, labs[2], ha="center", va="top", **kw)
    ax.text(0.5, 0.01, labs[3], ha="center", va="bottom", **kw)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    body = kit.base(DS.SHELL_PASS)
    OUT.mkdir(exist_ok=True)
    P = parts(body)
    leg = [Patch(fc="none", ec="k", lw=2, label="core shell (cut at z 271)"),
           Patch(fc=GREEN, alpha=0.5, label="green: yaw-to-pitch connector"),
           Patch(fc=YELLOW, alpha=0.5, label="yellow: pitch-to-roll connector"),
           Patch(fc=RED, alpha=0.5, label="red: roll motor (in the thigh)"),
           Patch(fc=BLUE, alpha=0.5, label="blue: thigh connector"),
           Patch(fc=MOTOR, alpha=0.6, label="other motors"),
           Patch(fc=STATIC, alpha=0.6, label="battery cradle, keel, cells")]
    if any(c == "SHELL" for _, _, c, _, _ in P):
        leg.append(Patch(fc="none", ec="#1f4fd6", lw=2, label="blue thigh shell (outline)"))
    for view in VIEWS:
        hl, vl = view[5], view[6]
        fig, ax = plt.subplots(figsize=((hl[1] - hl[0]) / 22, (vl[1] - vl[0]) / 22 + 0.6))
        draw_view(ax, P, view)
        ax.legend(handles=leg, loc="lower left", fontsize=8, framealpha=0.9, bbox_to_anchor=(0.03, 0.05))
        plt.tight_layout(); plt.savefig(OUT / f"case-{view[0]}.png", dpi=120, facecolor="white"); plt.close(fig)
    fig, axs = plt.subplots(2, 3, figsize=(30, 17))
    for ax, view in zip(axs.ravel(), [VIEWS[0], VIEWS[2], VIEWS[4], VIEWS[1], VIEWS[3], VIEWS[5]]):
        draw_view(ax, P, view)
    axs[0, 0].legend(handles=leg, loc="lower left", fontsize=10, framealpha=0.9, bbox_to_anchor=(0.03, 0.05))
    fig.suptitle("Your hip case from all six sides, standing. Roll motor (red) hangs down inside the thigh, roll axis "
                 "unchanged.", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig(OUT / "case-all-sides.png", dpi=70, facecolor="white"); plt.close(fig)
    print(OUT / "case-all-sides.png")


if __name__ == "__main__":
    main()
