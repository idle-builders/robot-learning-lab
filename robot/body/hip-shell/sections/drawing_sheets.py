"""Drawing sheets for the owner (T-czv2d8, 2026-11-16): clean cuts through the right hip of the current FDM body with
the 90-degree shell (shell-pass: belly and back shells with leg openings cut by the swept hips), for sketching over.

Shell = bold black outline on ivory, motors = grey, printed connectors = pale blue, other internals = pale beige.
Red + = hip centre C (yaw and pitch axes meet), blue x = hip roll axis (29 mm below). Faint 10 mm grid.
One PNG per cut and pose in sections/drawing-sheets/, plus an overview.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sections/drawing_sheets.py
"""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import kit  # noqa: E402

SHELL_PASS = kit.ROOT / "artifacts/robot/body/ape-proportions/shell-pass/assembly-parts.json"
# HIP_MANIFEST=<path> renders another manifest, e.g. the roll-flip variant (assembly-parts-roll-flip.json, T-fs5v6d)
if os.environ.get("HIP_MANIFEST"):
    SHELL_PASS = kit.ROOT / os.environ["HIP_MANIFEST"]
OUT = HERE / "drawing-sheets" / ("roll-flip" if os.environ.get("HIP_MANIFEST") else "")
ROLL_PT = np.array([56.0, 0.0, 190.0])


def side(x):
    return np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, x], [0, 0, 0, 1]], float)


def front(y):
    return np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, -y], [0, 0, 0, 1]], float)


SHEETS = [  # file, title, transform, horizontal label, xlim, zlim, pose
    ("side-x59-standing", "Side cut x = 59 (through hip centre C), standing", side(59), "forward  ->  (mm)", (-130, 130), (90, 340), {}),
    ("side-x59-flexion85", "Side cut x = 59, hip flexion -85, knee 90", side(59), "forward  ->  (mm)", (-130, 130), (90, 340),
     {"R_hip_pitch": -85.0, "R_knee": 90.0}),
    ("side-x36-standing", "Side cut x = 36 (through the hip-pitch servo), standing", side(36), "forward  ->  (mm)", (-130, 130), (90, 340), {}),
    ("side-x36-flexion85", "Side cut x = 36, hip flexion -85, knee 90", side(36), "forward  ->  (mm)", (-130, 130), (90, 340),
     {"R_hip_pitch": -85.0, "R_knee": 90.0}),
    ("front-y2-standing", "Front cut y = 2 (through the hip pitch axis), standing", front(2), "robot right  <-   (mm)  ->  robot left",
     (-170, 170), (90, 340), {}),
    ("front-y2-legout40", "Front cut y = 2, leg out (hip roll -40)", front(2), "robot right  <-   (mm)  ->  robot left",
     (-170, 170), (90, 340), {"R_hip_roll": -40.0}),
]
HIDE = ("R_thigh_front_hard_cover", "R_thigh_back_hard_cover", "L_thigh_front_hard_cover", "L_thigh_back_hard_cover")
# owner 2026-11-16: thigh shells left out so the upper-leg shell can be redrawn from scratch; the torso core shell stays
MOTOR = {"motor", "purchased_servo_case_family_reference", "purchased_servo_output_family_reference", "output", "horn",
         "output_horn", "actuator"}
CONNECTOR = {"printed_structural_connector", "printed_structural_carrier", "bearing", "compression_limit_sleeves"}


def style(p):
    r = p.get("role")
    if r == "body_shell":
        return dict(fc="#f5efe4", ec="#111111", lw=2.6, z=3)
    if r in MOTOR:
        return dict(fc="#9a9a9a", ec="#3a3a3a", lw=0.9, z=2)
    if r in CONNECTOR:
        return dict(fc="#cfe0ee", ec="#4a6a85", lw=0.8, z=1)
    return dict(fc="#ece6da", ec="#8a8070", lw=0.6, z=0)


# owner 2026-11-16: the torso shell ends where the two core parts meet: the spine chassis (stage 1, z 266-388) sits
# on the lower-pelvis battery cradle (stage 2, z 209-276); the cut is at the middle of their 10 mm overlap
CORE_CUT_Z = 271.0
TORSO_SHELLS = ("center_front_shell", "center_back_shell")


def mesh(body, n):
    p = body.parts[n]
    if n in TORSO_SHELLS:
        import trimesh
        return trimesh.intersections.slice_mesh_plane(body.mesh(n), [0, 0, 1], [0, 0, CORE_CUT_Z])
    if p.get("role") == "body_shell":
        return body.mesh(n)                 # shell-pass shells at full resolution
    return kit._mesh(body, n, 20000)


def draw(ax, body, T, q):
    from matplotlib.patches import Polygon
    for n, p in kit._hip_parts(body).items():
        if any(k in p["owning_link"] for k in ("shoulder", "elbow", "arm", "wrist", "hand", "grip", "finger", "palm")):
            continue
        if n in HIDE:
            continue
        m = mesh(body, n).copy()
        m.apply_transform(body.link_T(p["owning_link"], q))
        m.apply_transform(T)
        if m.bounds[0][2] > 0 or m.bounds[1][2] < 0:
            continue
        s = m.section(plane_origin=[0, 0, 0], plane_normal=[0, 0, 1])
        if s is None:
            continue
        st = style(p)
        for pts in s.discrete:
            pts = pts[:, :2]
            closed = len(pts) > 3 and np.linalg.norm(pts[0] - pts[-1]) < 1e-3
            if closed:
                ax.add_patch(Polygon(pts, closed=True, fc=st["fc"], ec=st["ec"], lw=st["lw"], zorder=st["z"]))
            else:
                ax.plot(pts[:, 0], pts[:, 1], color=st["ec"], lw=st["lw"], zorder=st["z"])


def sheet(ax, body, spec, small=False):
    name, title, T, hl, xlim, zlim, dq = spec
    q = {**body.standing, **dq}
    draw(ax, body, T, q)
    Cp = (T @ np.r_[kit.HIP_C, 1.0])[:2]
    ax.plot(*Cp, "+", color="#d62728", ms=16, mew=2.5, zorder=5)
    Rw = body.link_T("R_pitch_link", q) @ np.r_[ROLL_PT, 1.0]
    Rp = (T @ Rw)[:2]
    ax.plot(*Rp, "x", color="#1f5fd6", ms=11, mew=2.2, zorder=5)
    ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal")
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 10), minor=True); ax.set_yticks(np.arange(zlim[0], zlim[1] + 1, 10), minor=True)
    ax.grid(which="minor", color="#e8e8e8", lw=0.5); ax.grid(which="major", color="#d0d0d0", lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11 if small else 15)
    if not small:
        ax.set_xlabel(hl, fontsize=11); ax.set_ylabel("height z (mm)", fontsize=11)
        ax.text(0.01, 0.01, f"black = torso shell (90-deg version) cut at z {CORE_CUT_Z:.0f}, thigh shells left out   grey = motors   blue = printed connectors   "
                "red + = hip centre C   blue x = roll axis", transform=ax.transAxes, fontsize=10, color="#444")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = kit.base(SHELL_PASS)
    OUT.mkdir(exist_ok=True)
    for spec in SHEETS:
        w = (spec[4][1] - spec[4][0]) / 20; h = (spec[5][1] - spec[5][0]) / 20
        fig, ax = plt.subplots(figsize=(w, h))
        sheet(ax, body, spec)
        plt.tight_layout(); plt.savefig(OUT / f"{spec[0]}.png", dpi=130, facecolor="white"); plt.close(fig)
        print(OUT / f"{spec[0]}.png", flush=True)
    fig, axs = plt.subplots(2, 3, figsize=(30, 17), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    order = [SHEETS[0], SHEETS[2], SHEETS[4], SHEETS[1], SHEETS[3], SHEETS[5]]
    for ax, spec in zip(axs.ravel(), order):
        sheet(ax, body, spec, small=True)
    fig.suptitle("Hip cuts, current FDM body with the 90-deg shell (shell-pass). Black = shell, grey = motors, blue = "
                 "printed connectors; red + = hip centre C, blue x = roll axis", fontsize=15)
    plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(OUT / "overview.png", dpi=70, facecolor="white"); plt.close(fig)


if __name__ == "__main__" and "--xray" not in sys.argv:
    main()


# ------------------------------------------------------------------ X-ray views (owner 2026-11-16: "move like X-rays")
LINK_COL = {"torso": "#7f8c99", "R_yaw_link": "#2e9e5b", "R_pitch_link": "#e08a1e", "R_roll_link": "#2f6fd0",
            "R_shin": "#8e5bc9"}
LINK_NAME = {"torso": "torso (static)", "R_yaw_link": "yaw link", "R_pitch_link": "pitch link",
             "R_roll_link": "thigh (roll link)", "R_shin": "shin"}
PX = 0.25   # mm per pixel of the projection masks


def xray(ax, body, T, q, xlim, zlim, right_only):
    """Every part projected through the body: translucent fill + outline. Shells: bold black outline only."""
    from PIL import Image, ImageDraw
    from skimage.measure import find_contours
    from matplotlib.patches import Polygon
    W = int((xlim[1] - xlim[0]) / PX); Hh = int((zlim[1] - zlim[0]) / PX)
    items = []
    for n, p in kit._hip_parts(body).items():
        link = p["owning_link"]
        if any(k in link for k in ("shoulder", "elbow", "arm", "wrist", "hand", "grip", "finger", "palm")):
            continue
        if right_only and (n.startswith("L_") or link.startswith("L_")):
            continue
        if n in HIDE:
            continue
        m = mesh(body, n) if p.get("role") == "body_shell" else kit._mesh(body, n)
        v = m.vertices @ body.link_T(link, q)[:3, :3].T + body.link_T(link, q)[:3, 3]
        v = v @ T[:3, :3].T + T[:3, 3]
        u = (v[:, 0] - xlim[0]) / PX; w = (zlim[1] - v[:, 1]) / PX
        if u.max() < 0 or u.min() > W or w.max() < 0 or w.min() > Hh:
            continue
        img = Image.new("1", (W, Hh), 0); d = ImageDraw.Draw(img)
        for f in m.faces:
            d.polygon([(u[f[0]], w[f[0]]), (u[f[1]], w[f[1]]), (u[f[2]], w[f[2]])], fill=1)
        mask = np.array(img, float)
        if mask.sum() < 4:
            continue
        cs = find_contours(np.pad(mask, 1), 0.5)
        items.append((n, p, link, cs, float(v[:, 2].mean())))
    for n, p, link, cs, dep in sorted(items, key=lambda t: t[4]):
        role = p.get("role")
        col = LINK_COL.get(link, "#999999")
        if role == "body_shell":
            fc, ec, lw, a, z = "#f5efe4", "#111111", 2.4, 0.10, 6
        elif role in MOTOR:
            fc, ec, lw, a, z = "#808080", "#2a2a2a", 1.0, 0.35, 4
        elif role in CONNECTOR:
            fc, ec, lw, a, z = col, col, 1.3, 0.22, 5
        else:
            fc, ec, lw, a, z = "#d8d0c0", "#8a8070", 0.6, 0.15, 3
        for c in cs:
            if len(c) < 4:
                continue
            pts = np.c_[xlim[0] + (c[:, 1] - 1) * PX, zlim[1] - (c[:, 0] - 1) * PX]
            ax.add_patch(Polygon(pts, closed=True, fc=fc, alpha=a, ec="none", zorder=z))
            ax.add_patch(Polygon(pts, closed=True, fill=False, ec=ec, lw=lw, alpha=0.95, zorder=z + 0.5))


def xray_main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    body = kit.base(SHELL_PASS)
    OUT.mkdir(exist_ok=True)
    for name, title, T, hl, xlim, zlim, dq in SHEETS:
        if "-x36-" in name:
            continue                       # an X-ray shows the whole depth; one side view per pose is enough
        side_view = name.startswith("side")
        q = {**body.standing, **dq}
        w = (xlim[1] - xlim[0]) / 20; h = (zlim[1] - zlim[0]) / 20
        fig, ax = plt.subplots(figsize=(w, h))
        xray(ax, body, T, q, xlim, zlim, right_only=side_view)
        Cp = (T @ np.r_[kit.HIP_C, 1.0])[:2]
        Rp = (T @ (body.link_T("R_pitch_link", q) @ np.r_[ROLL_PT, 1.0]))[:2]
        ax.plot(*Cp, "+", color="#d62728", ms=18, mew=2.8, zorder=10)
        ax.plot(*Rp, "x", color="#1f5fd6", ms=12, mew=2.4, zorder=10)
        ax.set_xlim(*xlim); ax.set_ylim(*zlim); ax.set_aspect("equal")
        ax.set_xticks(np.arange(xlim[0], xlim[1] + 1, 10), minor=True); ax.set_yticks(np.arange(zlim[0], zlim[1] + 1, 10), minor=True)
        ax.grid(which="minor", color="#ececec", lw=0.5); ax.grid(which="major", color="#d4d4d4", lw=0.7); ax.set_axisbelow(True)
        what = "right leg and torso" if side_view else "both legs"
        pose = title.split(", ", 1)[1] if ", " in title else "standing"
        ax.set_title(f"X-ray {'side' if side_view else 'front'} view ({what}, every part projected through), {pose}", fontsize=14)
        ax.set_xlabel(hl, fontsize=11); ax.set_ylabel("height z (mm)", fontsize=11)
        leg = [Patch(fc="#f5efe4", ec="#111111", lw=2, label=f"torso shell (90-deg version) cut at z {CORE_CUT_Z:.0f}; thigh shells left out"),
               Patch(fc="#808080", ec="#2a2a2a", alpha=0.6, label="motor")]
        leg += [Patch(fc=c, ec=c, alpha=0.4, label=f"connector, {LINK_NAME[k]}") for k, c in LINK_COL.items()]
        ax.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.85)
        plt.tight_layout(); f = OUT / f"xray-{name.replace('-x59', '')}.png"
        plt.savefig(f, dpi=130, facecolor="white"); plt.close(fig)
        print(f, flush=True)


if __name__ == "__main__" and "--xray" in sys.argv:
    xray_main()
