"""Space for the owner's two hip shells (D-12), drawn in each shell's own frame as exact cuts (Idle T-czv2d8).

Green shell = yaw link (one per leg; holds the hip-pitch motor; its top slides inside the static core, cut at z 271).
Blue shell  = thigh (roll link; holds the knee motor and, flipped, the roll motor: case on the thigh, horn and idler
               bolted to the pitch-to-roll connector; roll axis unchanged at z 190).
Yellow      = pitch link (pitch-to-roll connector, pitch output, and after the flip the roll horn and idler).

For each shell frame, the parts that move with that shell are drawn solid; every other part is swept over the 151
realistic whole-leg poses (A-11) and shown as the region it occupies in that frame (X-ray: projected through all
depths would overstate it, so each part is cut by the sheet's plane in every pose and the sections are unioned).
A shell may go wherever no swept colour is; the yellow region on the green sheets is where the slits go.

Run: python3 infra/tools/memguard.py --limit-gb 6 -- .venv/bin/python -B robot/body/hip-shell/sections/shell_frames.py
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
import kit  # noqa: E402
import drawing_sheets as DS  # noqa: E402
from leg_motion import library_poses, q_of  # noqa: E402

OUT = HERE / "drawing-sheets"
FLIP = {"R_roll_servo_case_reference": "R_roll_link",          # roll motor case rides with the thigh
        "R_roll_upper_horn_reference": "R_pitch_link",          # its horn and idler bolt to the yellow connector
        "R_roll_passive_idler_reference": "R_pitch_link"}
GROUP = {"torso": "static core and the left leg", "L_": "static core and the left leg", "R_yaw_link": "green (yaw link)",
         "R_pitch_link": "yellow (pitch-to-roll connector)", "R_roll_link": "blue (thigh)", "R_shin": "shin and foot",
         "R_foot": "shin and foot"}
COL = {"static core and the left leg": "#8c8c8c", "green (yaw link)": "#2e9e5b", "yellow (pitch-to-roll connector)": "#e0a400",
       "blue (thigh)": "#2f6fd0", "shin and foot": "#8e5bc9"}
PX = 0.4


def group(link):
    for k, g in GROUP.items():
        if link == k or (k.endswith("_") and link.startswith(k)):
            return g
    return "static core and the left leg"


def parts(body):
    out = {}
    for n, p in kit._hip_parts(body).items():
        link = FLIP.get(n, p["owning_link"])
        if any(k in link for k in ("shoulder", "elbow", "arm", "wrist", "hand", "grip", "finger", "palm", "head", "neck")):
            continue
        if n in DS.HIDE or p.get("role") == "body_shell" and n not in DS.TORSO_SHELLS:
            continue                                   # no leg shells: they are what is being drawn
        b = np.array(p["bounds_mm"])
        if np.linalg.norm(np.clip(kit.HIP_C, b[0], b[1]) - kit.HIP_C) > 120 and not n.startswith("L_"):
            continue                                   # far from the right hip
        out[n] = (p, link)
    return out


def raster(body, n, p, link, q, frame, view, lim, mask=None, plane=0.0):
    """Cut one part (posed, moved into the shell frame) by the sheet's plane and OR the section into a mask."""
    from PIL import Image, ImageDraw
    m = _MESH.get(n)
    if m is None:
        m = kit._mesh(body, n)
        if n in DS.TORSO_SHELLS:
            import trimesh
            m = trimesh.intersections.slice_mesh_plane(m, [0, 0, 1], [0, 0, DS.CORE_CUT_Z])
        _MESH[n] = m
    Tw = body.link_T(link, q)
    Tf = np.linalg.inv(body.link_T(frame, q))
    T = Tf @ Tw
    v = m.vertices @ T[:3, :3].T + T[:3, 3]
    W = int((lim[0][1] - lim[0][0]) / PX); Hh = int((lim[1][1] - lim[1][0]) / PX)
    img = Image.new("1", (W, Hh), 0); d = ImageDraw.Draw(img)
    ax_i = 0 if view == "side" else 1                    # cut plane normal: x for side cuts, y for front cuts
    if v[:, ax_i].min() > plane or v[:, ax_i].max() < plane:
        return mask if mask is not None else np.array(img, bool)
    import trimesh
    mm = trimesh.Trimesh(v, m.faces, process=False)
    nrm = np.eye(3)[ax_i]
    try:
        sec = mm.section(plane_origin=nrm * plane, plane_normal=nrm)
    except Exception:  # noqa: BLE001
        sec = None
    if sec is not None:
        loops = [sec.vertices[e.points] for e in sec.entities]   # discrete drops open polylines of open meshes
        for loop in loops:
            h = -loop[:, 1] if view == "side" else -loop[:, 0]   # side: forward right; front: robot right left
            u = (h - lim[0][0]) / PX; w = (lim[1][1] - loop[:, 2]) / PX
            closed = len(loop) > 2 and np.linalg.norm(loop[0] - loop[-1]) < 1e-3
            if closed:
                d.polygon(list(zip(u, w)), fill=1)
            d.line(list(zip(u, w)), fill=1, width=3)           # open sections of open meshes still show
    a = np.array(img, bool)
    return a if mask is None else (mask | a)


_MESH = {}
SHEETS = [("green", "R_yaw_link", "side", 59.0, "x 59, through the hip centre"),
          ("green", "R_yaw_link", "side", 36.0, "x 36, through the pitch motor"),
          ("green", "R_yaw_link", "side", 78.0, "x 78, outboard of the pitch output"),
          ("green", "R_yaw_link", "front", 2.0, "y 2, through the pitch axis"),
          ("blue", "R_roll_link", "side", 59.0, "x 59, through the hip centre"),
          ("blue", "R_roll_link", "side", 40.0, "x 40, through the roll and knee motors"),
          ("blue", "R_roll_link", "front", 2.0, "y 2, through the pitch axis"),
          ("blue", "R_roll_link", "front", 10.0, "y 10, through the roll motor middle")]
LIM = {"side": ((-130, 130), (60, 330)), "front": ((-170, 120), (60, 330))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon
    from skimage.measure import find_contours
    body = kit.base(DS.SHELL_PASS)
    P = parts(body)
    poses = [q_of(body, p) for fam, p, end in library_poses()]
    q0 = q_of(body, (0, 0, 0, 0))
    for shell, frame, view, plane, plane_lab in SHEETS:
        lim = LIM[view]
        own = [n for n, (p, l) in P.items() if l == frame]
        swept = {}
        for n, (p, l) in P.items():
            if l == frame:
                continue
            g = group(l)
            if shell == "green" and n in DS.TORSO_SHELLS:
                continue                                 # the green shell slides inside the core shell (D-12)
            mk = swept.get(g)
            for q in poses:
                mk = raster(body, n, p, l, q, frame, view, lim, mk, plane)
            swept[g] = mk
        w = (lim[0][1] - lim[0][0]) / 20; h = (lim[1][1] - lim[1][0]) / 20
        fig, ax = plt.subplots(figsize=(w, h))
        ext = (lim[0][0], lim[0][1], lim[1][0], lim[1][1])
        order = ["static core and the left leg", "shin and foot", "blue (thigh)", "green (yaw link)",
                 "yellow (pitch-to-roll connector)"]
        for g in order:
            if g not in swept or swept[g] is None:
                continue
            rgba = np.zeros(swept[g].shape + (4,))
            c = matplotlib.colors.to_rgb(COL[g]); rgba[..., :3] = c; rgba[..., 3] = swept[g] * 0.28
            ax.imshow(rgba, extent=ext, origin="upper", zorder=1, interpolation="nearest")
            for cc in find_contours(np.pad(swept[g].astype(float), 1), 0.5):
                pts = np.c_[lim[0][0] + (cc[:, 1] - 1) * PX, lim[1][1] - (cc[:, 0] - 1) * PX]
                ax.plot(pts[:, 0], pts[:, 1], color=COL[g], lw=1.3, zorder=2)
        # the shell's own parts, solid, at the neutral pose
        for n in own:
            p, l = P[n]
            mk = raster(body, n, p, l, q0, frame, view, lim, None, plane)
            for cc in find_contours(np.pad(mk.astype(float), 1), 0.5):
                pts = np.c_[lim[0][0] + (cc[:, 1] - 1) * PX, lim[1][1] - (cc[:, 0] - 1) * PX]
                role = p.get("role")
                fc = "#7a7a7a" if role in DS.MOTOR else COL["green (yaw link)" if shell == "green" else "blue (thigh)"]
                ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="#222222", lw=1.0, alpha=0.75, zorder=4))
        if shell == "green":                              # core shell at standing, dashed (green slides inside)
            for n in DS.TORSO_SHELLS:
                p, l = P[n]
                mk = raster(body, n, p, l, q0, frame, view, lim, None, plane)
                for cc in find_contours(np.pad(mk.astype(float), 1), 0.5):
                    pts = np.c_[lim[0][0] + (cc[:, 1] - 1) * PX, lim[1][1] - (cc[:, 0] - 1) * PX]
                    ax.plot(pts[:, 0], pts[:, 1], color="k", lw=1.4, ls="--", zorder=5)
        C = kit.HIP_C
        hc = -C[1] if view == "side" else -C[0]
        ax.plot(hc, C[2], "+", color="#d62728", ms=18, mew=2.8, zorder=6)
        rp = DS.ROLL_PT
        if frame == "R_roll_link" or True:
            hr = -rp[1] if view == "side" else -rp[0]
            ax.plot(hr, rp[2], "x", color="#1f5fd6", ms=12, mew=2.4, zorder=6)
        ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_aspect("equal")
        ax.set_xticks(np.arange(lim[0][0], lim[0][1] + 1, 10), minor=True)
        ax.set_yticks(np.arange(lim[1][0], lim[1][1] + 1, 10), minor=True)
        ax.grid(which="minor", color="#ececec", lw=0.5); ax.grid(which="major", color="#d4d4d4", lw=0.7)
        ax.set_axisbelow(True)
        what = ("GREEN shell frame (turns with hip yaw; holds the pitch motor)" if shell == "green"
                else "BLUE shell frame (thigh; holds the knee motor and the flipped roll motor)")
        ax.set_title(f"{what}\n{view} cut at {plane_lab}. Solid = parts that move with this shell (standing). "
                     f"Tinted = where other parts pass through this cut over the 151 realistic poses."
                     + ("\nDashed = core shell at standing (cut at z 271); the green top slides inside it." if shell == "green" else ""),
                     fontsize=12)
        ax.set_xlabel("forward  ->  (mm)" if view == "side" else "robot right  <-   (mm)  ->  robot left")
        ax.set_ylabel("height z (mm)")
        leg = [Patch(fc=COL[g], alpha=0.4, ec=COL[g], label=f"swept: {g}") for g in order if g in swept and
               g != ("green (yaw link)" if shell == "green" else "blue (thigh)")]
        leg += [Patch(fc="#7a7a7a", ec="#222", label="motor of this shell"),
                Patch(fc=COL["green (yaw link)" if shell == "green" else "blue (thigh)"], ec="#222", alpha=0.75,
                      label="connector of this shell")]
        ax.legend(handles=leg, loc="lower left", fontsize=9, framealpha=0.9)
        plt.tight_layout()
        f = OUT / f"space-{shell}-{view}-{'x' if view == 'side' else 'y'}{plane:.0f}.png"
        plt.savefig(f, dpi=130, facecolor="white"); plt.close(fig)
        print(f, flush=True)


if __name__ == "__main__":
    main()
