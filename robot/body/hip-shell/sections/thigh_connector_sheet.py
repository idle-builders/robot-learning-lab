"""Owner drawing sheet of the roll-flip thigh connector (R_thigh_roll_flip): front (from -y), side (from +x), top (from +z),
section A-A at x 45.75 (inboard ear-screw column) and section B-B at z 143 (lateral knee screws), parts in their assembly
colours with the generated screws, used / removed screw sites marked, the blue thigh shell dashed in the sections, and an
info box (measurements typed in the sheet's JSON from the check reports).

  .venv/bin/python -B robot/body/hip-shell/sections/thigh_connector_sheet.py \
      artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json \
      robot/body/hip-shell/sections/drawing-sheets/roll-flip/thigh-connector-h-cage.png \
      robot/body/hip-shell/sections/thigh-connector-h-cage.json
(derived; recreate with this command)"""
import sys, json
from pathlib import Path
import numpy as np, trimesh
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, Rectangle
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import bodylib as BL
man_f, out_png, info_f = sys.argv[1], sys.argv[2], sys.argv[3]
INFO = json.loads(Path(info_f).read_text())
man = json.loads(Path(man_f).read_text())
P = {p["name"]: p for p in man["parts"]}
CONN = P["R_thigh_roll_flip"]["mesh"]
COL = dict(blue=(0.25, 0.55, 0.85), yellow=(0.93, 0.76, 0.16), red=(0.82, 0.22, 0.20), dark=(0.22, 0.23, 0.26), grey=(0.62, 0.64, 0.67),
           light=(0.80, 0.81, 0.83), steel=(0.85, 0.87, 0.90), shell=(0.55, 0.70, 0.95))
def M(path): return trimesh.load(ROOT / path, force="mesh") if not str(path).endswith(".npz") else BL.load_mesh(path)
parts = [  # (label, mesh, colour, alpha, z-bias)
    ("connector", M(CONN), COL["blue"], 1.0),
    ("connector sleeves", M(P["R_thigh_roll_flip_steel_sleeves"]["mesh"]), COL["steel"], 1.0),
    ("screws", trimesh.util.concatenate([BL.load_mesh(P[n]["mesh"]) for n in P if n.startswith("R_") and "screw_plan" in P[n] and P[n]["owning_link"] == "R_roll_link"]), (0.55, 0.57, 0.6), 1.0),
    ("roll case STS3250", BL.load_mesh(P["R_roll_servo_case_reference"]["mesh"]), COL["red"], 1.0),
    ("knee case ST3025", BL.load_mesh(P["R_knee_ST3025_case"]["mesh"]), COL["dark"], 1.0),
    ("Stage 5 (yellow)", BL.load_mesh(P["R_hip_pitch_to_roll_roll_flip"]["mesh"]), COL["yellow"], 0.55),
    ("roll horn / idler", None, COL["light"], 0.55),
    ("shin frame", BL.load_mesh(P["R_shin_combined"]["mesh"]), COL["grey"], 0.85),
    ("knee output", BL.load_mesh(P["R_knee_ST3025_output"]["mesh"]), COL["light"], 0.85),
]
hi = trimesh.util.concatenate([BL.load_mesh(P[n]["mesh"]) for n in ("R_roll_upper_horn_reference", "R_roll_passive_idler_reference")])
parts[6] = ("roll horn / idler", hi, COL["light"], 0.55)
CLIP = np.array([[20, -26, 108], [86, 40, 214]])
VIEWS = {"front": (1, (0, 2), -1, "FRONT (from -y): x lateral ->", False), "side": (0, (1, 2), +1, "SIDE (from +x, lateral): y rear ->", False),
         "top": (2, (0, 1), +1, "TOP (from +z): x lateral ->, y rear ^", False)}   # sgn: viewer direction along the depth axis
L = np.array([-0.4, -0.5, 0.75]); L /= np.linalg.norm(L)
def render(ax, view):
    dep_ax, (a, b), sgn, title, _ = VIEWS[view]
    polys, cols, deps = [], [], []
    for lab, m, c, al in parts:
        V, F = m.vertices, m.faces
        keep = np.all((V[F].mean(1) >= CLIP[0]) & (V[F].mean(1) <= CLIP[1]), axis=1)
        F = F[keep]; tri = V[F]; n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nn = np.linalg.norm(n, axis=1); ok = nn > 1e-9
        tri, n = tri[ok], n[ok] / nn[ok, None]
        shade = 0.5 + 0.5 * np.abs(n @ L)
        d = sgn * tri[:, :, dep_ax].mean(1)          # larger = nearer the viewer; drawn last
        polys.append(tri[:, :, [a, b]]); deps.append(d)
        cols.append(np.c_[np.clip(np.outer(shade, c), 0, 1), np.full(len(tri), al)])
    P_ = np.concatenate(polys); C_ = np.concatenate(cols); D_ = np.concatenate(deps); o = np.argsort(D_)
    ax.add_collection(PolyCollection(P_[o], facecolors=C_[o], edgecolors="none", antialiased=False))
    ax.set_xlim(CLIP[0][a], CLIP[1][a]); ax.set_ylim(CLIP[0][b], CLIP[1][b]); ax.set_aspect("equal"); ax.set_title(title, fontsize=9)
    ax.set_xlabel("xyz"[a] + " mm", fontsize=7); ax.set_ylabel("xyz"[b] + " mm", fontsize=7); ax.tick_params(labelsize=6); ax.grid(alpha=0.25, lw=0.4)
def section(ax, axis, value, uv, lims, title):
    for lab, m, c, al in parts:
        s = m.section(plane_origin=[value if axis == i else 0 for i in range(3)], plane_normal=[1 if axis == i else 0 for i in range(3)])
        if s is None: continue
        for ent in s.discrete:
            ent = np.asarray(ent)
            ax.fill(ent[:, uv[0]], ent[:, uv[1]], color=c, alpha=0.9 if al == 1 else 0.6, lw=0.3, ec="k")
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1]); ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("xyz"[uv[0]] + " mm", fontsize=7); ax.set_ylabel("xyz"[uv[1]] + " mm", fontsize=7); ax.tick_params(labelsize=6)
fig = plt.figure(figsize=(17, 11.5), dpi=130)
gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.12)
axs = {"front": fig.add_subplot(gs[0, 0]), "side": fig.add_subplot(gs[0, 1]), "top": fig.add_subplot(gs[0, 2]),
       "secx": fig.add_subplot(gs[1, 0]), "secz": fig.add_subplot(gs[1, 1]), "info": fig.add_subplot(gs[1, 2])}
for v in VIEWS: render(axs[v], v)
section(axs["secx"], 0, 45.75, (1, 2), ((-22, 36), (135, 205)), "SECTION A-A at x 45.75 (inboard ear-screw column), seen from +x")
section(axs["secz"], 2, 143.0, (0, 1), ((22, 86), (-22, 36)), "SECTION B-B at z 143 (lateral knee screws), seen from +z")
# screw sites
used, gone = INFO["screws_used"], INFO["screws_removed"]
def mark(ax, pts, used_):
    for (u, v) in pts:
        if used_: ax.add_patch(Circle((u, v), 1.75, fc="#18a558", ec="k", lw=0.6, zorder=20))
        else: ax.plot(u, v, marker="x", ms=7, mew=2, color="#e0201b", zorder=20)
for k, ax_name, ij in (("roll_front", "front", (0, 2)), ("roll_rear", "front", (0, 2)), ("knee_med", "side", (1, 2)), ("knee_lat", "side", (1, 2))):
    for flag, src in ((True, used), (False, gone)):
        pts = [(p[ij[0]], p[ij[1]]) for p in src.get(k, [])]; mark(axs[ax_name], pts, flag)
SHELLS = [BL.load_mesh(P[n]["mesh"]) for n in ("R_thigh_shell_front", "R_thigh_shell_back")]
def shell_outline(ax, axis, value, uv):
    for m in SHELLS:
        sct = m.section(plane_origin=[value if axis == i else 0 for i in range(3)], plane_normal=[1 if axis == i else 0 for i in range(3)])
        if sct is None: continue
        for ent in sct.discrete:
            ent = np.asarray(ent); ax.plot(ent[:, uv[0]], ent[:, uv[1]], color="#3d7cc9", lw=0.9, ls="--", zorder=15)
shell_outline(axs["secx"], 0, 45.75, (1, 2)); shell_outline(axs["secz"], 2, 143.0, (0, 1))
for ax_name, txts in INFO.get("callouts", {}).items():
    for t in txts:
        axs[ax_name].annotate(t["text"], xy=t["xy"], xytext=t["at"], fontsize=7, color="k", zorder=30,
                              arrowprops=dict(arrowstyle="->", lw=0.7, color="k"), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", lw=0.5, alpha=0.9))
ax = axs["info"]; ax.axis("off")
ax.text(0, 1.0, INFO["title"], fontsize=12, weight="bold", va="top", transform=ax.transAxes)
ax.text(0, 0.93, INFO["body"], fontsize=8, va="top", family="monospace", transform=ax.transAxes)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hand = [Patch(color=COL["blue"], label="R_thigh_roll_flip (blue, roll link)"), Patch(color=COL["yellow"], label="Stage 5 yoke (yellow, pitch link, moves)"),
        Patch(color=COL["red"], label="STS3250 roll case (red)"), Patch(color=COL["dark"], label="ST3025 knee case (dark grey)"),
        Patch(color=COL["grey"], label="shin frame (grey, moves)"), Patch(color=COL["light"], label="horn / idler / knee output"),
        Line2D([], [], marker="o", ls="", mfc="#18a558", mec="k", label="screw used"), Line2D([], [], marker="x", ls="", color="#e0201b", mew=2, label="screw site not used"),
        Line2D([], [], color="#3d7cc9", ls="--", label="blue thigh shell (sections)")]
ax.legend(handles=hand, loc="lower left", fontsize=7, frameon=True)
fig.suptitle(INFO["suptitle"], fontsize=13, weight="bold", x=0.01, ha="left")
fig.savefig(out_png, bbox_inches="tight"); print("wrote", out_png)
