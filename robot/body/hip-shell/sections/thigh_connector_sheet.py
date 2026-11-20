"""Owner drawing sheet of a roll-flip connector: front (from -y), side (from +x), top (from +z), two sections, parts in
their assembly colours with the generated screws, used / removed screw sites marked, the blue thigh shell dashed in the
sections, and an info box (measurements typed in the sheet's JSON from the check reports). Default: the thigh connector
(R_thigh_roll_flip; section A-A at x 45.75, B-B at z 143). The sheet JSON may set "focus" (the connector drawn opaque with
its _steel_sleeves; the yellow Stage 5 R_hip_pitch_to_roll_roll_flip draws the pitch-link screws and the green yaw-to-pitch
connector instead of the shin), "clip" and "sections" ([{axis, value, uv, lims, title}, ...]).

  .venv/bin/python -B robot/body/hip-shell/sections/thigh_connector_sheet.py \
      artifacts/robot/body/ape-proportions/shell-pass/assembly-parts-roll-flip.json \
      robot/body/hip-shell/sections/drawing-sheets/roll-flip/thigh-connector-h-cage.png \
      robot/body/hip-shell/sections/thigh-connector-h-cage.json
  (Stage 5: ... drawing-sheets/roll-flip/hip-pitch-to-roll-stage5.png robot/body/hip-shell/sections/hip-pitch-to-roll-stage5.json)
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
FOCUS = INFO.get("focus", "R_thigh_roll_flip")
STAGE5 = FOCUS == "R_hip_pitch_to_roll_roll_flip"
CONN = P[FOCUS]["mesh"]
COL = dict(blue=(0.25, 0.55, 0.85), yellow=(0.93, 0.76, 0.16), red=(0.82, 0.22, 0.20), dark=(0.22, 0.23, 0.26), grey=(0.62, 0.64, 0.67),
           light=(0.80, 0.81, 0.83), steel=(0.85, 0.87, 0.90), shell=(0.55, 0.70, 0.95))
def M(path): return trimesh.load(ROOT / path, force="mesh") if not str(path).endswith(".npz") else BL.load_mesh(path)
COL["green"] = (0.35, 0.68, 0.38)
hi = trimesh.util.concatenate([BL.load_mesh(P[n]["mesh"]) for n in ("R_roll_upper_horn_reference", "R_roll_passive_idler_reference")])
if STAGE5:   # the yellow yoke opaque with its sleeves and pitch-link screws; the blue cage, case and pitch motor around it
    parts = [  # (label, mesh, colour, alpha)
        ("connector", M(CONN), COL["yellow"], 1.0),
        ("connector sleeves", M(P[FOCUS + "_steel_sleeves"]["mesh"]), COL["steel"], 1.0),
        ("screws", trimesh.util.concatenate([BL.load_mesh(P[n]["mesh"]) for n in P if n.startswith("R_") and "screw_plan" in P[n] and P[n]["owning_link"] == "R_pitch_link"]), (0.55, 0.57, 0.6), 1.0),
        ("roll horn / idler", hi, COL["light"], 1.0),
        ("pitch output wheel", BL.load_mesh(P["R_hip_pitch_ST3025_output"]["mesh"]), COL["light"], 1.0),
        ("pitch case ST3025", BL.load_mesh(P["R_hip_pitch_ST3025_case"]["mesh"]), COL["dark"], 0.85),
        ("roll case STS3250", BL.load_mesh(P["R_roll_servo_case_reference"]["mesh"]), COL["red"], 0.85),
        ("H-cage (blue)", BL.load_mesh(P["R_thigh_roll_flip"]["mesh"]), COL["blue"], 0.55),
        ("yaw-to-pitch (green)", BL.load_mesh(P["R_hip_yaw_to_pitch_combined"]["mesh"]), COL["green"], 0.45),
    ]
else:
    parts = [  # (label, mesh, colour, alpha)
        ("connector", M(CONN), COL["blue"], 1.0),
        ("connector sleeves", M(P[FOCUS + "_steel_sleeves"]["mesh"]), COL["steel"], 1.0),
        ("screws", trimesh.util.concatenate([BL.load_mesh(P[n]["mesh"]) for n in P if n.startswith("R_") and "screw_plan" in P[n] and P[n]["owning_link"] == "R_roll_link"]), (0.55, 0.57, 0.6), 1.0),
        ("roll case STS3250", BL.load_mesh(P["R_roll_servo_case_reference"]["mesh"]), COL["red"], 1.0),
        ("knee case ST3025", BL.load_mesh(P["R_knee_ST3025_case"]["mesh"]), COL["dark"], 1.0),
        ("Stage 5 (yellow)", BL.load_mesh(P["R_hip_pitch_to_roll_roll_flip"]["mesh"]), COL["yellow"], 0.55),
        ("roll horn / idler", hi, COL["light"], 0.55),
        ("shin frame", BL.load_mesh(P["R_shin_combined"]["mesh"]), COL["grey"], 0.85),
        ("knee output", BL.load_mesh(P["R_knee_ST3025_output"]["mesh"]), COL["light"], 0.85),
    ]
CLIP = np.array(INFO.get("clip", [[20, -26, 108], [86, 40, 214]]))
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
SECS = INFO.get("sections", [dict(axis=0, value=45.75, uv=[1, 2], lims=[[-22, 36], [135, 205]], title="SECTION A-A at x 45.75 (inboard ear-screw column), seen from +x"),
                              dict(axis=2, value=143.0, uv=[0, 1], lims=[[22, 86], [-22, 36]], title="SECTION B-B at z 143 (all four knee screws), seen from +z")])
for key, sc in zip(("secx", "secz"), SECS):
    section(axs[key], sc["axis"], sc["value"], tuple(sc["uv"]), tuple(map(tuple, sc["lims"])), sc["title"])
# screw sites
used, gone = INFO.get("screws_used", {}), INFO.get("screws_removed", {})
def mark(ax, pts, used_, near=True):
    for (u, v) in pts:
        if used_ and near: ax.add_patch(Circle((u, v), 1.75, fc="#18a558", ec="k", lw=0.6, zorder=20))
        elif used_: ax.add_patch(Circle((u, v), 1.75, fc="none", ec="#18a558", lw=1.4, ls="--", zorder=20))
        elif near: ax.plot(u, v, marker="x", ms=7, mew=2, color="#e0201b", zorder=20)
# each view marks the screws on the face nearest the viewer filled and the far face's screws as dashed rings, so the
# two faces of one actuator never read as 4 screws on one face (owner 2026-11-20); removed sites only on the near face
for k, ax_name, ij, near in (("roll_front", "front", (0, 2), True), ("roll_rear", "front", (0, 2), False),
                             ("knee_lat", "side", (1, 2), True), ("knee_med", "side", (1, 2), False)):
    for flag, src in ((True, used), (False, gone)):
        pts = [(p[ij[0]], p[ij[1]]) for p in src.get(k, [])]; mark(axs[ax_name], pts, flag, near)
SHELLS = [BL.load_mesh(P[n]["mesh"]) for n in ("R_thigh_shell_front", "R_thigh_shell_back")]
def shell_outline(ax, axis, value, uv):
    for m in SHELLS:
        sct = m.section(plane_origin=[value if axis == i else 0 for i in range(3)], plane_normal=[1 if axis == i else 0 for i in range(3)])
        if sct is None: continue
        for ent in sct.discrete:
            ent = np.asarray(ent); ax.plot(ent[:, uv[0]], ent[:, uv[1]], color="#3d7cc9", lw=0.9, ls="--", zorder=15)
for key, sc in zip(("secx", "secz"), SECS):
    shell_outline(axs[key], sc["axis"], sc["value"], tuple(sc["uv"]))
for ax_name, txts in INFO.get("callouts", {}).items():
    for t in txts:
        axs[ax_name].annotate(t["text"], xy=t["xy"], xytext=t["at"], fontsize=7, color="k", zorder=30,
                              arrowprops=dict(arrowstyle="->", lw=0.7, color="k"), bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.5", lw=0.5, alpha=0.9))
ax = axs["info"]; ax.axis("off")
ax.text(0, 1.0, INFO["title"], fontsize=12, weight="bold", va="top", transform=ax.transAxes)
ax.text(0, 0.93, INFO["body"], fontsize=8, va="top", family="monospace", transform=ax.transAxes)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
hand = ([Patch(color=COL["yellow"], label="R_hip_pitch_to_roll_roll_flip (yellow, pitch link)"), Patch(color=COL["steel"], label="steel sleeves"),
         Patch(color=COL["light"], label="roll horn / idler, pitch output wheel"), Patch(color=COL["dark"], label="ST3025 hip-pitch case (yaw link)"),
         Patch(color=COL["red"], label="STS3250 roll case (red, roll link)"), Patch(color=COL["blue"], label="H-cage (blue, roll link)"),
         Patch(color=COL["green"], label="yaw-to-pitch connector (green, yaw link)")] if STAGE5 else
        [Patch(color=COL["blue"], label="R_thigh_roll_flip (blue, roll link)"), Patch(color=COL["yellow"], label="Stage 5 yoke (yellow, pitch link, moves)"),
        Patch(color=COL["red"], label="STS3250 roll case (red)"), Patch(color=COL["dark"], label="ST3025 knee case (dark grey)"),
        Patch(color=COL["grey"], label="shin frame (grey, moves)"), Patch(color=COL["light"], label="horn / idler / knee output")]) + [
        Line2D([], [], marker="o", ls="", mfc="#18a558", mec="k", label="screw on the face you see"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#18a558", mew=1.4, label="screw on the far face (hidden)"),
        Line2D([], [], marker="x", ls="", color="#e0201b", mew=2, label="screw site not used (face you see)"),
        Line2D([], [], color="#3d7cc9", ls="--", label="blue thigh shell (sections)")]
ax.legend(handles=hand, loc="lower left", fontsize=7, frameon=True)
fig.suptitle(INFO["suptitle"], fontsize=13, weight="bold", x=0.01, ha="left")
fig.savefig(out_png, bbox_inches="tight"); print("wrote", out_png)
