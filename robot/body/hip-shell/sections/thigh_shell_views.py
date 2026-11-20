"""Shaded look sheets of the blue thigh shell on the right leg (roll-flip hip; Idle T-czv2d8 / T-fs5v6d).

Opaque, lit surfaces (the outline sheets in owner_case_views.py do not show steps, lips or bulges). Owner colours: green
(yaw-to-pitch connector, hip-pitch motor), yellow (pitch-to-roll connector), red (roll motor case), blue (thigh shell), grey
(knee motor dark, shin). Row 1 standing; row 2 deep crouch (hip pitch -60, knee 95) to show the knee pit and the shell edges in use.

`closeup NAME` draws the owner's close-up angles instead (standing, cropped to the shell, grazing light so small ridges and
valleys show): from below-front, from the outboard side slightly low, and from the back slightly low.

  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 180 -- .venv/bin/python -B robot/body/hip-shell/sections/thigh_shell_views.py
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 180 -- .venv/bin/python -B robot/body/hip-shell/sections/thigh_shell_views.py closeup thigh-shell-closeup
  python3 infra/tools/memguard.py --limit-gb 4 --timeout-s 240 -- .venv/bin/python -B robot/body/hip-shell/sections/thigh_shell_views.py top thigh-shell-top
`top NAME` draws the top edge as the owner saw it (2026-11-20): both legs from the front slightly above, each leg from
the front-inboard side slightly above, and the right leg from the back-inboard side; the thigh connector and the yellow arm
hubs it covers are drawn light grey so what stands above the edge shows.

`--manifest PATH` draws another body manifest (e.g. an earlier shell for a before / after pair) and `--out NAME` names the sheet.
Writes drawing-sheets/roll-flip/thigh-shell-shaded.png or drawing-sheets/roll-flip/NAME.png (derived; recreate with this command).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "robot/body/leg"))
import roll_flip_checks as R  # noqa: E402

SHEETS = HERE / "drawing-sheets/roll-flip"
OUT = SHEETS / "thigh-shell-shaded.png"
SHELL = ("R_thigh_shell_front", "R_thigh_shell_back")
PARTS = {"R_thigh_shell_front": "#2f6fd0", "R_thigh_shell_back": "#2f6fd0",
         "R_hip_pitch_to_roll_roll_flip": "#e8b21e", "R_roll_servo_case_reference": "#d9473f",
         "R_hip_yaw_to_pitch_combined": "#4fae6e", "R_hip_pitch_ST3025_case": "#4fae6e",
         "R_shin_combined": "#9a9a9a", "R_shin_front_hard_cover": "#b4b4b4", "R_shin_back_hard_cover": "#b4b4b4",
         "R_knee_ST3025_output": "#8a8a8a", "R_knee_ST3025_case": "#3a3a3a"}
VIEWS = (("FRONT", (0, -1, 0), 0, 1), ("BACK", (0, 1, 0), 0, -1), ("RIGHT side (outboard)", (1, 0, 0), 1, 1),
         ("LEFT side (inboard)", (-1, 0, 0), 1, -1), ("TOP", (0, 0, 1), 0, 1), ("BOTTOM", (0, 0, -1), 0, -1))
POSES = (("standing", {}), ("crouch: hip pitch -60, knee 95", {"R_hip_pitch": -60.0, "R_knee": 95.0}))
# close-ups: direction toward the viewer (thigh frame: -y front, +x outboard, +z up), light direction
CLOSEUPS = (("from below-front", (0.25, -1.0, -0.75), (-0.5, -0.6, 0.35)),
            ("outboard side, slightly low", (1.0, -0.2, -0.3), (0.6, -0.7, 0.4)),
            ("from the back, slightly low", (0.25, 1.0, -0.3), (0.5, 0.5, 0.7)))

# top-edge views: (title, direction toward the viewer, light, legs framed)
TOPVIEWS = (("both legs, front, slightly above", (0.0, -1.0, 0.45), (0.2, -0.7, 0.7), "RL"),
            ("right leg, front-inboard, slightly above", (-0.75, -1.0, 0.55), (-0.5, -0.6, 0.6), "R"),
            ("left leg, front-inboard, slightly above", (0.75, -1.0, 0.55), (0.5, -0.6, 0.6), "L"),
            ("right leg, back-inboard, slightly above", (-0.75, 1.0, 0.55), (-0.5, 0.6, 0.6), "R"))
COVERED_GREY = {"R_thigh_roll_flip": "#cfcfcf", "R_roll_upper_horn_reference": "#e6e6e6", "R_roll_passive_idler_reference": "#e6e6e6"}


def _meshes(body, faces=20000, keep=(), parts=None):
    meshes = {}
    for n in parts or PARTS:
        if n in body.parts:
            m = body.mesh(n)
            meshes[n] = m.simplify_quadric_decimation(face_count=faces) if len(m.faces) > faces and n not in keep else m
    return meshes


def _draw(ax, body, meshes, q, w, light, crop=None, ambient=0.3, px=700):
    """Ray-cast (embree) orthographic image toward -w: exact visibility, smooth vertex-normal Lambert + a little specular."""
    import matplotlib
    import trimesh
    w = np.asarray(w, float); w /= np.linalg.norm(w)
    up = np.array([0.0, 0.0, 1.0]) if abs(w[2]) < 0.95 else np.array([0.0, np.sign(w[2]), 0.0])
    right = np.cross(up, w); right /= np.linalg.norm(right); up = np.cross(w, right)
    light = np.asarray(light, float); light /= np.linalg.norm(light)
    ms, cols = [], []
    for n, m in meshes.items():
        T = body.link_T(body.parts[n]["owning_link"], q)
        mm = m.copy(); mm.apply_transform(T); ms.append(mm)
        cols.append(np.tile(matplotlib.colors.to_rgb(COLOURS.get(n) or COLOURS["R" + n[1:]]), (len(mm.faces), 1)))
    M = trimesh.util.concatenate(ms); C = np.vstack(cols)
    V = M.vertices if crop is None else crop
    S = V @ np.c_[right, up]; lo, hi = S.min(0) - (2 if crop is None else 12), S.max(0) + (2 if crop is None else 12)
    s = max(hi - lo) / px
    gx, gy = np.arange(lo[0], hi[0], s), np.arange(lo[1], hi[1], s)
    X, Y = np.meshgrid(gx, gy)
    o = X.ravel()[:, None] * right + Y.ravel()[:, None] * up + w * (np.abs(M.vertices @ w).max() + 50.0)
    tri, ray, loc = M.ray.intersects_id(o, np.tile(-w, (len(o), 1)), multiple_hits=False, return_locations=True)
    img = np.ones((len(o), 3))
    bc = trimesh.triangles.points_to_barycentric(M.triangles[tri], loc)
    nv = M.vertex_normals[M.faces[tri]]
    nrm = np.einsum("ij,ijk->ik", bc, nv); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    fn = M.face_normals[tri] * np.where(M.face_normals[tri] @ w < 0, -1.0, 1.0)[:, None]   # seen side of the face
    nrm *= np.sign(np.einsum("ij,ij->i", nrm, fn))[:, None]                            # inner walls lit from inside
    bad = nrm @ w < 0.05
    nrm[bad] = fn[bad]                                                                 # sharp rims: the face normal
    h = light + w; h /= np.linalg.norm(h)
    sh = ambient + (1 - ambient) * np.clip(nrm @ light, 0, 1)
    sp = 0.25 * np.clip(nrm @ h, 0, 1) ** 40
    img[ray] = np.clip(C[tri] * sh[:, None] + sp[:, None], 0, 1)
    ax.imshow(img.reshape(len(gy), len(gx), 3), origin="lower", extent=(gx[0], gx[-1], gy[0], gy[-1]), interpolation="bilinear")
    ax.set_aspect("equal"); ax.axis("off")


COLOURS = {**PARTS, **COVERED_GREY}


def _opt(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


MANIFEST = Path(_opt("--manifest", str(R.VARIANT)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = R.kit.base(MANIFEST)
    meshes = _meshes(body)
    light = np.array([0.4, -0.6, 0.7])
    fig, axs = plt.subplots(len(POSES), len(VIEWS), figsize=(24, 9.5))
    for row, (pname, dq) in zip(axs, POSES):
        q = {**body.standing, **dict(zip(R.J, (0.0,) * 4)), **dq}
        for ax, (title, w, h, hs) in zip(row, VIEWS):
            _draw(ax, body, meshes, q, w, light)
            ax.set_title(f"{title} - {pname}", fontsize=10)
    fig.suptitle("Blue thigh shell on the right leg (shaded). Green: yaw-to-pitch + hip-pitch motor; yellow: pitch-to-roll; "
                 "red: roll motor; dark grey: knee motor; grey: shin", fontsize=13)
    out = SHEETS / f"{_opt('--out')}.png" if _opt("--out") else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, bbox_inches="tight"); print(out)


def closeup(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = R.kit.base(MANIFEST)
    meshes = _meshes(body, keep=SHELL)
    q = {**body.standing, **dict(zip(R.J, (0.0,) * 4))}
    crop = np.vstack([body.mesh(n).vertices @ body.link_T(body.parts[n]["owning_link"], q)[:3, :3].T
                      + body.link_T(body.parts[n]["owning_link"], q)[:3, 3] for n in SHELL])
    fig, axs = plt.subplots(1, len(CLOSEUPS), figsize=(8 * len(CLOSEUPS), 9))
    for ax, (title, w, light) in zip(axs, CLOSEUPS):
        _draw(ax, body, meshes, q, w, light, crop=crop, ambient=0.22, px=1000)
        ax.set_title(f"{title} (standing)", fontsize=12)
    fig.suptitle(f"Blue thigh shell close-up ({name}); grazing light shows ridges and valleys", fontsize=13)
    f = SHEETS / f"{name}.png"; f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=110, bbox_inches="tight"); print(f)


def top(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    body = R.kit.base(MANIFEST)
    names = [s_ + n[1:] for n in COLOURS for s_ in "RL" if s_ + n[1:] in body.parts]
    keep = [s_ + n[1:] for n in SHELL for s_ in "RL"]
    meshes = _meshes(body, keep=keep, parts=names)
    q = {**body.standing, **dict(zip(R.J, (0.0,) * 4))}
    def world(n):
        T = body.link_T(body.parts[n]["owning_link"], q); return body.mesh(n).vertices @ T[:3, :3].T + T[:3, 3]
    fig, axs = plt.subplots(1, len(TOPVIEWS), figsize=(36, 8.5), gridspec_kw=dict(width_ratios=[2.1] + [1] * (len(TOPVIEWS) - 1)))
    for ax, (title, w, light, legs) in zip(axs, TOPVIEWS):
        crop = np.vstack([world(s_ + n[1:]) for s_ in legs for n in SHELL + ("R_hip_pitch_to_roll_roll_flip",)])
        crop = crop[(crop[:, 2] > 160.0) & (crop[:, 2] < 235.0)]           # the upper thigh: shell top and the yellow arm
        sub = {n: m for n, m in meshes.items() if n[0] in legs}
        _draw(ax, body, sub, q, w, light, crop=crop, ambient=0.25, px=900)
        ax.set_title(f"{title} (standing)", fontsize=12)
    fig.suptitle(f"Thigh shell top edge ({name}). Blue: shell; yellow: pitch-to-roll arm; red: roll motor; green: hip pitch; "
                 "light grey: thigh connector and arm hubs", fontsize=13)
    f = SHEETS / f"{name}.png"; f.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f, dpi=100, bbox_inches="tight"); print(f)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "top":
        top(sys.argv[2])
    elif len(sys.argv) > 2 and sys.argv[1] == "closeup":
        closeup(sys.argv[2])
    else:
        main()
