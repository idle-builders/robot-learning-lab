"""Package the motor-refit body for the design dashboard: assembly GLB, printed-skeleton STEP and comparison renders.

  .venv/bin/python -B robot/body/motor-refit/package.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "robot/body"))
from checks import bodylib as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "robot/body/bom"))
import bom_config as BOM_CONFIG  # noqa: E402
CFG = BOM_CONFIG.section("motor-refit")   # configured values: robot/body/motor-refit/motor_refit.bom.yaml config:

R1 = ROOT / "artifacts/robot/body/assembly/assembly-parts.json"
OUT = ROOT / "artifacts/robot/body/motor-refit"
SKIP_ROLES = set(CFG["SKIP_ROLES"])


def scene(man):
    sc = trimesh.Scene()
    for p in man["parts"]:
        if not p.get("mesh") or p.get("role") in SKIP_ROLES:
            continue
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p["placement_offset_mm"])
        if len(m.faces) > 30000:  # display copy only; native STEP and the npz meshes stay exact
            m = m.simplify_quadric_decimation(face_count=30000)
        m.apply_scale(0.001)  # model units m, as r1
        m.visual.face_colors = np.array(p.get("color") or [180, 180, 180, 255], np.uint8)
        sc.add_geometry(m, node_name=p["name"], geom_name=p["name"])
    return sc


def skeleton_step(man, path):
    from build123d import import_step, export_step, Compound
    import cad_ops as C
    shapes = []
    for p in man["parts"]:
        if not p.get("native_step") or not (p.get("role", "").startswith("printed") or p.get("role") == "compression_limit_sleeves"):
            continue
        s = import_step(str(F.src(p["native_step"]))).wrapped
        if p.get("native_step_offset_mm"):
            s = C.translate(s, p["native_step_offset_mm"])
        shapes.append(s)
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound
    comp = TopoDS_Compound(); b = BRep_Builder(); b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s)
    export_step(Compound(comp), str(path))
    return len(shapes)


def project(man, ax, view, zmin, zmax, shells=True, title=""):
    """Orthographic painter's projection; view 'front' looks along +y, 'side' along -x."""
    polys, cols, depth = [], [], []
    light = np.array([0.35, -0.6, 0.7]); light /= np.linalg.norm(light)
    for p in man["parts"]:
        if not p.get("mesh") or p.get("role") in SKIP_ROLES or F.is_fastener(p["name"], p):
            continue
        if not shells and p.get("role") == "body_shell":
            continue
        b = np.array(p["bounds_mm"], float)
        if b[1][2] < zmin or b[0][2] > zmax:
            continue
        m = F.load_mesh(p["mesh"])
        if p.get("placement_offset_mm"):
            m.apply_translation(p["placement_offset_mm"])
        if len(m.faces) > 4000:
            m = m.simplify_quadric_decimation(face_count=4000)
        tri = m.vertices[m.faces]; n = m.face_normals
        if view == "front":
            uv, d, vis = tri[:, :, [0, 2]], tri[:, :, 1].mean(1), n[:, 1] < 0.05
        else:
            uv, d, vis = tri[:, :, [1, 2]], -tri[:, :, 0].mean(1), n[:, 0] > -0.05
        c = np.array(p.get("color") or [180, 180, 180, 255], float)[:3] / 255
        shade = 0.35 + 0.65 * np.clip(np.abs(n @ light), 0, 1)
        polys.extend(uv[vis]); depth.extend(d[vis]); cols.extend(np.clip(np.outer(shade[vis], c), 0, 1))
    order = np.argsort(-np.array(depth)) if view == "front" else np.argsort(np.array(depth))
    ax.add_collection(PolyCollection([polys[i] for i in order], facecolors=[cols[i] for i in order], edgecolors="none"))
    ax.set_aspect("equal"); ax.autoscale(); ax.set_ylim(zmin, zmax); ax.set_title(title, fontsize=10); ax.axis("off")


def main():
    man = json.load(open(OUT / "assembly-parts.json")); r1 = json.load(open(F.src(R1)))
    sc = scene(man); sc.export(str(OUT / "assembly-with-full-shell.glb"))
    print("glb", (OUT / "assembly-with-full-shell.glb").stat().st_size // 1024, "kB")
    n = skeleton_step(man, OUT / "complete-fdm-structural-skeleton.step"); print("skeleton step parts", n)
    (OUT / "renders").mkdir(exist_ok=True)
    for shells in (True, False):
        fig, axs = plt.subplots(1, 4, figsize=(20, 9))
        project(r1, axs[0], "front", 0, 420, shells, "r1 (C047 legs) - front")
        project(man, axs[1], "front", 0, 420, shells, "motor refit (ST3025 legs) - front")
        project(r1, axs[2], "side", 0, 420, shells, "r1 - side")
        project(man, axs[3], "side", 0, 420, shells, "motor refit - side")
        plt.tight_layout(); name = f"refit-vs-r1-{'shells' if shells else 'mechanism'}.png"
        plt.savefig(OUT / "renders" / name, dpi=90); plt.close(fig); print("render", name)
    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    project(r1, axs[0], "front", 170, 300, False, "r1 pelvis / hips (front)")
    project(man, axs[1], "front", 170, 300, False, "refit pelvis / hips: ST3025 hip pitch, horizontal cells (front)")
    plt.tight_layout(); plt.savefig(OUT / "renders" / "refit-vs-r1-pelvis.png", dpi=100); plt.close(fig); print("render pelvis")


if __name__ == "__main__":
    main()
