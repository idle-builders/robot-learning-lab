"""Overlay Microban's actual assembled CAD/MJCF on sketch-06 at original size.

This is a *reference fit screen*, not a design to manufacture or a training
model. Microban's hardware CAD is CC BY-NC-SA 4.0; keep the checkout local and
obtain permission before using its printed parts in a sold robot. Run:

  .venv/bin/python experiments/compact_morphology/ape_design/reference_core_fit.py \
      --reference /tmp/ape-microban-reference

The checkout should be pinned to d0a128d180ecfce69eaaf31b295a0fce31280bfd.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import build123d as bd
import mujoco
import numpy as np
import trimesh

import sketch_shell as shell


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-functional-cad"
PIN = "d0a128d180ecfce69eaaf31b295a0fce31280bfd"


def region_for(name: str) -> list[str]:
    side = "left" if "left" in name or name in (
        "shoulder_2", "radius_2", "hip_2", "foot_2") else "right"
    if name == "trunk":
        return ["torso_shell"]
    if name == "head":
        return ["head_shell"]
    if "shoulder" in name or "humerus" in name:
        return ["torso_shell", f"{side}_upper_arm_shell"]
    if "radius" in name:
        return [f"{side}_forearm_shell"]
    if "hip" in name:
        return ["torso_shell", f"{side}_thigh_shell"]
    if "femur" in name:
        return [f"{side}_thigh_shell"]
    if "tibia" in name:
        return [f"{side}_shin_shell"]
    if "ankle" in name:
        return [f"{side}_shin_shell", f"{side}_foot_shell"]
    if "foot" in name:
        return [f"{side}_foot_shell"]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                   cwd=args.reference, text=True).strip()
    if head != PIN:
        raise ValueError(f"Microban reference revision {head} != pinned {PIN}")
    source = args.reference / "src/model/mjcf/robot.xml"
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    shell.build()
    OUT.mkdir(parents=True, exist_ok=True)
    raw: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for gid in range(model.ngeom):
        if (model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH or
                model.geom_group[gid] != 2):
            continue
        mid = int(model.geom_dataid[gid])
        vertices = model.mesh_vert[
            model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        faces = model.mesh_face[
            model.mesh_faceadr[mid]:model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
        world = data.geom_xpos[gid] + vertices @ data.geom_xmat[gid].reshape(3, 3).T
        lo = np.minimum(lo, world.min(axis=0))
        hi = np.maximum(hi, world.max(axis=0))
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[gid]))
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
        raw.append((body, mesh_name, world, faces.copy()))
    # Floor-align *without scaling any manufactured part*. The reference is
    # 326.6 mm overall, leaving ~23 mm between its head and the 350 mm skin.
    shift = np.array((0., 0., -lo[2]))
    scene = trimesh.Scene()
    rows = []
    for index, (body, mesh_name, vertices, faces) in enumerate(raw):
        points = (vertices + shift) * 1000
        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
        color = ((55, 91, 117, 255) if "xl330" in mesh_name else
                 (88, 141, 98, 255) if "battery" in mesh_name else
                 (111, 113, 112, 255))
        mesh.visual.vertex_colors = color
        scene.add_geometry(mesh, geom_name=f"reference_{index}_{mesh_name}",
                           node_name=f"reference_{index}_{mesh_name}")
        names = region_for(body)
        sample = points[np.linspace(0, len(points) - 1,
                                    min(6, len(points)), dtype=int)]
        fitted = sum(any(shell.INTERIORS[name].is_inside(bd.Vector(*point))
                         for name in names if name in shell.INTERIORS)
                     for point in sample)
        rows.append({"body": body, "mesh": mesh_name,
                     "inside_surface_samples": fitted,
                     "surface_samples": len(sample), "target_regions": names})
    path = OUT / "microban-reference-neutral.glb"
    scene.export(path)
    by_body = {}
    for row in rows:
        entry = by_body.setdefault(row["body"], {"inside": 0, "sampled": 0})
        entry["inside"] += row["inside_surface_samples"]
        entry["sampled"] += row["surface_samples"]
    report = {
        "status": "reference fit only; not our assembly or a training model",
        "source": "https://github.com/Rhoban/microban",
        "source_revision": PIN,
        "source_model_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reference_hardware_license": "CC BY-NC-SA 4.0 per Microban README; commercial use requires permission",
        "reference_mjcf_license_note": "robot.xml header says Apache-2.0; the printed hardware CAD has separate noncommercial terms",
        "reference_axes": model.nu,
        "reference_urdf_mass_kg": float(sum(model.body_mass)),
        "reference_mechanical_cad_height_mm": float((hi[2] - lo[2]) * 1000),
        "alignment": "unaltered source components, floor aligned to sketch z=0, x/y unchanged",
        "sample_note": "Six deterministic mesh-surface vertices per visual CAD geom, tested against nominal inner shell. This is a quick diagnostic, not continuous clearance or material-overlap proof.",
        "body_surface_fit": by_body,
        "source_body_meshes": rows,
        "overlay_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (OUT / "microban-reference-fit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Microban core: {model.nu} axes, {sum(model.body_mass):.3f} kg URDF, "
          f"{(hi[2]-lo[2])*1000:.1f} mm high, {len(rows)} CAD geoms; "
          f"floor-aligned overlay -> {path}")
    for body, result in by_body.items():
        print(body, f"{result['inside']}/{result['sampled']} nominal interior samples")


if __name__ == "__main__":
    main()
