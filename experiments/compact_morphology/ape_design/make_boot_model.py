"""Export the approved sketch-06 boot into the ape's ankle frame and MJCF.

The contact geom is the convex hull of the CAD boot + cosmetic outsole because
MuJoCo treats a mesh geom as convex for collision.  Keep the exact shell and
outsole as separate visual meshes, and record the hull error rather than
pretending that a convex contact mesh is identical to the CAD union.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco
import numpy as np
import trimesh

import sketch_shell as shell


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_elbows_small_head.xml"
OUT = ROOT / "artifacts/ape-full-boot-control-20261002"
MESHES = OUT / "models"
PLA_KG_M3 = 1240  # Bambu PLA Basic Gradient density, a provisional shell material proxy.
TPU_KG_M3 = 1220  # Bambu TPU 95A HF nominal density, 1.22 g/cm3.
MOTOR_KG = 0.0745  # Feetech ST-3250-C001 listed mass, retained foot motor hypothesis.
ANKLE_CAD_MM = (0, 41, 32)  # Align CAD boot floor with MJCF sole bottom at -32 mm.
MOTOR_CENTER_CAD_MM = (2, 41, 15)
MOTOR_BOX_MM = (45.22, 34.92, 24.72)  # Case rotated so its short dimension is vertical.


def cad_outsole(side: int = 1) -> bd.Shape:
    """The approved boot's rounded, full-footprint outsole in CAD millimetres."""
    outsole = bd.Pos(14, side * 41, 1.25) * bd.Box(90, 60, 2.5)
    outsole = bd.fillet(outsole.edges().filter_by(bd.Axis.Z), 7)
    toe_edges = [edge for edge in outsole.edges()
                 if edge.center().X > 58 and edge.center().Z > 2.4
                 and edge.length > 20]
    return bd.fillet(toe_edges, 1.2)


def cad_mesh(shape: bd.Shape, *, mirror_y: bool = False) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(0.07, 0.07)
    points = np.array([[v.X, v.Y, v.Z] for v in vertices], dtype=float)
    if mirror_y:
        points[:, 1] *= -1
    points -= np.asarray(ANKLE_CAD_MM, dtype=float)
    result = trimesh.Trimesh(vertices=points / 1000, faces=np.asarray(faces), process=True)
    result.fix_normals()
    return result


def mass_properties(mesh: trimesh.Trimesh, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    props = mesh.mass_properties
    volume = float(props["volume"])
    if volume <= 0 or not mesh.is_watertight:
        raise ValueError("CAD mass mesh must be closed and have positive volume")
    return volume * density, np.asarray(props["center_mass"]), np.asarray(props["inertia"]) * density


def cad_mass_properties(shape: bd.Shape, density: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Use the exact BRep: tessellating a hollow shell loses inner-face winding."""
    mass = shape.volume * 1e-9 * density
    center = np.array(tuple(shape.center()))
    center = (center - np.asarray(ANKLE_CAD_MM)) / 1000
    inertia = np.asarray(shape.matrix_of_inertia) * density * 1e-15
    return mass, center, inertia


def combined_inertia(parts: list[tuple[float, np.ndarray, np.ndarray]]) -> tuple[float, np.ndarray, np.ndarray]:
    mass = sum(item[0] for item in parts)
    center = sum((item[0] * item[1] for item in parts), np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for part_mass, part_center, part_inertia in parts:
        offset = part_center - center
        inertia += part_inertia + part_mass * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return mass, center, inertia


def attr3(values: np.ndarray) -> str:
    return " ".join(f"{value:.17g}" for value in values)


def foot_inertial(tree: ET.ElementTree, side: str, mass: float,
                  center: np.ndarray, inertia: np.ndarray) -> None:
    body = tree.find(f".//body[@name='{side}_foot']")
    assert body is not None
    body.insert(0, ET.Element("inertial", {
        "pos": attr3(center), "mass": f"{mass:.17g}",
        "fullinertia": " ".join(f"{value:.17g}" for value in (
            inertia[0, 0], inertia[1, 1], inertia[2, 2],
            inertia[0, 1], inertia[0, 2], inertia[1, 2],
        )),
    }))


def add_boot_geoms(tree: ET.ElementTree, side: str, *, baseline_inertia: bool,
                   candidate: tuple[float, np.ndarray, np.ndarray],
                   reference: mujoco.MjModel) -> None:
    body = tree.find(f".//body[@name='{side}_foot']")
    sole = tree.find(f".//geom[@name='{side}_sole']")
    center_site = tree.find(f".//site[@name='{side}_sole_center']")
    touch_site = tree.find(f".//site[@name='{side}_sole_touch']")
    assert body is not None and sole is not None and center_site is not None and touch_site is not None
    if baseline_inertia:
        body_id = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, f"{side}_foot")
        body.insert(0, ET.Element("inertial", {
            "pos": attr3(reference.body_ipos[body_id]),
            "quat": attr3(reference.body_iquat[body_id]),
            "mass": f"{reference.body_mass[body_id]:.17g}",
            "diaginertia": attr3(reference.body_inertia[body_id]),
        }))
    else:
        foot_inertial(tree, side, *candidate)
    sole.set("type", "mesh")
    sole.set("mesh", "boot_contact")
    sole.set("pos", "0 0 0")
    sole.set("rgba", "0.45 0.53 0.50 0")
    sole.attrib.pop("size", None)
    center_site.set("pos", "0.014 0 -0.032")
    touch_site.set("pos", "0.013 0 -0.0155")
    touch_site.set("size", "0.046 0.036 0.0165")
    ET.SubElement(body, "geom", name=f"{side}_boot_visual", type="mesh",
                  mesh="boot_shell_visual", pos="0 0 0", mass="0",
                  contype="0", conaffinity="0", rgba="0.91 0.89 0.84 1")
    ET.SubElement(body, "geom", name=f"{side}_outsole_visual", type="mesh",
                  mesh="boot_outsole_visual", pos="0 0 0", mass="0",
                  contype="0", conaffinity="0", rgba="0.20 0.25 0.23 1")


def make_xml(name: str, *, baseline_inertia: bool,
             candidate: tuple[float, np.ndarray, np.ndarray],
             reference: mujoco.MjModel) -> Path:
    tree = ET.parse(SOURCE)
    asset = ET.Element("asset")
    for mesh_name, file_name in (
        ("boot_contact", "boot_contact_hull.stl"),
        ("boot_shell_visual", "boot_shell.stl"),
        ("boot_outsole_visual", "boot_outsole.stl"),
    ):
        ET.SubElement(asset, "mesh", name=mesh_name, file=file_name)
    tree.getroot().insert(4, asset)
    for side in ("left", "right"):
        add_boot_geoms(tree, side, baseline_inertia=baseline_inertia,
                       candidate=candidate, reference=reference)
    path = MESHES / f"{name}.xml"
    tree.write(path, encoding="unicode", xml_declaration=True)
    return path


def main() -> None:
    MESHES.mkdir(parents=True, exist_ok=True)
    shell.build()
    foot = next(part for part in shell.PARTS if part.name == "left_foot_shell")
    outer = foot.shape.fuse(shell.INTERIORS[foot.name])
    outsole = cad_outsole()
    union = outer.fuse(outsole)
    skin_mesh, outsole_mesh, union_mesh = (cad_mesh(shape) for shape in
                                          (foot.shape, outsole, union))
    hull_mesh = union_mesh.convex_hull
    for name, mesh in (("boot_shell.stl", skin_mesh),
                       ("boot_outsole.stl", outsole_mesh),
                       ("boot_contact_hull.stl", hull_mesh)):
        mesh.export(MESHES / name)
    motor = trimesh.creation.box(extents=np.asarray(MOTOR_BOX_MM) / 1000)
    motor.apply_translation((np.asarray(MOTOR_CENTER_CAD_MM)
                             - np.asarray(ANKLE_CAD_MM)) / 1000)
    motor_parts = mass_properties(motor, MOTOR_KG / motor.volume)
    motor_cad = bd.Pos(*MOTOR_CENTER_CAD_MM) * bd.Box(*MOTOR_BOX_MM)
    motor_inside_pct = 100 * (motor_cad & shell.INTERIORS[foot.name]).volume / motor_cad.volume
    candidate = combined_inertia([
        cad_mass_properties(foot.shape, PLA_KG_M3),
        cad_mass_properties(outsole, TPU_KG_M3),
        motor_parts,
    ])
    reference = mujoco.MjModel.from_xml_path(str(SOURCE))
    baseline_copy = MESHES / "baseline.xml"
    shutil.copy2(SOURCE, baseline_copy)
    controlled = make_xml("boot_geometry_locked", baseline_inertia=True,
                          candidate=candidate, reference=reference)
    physical = make_xml("boot_cad_mass", baseline_inertia=False,
                        candidate=candidate, reference=reference)
    report = {
        "source_cad": "sketch_shell.py:left_foot_shell + 90x60x2.5mm rounded outsole",
        "ankle_cad_origin_mm": ANKLE_CAD_MM,
        "contact_model": "convex hull of CAD outer boot + outsole, single sole geom",
        "cad_union_volume_cm3": union.volume / 1000,
        "contact_hull_volume_cm3": hull_mesh.volume * 1e6,
        "contact_hull_extra_volume_fraction": hull_mesh.volume / (union.volume * 1e-9) - 1,
        "visual_shell_volume_cm3": foot.shape.volume / 1000,
        "outsole_volume_cm3": outsole.volume / 1000,
        "mass_method": "exact build123d/OpenCascade BRep; hollow shell STL winding is unsuitable for mass",
        "boot_mass_components_kg": {
            "PLA_shell": cad_mass_properties(foot.shape, PLA_KG_M3)[0],
            "TPU_outsole": cad_mass_properties(outsole, TPU_KG_M3)[0],
            "Feetech_motor": MOTOR_KG,
        },
        "motor_case_candidate": {
            "center_cad_mm": MOTOR_CENTER_CAD_MM,
            "size_mm": MOTOR_BOX_MM,
            "nominal_inside_shell_pct": motor_inside_pct,
            "status": "case-envelope screen only; output shaft, mounts and wires unverified",
        },
        "cad_foot_mass_kg": candidate[0],
        "cad_foot_center_m": candidate[1].tolist(),
        "cad_foot_inertia_kg_m2": candidate[2].tolist(),
        "retained_foot_mass_kg": float(reference.body_mass[
            mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, "left_foot")]),
        "friction": "retained MJCF friction=1.0; TPU-floor friction unmeasured",
        "sources": {
            "PLA_density": "https://eu.store.bambulab.com/nl-nl/products/pla-basic-gradient?variant=53638219006300",
            "TPU_density": "https://uk.store.bambulab.com/products/tpu-95a-hf",
            "motor_mass": "https://www.feetechrc.com/en/562636.html",
        },
        "files": {},
    }
    for path in [baseline_copy, controlled, physical, *MESHES.glob("*.stl")]:
        report["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "model.json").write_text(json.dumps(report, indent=2) + "\n")
    for path in (controlled, physical):
        compiled = mujoco.MjModel.from_xml_path(str(path))
        body_id = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        print(path.name, "foot mass", float(compiled.body_mass[body_id]),
              "contact hull excess", round(report["contact_hull_extra_volume_fraction"], 4))
    print(OUT / "model.json")


if __name__ == "__main__":
    main()
