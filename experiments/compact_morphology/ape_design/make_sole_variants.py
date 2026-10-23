"""Make single-change sole variants for a saved-policy contact replay.

The retained ape has a 90 x 60 x 24 mm collision box.  The sketch's visible
boot bounds are about 78 x 60 mm in plan at 35 cm.  Keep the sole's published
mass and bottom height fixed; first change only thickness, then footprint.
These are collision proxies, not a CAD mesh or a material model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/compact_morphology/playground_models/ape_sourced_v3_elbows_small_head.xml"
OUT = ROOT / "artifacts/ape-sole-policy-replay-20261002/models"


def lock_foot_inertia(tree: ET.ElementTree, reference: mujoco.MjModel) -> None:
    """Keep the *compiled* original foot mass properties across contact shapes."""
    for side in ("left", "right"):
        name = f"{side}_foot"
        body_id = mujoco.mj_name2id(reference, mujoco.mjtObj.mjOBJ_BODY, name)
        body = tree.find(f".//body[@name='{name}']")
        assert body is not None and body_id >= 0
        inertial = ET.Element("inertial", {
            "pos": " ".join(f"{x:.17g}" for x in reference.body_ipos[body_id]),
            "quat": " ".join(f"{x:.17g}" for x in reference.body_iquat[body_id]),
            "mass": f"{reference.body_mass[body_id]:.17g}",
            "diaginertia": " ".join(f"{x:.17g}" for x in reference.body_inertia[body_id]),
        })
        body.insert(0, inertial)


def make_variant(name: str, *, center_x: float, half_length: float,
                 half_thickness: float, reference: mujoco.MjModel,
                 rounded_mesh: bool = False) -> dict:
    tree = ET.parse(SOURCE)
    lock_foot_inertia(tree, reference)
    if rounded_mesh:
        asset = ET.Element("asset")
        ET.SubElement(asset, "mesh", name="rounded_outsole",
                      file="rounded_outsole.stl", scale="0.001 0.001 0.001")
        tree.getroot().insert(4, asset)
    for side in ("left", "right"):
        sole = tree.find(f".//geom[@name='{side}_sole']")
        center = tree.find(f".//site[@name='{side}_sole_center']")
        touch = tree.find(f".//site[@name='{side}_sole_touch']")
        assert sole is not None and center is not None and touch is not None
        center_z = -0.032 + half_thickness
        sole.set("pos", f"{center_x} 0 {center_z}")
        if rounded_mesh:
            sole.set("type", "mesh")
            sole.set("mesh", "rounded_outsole")
            sole.attrib.pop("size", None)
        else:
            sole.set("size", f"{half_length} 0.030 {half_thickness}")
        center.set("pos", f"{center_x} 0 -0.032")
        touch.set("pos", f"{center_x} 0 {center_z}")
        touch.set("size", f"{half_length} 0.030 {half_thickness}")
    path = OUT / f"{name}.xml"
    tree.write(path, encoding="unicode", xml_declaration=True)
    return {"variant": name, "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "footprint_mm": [round(2 * half_length * 1000), 60],
            "thickness_mm": round(2 * half_thickness * 1000, 1),
            "center_x_mm": round(center_x * 1000, 1),
            "sole_mass_each_kg": 0.010118834,
            "rounded_mesh": rounded_mesh}


def make_rounded_mesh() -> None:
    """Export the thin CAD outsole's contact shape in millimetres."""
    outsole = bd.Box(90, 60, 2.5)
    outsole = bd.fillet(outsole.edges().filter_by(bd.Axis.Z), 7)
    toe_edges = [edge for edge in outsole.edges()
                 if edge.center().X > 44 and edge.center().Z > 1.2
                 and edge.length > 20]
    outsole = bd.fillet(toe_edges, 1.2)
    bd.export_stl(outsole, OUT / "rounded_outsole.stl", tolerance=0.15)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reference = mujoco.MjModel.from_xml_path(str(SOURCE))
    source_copy = OUT / "baseline.xml"
    shutil.copy2(SOURCE, source_copy)
    manifest = [{"variant": "baseline", "path": str(source_copy),
                 "sha256": hashlib.sha256(source_copy.read_bytes()).hexdigest(),
                 "footprint_mm": [90, 60], "thickness_mm": 24,
                 "center_x_mm": 14, "sole_mass_each_kg": 0.010118834}]
    manifest.append(make_variant("thin_full_footprint_locked", center_x=0.014,
                                 half_length=0.045, half_thickness=0.00125,
                                 reference=reference))
    manifest.append(make_variant("thick_visible_boot_locked", center_x=0.007,
                                 half_length=0.039, half_thickness=0.012,
                                 reference=reference))
    manifest.append(make_variant("thin_visible_boot_locked", center_x=0.007,
                                 half_length=0.039, half_thickness=0.00125,
                                 reference=reference))
    make_rounded_mesh()
    manifest.append(make_variant("rounded_thin_full_locked", center_x=0.014,
                                 half_length=0.045, half_thickness=0.00125,
                                 reference=reference, rounded_mesh=True))
    (OUT.parent / "models.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
