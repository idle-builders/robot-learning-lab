"""Overlay a retained cover candidate onto its matching full-robot CAD render.

Visual review only. This does not establish full-body collision or mass validity.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import mujoco
from PIL import Image
from build123d import Compound, export_step, import_step
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.cad_identity import geometry_agrees, geometry_record


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(assembly, cover, output):
    config = json.loads((assembly / "resolved-config.json").read_text())
    if config != json.loads((cover / "resolved-config.json").read_text()):
        raise ValueError("cover configuration differs from full robot")
    r = json.loads((cover / "report.json").read_text())
    base = json.loads((assembly / "report.json").read_text())
    for directory, report in [(assembly, base), (cover, r)]:
        for name, h in report["output_hashes"].items():
            if sha(directory / name) != h:
                raise ValueError(f"changed source output: {name}")
    output.mkdir(parents=True, exist_ok=False)
    root = ET.parse(assembly / "mechanism.xml").getroot()
    candidate = ET.parse(cover / "mechanism.xml").getroot()
    changed = set(r["geometry"])
    retained = []
    for mesh in list(root.find("asset")):
        if mesh.tag != "mesh":
            continue
        if mesh.get("name") in changed:
            root.find("asset").remove(mesh)
        else:
            p = (assembly / mesh.get("file")).absolute()
            mesh.set("file", str(p))
            retained.append(p)
    for g in list(root.find("worldbody")):
        if g.tag == "geom" and g.get("mesh") in changed:
            root.find("worldbody").remove(g)
    for mesh in candidate.find("asset"):
        if mesh.tag == "mesh" and mesh.get("name") in changed:
            m = deepcopy(mesh)
            p = (cover / m.get("file")).absolute()
            m.set("file", str(p))
            retained.append(p)
            root.find("asset").append(m)
    for g in candidate.find("worldbody"):
        if g.tag == "geom" and g.get("mesh") in changed:
            root.find("worldbody").append(deepcopy(g))
    ET.ElementTree(root).write(output / "preview.xml")
    m = mujoco.MjModel.from_xml_path(str(output / "preview.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = m.stat.center
    camera.distance = m.stat.extent * 1.65
    camera.azimuth = 135
    camera.elevation = -15
    with mujoco.Renderer(m, height=800, width=960) as renderer:
        renderer.update_scene(d, camera=camera)
        Image.fromarray(renderer.render()).save(output / "preview.png")
    # Producer parts use separate owner-local frames. Supply a placed STEP and
    # an explicit frame/role manifest for downstream composition.
    for name in ["lab/config.py", "lab/robot.py", "lab/leg_mechanism.py"]:
        if sha(Path(name)) != r["source_hashes"][name]:
            raise ValueError(f"kinematic source changed: {name}")
    robot = make_robot(RobotConfig.model_validate(config))
    frames = robot.forward_kinematics()
    local = import_step(cover / "cover-assembly.step")
    if len(local.children) != len(r["geometry"]) or {
        s.label for s in local.children
    } != set(r["geometry"]):
        raise ValueError("cover STEP inventory differs")
    placed = []
    parts = []
    for shape in local.children:
        name = shape.label
        side = name.split("_")[0]
        if side not in ["left", "right"]:
            raise ValueError("unknown cover owner")
        owner = f"{side}_forearm"
        if name == f"{side}_forearm_spine":
            role = "spine"
        elif name.startswith(f"{side}_cover_halves_"):
            role = "shell"
        elif name.startswith(f"{side}_cover_screws_"):
            role = "screw"
        elif name.startswith(f"{side}_cover_inserts_"):
            role = "insert"
        else:
            raise ValueError(f"unknown part role: {name}")
        world = transform_shape(shape, frames[owner])
        world.label = name
        placed.append(world)
        parts.append(
            dict(
                name=name,
                owner=owner,
                role=role,
                neutral_owner_to_world=frames[owner].tolist(),
            )
        )
    export_step(Compound(children=placed), output / "posed-cover-parts.step")
    restored = import_step(output / "posed-cover-parts.step")
    restored_by_name = {p.label: p for p in restored.children}
    if len(restored.children) != len(placed) or set(restored_by_name) != {
        p.label for p in placed
    }:
        raise ValueError("posed STEP inventory mismatch")
    comparisons = [
        dict(
            name=p.label,
            expected=geometry_record(p),
            actual=geometry_record(restored_by_name[p.label]),
        )
        for p in placed
    ]
    if not all(geometry_agrees(c["actual"], c["expected"]) for c in comparisons):
        raise ValueError("posed STEP geometry mismatch")
    manifest = dict(
        schema_version=1,
        step_coordinates="neutral world",
        step_units="mm",
        transform_translation_units="m",
        rotation="3x3 right-handed in 4x4 matrix",
        source_coordinates="owner-local millimetres",
        parts=parts,
        replaces=["left_forearm_spine", "right_forearm_spine"],
        scope="Placement and role mapping only; no full-body clearance claim",
        step_roundtrip=comparisons,
    )
    (output / "part-frames.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bindings = {str(p): sha(p) for p in retained}
    for directory in [assembly, cover]:
        for name in ["report.json", "resolved-config.json", "mechanism.xml"]:
            p = directory / name
            bindings[str(p)] = sha(p)
    bindings[str(cover / "cover-assembly.step")] = sha(cover / "cover-assembly.step")
    source_hashes = {}
    for source in [Path(__file__).resolve(), Path("tools/cad_identity.py").resolve()]:
        name = source.relative_to(Path.cwd())
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target)
        source_hashes[str(name)] = sha(target)
    write = dict(
        status="rendered",
        scope="visual overlay; no full-body collision or dynamics claim",
        cover_status=r["status"],
        changed_groups=sorted(changed),
        source_bindings=bindings,
        source_hashes=source_hashes,
        output_hashes={
            name: sha(output / name)
            for name in [
                "preview.xml",
                "preview.png",
                "posed-cover-parts.step",
                "part-frames.json",
            ]
        },
    )
    (output / "report.json").write_text(json.dumps(write, indent=2) + "\n")
    print(json.dumps(dict(status="rendered", output=str(output))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["assembly", "cover", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    run(a.assembly, a.cover, a.output)
