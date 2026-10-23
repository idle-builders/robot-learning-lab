"""Consolidate sketch-06 CAD and the 21-axis layout into one *auditable draft*.

This does not invent mounts for colliding motor boxes. The exported assembly
stays explicitly blocked until shafts, bearings, brackets, cables and moving
skins are connected and cleared. Run from the repository root with .venv.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import warnings

import build123d as bd
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import sketch_shell as shell
from check_shell_sweep import PAIRS, volume
from make_boot_model import cad_outsole
from replay_shell_cases import owner


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-functional-cad"
SKELETON = ROOT / "artifacts/ape-sketch-shell-kinematics-20261002/shell-aligned-21.xml"
CASES = ROOT / "artifacts/ape-sketch-shell-kinematics-20261002/case-screen-21.json"
PLA_DENSITY_G_CM3 = 1.24  # Bambu PLA Basic Gradient; provisional print material.
TPU_DENSITY_G_CM3 = 1.22  # Bambu TPU 95A HF; provisional outsole material.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cad_mesh(shape: bd.Shape, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(0.12, 0.12)
    result = trimesh.Trimesh(
        vertices=np.array([[v.X, v.Y, v.Z] for v in vertices]),
        faces=np.asarray(faces), process=False,
    )
    result.merge_vertices()
    result.visual.vertex_colors = rgba
    return result


def material_item(name: str, body: str, shape: bd.Shape,
                  density_g_cm3: float, material: str) -> dict:
    mass = shape.volume * density_g_cm3 * 1e-6
    com = np.array(tuple(shape.center(bd.CenterOf.MASS))) / 1000
    inertia = np.array(shape.matrix_of_inertia) * density_g_cm3 * 1e-12
    return {"name": name, "body": body, "material": material,
            "mass_kg": mass, "com_world_m": com,
            "inertia_com_world_kg_m2": inertia}


def box_item(name: str, body: str, center_mm: np.ndarray,
             size_mm: np.ndarray, rotation: np.ndarray,
             mass_g: float, material: str) -> dict:
    mass = mass_g / 1000
    x, y, z = size_mm / 1000
    inertia_local = mass / 12 * np.diag((y*y + z*z, x*x + z*z, x*x + y*y))
    return {"name": name, "body": body, "material": material,
            "mass_kg": mass, "com_world_m": center_mm / 1000,
            "inertia_com_world_kg_m2": rotation @ inertia_local @ rotation.T}


def aggregate(items: list[dict], origin_m: np.ndarray) -> dict:
    mass = sum(item["mass_kg"] for item in items)
    if mass == 0:
        return {"mass_kg": 0, "com_local_m": [0, 0, 0],
                "inertia_com_local_kg_m2": np.zeros((3, 3)).tolist()}
    com = sum((item["mass_kg"] * item["com_world_m"] for item in items),
              np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for item in items:
        delta = item["com_world_m"] - com
        inertia += item["inertia_com_world_kg_m2"] + item["mass_kg"] * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return {"mass_kg": mass, "com_local_m": (com - origin_m).tolist(),
            "inertia_com_local_kg_m2": inertia.tolist()}


def shape_box(center_mm: np.ndarray, size_mm: np.ndarray,
              rotation: np.ndarray) -> bd.Shape:
    # All three orientation columns are orthonormal rotations of a datasheet
    # envelope. An enclosure is never inferred from this box alone.
    # build123d's Rot(x, y, z) is not SciPy's extrinsic xyz convention. Apply
    # the three intrinsic shape rotations explicitly, as in build.py.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        x, y, z = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    return (bd.Pos(*center_mm) * bd.Rot(0, 0, z) * bd.Rot(0, y, 0) *
            bd.Rot(x, 0, 0) * bd.Box(*size_mm))


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shell.build()
    model = mujoco.MjModel.from_xml_path(str(SKELETON))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)
    case_screen = json.loads(CASES.read_text())

    scene = trimesh.Scene()
    items: list[dict] = []
    shapes: dict[str, bd.Shape] = {}
    shell_parts = {part.name: part for part in shell.PARTS}
    for part in shell.PARTS:
        body = owner(part.name)
        shapes[part.name] = part.shape
        items.append(material_item(part.name, body, part.shape,
                                   PLA_DENSITY_G_CM3, "PLA shell proxy"))
        scene.add_geometry(shell.mesh(part), geom_name=part.name, node_name=part.name)

    for side, sign in (("left", 1), ("right", -1)):
        name = f"{side}_outsole"
        shape = cad_outsole(sign)
        shapes[name] = shape
        items.append(material_item(name, f"{side}_foot", shape,
                                   TPU_DENSITY_G_CM3, "TPU outsole proxy"))
        scene.add_geometry(cad_mesh(shape, (63, 67, 62, 255)),
                           geom_name=name, node_name=name)

    case_names = []
    for case in case_screen["cases"]:
        joint = case["joint"]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        driven_body = int(model.jnt_bodyid[jid])
        fixed_body = int(model.body_parentid[driven_body])
        fixed_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, fixed_body)
        center = np.array(case["center_mm"], dtype=float)
        size = np.array(case["size_mm"], dtype=float)
        rotation = np.array(case["rotation_matrix"], dtype=float)
        name = f"{joint}_motor_envelope"
        shape = shape_box(center, size, rotation)
        shapes[name] = shape
        case_names.append(name)
        items.append(box_item(name, fixed_name, center, size, rotation,
                              case["mass_g"], case["source_case"]))
        scene.add_geometry(cad_mesh(shape, (66, 105, 145, 180)),
                           geom_name=name, node_name=name)

    battery_center = np.array((0., 0., 191.))
    battery_size = np.array((60., 30., 30.5))
    battery_name = "battery_4s_850_envelope"
    battery = bd.Pos(*battery_center) * bd.Box(*battery_size)
    shapes[battery_name] = battery
    items.append(box_item(battery_name, "pelvis", battery_center, battery_size,
                          np.eye(3), 107, "Tattu 4S 850 mAh candidate"))
    scene.add_geometry(cad_mesh(battery, (83, 148, 96, 185)),
                       geom_name=battery_name, node_name=battery_name)

    for name, shape in shapes.items():
        shape.label = name
        part = shell_parts.get(name)
        color = (shell.COLORS[part.color] if part else
                 (63, 67, 62, 255) if name.endswith("_outsole") else
                 (83, 148, 96, 255) if name == battery_name else
                 (66, 105, 145, 255))
        shape.color = bd.Color(*(channel / 255 for channel in color[:3]))
    bd.export_step(bd.Compound(children=list(shapes.values())),
                   str(OUT / "assembly-neutral.step"))
    scene.export(OUT / "assembly-neutral.glb")

    joints = []
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name == "root":
            continue
        bid = int(model.jnt_bodyid[jid])
        parent = int(model.body_parentid[bid])
        joints.append({
            "name": name,
            "parent_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent),
            "moving_body": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid),
            "axis_world_neutral": np.asarray(data.xaxis[jid]).tolist(),
            "pivot_world_mm_neutral": (np.asarray(data.xanchor[jid]) * 1000).tolist(),
            "proxy_range_rad_not_cad_verified": np.asarray(model.jnt_range[jid]).tolist(),
            "mechanical_interface": "UNRESOLVED: no validated output shaft, bearing and bracket",
        })
    body_ledger = {}
    for bid in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        body_ledger[name] = aggregate([item for item in items if item["body"] == name],
                                      np.asarray(data.xpos[bid]))
    total = aggregate(items, np.zeros(3))

    # Exact neutral BRep overlap complements the older corner-only motor
    # screen. Do not subtract intersecting shell mass: it needs redesign.
    shell_overlap = {}
    for a, b in PAIRS:
        hit = volume(shapes[a].intersect(shapes[b]))
        if hit > 1:
            shell_overlap[f"{a} ↔ {b}"] = round(hit, 1)
    case_shell_hits = case_screen["shell_hits_mm3"]
    report = {
        "stage": "mechanical assembly draft: BLOCKED, no gym export or training",
        "design_source": "sketch_shell.py (approved sketch-06 exterior)",
        "skeleton_source": str(SKELETON.relative_to(ROOT)),
        "skeleton_source_sha256": sha(SKELETON),
        "case_layout_source": str(CASES.relative_to(ROOT)),
        "case_layout_sha256": sha(CASES),
        "source_file_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in (
                Path(shell.__file__), Path(__file__),
                Path(__file__).with_name("make_boot_model.py"),
                ROOT / "docs/design/ape-concepts/06-tumble-compact-shoulder.png",
            )
        },
        "source_links": {
            "PLA_density": "https://eu.store.bambulab.com/nl-nl/products/pla-basic-gradient?variant=53638219006300",
            "TPU_density": "https://uk.store.bambulab.com/products/tpu-95a-hf",
            "motor_catalog": "docs/design/ape-concepts/drive-catalog.md",
            "battery": "https://genstattu.com/tattu-850mah-14-8v-75c-4s1p-lipo-battery-pack-with-xt60-plug.html",
        },
        "shell_part_count": len(shell.PARTS),
        "axis_count": len(joints),
        "envelope_count": len(case_names),
        "parts": [{**{k: v for k, v in item.items() if k not in (
            "com_world_m", "inertia_com_world_kg_m2")},
                   "com_world_m": item["com_world_m"].tolist()}
                  for item in items],
        "joints": joints,
        "body_mass_properties_provisional": body_ledger,
        "whole_assembly_mass_properties_provisional": total,
        "neutral_case_pair_clashes_at_1mm": case_screen["neutral_case_clashes_1mm"],
        "neutral_case_shell_intersections_mm3": case_shell_hits,
        "neutral_shell_pair_intersections_mm3": shell_overlap,
        "missing_mechanics": [
            f"{len(joints)} supported output shafts/bearings/brackets and vendor horn coordinates",
            "Battery tray/retention, power conversion and serviceable wire path",
            "Working opposed-jaw hand; the current smooth hand is one closed skin",
            "Actual joint hard stops and swept skin/structure clearance",
            "Printed split, fastener bosses and tolerances",
        ],
        "mass_note": "BRep print volume plus uniform motor/battery boxes; includes mutually overlapping solids and omits structure, hardware, wiring and electronics. This ledger must not be promoted to the gym yet.",
        "files_sha256": {p.name: sha(p) for p in
                         (OUT / "assembly-neutral.step", OUT / "assembly-neutral.glb")},
    }
    (OUT / "assembly.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(CASES, OUT / "case-layout.json")
    shutil.copy2(ROOT / "artifacts/ape-sketch-shell-kinematics-20261002/motion-screen-21.json",
                 OUT / "pose-screen.json")
    shutil.copy2(Path(__file__).with_name("functional_cad_viewer.html"), OUT / "index.html")
    print(f"{len(shapes)} CAD items, {len(joints)} axes, "
          f"{total['mass_kg']:.3f} kg provisional, "
          f"{len(case_screen['neutral_case_clashes_1mm'])} motor clashes, "
          f"{len(shell_overlap)} shell-pair overlaps -> {OUT}")


if __name__ == "__main__":
    build()
