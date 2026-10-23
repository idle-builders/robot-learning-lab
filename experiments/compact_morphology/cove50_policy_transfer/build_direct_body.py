"""Derive a provisional physics body from the selected Cove-50 Direct Frame.

The selected CAD is an exact geometry source, but print density, hub material,
inertia from each part's bounding box, and the contact/servo law are explicit
simulation assumptions. This builder leaves the selected source artifacts alone.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.compact_morphology.ape_frame50 import direct_skeleton_candidate as cad


SELECTED = ROOT / "robot/ape-current-design.json"
OUT = ROOT / "artifacts/ape-cove50-transfer-20261008"
PETG_KG_PER_MM3 = 1.28e-6
STEEL_KG_PER_MM3 = 7.85e-6
CATALOG_MASS = {"ST3250": 0.0745, "ST3215": 0.055, "XL330": 0.018}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(values) -> str:
    return " ".join(f"{float(x):.10g}" for x in values)


def selected_source() -> tuple[dict, dict, Path]:
    chosen = json.loads(SELECTED.read_text())
    if chosen["id"] != "cove-50-direct-frame":
        raise ValueError("The selected design changed; review it before building")
    checks = [
        (chosen["generator"]["path"], chosen["generator"]["sha256"]),
        (chosen["evidence"]["report"], chosen["evidence"]["report_sha256"]),
        (chosen["evidence"]["visual_body"], chosen["evidence"]["visual_body_sha256"]),
        (chosen["physics"]["source_body"], chosen["physics"]["source_body_sha256"]),
    ]
    for name, expected in checks:
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError(f"Pinned selection hash mismatch: {name}")
    report = json.loads((ROOT / chosen["evidence"]["report"]).read_text())
    source = ROOT / chosen["evidence"]["visual_body"]
    return chosen, report, source


def part_mass(name: str, shape, cases: dict[str, str]) -> tuple[float, str]:
    if name in cases:
        model = cases[name]
        return CATALOG_MASS[model], f"catalog {model} mass"
    if name == "battery_4s850":
        return 0.107, "catalog 4S850 battery mass"
    if name == "camera_module":
        return 0.004, "catalog Camera Module 3 mass"
    if name == "imu_reserve":
        return 0.005, "0.7 g board plus assumed mount and harness"
    density = STEEL_KG_PER_MM3 if name.endswith("_rotating_hub") else PETG_KG_PER_MM3
    material = "assumed solid steel hub" if name.endswith("_rotating_hub") else "gross solid PETG-HF stock"
    return float(shape.volume) * density, material


def main() -> None:
    selected, source_report, source = selected_source()
    physics = mujoco.MjModel.from_xml_path(str(source))
    state = mujoco.MjData(physics)
    state.qpos[:] = physics.qpos0
    mujoco.mj_forward(physics, state)
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", "ape_cove50_direct_frame_provisional")
    world = root.find("worldbody")
    assert world is not None
    bodies = {b.get("name"): b for b in world.iter("body")}
    drives = cad.drives()
    cases = {f"{d.name}_{d.model}_case": d.model for d in drives}
    rows: list[dict] = []
    moments: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {name: [] for name in bodies}

    for _, body_name, parts in cad.visual_parts():
        bid = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise ValueError(f"Missing CAD body: {body_name}")
        rotation = state.xmat[bid].reshape(3, 3)
        origin = state.xpos[bid]
        for name, shape in parts.items():
            mass, source_name = part_mass(name, shape, cases)
            bounds = shape.bounding_box()
            center = np.asarray(tuple(shape.center()), dtype=float) / 1000
            size = np.asarray(tuple(bounds.size), dtype=float) / 1000
            local = (center - origin) @ rotation
            box_i_world = np.diag(mass * np.array([
                size[1] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[2] ** 2,
                size[0] ** 2 + size[1] ** 2,
            ]) / 12)
            box_i_local = rotation.T @ box_i_world @ rotation
            moments[body_name].append((mass, local, box_i_local))
            rows.append({"body": body_name, "part": name, "mass_kg": round(mass, 8),
                         "source_or_assumption": source_name,
                         "center_world_m": center.round(6).tolist(),
                         "bbox_size_m": size.round(6).tolist()})

    # An explicit 150 g compute/power allocation existed in the prior body.
    # Retain it as a visible assumption on the torso, separate from sourced CAD.
    torso_id = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_r = state.xmat[torso_id].reshape(3, 3)
    reserve_center = np.asarray([0.0, 0.0, 0.31])
    reserve_local = (reserve_center - state.xpos[torso_id]) @ torso_r
    reserve_size = np.asarray([0.04, 0.05, 0.04])
    reserve_i = np.diag(0.15 * np.array([
        reserve_size[1] ** 2 + reserve_size[2] ** 2,
        reserve_size[0] ** 2 + reserve_size[2] ** 2,
        reserve_size[0] ** 2 + reserve_size[1] ** 2,
    ]) / 12)
    moments["torso"].append((0.15, reserve_local, torso_r.T @ reserve_i @ torso_r))
    rows.append({"body": "torso", "part": "compute_power_reserve", "mass_kg": 0.15,
                 "source_or_assumption": "unselected compute/power mass allowance",
                 "center_world_m": reserve_center.tolist(), "bbox_size_m": reserve_size.tolist()})

    # Replace the previous body's inertial distribution, not its joint chain.
    for name, body in bodies.items():
        items = moments[name]
        if not items:
            raise ValueError(f"CAD did not supply mass for {name}")
        total = sum(m for m, _, _ in items)
        com = sum((m * p for m, p, _ in items), np.zeros(3)) / total
        inertia = np.zeros((3, 3))
        for mass, center, own in items:
            d = center - com
            inertia += own + mass * ((d @ d) * np.eye(3) - np.outer(d, d))
        values = np.linalg.eigvalsh(inertia)
        if total <= 0 or min(values) <= 0:
            raise ValueError(f"Nonphysical inertial estimate for {name}: {values}")
        for old in body.findall("inertial"):
            body.remove(old)
        ET.SubElement(body, "inertial", {
            "pos": fmt(com), "mass": f"{total:.10g}",
            "fullinertia": fmt([inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                inertia[0, 1], inertia[0, 2], inertia[1, 2]]),
        })
        for geom in body.findall("geom"):
            geom.set("mass", "0")

    # The former belt body's hip contact pairs refer to absent belt hardware.
    # Remove them; direct-drive case pairs are the first physical collision set.
    contact = root.find("contact")
    assert contact is not None
    old_pair_count = len(contact.findall("pair"))
    for pair in contact.findall("pair"):
        contact.remove(pair)
    old_proxy_count = 0
    for body in bodies.values():
        for geom in list(body.findall("geom")):
            if geom.get("name", "").endswith("_proxy"):
                body.remove(geom)
                old_proxy_count += 1
    motor_geoms = {}
    for body in bodies.values():
        for geom in body.findall("geom"):
            name = geom.get("name", "")
            if name.endswith("_motor_case"):
                motor_geoms[name.removesuffix("_motor_case")] = (body.get("name"), geom)
    for drive in drives:
        parent, geom = motor_geoms[drive.name]
        if parent != drive.parent:
            raise ValueError(f"Motor parent changed: {drive.name}")
        bid = mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_BODY, parent)
        rotation = state.xmat[bid].reshape(3, 3)
        local = (np.asarray(drive.center) / 1000 - state.xpos[bid]) @ rotation
        geom.set("pos", fmt(local))
        geom.set("size", fmt(np.asarray(drive.size) / 2000))
    if set(motor_geoms) != {d.name for d in drives}:
        raise ValueError("Motor case set differs from selected direct frame")
    for i, a in enumerate(drives):
        for b in drives[i + 1:]:
            if a.parent != b.parent:
                ET.SubElement(contact, "pair", {
                    "name": f"direct_cases_{a.name}_{b.name}",
                    "geom1": f"{a.name}_motor_case", "geom2": f"{b.name}_motor_case",
                    "condim": "1",
                })

    # The direct CAD plate is 120 x 70 x 9 mm at world z=5 mm in neutral.
    for side in ("left", "right"):
        sole = next(g for g in bodies[f"{side}_foot"].findall("geom")
                    if g.get("name") == f"{side}_sole")
        sole.set("pos", "0.012 0 -0.038")
        sole.set("size", "0.060 0.035 0.0045")
    assets = root.find("asset")
    assert assets is not None
    for mesh in assets.findall("mesh"):
        mesh.set("file", "../ape-frame50-direct-skeleton/" + mesh.get("file"))

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "cove50-direct-provisional.xml"
    ET.indent(tree, space="  ")
    tree.write(target, encoding="unicode", xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(target))
    if model.nu != 21 or model.nq != physics.nq:
        raise ValueError("Direct-frame body changed joint or action dimensions")
    # Repeated native training does not need the zero-mass, noncontact CAD
    # display meshes. Keep the full model for articulated visualization and a
    # lean body with identical physics for GPU rollouts.
    lean_tree = ET.parse(target)
    lean_root = lean_tree.getroot()
    lean_root.set("model", "ape_cove50_direct_frame_training")
    lean_assets = lean_root.find("asset")
    if lean_assets is not None:
        lean_root.remove(lean_assets)
    lean_world = lean_root.find("worldbody")
    assert lean_world is not None
    for body in lean_world.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("name", "").startswith("cad_"):
                body.remove(geom)
    lean = OUT / "cove50-direct-training.xml"
    ET.indent(lean_tree, space="  ")
    lean_tree.write(lean, encoding="unicode", xml_declaration=True)
    lean_model = mujoco.MjModel.from_xml_path(str(lean))
    if lean_model.nu != model.nu or lean_model.nq != model.nq or not np.allclose(
        lean_model.body_mass, model.body_mass, atol=1e-9
    ):
        raise ValueError("Training body changed mass or policy interface")
    # Names and contact-relevant parameters must remain the same after
    # display-only meshes are dropped.
    physics_names = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
                               for i in range(m.ngeom)
                               if not mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i).startswith("cad_")]
    if physics_names(model) != physics_names(lean_model) or model.npair != lean_model.npair:
        raise ValueError("Training body changed contact geometry")
    total = float(model.body_mass.sum())
    ledger = {
        "selected_design_id": selected["id"],
        "selected_design_sha256": digest(SELECTED),
        "source_report_sha256": selected["evidence"]["report_sha256"],
        "source_visual_body_sha256": selected["evidence"]["visual_body_sha256"],
        "cad_generator_sha256": selected["generator"]["sha256"],
        "generated_body": str(target.relative_to(ROOT)),
        "generated_body_sha256": digest(target),
        "training_body": str(lean.relative_to(ROOT)),
        "training_body_sha256": digest(lean),
        "compiled_mass_kg": round(total, 6),
        "source_body_mass_kg": source_report["mass_kg"],
        "powered_axes": model.nu,
        "old_belt_contact_pairs_removed": old_pair_count,
        "old_belt_contact_proxies_removed": old_proxy_count,
        "direct_motor_case_pairs": len(contact.findall("pair")),
        "mass_model": "exact CAD solid volumes and part centers; each part's bounding box approximates inertia",
        "contact_model": "CAD plate sized sole plus direct motor case pairs; other frame contacts remain proxy or absent",
        "drive_model": "existing rated-effort-capped position servos; unmeasured impedance and duty",
        "parts": rows,
    }
    (OUT / "body-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    print(json.dumps({k: v for k, v in ledger.items() if k != "parts"}, indent=2))


if __name__ == "__main__":
    main()
