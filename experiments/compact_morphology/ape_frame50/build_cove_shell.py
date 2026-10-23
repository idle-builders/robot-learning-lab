"""Attach the approved 500 mm Cove shell to the selected direct frame.

This is an articulated visual/CAD study. The saved controller, masses, contacts,
and drive law remain those of the selected frame's earlier physics body.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco
import numpy as np
import trimesh

from experiments.compact_morphology.ape_frame50 import direct_skeleton_candidate as direct


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/sketch-ape-shell/exports/500mm"
FRAME = ROOT / "artifacts/ape-frame50-direct-skeleton"
DEFAULT_OUT = ROOT / "artifacts/ape-frame50-cove-shell-v1"
COLORS = {
    "shell": (0.80, 0.77, 0.70, 1),
    "face": (0.64, 0.52, 0.43, 1),
    "muzzle": (0.92, 0.86, 0.75, 1),
    "joint": (0.37, 0.36, 0.33, 1),
    "soft": (0.25, 0.27, 0.26, 1),
    "eye": (0.025, 0.035, 0.04, 1),
    "accent": (0.80, 0.45, 0.27, 1),
    "window": (0.42, 0.35, 0.29, 0.12),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_body(name: str) -> str | None:
    if name.startswith("torso_") or name == "chest_indicator":
        return "torso"
    if name.startswith("pelvis_"):
        return "pelvis"
    if name == "neck_collar":
        return "head_yaw_body"
    if name.startswith(("head_", "face_", "muzzle_", "mouth_")) or "eye_" in name or "nostril_" in name:
        return "head"
    if not name.startswith(("L_", "R_")):
        raise ValueError(f"unassigned shell part: {name}")
    side = "left" if name.startswith("R_") else "right"  # source X -> simulator Y
    suffix = name[2:]
    if suffix.startswith("shoulder_trim"):
        return f"{side}_shoulder_yaw_body"
    if suffix.startswith("upper_arm") or suffix.startswith("elbow_cuff"):
        return f"{side}_upper_arm"
    if suffix.startswith(("forearm", "wrist_cuff", "paw")):
        return f"{side}_forearm"
    if suffix.startswith(("finger", "thumb", "palm")):
        return None  # the user's simpler smooth-paw direction
    if suffix.startswith("thigh"):
        return f"{side}_thigh"
    if suffix.startswith("knee_cuff") or suffix.startswith("shin"):
        return f"{side}_shin"
    if suffix.startswith(("ankle_cuff", "foot_", "toe_", "sole")):
        if suffix.startswith(("foot_dorsal", "toe_")):
            return None  # replaced by one soft, unsegmented boot
        return f"{side}_foot"
    raise ValueError(f"unassigned shell part: {name}")


def color_role(name: str) -> str:
    if "eye_lens" in name:
        return "eye"
    if "face_mask" in name or "eye_bezel" in name:
        return "face"
    if "muzzle" in name:
        return "muzzle"
    if any(x in name for x in ("collar", "cuff", "trim", "sole", "mouth", "nostril")):
        return "joint"
    if "indicator" in name:
        return "accent"
    return "shell"


def source_to_sim(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    v = np.asarray(mesh.vertices).copy()
    mesh.vertices = np.stack((-v[:, 1], v[:, 0], v[:, 2]), axis=1)
    return mesh


def source_shape_to_sim(shape: bd.Shape) -> bd.Shape:
    return shape.rotate(bd.Axis.Z, 90)


def hollow_ellipsoid(center, radii, wall=2.0):
    outer = bd.Solid.make_sphere(1).scale(radii).translate(center)
    inner = bd.Solid.make_sphere(1).scale(tuple(r - wall for r in radii)).translate(center)
    return outer - inner


def hollow_loft(stations, wall=2.0):
    """Closed curved envelope from z/x/y and X/Y ellipse radii in millimetres."""
    def loft(data):
        return bd.Solid.make_loft([
            bd.Wire.make_ellipse(rx, ry, bd.Plane(origin=(x, y, z)))
            for z, x, y, rx, ry in data])
    inner = [(z + (wall if i == 0 else -wall if i == len(stations) - 1 else 0),
              x, y, rx - wall, ry - wall)
             for i, (z, x, y, rx, ry) in enumerate(stations)]
    return loft(stations) - loft(inner)


def camera_port(shape: bd.Shape) -> bd.Shape:
    # Both source-body cameras face forward from x=49 mm. Keep the opening clear
    # through the decorative face instead of silently blinding visual simulation.
    for y, z in ((15, 457), (-15, 433)):
        shape = shape - bd.Solid.make_cylinder(
            9, 55, bd.Plane(origin=(22, y, z), z_dir=(1, 0, 0)))
    return shape


def rigid_panel_relief(name: str, body: str, shape: bd.Shape) -> tuple[bd.Shape, float]:
    """Remove immediate motor pockets and reserve moving hip/knee seam bands."""
    before = shape.volume
    if name.endswith("_thigh_back"):
        # The saved gait's knee/shin cut-through is at the *rear* distal thigh
        # (z=124..151 mm). Preserve the front sculpted knee instead of slicing
        # the whole leg into visibly disconnected cylinders.
        shape = shape & (bd.Pos(0, 0, 327.5) * bd.Box(500, 500, 345))
    if name.startswith("pelvis_"):
        # The direct-frame hip axes lie outboard of the old shell-study sockets.
        # A rounded relieved socket admits the upper thigh during the saved
        # walking flexion while keeping the V-shaped central belly intact.
        for sign in (-1, 1):
            socket = bd.Solid.make_sphere(1).scale((29, 41, 42)).translate((11, sign * 56, 215))
            shape = shape - socket
    for drive in direct.drives():
        if drive.parent == body:
            shape = shape - direct.box(drive.center, tuple(s + 2 for s in drive.size))
    if shape.volume <= 0:
        raise ValueError(f"relief removed panel: {name}")
    return shape, before - shape.volume


def added_fairings() -> list[tuple[str, str, bd.Shape, str]]:
    result = []
    for side, sign in (("left", 1), ("right", -1)):
        # The original study establishes the silhouette. Separate spherical
        # pods make the motors look like detached ornaments, so hard packaging
        # gaps remain explicit in the report until a joint-specific enclosure
        # can be designed with a real overlap/cuff and a swept-clearance check.
        # One tapered paw replaces five bead-like finger caps and the detached
        # palm. The underside/front slot is reserved for the active grip.
        paw = hollow_loft([
            (105, 18, sign * 151, 20, 22),
            (111, 18, sign * 151, 24, 25),
            (130, 15, sign * 149, 27, 27),
            (150, 10, sign * 146, 24, 24),
            (171, 7, sign * 143, 16, 17),
            (178, 6, sign * 140, 11, 13),
        ])
        paw = paw - (bd.Pos(19, sign * 151, 96) * bd.Box(28, 24, 35))
        result.append((f"{side}_smooth_paw", f"{side}_forearm", paw, "shell"))
        # Broad but low continuous boots keep the original planted silhouette
        # without front-facing artificial toe seams.
        foot = hollow_loft([
            (2, 11, sign * 66, 48, 43),
            (7, 10, sign * 66, 49, 44),
            (17, 9, sign * 65, 43, 39),
            (28, 3, sign * 65, 34, 32),
            (37, -4, sign * 64, 24, 26),
        ])
        result.append((f"{side}_smooth_boot", f"{side}_foot", foot, "shell"))
        # The sketch reads as one enclosed character because the pale rigid
        # panels disappear into dark flexible zones. These are visual gaiter
        # *envelopes*, not a claim that a printed elastomer will clear every
        # coupled pose. A thin, recessed high-hip sleeve bridges the relieved
        # V socket without adding another ivory motor pod.
        hip_gaiter = hollow_loft([
            (190, -17, sign * 62, 16, 22),
            (201, -17, sign * 62, 20, 25),
            (219, -17, sign * 62, 21, 27),
            (237, -18, sign * 62, 19, 25),
            (250, -18, sign * 62, 15, 20),
        ], wall=1.3)
        result.append((f"{side}_hip_flex_envelope", f"{side}_thigh", hip_gaiter, "soft"))
        # A long, tapered knee gaiter spans the actual knee axis. The previous
        # short cuff sat below it and left a black cutout during flexion.
        knee_gaiter = hollow_loft([
            (121, -15, sign * 61, 16, 23),
            (133, -15, sign * 61, 19, 25),
            (148, -15, sign * 61, 20, 27),
            (159, -15, sign * 61, 19, 25),
            (168, -15, sign * 61, 15, 21),
        ], wall=1.3)
        result.append((f"{side}_knee_flex_envelope", f"{side}_shin", knee_gaiter, "soft"))
        ankle_gaiter = hollow_loft([
            (27, -11, sign * 63, 15, 21),
            (39, -11, sign * 63, 18, 24),
            (53, -11, sign * 63, 18, 24),
            (62, -11, sign * 63, 15, 21),
        ], wall=1.2)
        result.append((f"{side}_ankle_flex_envelope", f"{side}_foot", ankle_gaiter, "soft"))
    for side, sign in (("left", 1), ("right", -1)):
        # Recessed dark eyes on the warm mask: the original pale-bezel circles
        # sat behind the mask and read as two raised white discs in simulation.
        eye = bd.Solid.make_sphere(1).scale((4.5, 7.2, 7.2)).translate((44.5, sign * 26, 445))
        result.append((f"{side}_dark_eye", "head", eye, "eye"))
    # Leave the two real camera bores open. An extra decorative disk above one
    # eye read as a third eye, and an opaque cap would silently blind the sim.
    return result


def mesh_shape(shape: bd.Shape) -> trimesh.Trimesh:
    verts, faces = shape.tessellate(0.55, 0.45)
    return trimesh.Trimesh(vertices=np.asarray([tuple(v) for v in verts]),
                           faces=np.asarray(faces), process=False)


def link_cas_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if target.exists() or target.is_symlink():
            continue
        if not item.is_symlink():
            raise ValueError(f"selected frame asset is not immutable CAS evidence: {item}")
        # Both folders are one level below their artifact root.
        target.symlink_to(os.readlink(item))


def build(out: Path) -> dict:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() for p in out.rglob("*")):
        raise ValueError("sealed output: choose a fresh --out")
    selection = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if selection["id"] != "cove-50-direct-frame":
        raise ValueError("shell must be reviewed against the locked Cove-50 Direct Frame")
    if digest(ROOT / selection["reference_sketch"]["path"]) != selection["reference_sketch"]["sha256"]:
        raise ValueError("approved sketch changed")
    for key in ("report", "visual_body"):
        if digest(ROOT / selection["evidence"][key]) != selection["evidence"][f"{key}_sha256"]:
            raise ValueError(f"selected {key} changed")
    frame_report = json.loads((FRAME / "report.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(FRAME / "ape-frame50-direct-visual.xml"))
    neutral = mujoco.MjData(model)
    neutral.qpos[:] = model.qpos0
    mujoco.mj_forward(model, neutral)
    source_parts = {p.label: p for p in bd.import_step(str(SOURCE / "ape_shell_assembly.step")).children}
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    selected_parts: list[tuple[str, str, bd.Shape, trimesh.Trimesh, str]] = []
    for entry in manifest["parts"]:
        name = entry["name"]
        body = target_body(name)
        if body is None or "eye_bezel" in name or "eye_lens" in name:
            continue
        shape = source_shape_to_sim(source_parts[name])
        # The finished face is a functional sensor surface, not an opaque cap.
        if name in ("head_cranium_front", "face_mask", "muzzle_shell"):
            cut = camera_port(shape)
            if len(cut.solids()) == 1 and cut.volume > 0:
                shape = cut
        shape, removed = rigid_panel_relief(name, body, shape)
        if shape.volume <= 0:
            raise ValueError(f"empty shell panel: {name}")
        # Existing exported STL is already a valid, watertight part. Re-mesh
        # the few panels altered by sensor apertures from the actual CAD solid.
        tri = (mesh_shape(shape) if removed > 0.1 or name in ("head_cranium_front", "face_mask", "muzzle_shell")
               else source_to_sim(trimesh.load_mesh(SOURCE / "stl" / f"{name}.stl", process=False)))
        selected_parts.append((name, body, shape, tri, color_role(name)))
    for name, body, shape, role in added_fairings():
        selected_parts.append((name, body, shape, mesh_shape(shape), role))
    link_cas_files(FRAME / "meshes", out / "meshes")
    link_cas_files(FRAME / "cad-glb", out / "frame-glb")
    (out / "kinematics.json").symlink_to(os.readlink(FRAME / "kinematics.json"))
    tree = ET.parse(FRAME / "ape-frame50-direct-visual.xml")
    root = tree.getroot()
    assets = root.find("asset")
    world = root.find("worldbody")
    assert assets is not None and world is not None
    body_nodes = {n.get("name"): n for n in world.iter("body")}
    shell_mesh_dir = out / "shell-meshes"
    shell_mesh_dir.mkdir()
    scenes: dict[str, trimesh.Scene] = {}
    records = []
    for name, body, shape, tri, role in selected_parts:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        if body_id < 0:
            raise ValueError(f"missing frame body: {body}")
        rotation = neutral.xmat[body_id].reshape(3, 3)
        origin = neutral.xpos[body_id]
        tri.vertices = (np.asarray(tri.vertices) / 1000 - origin) @ rotation
        tri.visual.vertex_colors = tuple(round(255 * x) for x in COLORS[role])
        filename = f"shell_{name}.obj"
        mesh_path = shell_mesh_dir / filename
        tri.export(mesh_path)
        ET.SubElement(assets, "mesh", {"name": f"shell_{name}", "file": f"shell-meshes/{filename}"})
        ET.SubElement(body_nodes[body], "geom", {"name": f"shell_{name}",
            "type": "mesh", "mesh": f"shell_{name}",
            "rgba": " ".join(str(x) for x in COLORS[role]), "mass": "0",
            "contype": "0", "conaffinity": "0", "group": "2"})
        scenes.setdefault(body, trimesh.Scene()).add_geometry(tri, geom_name=name, node_name=name)
        density = 1.20 if role == "soft" else 1.27
        mass_kg = shape.volume / 1000 * density / 1000
        center = shape.center(bd.CenterOf.MASS)
        records.append({"name": name, "body": body, "role": role,
                        "volume_cm3": round(shape.volume / 1000, 3),
                        "illustrative_mass_kg": round(mass_kg, 5),
                        "neutral_center_mm": [round(float(x), 2) for x in center],
                        "mesh_sha256": digest(mesh_path)})
    glb_dir = out / "shell-glb"
    glb_dir.mkdir()
    shell_glbs = {}
    for body, scene in scenes.items():
        path = glb_dir / f"{body}.glb"
        scene.export(path)
        shell_glbs[body] = {"file": str(path.relative_to(out)), "sha256": digest(path)}
    visual_path = out / "cove50-shell-visual.xml"
    ET.indent(tree, space="  ")
    tree.write(visual_path, encoding="unicode", xml_declaration=True)
    combined = mujoco.MjModel.from_xml_path(str(visual_path))
    if combined.nq != model.nq or combined.nu != model.nu or not np.isclose(combined.body_mass.sum(), model.body_mass.sum(), atol=1e-6):
        raise ValueError("shell changed selected control or physics body")
    # An articulated STEP assembly is the editable envelope for future fit
    # passes; it retains each shell panel as a separate solid.
    assembly = bd.Compound(label="Cove-50 articulated shell at neutral",
                           children=[p[2] for p in selected_parts])
    step_path = out / "cove50-shell-neutral.step"
    bd.export_step(assembly, str(step_path))
    report = {
        "title": "Cove-50 shell on Direct Frame", "kind": "Engineering",
        "summary": "Sketch-derived enclosed shell, animated on the selected direct frame with joint-range explorer.",
        "selection_id": selection["id"], "selection_sha256": digest(ROOT / "robot/ape-current-design.json"),
        "source_sketch_sha256": selection["reference_sketch"]["sha256"],
        "source_shell_step_sha256": digest(SOURCE / "ape_shell_assembly.step"),
        "source_shell_manifest_sha256": digest(SOURCE / "manifest.json"),
        "frame_report_sha256": digest(FRAME / "report.json"),
        "source_physics_body_sha256": frame_report["source_body_sha256"],
        "source_visual_body_sha256": digest(FRAME / "ape-frame50-direct-visual.xml"),
        "shell_visual_body_sha256": digest(visual_path),
        "shell_step_sha256": digest(step_path),
        "shell_parts": len(records), "shell_parts_by_body": records,
        "frame_body_glbs": {body: {"file": "frame-glb/" + Path(info["file"]).name,
                                   "sha256": info["sha256"]} for body, info in frame_report["body_glbs"].items()},
        "shell_body_glbs": shell_glbs,
        "old_model_mass_kg": float(model.body_mass.sum()),
        "shell_nominal_material_cm3": round(sum(p[2].volume for p in selected_parts) / 1000, 2),
        "illustrative_shell_mass_kg": round(sum(p["illustrative_mass_kg"] for p in records), 4),
        "illustrative_density_g_cm3": {"rigid_panels": 1.27, "soft_joint_envelopes": 1.20},
        "mass_estimate_status": "CAD volume times stated assumed density; not added to simulator dynamics, excludes fasteners/paint/liner and any redesign for actual printability",
        "limits": ["This is visual geometry: shell collision and mass are not active in MJLab.",
                   "The saved policy trace predates this direct frame and shell; this is articulated replay, not newly learned control.",
                   "Nominal 2 mm skins and dark joint gaiters are CAD envelopes, not printable fastening or validated flexible-seal designs.",
                   "Independent simulator joint ranges are not mechanically safe coupled ranges; inspect shell conflicts before actuating hardware."],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "inspector.json").write_text(json.dumps({k: report[k] for k in ("title", "kind", "summary")}, indent=2) + "\n")
    shutil.copyfile(Path(__file__).with_name("cove_shell_viewer.html"), out / "index.html")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build(args.out)
    print(json.dumps({"out": str(args.out), "shell_parts": report["shell_parts"],
                      "step_sha256": report["shell_step_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
