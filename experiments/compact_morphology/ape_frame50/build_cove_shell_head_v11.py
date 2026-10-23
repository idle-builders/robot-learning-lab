"""Build an unselected, analytic replacement for the Cove-50 front cranium.

The selected v10 STEP's imported head-front topology is wrong. Reconstruct the
dimensioned sketch loft, face aperture and split *before* STEP exchange instead
of transforming the imported v10 B-rep. All dimensions are millimetres.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import warnings

import build123d as bd
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import trimesh
from trimesh.proximity import closest_point

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.sketch_ape_shell.build import box, cranial_loft, prism_xz


SOURCE_SCRIPT = ROOT / "experiments/sketch_ape_shell/build.py"
SOURCE_STL = ROOT / "artifacts/sketch-ape-shell/exports/500mm/stl/head_cranium_front.stl"
SELECTED = ROOT / "artifacts/ape-frame50-cove-shell-v10"
AUDIT = ROOT / "studies/cove50-actuators/shell-head-audit"
WB_SCRIPT = ROOT / "studies/cove50-actuators/cad-variants/whole-body/build.py"
WB_OUTPUT = WB_SCRIPT.parent / "output"
ARCH = ROOT / "studies/cove50-actuators/whole-body-architectures/architectures.json"
FRAME = ROOT / "artifacts/ape-frame50-direct-skeleton/ape-frame50-direct-visual.xml"
DEFAULT_OUT = ROOT / "artifacts/ape-frame50-cove-shell-head-v11"
NAME = "head_cranium_front"
SCALE = 500 / 350
WALL_MM = 2.0
SOURCE_GAP_MM = 0.65
FACE_OUTLINE_XZ_350 = [
    (0, 328.5), (15, 331), (24, 329), (30, 321), (31, 313),
    (27, 303), (23, 298), (22, 287), (16, 279), (0, 276.7),
    (-16, 279), (-22, 287), (-23, 298), (-27, 303), (-31, 313),
    (-30, 321), (-24, 329), (-15, 331),
]
CAMERAS = (("upper", 15, 457), ("lower", -15, 433))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mesh(shape: bd.Shape) -> trimesh.Trimesh:
    vertices, faces = shape.tessellate(0.15, 0.15)
    return trimesh.Trimesh(vertices=np.asarray([tuple(v) for v in vertices]),
                           faces=np.asarray(faces), process=False)


def bounds(vertices) -> list[list[float]]:
    a = np.asarray(vertices)
    return np.stack((a.min(axis=0), a.max(axis=0))).round(3).tolist()


def sampled_surface_distance(a: trimesh.Trimesh, b: trimesh.Trimesh) -> dict:
    # Each direction samples the vertices of one tessellation against the
    # triangular surface of the other. The maximum is not an analytic bound.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ab = closest_point(b, a.vertices)[1]
        ba = closest_point(a, b.vertices)[1]
    assert np.isfinite(ab).all() and np.isfinite(ba).all()
    return {"a_to_b_max_mm": round(float(ab.max()), 3),
            "b_to_a_max_mm": round(float(ba.max()), 3),
            "a_to_b_p95_mm": round(float(np.percentile(ab, 95)), 3),
            "b_to_a_p95_mm": round(float(np.percentile(ba, 95)), 3)}


def intersection_mm3(a: bd.Shape, b: bd.Shape) -> float | dict:
    try:
        hit = a & b
        return round(float(hit.volume), 3) if hit is not None else 0.0
    except Exception as exc:
        return {"unresolved": str(exc)[:200]}


def candidate_head_front() -> bd.Shape:
    # The source sketch is dimensioned at 350 mm, X lateral, Y rear, Z up.
    # Construct its analytic loft and split, then transform this fresh solid to
    # selected coordinates (X forward, Y lateral, Z up). Importing and rotating
    # the old source/selected STEP panel reproduces the exchange defect.
    center = (0, 0, 313)
    radii = (39, 35, 37)
    t = WALL_MM / SCALE
    outer = cranial_loft(center, radii)
    inner = cranial_loft(center, tuple(r - t for r in radii))
    face_aperture = prism_xz(FACE_OUTLINE_XZ_350, length=86)
    split_y = 2 - (SOURCE_GAP_MM / SCALE) / 2
    front = ((outer - inner - face_aperture) &
             box(-180, 180, -100, split_y)).clean()
    front = front.scale(SCALE).rotate(bd.Axis.Z, 90)
    # The two old 9 mm radius ports do not intersect this cranium panel: they
    # lie in its open face aperture. Re-apply them to retain the design intent.
    for _, y, z in CAMERAS:
        cut = front - bd.Solid.make_cylinder(
            9, 55, bd.Plane(origin=(22, y, z), z_dir=(1, 0, 0)))
        if cut is not None and len(cut.solids()) == 1:
            front = cut
    front.label = NAME
    assert front.is_valid and len(front.solids()) == 1
    return front


def load_wb():
    spec = importlib.util.spec_from_file_location("cove_whole_body_head_v11", WB_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def contact_rescreen(front: bd.Shape, faceted: bd.Shape) -> list[dict]:
    wb = load_wb()
    arch = json.loads(ARCH.read_text())
    model = mujoco.MjModel.from_xml_path(str(FRAME))
    neutral = wb.pose_data(model, {})
    rows = []
    for assignment in arch["architectures"]:
        for layout in ("inward", "outward"):
            reference = json.loads((WB_OUTPUT / f"{assignment['id']}-{layout}-dorsal/report.json").read_text())
            affected = [(r["joint"], hit) for r in reference["neutral_case_shell_BRep"]
                        for hit in r["hits"] if hit["panel"] == NAME]
            for joint, old in affected:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                pivot = neutral.xanchor[jid] * 1000
                part_id = assignment["assignments"][joint]
                case, case_size, caveats = wb.motor_shape(
                    joint, arch["parts"][part_id], pivot, layout, assignment["id"])
                rows.append({
                    "architecture": assignment["id"], "layout": layout,
                    "joint": joint, "part_id": part_id, "case_size_mm": case_size,
                    "case_assumptions": caveats,
                    "v10_screen_status": old["overlap_mm3"],
                    "analytic_v11_overlap_mm3": intersection_mm3(case, front),
                    "faceted_audit_overlap_mm3": intersection_mm3(case, faceted),
                    "repeated_battery_layouts": 3,
                })
    return rows


def neutral_visual_mesh(name: str, report_rows: list[dict], model, neutral) -> trimesh.Trimesh:
    row = next(r for r in report_rows if r["name"] == name)
    raw = trimesh.load_mesh(SELECTED / "shell-meshes" / f"shell_{name}.obj", process=False)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, row["body"])
    raw.vertices = (np.asarray(raw.vertices) @ neutral.xmat[bid].reshape(3, 3).T
                    + neutral.xpos[bid]) * 1000
    return raw


def render_head(items: list[tuple[str, trimesh.Trimesh, tuple[int, int, int]]],
                view: str, path: Path) -> None:
    direction = {"front": (1, 0, .05), "side": (0, 1, .04),
                 "three-quarter": (1, -.72, .22)}[view]
    forward = np.asarray(direction, float); forward /= np.linalg.norm(forward)
    up = np.asarray((0., 0., 1.))
    right = np.cross(up, forward); right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    size = (540, 520)
    image = Image.new("RGB", size, (242, 238, 229))
    pen = ImageDraw.Draw(image)
    polygons = []
    for _, part, color in items:
        verts = np.asarray(part.vertices)
        rel = verts - np.asarray((4, 0, 447))
        xy = np.stack((rel @ right, rel @ up), axis=1) * 3.35
        xy += np.asarray((size[0]/2, size[1]/2))
        xy[:, 1] = size[1] - xy[:, 1]
        depth = rel @ forward
        for face in part.faces:
            polygons.append((float(np.mean(depth[face])), tuple(map(tuple, xy[face])), color))
    for _, poly, color in sorted(polygons, key=lambda p: p[0]):
        pen.polygon(poly, fill=color)
    image.save(path, optimize=True)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "robot/ape-current-shell.json").read_text())
    selected_step = SELECTED / "cove50-shell-neutral.step"
    assert sha(selected_step) == selection["evidence"]["neutral_cad"]["sha256"]
    selected_report = json.loads((SELECTED / "report.json").read_text())
    rows = selected_report["shell_parts_by_body"]
    index = next(i for i, row in enumerate(rows) if row["name"] == NAME)
    old = bd.import_step(selected_step)
    assert len(old.children) == len(rows) == 53
    front = candidate_head_front()
    original = old.children[index]
    parts = list(old.children)
    parts[index] = front
    standalone = out / "head-cranium-front-analytic.step"
    assembly = out / "cove50-shell-head-v11-neutral.step"
    bd.export_step(bd.Compound(children=[front]), standalone)
    bd.export_step(bd.Compound(children=parts), assembly)
    single_rt = bd.import_step(standalone)
    assembly_rt = bd.import_step(assembly)
    new = assembly_rt.children[index]
    assert len(assembly_rt.children) == 53
    assert assembly_rt.is_valid
    assert all(child.is_valid and len(child.solids()) == 1 for child in assembly_rt.children)
    assert single_rt.is_valid and len(single_rt.solids()) == 1
    assert new.is_valid and len(new.solids()) == 1
    assert abs(new.volume - front.volume) < .01
    other_deltas = [abs(a.volume - b.volume) for i, (a, b) in
                    enumerate(zip(parts, assembly_rt.children)) if i != index]
    assert max(other_deltas) < 1.0

    analytic_mesh = mesh(new)
    faceted = bd.import_step(AUDIT / "head-front-faceted-cad-candidate.step")
    faceted_mesh = mesh(faceted)
    source_mesh = trimesh.load_mesh(AUDIT / "head-front-visual-datum-in-selected-frame.stl", process=False)
    source_mesh.merge_vertices()
    model = mujoco.MjModel.from_xml_path(str(FRAME))
    neutral = mujoco.MjData(model)
    neutral.qpos[:] = model.qpos0
    mujoco.mj_forward(model, neutral)
    selected_visual = neutral_visual_mesh(NAME, rows, model, neutral)
    av = np.asarray(analytic_mesh.vertices)
    source_comparison = sampled_surface_distance(analytic_mesh, source_mesh)
    selected_visual_comparison = sampled_surface_distance(analytic_mesh, selected_visual)

    named = {r["name"]: s for r, s in zip(rows, assembly_rt.children)}
    seams = {}
    for other_name in ("head_cranium_back", "face_mask", "muzzle_shell"):
        neighbor = named[other_name]
        other_mesh = mesh(neighbor)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            distance = float(closest_point(other_mesh, av)[1].min())
        seams[other_name] = {
            "solid_overlap_mm3": intersection_mm3(new, neighbor),
            "min_analytic_vertex_to_neighbor_surface_mm": round(distance, 4),
        }

    camera = []
    for label, y, z in CAMERAS:
        bore = bd.Solid.make_cylinder(9, 55, bd.Plane(origin=(22, y, z), z_dir=(1, 0, 0)))
        module = bd.Solid.make_box(25, 24, 13,
                                   bd.Plane(origin=(49-12.5, y-12, z-6.5)))
        camera.append({
            "name": label, "center_mm": [49, y, z],
            "bore_radius_mm": 9, "module_envelope_mm": [25, 24, 13],
            "bore_overlaps_mm3": {name: intersection_mm3(named[name], bore)
                                   for name in (NAME, "head_cranium_back", "face_mask", "muzzle_shell")},
            "module_overlaps_mm3": {name: intersection_mm3(named[name], module)
                                     for name in (NAME, "head_cranium_back", "face_mask", "muzzle_shell")},
        })

    colors = {NAME: (221, 214, 196), "head_cranium_back": (204, 197, 180),
              "face_mask": (178, 157, 128), "muzzle_shell": (232, 219, 190)}
    visual = {name: neutral_visual_mesh(name, rows, model, neutral) for name in colors}
    variants = {"selected_visual_v10": selected_visual,
                "analytic_v11": analytic_mesh, "faceted_audit": faceted_mesh}
    for view in ("front", "side", "three-quarter"):
        sheet = Image.new("RGB", (540*3, 570), "white")
        pen = ImageDraw.Draw(sheet)
        for col, (label, front_mesh) in enumerate(variants.items()):
            items = [(name, front_mesh if name == NAME else visual[name], colors[name])
                     for name in colors]
            tile = out / f"{label}-{view}.png"
            render_head(items, view, tile)
            sheet.paste(Image.open(tile), (col*540, 50))
            pen.text((col*540+15, 18), f"{label.replace('_', ' ')} / {view}", fill=(35, 32, 28))
        sheet.save(out / f"comparison-{view}.png", optimize=True)

    scene = trimesh.Scene()
    for name, color in colors.items():
        m = analytic_mesh.copy() if name == NAME else visual[name].copy()
        m.visual.vertex_colors = (*color, 255)
        scene.add_geometry(m, geom_name=name, node_name=name)
    scene.export(out / "head-v11-analytic.glb")

    contact = contact_rescreen(new, faceted)
    result = {
        "schema": "cove50.shell-head-analytic-candidate.v1",
        "status": "unselected_candidate; selected shell pointer unchanged",
        "selected_shell_id": selection["id"],
        "inputs_sha256": {
            "selected_shell_step": sha(selected_step),
            "source_sketch_generator": sha(SOURCE_SCRIPT),
            "source_front_stl": sha(SOURCE_STL),
            "source_stl_selected_frame_audit": sha(AUDIT / "head-front-visual-datum-in-selected-frame.stl"),
            "faceted_audit_step": sha(AUDIT / "head-front-faceted-cad-candidate.step"),
            "whole_body_generator": sha(WB_SCRIPT),
            "whole_body_architectures": sha(ARCH),
            "frame_visual_mjcf": sha(FRAME),
        },
        "parameters": {"height_mm": 500, "sketch_height_mm": 350,
                       "nominal_wall_mm": WALL_MM, "sketch_split_gap_mm": SOURCE_GAP_MM,
                       "head_center_350_mm": [0, 0, 313],
                       "head_radii_350_mm": [39, 35, 37],
                       "face_outline_xz_350_mm": FACE_OUTLINE_XZ_350},
        "round_trip": {
            "selected_v10_import_valid": original.is_valid,
            "selected_v10_import_solids": len(original.solids()),
            "selected_v10_import_volume_cm3": round(original.volume/1000, 3),
            "source_in_memory_nominal_cm3": rows[index]["volume_cm3"],
            "analytic_in_memory_valid": front.is_valid,
            "analytic_in_memory_solids": len(front.solids()),
            "analytic_in_memory_volume_cm3": round(front.volume/1000, 3),
            "standalone_step_valid": single_rt.is_valid,
            "standalone_step_solids": len(single_rt.solids()),
            "assembly_step_valid_head": new.is_valid,
            "assembly_step_solids_head": len(new.solids()),
            "assembly_step_volume_cm3": round(new.volume/1000, 3),
            "full_assembly_children": len(assembly_rt.children),
            "full_assembly_valid": assembly_rt.is_valid,
            "all_assembly_children_valid_one_solid": True,
            "max_other_panel_roundtrip_volume_delta_mm3": round(max(other_deltas), 5),
            "faceted_candidate_volume_cm3": round(faceted.volume/1000, 3),
            "analytic_brep_face_count": len(new.faces()),
            "faceted_brep_face_count": len(faceted.faces()),
        },
        "visual_comparison": {
            "analytic_tessellation_bounds_mm": bounds(analytic_mesh.vertices),
            "selected_visual_bounds_mm": bounds(selected_visual.vertices),
            "faceted_tessellation_bounds_mm": bounds(faceted_mesh.vertices),
            "source_stl_bounds_mm": bounds(source_mesh.vertices),
            "analytic_vs_source_stl_sampled_surface_mm": source_comparison,
            "analytic_vs_selected_visual_sampled_surface_mm": selected_visual_comparison,
            "metric_limit": "Bidirectional sampled triangle-surface distances are tessellation diagnostics, not a continuous-surface Hausdorff guarantee. The selected visual is coarser near the crown than the source STL.",
        },
        "seams": seams,
        "cameras": camera,
        "affected_motor_head_contacts_unique": contact,
        "affected_whole_body_report_entries": sum(x["repeated_battery_layouts"] for x in contact),
        "output_sha256": {"standalone_step": sha(standalone),
                          "assembly_step": sha(assembly),
                          "head_glb": sha(out / "head-v11-analytic.glb")},
        "qualification_limits": [
            "This is a new head-panel candidate, not a replacement of robot/ape-current-shell.json.",
            "Neutral motor cases are dimension envelopes on the old frame pivots; contacts do not qualify motion, mounting or torque.",
            "Camera bores clear only the 9 mm-radius apertures. The 25x24x13 mm illustrative camera module envelope may still collide with face or muzzle panels.",
            "Panel seams are neutral geometry checks, not manufacturing tolerances, service access or moving clearance.",
        ],
    }
    (out / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    sections = "".join(
        f'<section><h2>{view.replace("-", " ").title()}</h2>'
        f'<img src="comparison-{view}.png" alt="Selected visual, analytic candidate and faceted audit compared from {view}"></section>'
        for view in ("front", "side", "three-quarter"))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8">'
            '<title>Cove-50 analytic head candidate</title>'
            '<style>body{font:16px system-ui;background:#eee9de;color:#252622;margin:0}'
            'header{background:#353935;color:#fff;padding:26px}main{max-width:1660px;margin:auto;padding:20px}'
            'section{background:#fff;padding:15px;margin:20px 0}img{width:100%;height:auto}'
            'a{color:#91542d}.warning{background:#fff2d9;padding:12px;border-left:5px solid #b46d2f}'
            '</style><header><h1>Cove-50 analytic front cranium / v11 candidate</h1>'
            '<p>Editable loft B-rep · 53-panel STEP · unselected</p></header><main>'
            '<p class="warning">The candidate fixes the head-front STEP exchange defect only. '
            'It still intersects the illustrative head-yaw motor case, and the camera module envelopes '
            'intersect face/muzzle panels. No fit or motion qualification follows.</p>'
            '<p><a href="report.json">Machine-readable report</a> · '
            '<a href="cove50-shell-head-v11-neutral.step">Full candidate STEP</a> · '
            '<a href="head-cranium-front-analytic.step">Editable head STEP</a> · '
            '<a href="head-v11-analytic.glb">Head GLB</a></p>' + sections +
            '</main></html>')
    (out / "index.html").write_text(html)
    print(json.dumps({"candidate_cm3": result["round_trip"]["assembly_step_volume_cm3"],
                      "affected_entries": result["affected_whole_body_report_entries"],
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    main(parser.parse_args().out)
