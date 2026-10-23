"""Render a neck-skirt study with bound display meshes and exported cover parts."""

import argparse
import json
import traceback
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import numpy as np
import trimesh
import mujoco
from PIL import Image
from build123d import import_step, Pos
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepTools import BRepTools
from OCP.StlAPI import StlAPI_Writer

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import sha, write, copy_source
from tools.compose_forearm_enclosure import verify_report


def render(meshes, colors, output, label, center):
    root = Element("mujoco")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="900", offheight="760")
    SubElement(
        visual,
        "headlight",
        ambient=".55 .55 .55",
        diffuse=".6 .6 .6",
        specular=".1 .1 .1",
    )
    asset = SubElement(root, "asset")
    world = SubElement(root, "worldbody")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1=".9 .91 .88",
        rgb2=".9 .91 .88",
        width="512",
        height="512",
    )
    SubElement(world, "light", pos=".5 -.7 1", dir="-.2 .2 -1", ambient=".55 .55 .55")
    for i, (name, mesh) in enumerate(meshes.items()):
        filename = f"{label}-{i}.stl"
        mesh.export(output / filename)
        SubElement(asset, "mesh", name=f"m{i}", file=filename, scale=".001 .001 .001")
        SubElement(
            world,
            "geom",
            type="mesh",
            mesh=f"m{i}",
            rgba=" ".join(
                map(str, colors[name] if len(colors[name]) == 4 else [*colors[name], 1])
            ),
            contype="0",
            conaffinity="0",
        )
    xml = output / f"{label}.xml"
    ElementTree(root).write(xml)
    m = mujoco.MjModel.from_xml_path(str(xml))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    c = mujoco.MjvCamera()
    c.lookat[:] = center
    c.distance = 0.37
    c.azimuth = 145
    c.elevation = -12
    with mujoco.Renderer(m, height=760, width=900) as renderer:
        renderer.update_scene(d, camera=c)
        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        Image.fromarray(renderer.render()).save(output / f"{label}.png")


def run(studies, output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                *Path("lab").glob("*.py"),
                Path("tools/review_neck_skirt.py"),
                Path("tools/check_camera_support.py"),
                Path("tools/compose_forearm_enclosure.py"),
                Path("requirements-macos-arm64.lock"),
            ]
        },
    )

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    base = Path("artifacts/head-display-v7")
    view = verify_report(base)
    bind(base / "report.json")
    data = json.loads(bind(base / "dataset.json").read_text())
    sections = []
    results = []
    for study in studies:
        report = verify_report(study)
        bind(study / "report.json")
        name = next(n for n in ("pip", "scout") if study.name.startswith(n + "-"))
        r = next(r for r in data["robots"] if r["name"].lower() == name)
        # Display source must be the exact assembly used by the study.
        assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
        if view["source_bindings"][assembly] != report["source_bindings"][assembly]:
            raise ValueError("Display and study source assemblies differ")
        cfg = RobotConfig.model_validate_json(
            bind(study / "33mm/resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        rows = []
        for row in report["candidates"]:
            height = row["height_mm"]
            folder = output / f"{name}-{height:g}mm"
            folder.mkdir()
            failures = [p for p in row["motion"] if p["status"] == "fail"]
            shown = (
                failures[0]
                if failures
                else next(p for p in row["motion"] if p["name"] == "neck-0.00-0.00")
            )
            posed = robot.forward_kinematics(shown["pose_rad"], enforce_limits=False)
            meshes = {}
            colors = {}
            flagged = {
                c["part"]
                for c in shown["collisions"] + shown.get("clearance_collisions", [])
            }
            for part in r["parts"]:
                if part["role"] == "scene" or part["name"].startswith("neck_collar_"):
                    continue
                if part["owner"] not in ("head", "neck_yaw") and part["name"] not in (
                    "torso_front",
                    "torso_rear",
                    "neck_yaw_housing",
                    "neck_yaw_clamp_lower",
                    "neck_yaw_clamp_upper",
                ):
                    continue
                a = data["assets"][part["asset"]]
                vertices = (np.array(a[0]).reshape(-1, 3) / 10) @ posed[part["owner"]][
                    :3, :3
                ].T + posed[part["owner"]][:3, 3] * 1000
                mesh = trimesh.Trimesh(
                    vertices, np.array(a[1]).reshape(-1, 3), process=False
                )
                mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
                meshes[part["name"]] = mesh
                colors[part["name"]] = (
                    [0.8, 0.22, 0.14, 1] if part["name"] in flagged else part["color"]
                )
            exports = []
            for half in ("front", "rear"):
                path = bind(study / f"{height:g}mm/{half}.step")
                shape = import_step(path)
                BRepTools.Clean_s(shape.wrapped)
                local = Pos(0, 0, -cfg.dimensions.torso_height_m * 1000) * shape
                mesher = BRepMesh_IncrementalMesh(local.wrapped, 0.03, False, 0.1, True)
                writer = StlAPI_Writer()
                writer.ASCIIMode = False
                stl = folder / f"{half}.stl"
                if not mesher.IsDone() or not writer.Write(local.wrapped, str(stl)):
                    raise ValueError("Cover surface export failed")
                mesh = trimesh.load_mesh(stl)
                if (
                    not mesh.is_watertight
                    or abs(mesh.volume - local.volume) / local.volume > 0.01
                ):
                    raise ValueError("Cover STL identity failure")
                exports.append(
                    dict(
                        part=half,
                        stl=str(stl.relative_to(output)),
                        watertight=True,
                        volume_mm3=local.volume,
                    )
                )
                mesh.vertices += np.array([0, 0, cfg.dimensions.torso_height_m * 1000])
                mesh.vertices = (
                    mesh.vertices @ neutral["torso"][:3, :3].T
                    + neutral["torso"][:3, 3] * 1000
                )
                meshes["neck_collar_" + half] = mesh
                colors["neck_collar_" + half] = [0.25, 0.64, 0.49, 1]
            center = neutral["head"][:3, 3].copy()
            center[2] -= 0.03
            render(meshes, colors, folder, "pose", center)
            margin = report["clearance_margin_mm"]
            clearance = (
                "pass"
                if all(p["moving_clearance_status"] == "pass" for p in row["motion"])
                and len(row["motion"]) == row["requested_pose_count"]
                else "fail / incomplete"
            )
            unrun = len(row.get("unrun_pose_names", []))
            rows.append(
                f'<article><h3>{height:g} mm · {row["status"].upper()}</h3><img src="{folder.name}/pose.png" alt="{name} collar candidate at its most limiting sampled pose"><p>Shown: {shown["name"]}. {margin:g} mm moving-clearance envelope: {clearance}. Motion failures: {len(failures)}/{len(row["motion"])} evaluated; {unrun} poses not run after rejection. Service: {row.get("service_status", "not_run")}.</p><p>Added solid volume: {row["volume_added_mm3"]:.1f} mm³. Material, printing and physical strength unqualified.</p><a href="{folder.name}/front.stl">Front STL</a> · <a href="{folder.name}/rear.stl">Rear STL</a></article>'
            )
            results.append(
                dict(
                    robot=name,
                    height_mm=height,
                    status=row["status"],
                    shown_pose=shown,
                    clearance_margin_mm=margin,
                    clearance_status=clearance,
                    exports=exports,
                )
            )
        sections.append(
            f"<h2>{name.capitalize()}</h2><p>Inherited assembly: {report['inherited_assembly_status']}. Nominal collision/service selection: {report['selected_height_mm']} mm. This does not establish complete protection.</p><section>"
            + "".join(rows)
            + "</section>"
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Neck protection study</title><style>body{font:16px system-ui;background:#f5f4ef;color:#23392e;margin:32px auto;max-width:1250px;padding:0 20px}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}img{width:100%}article{border-top:1px solid #bbb}a{color:#285d46}</style><h1>More neck protection with the existing mounts</h1><p>Fixed torso collar, 54 mm outside radius, 2 mm wall and four retained mounting positions. Twenty-five neck poses plus twelve recorded task endpoints are requested per height; a rejected height stops at its first failure. Red shows body parts intersecting the cover or its clearance envelope at the displayed sample; the cover is green. Body meshes are simplified for display; exact solids supplied the clearance results. Failed designs remain visible and downloadable for inspection, not approved printing.</p>'
        + "".join(sections)
        + "</html>"
    )
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Bound presentation and STL exports; geometry/design outcomes retained separately",
            source_hashes=sources,
            source_bindings=bindings,
            results=results,
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file() and not p.is_relative_to(output / "source")
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studies", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = vars(p.parse_args())
    existed = args["output"].exists()
    try:
        run(**args)
    except Exception:
        if not existed and args["output"].exists():
            write(
                args["output"] / "report.json",
                dict(status="fail", error=traceback.format_exc()),
            )
        raise
