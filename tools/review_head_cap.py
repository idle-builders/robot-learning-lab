"""Render source-bound, unmounted head-cap candidates for product review."""

import argparse
import html
import json
from pathlib import Path
import traceback

import numpy as np
import trimesh
from build123d import import_step

from tools.cap_mesh import export_checked_stl

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import sha, write, copy_source
from tools.compose_forearm_enclosure import verify_report
from tools.review_neck_skirt import render


def run(studies, output):
    output.mkdir(parents=True, exist_ok=False)
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                *Path("lab").glob("*.py"),
                Path("tools/review_head_cap.py"),
                Path("tools/review_neck_skirt.py"),
                Path("tools/check_camera_support.py"),
                Path("tools/cap_mesh.py"),
                Path("tools/compose_forearm_enclosure.py"),
            ]
        },
    )
    bindings = {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    base = Path("artifacts/head-display-v7")
    view = verify_report(base)
    bind(base / "report.json")
    data = json.loads(bind(base / "dataset.json").read_text())
    results, presentation, sections = [], [], []
    for study in studies:
        report = verify_report(study)
        bind(study / "report.json")
        name = next(
            (n for n in ("pip", "scout") if study.name.startswith(n + "-")), None
        )
        if not name:
            raise ValueError("Unknown robot study")
        assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
        if report["source_bindings"][assembly] != view["source_bindings"][assembly]:
            raise ValueError("Display and study source assemblies differ")
        cfg = RobotConfig.model_validate_json(
            bind(study / "resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        r = next(r for r in data["robots"] if r["name"].lower() == name)
        depth = report.get("depth_mm")
        tag = f"{name}-{depth:g}mm" if depth is not None else f"{name}-full"
        relieved = bool(report.get("relief"))
        if relieved:
            tag += "-relieved"
        folder = output / tag
        folder.mkdir(exist_ok=False)
        (folder / "study.json").write_bytes((study / "report.json").read_bytes())
        presentation.append(f"{tag}/study.json")
        exports, cap_meshes = [], {}
        for record in report["parts"]:
            key = record["name"]
            path = bind(study / f"{key}.step")
            shape = import_step(path)
            stl = folder / f"{key}.stl"
            mesh, cleanup = export_checked_stl(shape, stl)
            error = cleanup["volume_error_fraction"]
            cap_meshes[key] = mesh
            exports.append(
                dict(
                    part=key,
                    mesh_cleanup=cleanup,
                    stl=str(stl.relative_to(output)),
                    watertight=True,
                    volume_error_fraction=error,
                    volume_mm3=shape.volume,
                )
            )
            presentation.append(str(stl.relative_to(output)))
        motions = report["motion"]
        zero = next(m for m in motions if m["name"] == "neck-0.00-0.00")
        worst = max(
            motions, key=lambda m: (sum(c["overlap_mm3"] for c in m["collisions"]), sum(abs(v) for v in m["pose_rad"].values()))
        )
        shown = []
        for label, record in [("neutral", zero), ("worst", worst)]:
            fk = robot.forward_kinematics(record["pose_rad"], enforce_limits=False)
            flagged = {
                c["part"]
                for c in record["collisions"]
                + record["conservative_clearance_collisions"]
            }
            meshes, colors = {}, {}
            for part in r["parts"]:
                if part["role"] == "scene" or part["owner"] not in (
                    "head",
                    "neck_yaw",
                    "torso",
                ):
                    continue
                vertices, faces = data["assets"][part["asset"]]
                vertices = np.array(vertices).reshape(-1, 3) / 10
                vertices = (
                    vertices @ fk[part["owner"]][:3, :3].T
                    + fk[part["owner"]][:3, 3] * 1000
                )
                mesh = trimesh.Trimesh(
                    vertices, np.array(faces).reshape(-1, 3), process=False
                )
                mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
                meshes[part["name"]] = mesh
                colors[part["name"]] = (
                    [0.8, 0.22, 0.14, 1] if part["name"] in flagged else part["color"]
                )
            for key, original in cap_meshes.items():
                mesh = original.copy()
                mesh.vertices = (
                    mesh.vertices @ fk["head"][:3, :3].T + fk["head"][:3, 3] * 1000
                )
                meshes[key] = mesh
                colors[key] = [0.2, 0.68, 0.34, 1]
            center = neutral["head"][:3, 3].copy()
            center[2] -= 0.03
            render(meshes, colors, folder, label, center)
            presentation.append(f"{tag}/{label}.png")
            shown.append(dict(label=label, **record))
        actual = sum(bool(m["collisions"]) for m in motions)
        margin = sum(bool(m["conservative_clearance_collisions"]) for m in motions)
        links = " · ".join(
            f'<a href="{e["stl"]}">{html.escape(e["part"].replace("head_lower_", "").replace("_", " "))}</a>'
            for e in exports
        )
        decision = "SAMPLED CLEARANCE PASSES" if actual == margin == 0 else "REJECTED"
        design_label = " · local relief" if relieved else " · original stock"
        control_label = "1 mm" if relieved else "20 mm"
        pose_label = "greatest total actual overlap" if actual else "extreme checked pose"
        sections.append(
            f'<article><h2>{name.capitalize()} · {"full depth" if depth is None else f"{depth:g} mm"}{design_label}</h2><p><b>{decision} · unmounted</b>. Actual interference in {actual}/{len(motions)} poses; 1.5 mm clearance envelope fails in {margin}/{len(motions)}. Study {control_label} control: {report["negative_control"]["status"]}.</p><div class="poses"><figure><img src="{tag}/neutral.png" alt="{name} unmounted cap at neutral"><figcaption>Neutral</figcaption></figure><figure><img src="{tag}/worst.png" alt="{name} unmounted cap at selected extreme pose"><figcaption>{html.escape(worst["name"])} · {pose_label}</figcaption></figure></div><p>Inspection STL files: {links}</p><p><a href="{tag}/study.json">Original study report</a> · mounting, removal and physical tests: not_run. Inherited assembly: {report["inherited_assembly_status"]}.</p></article>'
        )
        results.append(
            dict(
                robot=name,
                depth_mm=depth,
                status=report["status"],
                exports=exports,
                shown=shown,
            )
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lower-head cover experiments</title><style>body{max-width:1180px;margin:32px auto;padding:0 22px;font:16px/1.5 system-ui;background:#f5f4ef;color:#23392e}article{border-top:1px solid #b8c1b9;margin:32px 0}.poses{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}figure{margin:0}img{width:100%}a{color:#285d46}</style><h1>Lower-head cover experiments</h1><p>Four recovered shell wings move with the head. Green shows the experimental wings; red marks body parts that intersect them or the conservative clearance envelope at the shown pose. Original full-depth and 8 mm stock fail. Locally relieved candidates are shown separately; sampled clearance does not qualify their attachment or physical fit. These are inspection files, not approved prints. Exact CAD supplied the checks; other body display meshes remain simplified.</p><p>The original 20 mm controls can jump past stock; a separate, source-bound 1 mm audit passes for all four original stock candidates. That audit does not clear their installed-position collisions. The relieved studies use their own 1 mm control. No cap attachment is designed or verified.</p>'
        + "".join(sections)
        + "</html>"
    )
    presentation.append("index.html")
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Source-bound gallery and STL exports; engineering status remains separate; no attachment or simulation release",
            source_hashes=sources,
            source_bindings=bindings,
            results=results,
            presentation_files=presentation,
            output_hashes={
                str(p.relative_to(output)): sha(p)
                for p in output.rglob("*")
                if p.is_file() and not p.is_relative_to(output / "source")
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run(args.studies, args.output)
    except Exception:
        if not existed and args.output.exists():
            write(
                args.output / "report.json",
                dict(status="fail", error=traceback.format_exc()),
            )
        raise
