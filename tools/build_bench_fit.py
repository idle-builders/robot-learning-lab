"""Export inspectable first fit specimens with CAD/STL checks and bench records."""

import argparse
import json
from pathlib import Path

from build123d import Compound, Cylinder, Pos, Rot, export_step, export_stl, import_step
import trimesh
import yaml

from lab.bench_fit import actuator_cradle, insert_stack
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_camera_support import copy_source, sha, write


def volume(shape):
    return sum(s.volume for s in shape.solids()) if shape is not None else 0.0


def export_part(shape, path):
    if not shape.is_valid or len(shape.solids()) != 1:
        raise ValueError(f"Disconnected or invalid specimen: {path.name}")
    # Printable files are individual solids with their bottom at Z=0.
    local = Pos(0, 0, -shape.bounding_box().min.Z) * shape
    export_step(local, path.with_suffix(".step"))
    export_stl(local, path.with_suffix(".stl"), tolerance=0.025, angular_tolerance=0.08)
    imported = import_step(path.with_suffix(".step"))
    if not geometry_agrees(geometry_record(local), geometry_record(imported)):
        raise ValueError(f"STEP reimport mismatch: {path.name}")
    mesh = trimesh.load_mesh(path.with_suffix(".stl"))
    error = abs(mesh.volume - local.volume) / local.volume
    if not mesh.is_watertight or not mesh.is_winding_consistent or error > 0.01:
        raise ValueError(f"STL geometry mismatch: {path.name}")
    return dict(
        name=path.name,
        status="pass",
        cad_volume_mm3=local.volume,
        mesh_volume_relative_error=error,
        watertight=bool(mesh.is_watertight),
        bounds_mm=mesh.bounds.tolist(),
        step=path.with_suffix(".step").name,
        stl=path.with_suffix(".stl").name,
        print_orientation="Flat base at Z=0; process and strength unqualified",
    )


def render(parts, output, name, target, distance):
    import mujoco
    from PIL import Image
    from xml.etree.ElementTree import Element, SubElement, ElementTree

    root = Element("mujoco")
    visual = SubElement(root, "visual")
    SubElement(visual, "global", offwidth="1000", offheight="700")
    asset = SubElement(root, "asset")
    SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="flat",
        rgb1=".88 .90 .88",
        rgb2=".88 .90 .88",
        width="512",
        height="512",
    )
    world = SubElement(root, "worldbody")
    SubElement(world, "light", pos=".1 -.2 .4", dir="0 0 -1", ambient=".5 .5 .5")
    colors = [".42 .66 .56 1", ".81 .86 .78 1", ".7 .53 .22 1", ".45 .47 .48 1"]
    for i, (label, shape) in enumerate(parts.items()):
        filename = f"{name}-view-{label}.stl"
        export_stl(shape, output / filename, tolerance=0.05, angular_tolerance=0.1)
        SubElement(asset, "mesh", name=label, file=filename, scale=".001 .001 .001")
        SubElement(
            world,
            "geom",
            type="mesh",
            mesh=label,
            rgba=colors[i % len(colors)],
            contype="0",
            conaffinity="0",
        )
    xml = output / f"{name}-view.xml"
    ElementTree(root).write(xml)
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = target
    camera.distance = distance
    camera.azimuth = 130
    camera.elevation = -28
    with mujoco.Renderer(model, height=700, width=1000) as renderer:
        renderer.update_scene(data, camera=camera)
        Image.fromarray(renderer.render()).save(output / f"{name}.png")


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                Path("lab/bench_fit.py"),
                Path("lab/joint_module.py"),
                Path("lab/logic_mount.py"),
                Path("lab/forearm_cover.py"),
                Path("robot/bench-fit.yaml"),
                Path("robot/forearm-cover.yaml"),
                Path("robot/actuator-catalog.yaml"),
                Path("robot/fastener-catalog.yaml"),
                Path("tools/build_bench_fit.py"),
                Path("tools/cad_identity.py"),
            ]
        },
    )
    try:
        cover = yaml.safe_load(Path("robot/forearm-cover.yaml").read_text())
        bench = yaml.safe_load(Path("robot/bench-fit.yaml").read_text())
        write(output / "resolved-bench.json", bench)
        coupons = []
        manifest = []
        checks = []
        for diameter in bench["insert_coupon"]["pilot_diameters"]:
            parts, metrics = insert_stack(cover, bench, diameter)
            label = f"pilot-{diameter:.1f}".replace(".", "p")
            manifest.append(export_part(parts["block"], output / label))
            overlap = volume(parts["block"] & parts["cover"])
            shaft = volume(parts["screw"] & parts["cover"])
            bottom = volume(parts["screw"] & parts["block"])
            checks.append(
                dict(
                    name=label,
                    cover_overlap_mm3=overlap,
                    screw_to_cover_mm3=shaft,
                    screw_to_block_mm3=bottom,
                    status="pass" if max(overlap, shaft, bottom) < 1e-6 else "fail",
                )
            )
            coupons.append(
                dict(
                    id=label,
                    replicates=bench["insert_coupon"][
                        "physical_replicates_per_diameter"
                    ],
                    **metrics,
                )
            )
            if diameter == 4.0:
                manifest.append(export_part(parts["cover"], output / "cover-stack"))
                export_step(
                    Compound(children=list(parts.values())),
                    output / "nominal-stack.step",
                )
                render(
                    {
                        k: Pos(i * 20, 0, 0) * v
                        for i, (k, v) in enumerate(parts.items())
                    },
                    output,
                    "coupon",
                    [0.025, 0, 0.008],
                    0.14,
                )
                # Deliberate failures exercise the same geometric obstruction screen.
                fat_shaft = Pos(0, 0, metrics["stack_height_mm"] - 3) * Cylinder(
                    1.5, 10
                )
                long_screw = Pos(0, 0, -3) * parts["screw"]
                checks.extend(
                    [
                        dict(
                            name="negative-oversize-shaft",
                            expected="interference",
                            overlap_mm3=volume(fat_shaft & parts["cover"]),
                            status="pass"
                            if volume(fat_shaft & parts["cover"]) > 0.01
                            else "fail",
                        ),
                        dict(
                            name="negative-long-screw",
                            expected="bottoming",
                            overlap_mm3=volume(long_screw & parts["block"]),
                            status="pass"
                            if volume(long_screw & parts["block"]) > 0.01
                            else "fail",
                        ),
                    ]
                )
        cradle, metrics = actuator_cradle(bench)
        for index, (x, y) in enumerate(metrics["clamp_bolt_centers_xy_mm"]):
            shaft = Pos(x, y, 25) * Cylinder(1.5, 70)
            overlap = sum(volume(cradle[name].intersect(shaft))
                          for name in ("cradle", "clamp_cap"))
            checks.append(dict(name=f"clamp-bolt-path-{index}", overlap_mm3=overlap,
                               status="pass" if overlap < 1e-6 else "fail"))
        for name in ["cradle", "clamp_cap"]:
            printable = (
                Rot(180, 0, 0) * cradle[name] if name == "clamp_cap" else cradle[name]
            )
            manifest.append(export_part(printable, output / name))
            if name == "clamp_cap":
                manifest[-1]["print_orientation"] = (
                    "Inverted relative to assembly, closed back on Z=0; avoids bridging the case cavity"
                )
        for fixed in ["cradle", "clamp_cap"]:
            overlap = volume(cradle[fixed] & cradle["case_reference"])
            checks.append(
                dict(
                    name=f"case-{fixed}",
                    overlap_mm3=overlap,
                    status="pass" if overlap < 1e-6 else "fail",
                )
            )
        export_step(
            Compound(children=list(cradle.values())), output / "cradle-assembly.step"
        )
        shifted_case_overlap = volume(
            (Pos(1, 0, 0) * cradle["case_reference"]) & cradle["cradle"]
        )
        checks.append(
            dict(
                name="negative-shifted-case",
                expected="interference",
                overlap_mm3=shifted_case_overlap,
                status="pass" if shifted_case_overlap > 0.01 else "fail",
            )
        )
        # Reference case shown offset for inspection, never listed as a printable part.
        render(
            {
                "cradle": cradle["cradle"],
                "cap": Pos(0, 0, 18) * cradle["clamp_cap"],
                "case": Pos(65, 0, 0) * cradle["case_reference"],
            },
            output,
            "cradle",
            [0.018, 0, 0.025],
            0.25,
        )
        result = dict(
            status="pass" if all(c["status"] == "pass" for c in checks) else "fail",
            scope="Nominal CAD/STL dimensional fit specimens only; physical fit, strength and powered joint operation not_run",
            source_hashes=sources,
            parts=manifest,
            coupons=coupons,
            cradle=metrics,
            checks=checks,
            physical_status="not_run",
            not_run=[
                "print/slicer execution",
                "insert installation and retention",
                "material/anisotropy/creep",
                "actuator case/horn interchange",
                "clamp preload and housing crush",
                "powered or loaded operation",
            ],
        )
        rows = []
        for c in coupons:
            for rep in range(1, c["replicates"] + 1):
                rows.append(
                    dict(
                        specimen_id=f"{c['id']}-r{rep}",
                        source_part=c["id"],
                        pilot_nominal_mm=c["pilot_diameter_mm"],
                        material=None,
                        process=None,
                        orientation=None,
                        layer_height_mm=None,
                        measured_pilot_mm=None,
                        insert_depth_mm=None,
                        installation_temperature_c=None,
                        cracking=None,
                        insert_rotation=None,
                        screw_bottoming=None,
                        retention_force_n=None,
                        notes=None,
                        status="not_run",
                    )
                )
        write(
            output / "measurements.template.json",
            dict(
                schema_version=1,
                scope="Empty physical measurement form; null is not zero",
                specimens=rows,
            ),
        )
        html = '<!doctype html><html><head><meta charset="utf-8"><title>First fit specimens</title><style>body{font:16px/1.5 system-ui;margin:40px auto;max-width:1000px;padding:0 20px;color:#25382e}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccd4ce}code{overflow-wrap:anywhere}</style></head><body><h1>First printable fit specimens</h1><p>Five pilot diameters around the forearm insert stack, one matching cover coupon, and an actuator fit cradle. CAD checks do not establish print fit or load capacity.</p><h2>Insert and cover stack</h2><p>Exploded inspection view: block, cover, insert envelope and screw. Only the block and cover are printable.</p><img src="coupon.png" alt="Exploded insert coupon components"><h2>Actuator fit cradle</h2><p>The robot clamp recipe receives a fixture plate. The separate box represents the catalog actuator keep-out. No powered operation is qualified.</p><img src="cradle.png" alt="Actuator clamp and fixture plate with separate reference case"><h2>Print inventory</h2><table><tr><th>Part</th><th>STL</th><th>STEP</th><th>Solid volume (mm³)</th></tr>'
        for p in manifest:
            html += f'<tr><td>{p["name"]}</td><td><a href="{p["stl"]}">STL</a></td><td><a href="{p["step"]}">STEP</a></td><td>{p["cad_volume_mm3"]:.1f}</td></tr>'
        html += '</table><p><a href="report.json">CAD checks and scope</a> · <a href="measurements.template.json">Physical measurement form</a></p><p>Three specimens per pilot diameter are proposed. Do not treat an empty measurement form as evidence. Print material and process remain to select.</p></body></html>'
        (output / "index.html").write_text(html)
        result["output_hashes"] = {
            str(p.relative_to(output)): sha(p) for p in output.iterdir() if p.is_file()
        }
        write(output / "report.json", result)
        print(
            json.dumps(
                dict(status=result["status"], parts=len(manifest), checks=checks)
            )
        )
    except Exception as exc:
        write(
            output / "report.json",
            dict(
                status="fail",
                scope="Incomplete fit-specimen build",
                source_hashes=sources,
                error=str(exc),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(**vars(parser.parse_args()))
