"""Inspect source-bound unmounted cap wings at their exact evaluated poses."""

import argparse
import base64
from copy import deepcopy
import gzip
import json
from pathlib import Path
import traceback

import numpy as np
import trimesh
from build123d import import_step

from tools.cap_mesh import export_checked_stl

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import sha, copy_source, write
from tools.compose_forearm_enclosure import verify_report
from tools.build_product_review import wrapper


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    source_paths = [
        *Path("lab").glob("*.py"),
        Path("ui/head-cap.fragment.html"),
        Path("ui/product-review.css"),
        Path("tools/build_head_cap_inspector.py"),
        Path("tools/build_product_review.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("tools/check_camera_support.py"),
        Path("tools/cap_mesh.py"),
    ]
    sources = copy_source(output, {str(p): p for p in source_paths})
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    base = Path("artifacts/head-display-v7")
    original = verify_report(base)
    bind(base / "report.json")
    data = json.loads(bind(base / "dataset.json").read_text())
    robots, exports = [], []
    for name in ("pip", "scout"):
        for version in ("relief-v3", "stock-v2", "stock-v1"):
            study = Path(f"artifacts/{name}-head-cap-{version}")
            report = verify_report(study)
            bind(study / "report.json")
            assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
            if (
                report["source_bindings"][assembly]
                != original["source_bindings"][assembly]
            ):
                raise ValueError("Cap and display source assemblies differ")
            cfg = RobotConfig.model_validate_json(
                bind(study / "resolved-config.json").read_text()
            )
            robot = make_robot(cfg)
            candidate = deepcopy(
                next(r for r in data["robots"] if r["name"].lower() == name)
            )
            candidate["name"] = (
                f"{name.capitalize()} · {'full depth' if version == 'stock-v1' else '8 mm · local relief' if version.startswith('relief') else '8 mm · original stock'}"
            )
            for stale_key in ("physics_status", "scene_status"):
                candidate.pop(stale_key, None)
            candidate["parts"] = [p for p in candidate["parts"] if p["role"] != "scene"]
            candidate["owners"] = [link.name for link in robot.links]
            candidate["source"] = str(study)
            candidate["inherited_status"] = report["inherited_assembly_status"]
            for part in report["parts"]:
                key = part["name"]
                path = bind(study / f"{key}.step")
                shape = import_step(path)
                dest = output / f"{name}-{version}-{key}.stl"
                mesh, cleanup = export_checked_stl(shape, dest)
                volume_error = cleanup["volume_error_fraction"]
                quantized = np.round(mesh.vertices * 10).astype(int)
                max_displacement = float(
                    np.linalg.norm(quantized / 10 - mesh.vertices, axis=1).max()
                )
                display = trimesh.Trimesh(quantized / 10, mesh.faces, process=False)
                display.update_faces(display.nondegenerate_faces(height=1e-8))
                if (
                    max_displacement > 0.087
                    or abs(display.volume - mesh.volume) / mesh.volume > 0.02
                ):
                    raise ValueError("Quantized cap display differs")
                candidate["parts"].append(
                    dict(
                        name=key,
                        owner="head",
                        role="unmounted_cap",
                        asset=len(data["assets"]),
                        color=[0.2, 0.68, 0.34],
                    )
                )
                data["assets"].append(
                    [quantized.flatten().tolist(), display.faces.flatten().tolist()]
                )
                exports.append(
                    dict(
                        robot=name,
                        version=version,
                        part=key,
                        stl=dest.name,
                        volume_error_fraction=volume_error,
                        max_quantization_displacement_mm=max_displacement,
                        source_geometry=part["geometry"],
                        mesh_cleanup=cleanup,
                    )
                )
            candidate["samples"] = []
            for record in report["motion"]:
                fk = robot.forward_kinematics(record["pose_rad"], enforce_limits=False)
                collisions = record["collisions"]
                margins = record["conservative_clearance_collisions"]
                flagged = sorted(
                    {
                        key
                        for hit in collisions + margins
                        for key in (hit["cover"], hit["part"])
                    }
                )
                pair = (
                    [collisions[0]["cover"], collisions[0]["part"]]
                    if collisions
                    else [margins[0]["cover"], margins[0]["part"]]
                    if margins
                    else ["None in this sample"]
                )
                candidate["samples"].append(
                    dict(
                        name=record["name"],
                        kind="pose",
                        status=record["status"],
                        phase="",
                        actual_status="fail" if collisions else "pass",
                        clearance_status="fail" if margins else "pass",
                        flagged=flagged,
                        pair=pair,
                        frames=[
                            fk[owner].flatten().tolist()
                            for owner in candidate["owners"]
                        ],
                    )
                )
            robots.append(candidate)
    data["robots"] = robots
    used = sorted({p["asset"] for r in robots for p in r["parts"]})
    remap = {old: i for i, old in enumerate(used)}
    data["assets"] = [data["assets"][i] for i in used]
    for r in robots:
        for p in r["parts"]:
            p["asset"] = remap[p["asset"]]
    write(output / "dataset.json", data)
    packed = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    ).decode()
    fragment = (
        Path("ui/head-cap.fragment.html")
        .read_text()
        .replace("__HEAD_CAP_DATA__", packed)
    )
    (output / "index.html").write_text(wrapper(fragment))
    (output / "review.css").write_bytes(
        bind(Path("ui/product-review.css")).read_bytes()
    )
    presentation = ["index.html", "review.css", *[p["stl"] for p in exports]]
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Interactive display of unmounted cap studies with per-pose outcomes, not an assembly or simulation release",
            source_hashes=sources,
            source_bindings=bindings,
            exports=exports,
            candidates=[
                dict(name=r["name"], groups=len(r["parts"]), samples=len(r["samples"]))
                for r in robots
            ],
            presentation_files=presentation,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    existed = args.output.exists()
    try:
        run(args.output)
    except Exception:
        if not existed and args.output.exists():
            write(
                args.output / "report.json",
                dict(status="fail", error=traceback.format_exc()),
            )
        raise
