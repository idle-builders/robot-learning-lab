"""Restore head display geometry from bound source meshes and measure its error."""

import argparse
from copy import deepcopy
import base64
import gzip
import json
import traceback
from pathlib import Path

import numpy as np
import trimesh
from build123d import import_step
from OCP.BRepTools import BRepTools
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer
from tools.check_arm_episode import parts_from_step

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def distances(mesh, points):
    mesh = mesh.copy()
    mesh.update_faces(mesh.nondegenerate_faces(height=1e-8))
    return np.concatenate(
        [
            trimesh.proximity.closest_point_naive(mesh, points[i : i + 16])[1]
            for i in range(0, len(points), 16)
        ]
    )


def error_metrics(original, candidate, samples=256):
    a = trimesh.sample.sample_surface(original, samples, seed=0)[0]
    b = trimesh.sample.sample_surface(candidate, samples, seed=1)[0]
    errors = np.concatenate([distances(candidate, a), distances(original, b)])
    return dict(
        sample_count_each_direction=samples,
        sampled_surface_p95_mm=float(np.quantile(errors, 0.95)),
        sampled_surface_max_mm=float(errors.max()),
        bounds_max_error_mm=float(np.abs(original.bounds - candidate.bounds).max()),
        signed_volume_relative_error=float(
            abs(original.volume - candidate.volume) / abs(original.volume)
        ),
        triangles=len(candidate.faces),
    )


def acceptable(metrics):
    return (
        metrics["sampled_surface_max_mm"] <= 0.75
        and metrics["sampled_surface_p95_mm"] <= 0.2
        and metrics["bounds_max_error_mm"] <= 0.5
        and metrics["signed_volume_relative_error"] <= 0.02
    )


def run(output, fragment):
    output.mkdir(parents=True, exist_ok=False)
    sources = copy_source(
        output,
        {
            str(p): p
            for p in [
                Path("tools/repair_head_display.py"),
                Path("ui/distal-grip-motion.fragment.html"),
                Path("lab/robot.py"),
                Path("lab/config.py"),
                Path("requirements-macos-arm64.lock"),
            ]
        },
    )
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    baseline = Path("artifacts/distal-grip-view-v2")
    old = verify_report(baseline)
    bind(baseline / "report.json")
    data = json.loads(bind(baseline / "dataset.json").read_text())
    if sha(baseline / "dataset.json") != old["output_hashes"]["dataset.json"]:
        raise ValueError("Display dataset changed")
    for p, digest in old["source_bindings"].items():
        if sha(bind(Path(p))) != digest:
            raise ValueError("Display source drift: " + p)
    metrics = []
    for robot_data in data["robots"]:
        name = robot_data["name"].lower()
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        report = verify_report(assembly)
        config = bind(assembly / "resolved-config.json")
        robot = make_robot(RobotConfig.model_validate_json(config.read_text()))
        fk = robot.forward_kinematics()
        bind(assembly / "report.json")
        shapes = {
            p.name: p.shape
            for p in parts_from_step(
                list(import_step(bind(assembly / "assembly.step")).children),
                report["parts"],
            )
        }
        source_parts = {p["name"]: p for p in report["parts"]}
        mesh_folder = output / name
        mesh_folder.mkdir()
        for part in robot_data["parts"]:
            if part["role"] != "head_polymer":
                continue
            path = mesh_folder / (part["name"] + ".stl")
            shape = shapes[part["name"]]
            if not np.isclose(
                shape.volume,
                source_parts[part["name"]]["volume_mm3"],
                rtol=0.001,
                atol=0.02,
            ):
                raise ValueError(
                    f"STEP volume differs for {part['name']}: {shape.volume} vs {source_parts[part['name']]['volume_mm3']}"
                )
            BRepTools.Clean_s(shape.wrapped)
            mesher = BRepMesh_IncrementalMesh(shape.wrapped, 0.03, False, 0.12, True)
            writer = StlAPI_Writer()
            writer.ASCIIMode = False
            if not mesher.IsDone() or not writer.Write(shape.wrapped, str(path)):
                raise ValueError("Head STL export failed")
            original = trimesh.load_mesh(path, process=True)
            if (
                abs(original.volume - source_parts[part["name"]]["volume_mm3"])
                / source_parts[part["name"]]["volume_mm3"]
                > 0.01
            ):
                raise ValueError(
                    "Head source volume differs from complete CAD manifest"
                )
            frame = fk[part["owner"]]
            old_asset = data["assets"][part["asset"]]
            old_vertices = (np.asarray(old_asset[0]).reshape(-1, 3) / 10) @ frame[
                :3, :3
            ].T + frame[:3, 3] * 1000
            old_mesh = trimesh.Trimesh(
                old_vertices, np.asarray(old_asset[1]).reshape(-1, 3), process=False
            )
            before = error_metrics(original, old_mesh)
            attempts = []
            for target in [2200, 4400, 8800, 16000, len(original.faces)]:
                candidate = (
                    original.copy()
                    if len(original.faces) <= target
                    else original.simplify_quadric_decimation(
                        face_count=target, aggression=5
                    )
                )
                local = (candidate.vertices - frame[:3, 3] * 1000) @ frame[:3, :3]
                quantized = np.rint(local * 10).astype(int)
                reconstructed = trimesh.Trimesh(
                    quantized / 10 @ frame[:3, :3].T + frame[:3, 3] * 1000,
                    candidate.faces,
                    process=False,
                )
                reconstructed.update_faces(
                    reconstructed.nondegenerate_faces(height=1e-8)
                )
                after = error_metrics(original, reconstructed)
                attempts.append(after)
                if acceptable(after):
                    break
            write(output / f"{name}-{part['name']}-attempts.json", attempts)
            if not acceptable(after):
                raise ValueError(f"No acceptable display for {name} {part['name']}")
            part["asset"] = len(data["assets"])
            data["assets"].append(
                [quantized.ravel().tolist(), reconstructed.faces.ravel().tolist()]
            )
            row = dict(
                robot=name,
                part=part["name"],
                source=str(path),
                source_triangles=len(original.faces),
                before=before,
                after=after,
                attempts=attempts,
            )
            metrics.append(row)
            write(output / "progress.json", metrics)
            print(
                name,
                part["name"],
                f"{before['sampled_surface_max_mm']:.2f} -> {after['sampled_surface_max_mm']:.2f} mm",
                flush=True,
            )
    used = sorted({p["asset"] for r in data["robots"] for p in r["parts"]})
    mapping = {old: i for i, old in enumerate(used)}
    data["assets"] = [data["assets"][i] for i in used]
    for r in data["robots"]:
        for p in r["parts"]:
            p["asset"] = mapping[p["asset"]]
    packed = base64.b64encode(
        gzip.compress(json.dumps(data, separators=(",", ":")).encode(), mtime=0)
    ).decode()
    text = (
        Path("ui/distal-grip-motion.fragment.html")
        .read_text()
        .replace("__FOREARM_DATA__", packed)
    )
    text = text.replace(
        "distal-grip-motion-inspector", "head-restored-motion-inspector"
    )
    text = text.replace(
        "Pip and Scout · distal-grip trial", "Pip and Scout · assembly replay"
    )
    text = text.replace(
        "simplified display meshes",
        "simplified display meshes · restored head detail checked against retained source meshes",
    )
    inline_text = text
    inline_robots = [r["name"] for r in data["robots"]]
    if len(inline_text.encode()) >= 1_000_000:
        # The local review keeps both robots at the verified fidelity. The chat
        # preview uses Pip alone to stay within the inline payload limit.
        preview = deepcopy(data)
        preview["robots"] = preview["robots"][:1]
        used = sorted({p["asset"] for p in preview["robots"][0]["parts"]})
        remap = {old: i for i, old in enumerate(used)}
        preview["assets"] = [preview["assets"][i] for i in used]
        for part in preview["robots"][0]["parts"]:
            part["asset"] = remap[part["asset"]]
        preview_packed = base64.b64encode(
            gzip.compress(json.dumps(preview, separators=(",", ":")).encode(), mtime=0)
        ).decode()
        inline_text = (
            text.replace(packed, preview_packed)
            .replace("head-restored-motion-inspector", "pip-restored-motion-inspector")
            .replace("Pip and Scout · assembly replay", "Pip · assembly replay")
        )
        inline_robots = [preview["robots"][0]["name"]]
        if len(inline_text.encode()) >= 1_000_000:
            raise ValueError(
                f"Inline display exceeds budget: {len(inline_text.encode())}"
            )
    if fragment.exists():
        raise FileExistsError(fragment)
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(inline_text)
    (output / "motion.fragment.html").write_text(text)
    write(output / "dataset.json", data)
    write(
        output / "head-error.json",
        dict(
            status="pass",
            scope="Bidirectional sampled mesh error, not Hausdorff distance or manufacturing tolerance",
            rows=metrics,
        ),
    )
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Head display restoration only; robot CAD, motion and engineering outcomes unchanged",
            source_hashes=sources,
            source_bindings=bindings,
            output_hashes={
                n: sha(output / n)
                for n in [
                    "dataset.json",
                    "head-error.json",
                    "motion.fragment.html",
                    *[str(p.relative_to(output)) for p in output.glob("*/*.stl")],
                ]
            },
            fragment=str(fragment),
            fragment_sha256=sha(fragment),
            bytes=len(text.encode()),
            inline_bytes=len(inline_text.encode()),
            inline_robots=inline_robots,
            restored_parts=len(metrics),
            limitations=[
                "Source STL seams are retained; source meshes are not asserted watertight",
                "Surface samples do not guarantee worst-case error",
                "Other parts retain earlier simplified display geometry",
                "No physical geometry or simulation result was changed",
            ],
        ),
    )
    print("restored view", len(text.encode()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fragment", type=Path, required=True)
    args = vars(parser.parse_args())
    output_existed = args["output"].exists()
    try:
        run(**args)
    except Exception:
        output = args["output"]
        if (
            not output_existed
            and output.exists()
            and not (output / "report.json").exists()
        ):
            write(
                output / "report.json",
                dict(status="fail", error=traceback.format_exc()),
            )
        raise
