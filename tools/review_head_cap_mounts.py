"""Inspect selected unbored head-cap mount stock without implying assembly release."""

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

from lab.config import RobotConfig
from lab.robot import make_robot
from tools.cap_mesh import export_checked_stl
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.build_product_review import wrapper


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__)
    try:
        report["source_hashes"] = copy_source(
            output,
            {
                str(p): p
                for p in [
                    *Path("lab").glob("*.py"),
                    Path("tools/review_head_cap_mounts.py"),
                    Path("tools/cap_mesh.py"),
                    Path("tools/check_camera_support.py"),
                    Path("tools/compose_forearm_enclosure.py"),
                    Path("tools/build_product_review.py"),
                    Path("ui/head-mount.fragment.html"),
                    Path("ui/product-review.css"),
                ]
            },
        )
        bindings = {}

        def bind(p):
            bindings[str(p)] = sha(p)
            return p

        base = Path("artifacts/head-display-v7")
        view = verify_report(base)
        bind(base / "report.json")
        data = json.loads(bind(base / "dataset.json").read_text())
        robots, exports, summaries = [], [], []
        for name in ("pip", "scout"):
            study = Path(f"artifacts/{name}-head-cap-mount-stock-v3")
            prior = verify_report(study)
            bind(study / "report.json")
            assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
            if prior["source_bindings"][assembly] != view["source_bindings"][assembly]:
                raise ValueError("Display and stock assemblies differ")
            config = RobotConfig.model_validate_json(
                bind(study / "resolved-config.json").read_text()
            )
            robot = make_robot(config)
            selected = {}
            for key in prior["eligible_by_cover"]:
                cases = [
                    row
                    for row in prior["cases"]
                    if row["cover"] == key and row["status"] == "pass"
                ]
                if cases:
                    selected[key] = max(
                        cases,
                        key=lambda r: min(
                            r["boss_shell_overlap_mm3"], r["tab_cap_overlap_mm3"]
                        ),
                    )
            if not selected:
                summaries.append(dict(robot=name, selected={}, status=prior["status"]))
                continue
            candidate = deepcopy(
                next(r for r in data["robots"] if r["name"].lower() == name)
            )
            for stale in ("physics_status", "scene_status", "samples"):
                candidate.pop(stale, None)
            candidate["name"] = name.capitalize() + " · unbored stock"
            candidate["parts"] = [p for p in candidate["parts"] if p["role"] != "scene"]
            candidate["owners"] = [link.name for link in robot.links]
            candidate["source"] = str(study)
            candidate["inherited_status"] = prior["inherited_assembly_status"]
            candidate["stock_summary"] = (
                f"{len(selected)}/4 wing locations have individually screened stock; no combined attachment qualification"
            )
            relief = Path(f"artifacts/{name}-head-cap-relief-v3")
            caps = verify_report(relief)
            bind(relief / "report.json")
            pieces = [
                (
                    p["name"],
                    relief / f"{p['name']}.step",
                    "unmounted_cap",
                    [0.2, 0.68, 0.34],
                )
                for p in caps["parts"]
            ]
            for key, row in selected.items():
                x, y, _ = row["center_head_mm"]
                for part, color in [
                    ("boss", [0.3, 0.55, 0.82]),
                    ("tab", [0.9, 0.5, 0.2]),
                ]:
                    path = study / key / f"x{x:g}-y{y:g}" / f"{part}.step"
                    pieces.append((key + "_" + part, path, "unbored_" + part, color))
            for key, path, role, color in pieces:
                shape = import_step(bind(path))
                dest = output / f"{name}-{key}.stl"
                mesh, cleanup = export_checked_stl(shape, dest)
                quantized = np.round(mesh.vertices * 10).astype(int)
                display = trimesh.Trimesh(quantized / 10, mesh.faces, process=False)
                display.update_faces(display.nondegenerate_faces(height=1e-8))
                if abs(display.volume - mesh.volume) / mesh.volume > 0.02:
                    raise ValueError("Display quantization changes stock volume")
                candidate["parts"].append(
                    dict(
                        name=key,
                        owner="head",
                        role=role,
                        download=dest.name,
                        asset=len(data["assets"]),
                        color=color,
                    )
                )
                data["assets"].append(
                    [quantized.flatten().tolist(), display.faces.flatten().tolist()]
                )
                exports.append(
                    dict(robot=name, part=key, stl=dest.name, mesh_cleanup=cleanup)
                )
            version = "v5" if name == "pip" else "v2"
            motion_dir = Path(f"artifacts/{name}-head-cap-mount-motion-{version}")
            motion = verify_report(motion_dir)
            bind(motion_dir / "report.json")
            if (
                motion["source_bindings"][assembly]
                != prior["source_bindings"][assembly]
            ):
                raise ValueError("Motion and display assemblies differ")
            if len(motion["poses"]) != 37:
                raise ValueError("Incomplete stock pose screen")
            if {c["cover"]: c["center_head_mm"] for c in motion["selected_cases"]} != {
                key: row["center_head_mm"] for key, row in selected.items()
            }:
                raise ValueError("Displayed stock selection differs from motion")
            witnesses = []
            audit_dir = Path(f"artifacts/{name}-head-cap-mount-gap-v1")
            audit = verify_report(audit_dir)
            bind(audit_dir / "report.json")
            if audit["source_bindings"][str(motion_dir / "report.json")] != sha(
                motion_dir / "report.json"
            ):
                raise ValueError("Distance audit differs from displayed motion")
            witnesses = audit["cases"]
            candidate["samples"] = []
            for pose in motion["poses"]:
                fk = robot.forward_kinematics(pose["pose_rad"], enforce_limits=False)
                hits = (
                    pose["exact_collisions"] + pose["conservative_clearance_collisions"]
                )
                flagged = sorted(
                    {
                        key
                        for hit in hits
                        for key in [
                            hit["cover"] + "_boss",
                            hit["cover"] + "_tab",
                            hit["part"],
                        ]
                    }
                )
                candidate["samples"].append(
                    dict(
                        name=pose["name"],
                        gap_note=(
                            "Exact flagged-pair gaps: "
                            + "; ".join(
                                f"{row['target']}: {row['distance_mm']:.2f} mm ({row['status']})"
                                for row in witnesses
                                if row["pose"] == pose["name"] and "distance_mm" in row
                            )
                        )
                        if any(
                            row["pose"] == pose["name"] and "distance_mm" in row
                            for row in witnesses
                        )
                        else "",
                        kind="pose",
                        status=pose["status"],
                        phase="",
                        actual_status="fail" if pose["exact_collisions"] else "pass",
                        clearance_status="fail"
                        if pose["conservative_clearance_collisions"]
                        else "pass",
                        flagged=flagged,
                        pair=[hits[0]["cover"], hits[0]["part"]]
                        if hits
                        else ["None in this sample"],
                        frames=[
                            fk[owner].flatten().tolist()
                            for owner in candidate["owners"]
                        ],
                    )
                )
            candidate["source"] += " · " + str(motion_dir)
            robots.append(candidate)
            summaries.append(
                dict(robot=name, selected=selected, status=prior["status"])
            )
        if not robots:
            raise ValueError("No nominal stock candidate to display")
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
        (output / "index.html").write_text(
            wrapper(
                Path("ui/head-mount.fragment.html")
                .read_text()
                .replace("__HEAD_MOUNT_DATA__", packed)
            )
        )
        (output / "review.css").write_bytes(Path("ui/product-review.css").read_bytes())
        write(output / "selected-stock.json", summaries)
        report.update(
            status="pass",
            source_bindings=bindings,
            selected=summaries,
            exports=exports,
            presentation_files=[
                "index.html",
                "review.css",
                "selected-stock.json",
                *[e["stl"] for e in exports],
            ],
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        )
        write(output / "report.json", report)
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
