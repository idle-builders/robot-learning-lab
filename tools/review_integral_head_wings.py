"""Present source-bound integral lower-wing candidates and their CAD evidence."""

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
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/cap_mesh.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/build_product_review.py"),
            Path("ui/integral-head-wings.fragment.html"),
            Path("ui/product-review.css"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        bindings = {}

        def bind(path):
            bindings[str(path)] = sha(path)
            return path

        base = Path("artifacts/head-display-v7")
        prior = verify_report(base)
        bind(base / "report.json")
        data = json.loads(bind(base / "dataset.json").read_text())
        robots = []
        exports = []
        for name in ("pip", "scout"):
            version = "v5" if name == "pip" else "v4"
            study = Path(f"artifacts/{name}-integral-head-wings-{version}")
            evidence = verify_report(study)
            bind(study / "report.json")
            cfg = RobotConfig.model_validate_json(
                bind(study / "resolved-config.json").read_text()
            )
            robot = make_robot(cfg)
            assembly = f"artifacts/{name}-distal-grip-integrated-v1/assembly.step"
            if (
                evidence["source_bindings"][assembly]
                != prior["source_bindings"][assembly]
            ):
                raise ValueError("Assembly/display identity mismatch")
            candidate = deepcopy(
                next(r for r in data["robots"] if r["name"].lower() == name)
            )
            candidate["name"] = name.capitalize() + " · integral lower wings"
            candidate["parts"] = [p for p in candidate["parts"] if p["role"] != "scene"]
            candidate["owners"] = [link.name for link in robot.links]
            candidate["source"] = str(study)
            candidate["inherited_status"] = evidence["inherited_assembly_status"]
            candidate["stock_summary"] = (
                f"Nominal joining {evidence['nominal_status'].upper()} · sampled neck clearance {evidence['motion_status'].upper()} · conditional cover/board removal {evidence['service_status'].upper()} · new lower-cover fasteners: zero"
            )
            for key in ("head_shell", "head_rear_cover"):
                shape = import_step(bind(study / f"{key}.step"))
                dest = output / f"{name}-{key}.stl"
                mesh, cleanup = export_checked_stl(shape, dest)
                quantized = np.round(mesh.vertices * 10).astype(int)
                display = trimesh.Trimesh(quantized / 10, mesh.faces, process=False)
                display.update_faces(display.nondegenerate_faces(height=1e-8))
                if abs(display.volume - mesh.volume) / mesh.volume > 0.02:
                    raise ValueError("Display quantization changes volume")
                part = next(p for p in candidate["parts"] if p["name"] == key)
                part.update(
                    role="integral_shell_candidate",
                    download=dest.name,
                    asset=len(data["assets"]),
                    color=[0.3, 0.64, 0.53],
                )
                data["assets"].append(
                    [quantized.flatten().tolist(), display.faces.flatten().tolist()]
                )
                exports.append(
                    dict(robot=name, part=key, stl=dest.name, mesh_cleanup=cleanup)
                )
            candidate["samples"] = []
            for pose in evidence["poses"]:
                fk = robot.forward_kinematics(pose["pose_rad"], enforce_limits=False)
                failures = pose["failures"]
                flagged = sorted(
                    {
                        part
                        for f in failures
                        for part in [
                            "head_shell"
                            if "_front_" in f["wing"]
                            else "head_rear_cover",
                            f["target"],
                        ]
                    }
                )
                candidate["samples"].append(
                    dict(
                        name=pose["name"],
                        kind="pose",
                        phase="",
                        status=pose["status"],
                        clearance_status=pose["status"],
                        flagged=flagged,
                        pair=flagged or ["None in this sample"],
                        gap_note="; ".join(
                            f"{f['wing']} / {f['target']}: {f['distance_mm']:.2f} mm"
                            for f in failures
                        ),
                        frames=[
                            fk[owner].flatten().tolist()
                            for owner in candidate["owners"]
                        ],
                    )
                )
            if len(candidate["samples"]) != 37:
                raise ValueError("Requires completed 37-pose evidence")
            robots.append(candidate)
        data["robots"] = robots
        encoded = base64.b64encode(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode())
        ).decode()
        fragment = (
            Path("ui/integral-head-wings.fragment.html")
            .read_text()
            .replace("__HEAD_MOUNT_DATA__", encoded)
        )
        (output / "index.html").write_text(wrapper(fragment))
        (output / "review.css").write_text(Path("ui/product-review.css").read_text())
        report.update(
            status="pass",
            source_bindings=bindings,
            exports=exports,
            presentation_files=["index.html", "review.css"]
            + [p["stl"] for p in exports],
        )
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "report.json"
        }
        write(output / "report.json", report)
        return report
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
