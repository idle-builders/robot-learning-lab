"""Present retained service-tool allocations, failures and exact sampled motions."""

import argparse
import base64
import gzip
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import import_step

from lab.config import RobotConfig
from lab.head_service import pi_reservation
from lab.robot import make_robot
from tools.build_product_review import wrapper
from tools.cap_mesh import export_checked_stl
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def embedded_dataset(path):
    marker = '<script type="application/octet-stream" data-id="data">'
    encoded = path.read_text().split(marker, 1)[1].split("</script>", 1)[0].strip()
    return json.loads(gzip.decompress(base64.b64decode(encoded)))


def add_asset(data, shape, path):
    mesh, cleanup = export_checked_stl(shape, path)
    # Keep tool precision finer than the body's 0.1 mm display quantization.
    # Dataset coordinates are still units of 0.1 mm; fractional values allowed.
    vertices = np.round(mesh.vertices * 10, 3)
    asset = len(data["assets"])
    data["assets"].append([vertices.flatten().tolist(), mesh.faces.flatten().tolist()])
    return asset, cleanup


def service_samples(report):
    tools = {row["part"]: row for row in report["tools"]}
    samples = []
    for kind, rows in [
        ("seated", report["tools"]),
        ("withdrawal", report["withdrawal"]),
        ("swing", report.get("swing", [])),
    ]:
        for row in rows:
            screw = row["part"]
            tool = tools[screw]
            action = row.get("kind", kind)
            profile = tool.get("profile", "straight_driver_allocation")
            samples.append(
                dict(
                    screw=screw,
                    kind=action,
                    stage=row["stage"],
                    axis=tool["axis"],
                    center_head_mm=tool["center_head_mm"],
                    status=row["status"],
                    hits=row["hits"],
                    distance_mm=row.get("distance_mm", 0),
                    angle_deg=row.get("angle_deg", 0),
                    profile=profile,
                    label=f"{action} · {row.get('distance_mm', 0)} mm · {row.get('angle_deg', 0)}°",
                )
            )
    return samples


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__)
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/build_product_review.py"),
            Path("tools/cap_mesh.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("ui/integral-head-service.fragment.html"),
            Path("ui/product-review.css"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        bindings = {}

        def bind(path):
            bindings[str(path)] = sha(path)
            return path

        view = Path("artifacts/integral-head-wings-review-v2")
        view_report = verify_report(view)
        bind(view / "report.json")
        data = embedded_dataset(bind(view / "index.html"))
        summaries = []
        for name in ("pip", "scout"):
            robot_data = next(
                r for r in data["robots"] if r["name"].lower().startswith(name)
            )
            study = Path(robot_data["source"])
            wing_report = verify_report(study)
            bind(study / "report.json")
            cfg_path = study / "resolved-config.json"
            cfg = RobotConfig.model_validate_json(bind(cfg_path).read_text())
            robot = make_robot(cfg)
            robot_data["neutral_frames"] = [
                robot.forward_kinematics()[owner].flatten().tolist()
                for owner in robot_data["owners"]
            ]
            robot_data.pop("samples")
            for part in robot_data["parts"]:
                part.pop("download", None)
            asset, _ = add_asset(
                data,
                pi_reservation(cfg.head_service),
                output / f"{name}-reservation.stl",
            )
            robot_data["parts"].append(
                dict(
                    name="pi_component_reservation",
                    owner="head",
                    role="reservation",
                    asset=asset,
                    color=[0.85, 0.62, 0.18],
                )
            )
            robot_data["variants"] = []
            assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1/assembly.step")
            for v in (1, 2):
                folder = Path(f"artifacts/{name}-integral-head-service-v{v}")
                service = verify_report(folder)
                bind(folder / "report.json")
                service_cfg = RobotConfig.model_validate_json(
                    bind(folder / "resolved-config.json").read_text()
                )
                if service_cfg != cfg:
                    raise ValueError("Service/view configuration mismatch")
                # Both assembly and replaced shell identities must match the view.
                for path in [
                    assembly,
                    study / "head_shell.step",
                    study / "head_rear_cover.step",
                ]:
                    expected = (
                        wing_report["source_bindings"][str(path)]
                        if path == assembly
                        else view_report["source_bindings"][str(path)]
                    )
                    if service["source_bindings"][str(path)] != expected:
                        # The view binds original assembly transitively through its
                        # wing report, whose live bindings verify_report checked.
                        raise ValueError(f"Service/view geometry mismatch: {path}")
                    bind(path)
                tools = []
                for row in service["tools"]:
                    path = bind(folder / f"{row['part']}-driver.step")
                    asset, _ = add_asset(
                        data,
                        import_step(path),
                        output / f"{name}-v{v}-{row['part']}.stl",
                    )
                    tools.append(
                        dict(
                            screw=row["part"],
                            asset=asset,
                            axis=row["axis"],
                            center_head_mm=row["center_head_mm"],
                        )
                    )
                robot_data["variants"].append(
                    dict(
                        name="Straight driver · retained failure"
                        if v == 1
                        else "Right-angle Pi key · candidate",
                        version=v,
                        status=service["status"],
                        inherited_status=service["inherited_assembly_status"],
                        source=str(folder / "report.json"),
                        tools=tools,
                        samples=service_samples(service),
                        checks={
                            k: service.get(k, "not_run")
                            for k in [
                                "bearing_status",
                                "driver_status",
                                "withdrawal_status",
                                "swing_status",
                                "bridge_access_status",
                                "harness_status",
                                "physical_status",
                            ]
                        },
                    )
                )
                summaries.append(
                    dict(
                        robot=name,
                        version=v,
                        samples=len(robot_data["variants"][-1]["samples"]),
                        status=service["status"],
                    )
                )
        encoded = base64.b64encode(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode())
        ).decode()
        fragment = (
            Path("ui/integral-head-service.fragment.html")
            .read_text()
            .replace("__INTEGRAL_HEAD_SERVICE_DATA__", encoded)
        )
        (output / "index.html").write_text(wrapper(fragment))
        (output / "review.css").write_text(Path("ui/product-review.css").read_text())
        report.update(
            status="pass",
            source_bindings=bindings,
            cases=summaries,
            presentation_files=["index.html", "review.css"],
            limits=[
                "Presentation integrity only; exact evidence remains in service reports.",
                "V1 viewer draft retained; source audit and UI review found incomplete frames, selection and tool transforms. This fresh viewer supersedes it.",
            ],
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
