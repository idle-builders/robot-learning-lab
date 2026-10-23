"""Inspect head-bridge tool and fastener paths with their checked service stages."""

import argparse
import base64
import gzip
import json
from pathlib import Path
import traceback
from build123d import import_step
from tools.review_integral_head_service import embedded_dataset, add_asset
from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__)
    try:
        paths = [
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/review_integral_head_service.py"),
            Path("tools/build_product_review.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/cap_mesh.py"),
            Path("ui/head-bridge-service.fragment.html"),
            Path("ui/product-review.css"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        bindings = {}

        def bind(p):
            bindings[str(p)] = sha(p)
            return p

        source = Path("artifacts/integral-head-service-review-v2")
        prior = verify_report(source)
        bind(source / "report.json")
        data = embedded_dataset(bind(source / "index.html"))
        summary = []
        for name in ["pip", "scout"]:
            robot = next(
                r for r in data["robots"] if r["name"].lower().startswith(name)
            )
            folder = Path(f"artifacts/{name}-head-bridge-service-v3")
            r = verify_report(folder)
            bind(folder / "report.json")
            for path in r["source_bindings"]:
                if (
                    path.endswith("assembly.step")
                    or path.endswith("head_shell.step")
                    or path.endswith("head_rear_cover.step")
                ):
                    if r["source_bindings"][path] != prior["source_bindings"][path]:
                        raise ValueError("Viewer/bridge geometry mismatch")
                    bind(Path(path))
            if sha(folder / "resolved-config.json") != sha(
                Path(robot["source"]) / "resolved-config.json"
            ):
                raise ValueError("Viewer/bridge config mismatch")
            bind(folder / "resolved-config.json")
            robot["parts"] = [
                p
                for p in robot["parts"]
                if not p["name"].startswith("head_bridge_bolt_")
            ]
            assets = {}
            for x in [-5, 5]:
                key = f"head_bridge_bolt_{x}"
                for suffix in ["screw", "nut", "upper_washer", "lower_washer"]:
                    part = f"{key}_{suffix}"
                    path = bind(folder / f"{part}.step")
                    asset, _ = add_asset(
                        data, import_step(path), output / f"{name}-{part}.stl"
                    )
                    robot["parts"].append(
                        dict(
                            name=part,
                            owner="head",
                            role="nominal_bridge_hardware",
                            color=[0.55, 0.57, 0.6],
                            asset=asset,
                        )
                    )
                for suffix in ["driver", "wrench--1", "wrench-1"]:
                    asset, _ = add_asset(
                        data,
                        import_step(bind(folder / f"{key}-{suffix}.step")),
                        output / f"{name}-{key}-{suffix}.stl",
                    )
                    assets[f"{key}-{suffix}"] = asset
            variants = []
            for staged in [False, True]:
                samples = []
                tools = []

                def add(
                    case,
                    part,
                    kind,
                    row,
                    shift,
                    moving=(),
                    absent=(),
                    tool_asset=None,
                    stage="Closed head, neutral robot",
                ):
                    if not any(t["screw"] == case for t in tools):
                        tools.append(
                            dict(
                                screw=case,
                                label=case.replace("_", " "),
                                asset=tool_asset
                                if tool_asset is not None
                                else assets[part + "-driver"],
                            )
                        )
                    samples.append(
                        dict(
                            screw=case,
                            kind=kind,
                            stage="bridge_service",
                            stage_label=stage,
                            axis="z",
                            center_head_mm=[0, 0, 0],
                            status=row["status"],
                            hits=row["hits"],
                            distance_mm=max(abs(v) for v in shift),
                            angle_deg=0,
                            profile="nominal bridge/tool allocation",
                            shift_mm=shift,
                            moving_parts=list(moving),
                            absent_parts=list(absent),
                            show_tool=tool_asset is not None,
                        )
                    )

                if not staged:
                    for row in r["drivers"]:
                        part = row["part"]
                        add(
                            part + "_driver",
                            part,
                            "seated driver",
                            row,
                            [0, 0, 0],
                            tool_asset=assets[part + "-driver"],
                        )
                    for row in r["withdrawal"]:
                        part = row["part"]
                        kind = row["kind"]
                        absent = [part + "_nut"]
                        if kind == "driver":
                            add(
                                part + "_driver",
                                part,
                                kind,
                                row,
                                [0, 0, row["distance_mm"]],
                                absent=absent
                                + [part + "_screw", part + "_upper_washer"],
                                tool_asset=assets[part + "-driver"],
                            )
                        else:
                            add(
                                part + "_screw",
                                part,
                                kind,
                                row,
                                [0, 0, row["distance_mm"]],
                                moving=[part + "_screw", part + "_upper_washer"],
                                absent=absent,
                            )
                    for row in r["wrenches"]:
                        part = row["part"]
                        direction = row["direction"]
                        add(
                            part + f"_wrench_{direction}",
                            part,
                            "seated wrench",
                            row,
                            [0, 0, 0],
                            tool_asset=assets[part + f"-wrench-{direction}"],
                        )
                    for row in r["nut_insertion"]:
                        part = row["part"]
                        add(
                            part + "_nut",
                            part,
                            "axial nut insertion",
                            row,
                            [0, 0, -row["distance_mm"]],
                            moving=[part + "_nut"],
                            absent=[part + "_screw"],
                        )
                else:
                    for row in r["staged_service"]:
                        if row["kind"] == "simultaneous_tools":
                            continue
                        part = row["part"]
                        kind = row["kind"]
                        absent = list(row["absent_parts"])
                        stage = "Rear cover/Pi removed · " + (
                            "front stack first; rear stack absent"
                            if row["stage"] == "front_first"
                            else "rear stack second; front stack installed"
                        )
                        if kind == "wrench":
                            add(
                                part + "_wrench",
                                part,
                                kind,
                                row,
                                [-row["rear_offset_mm"], 0, 0],
                                absent=absent,
                                tool_asset=assets[part + "-wrench--1"],
                                stage=stage,
                            )
                        else:
                            add(
                                part + "_nut_washer",
                                part,
                                kind,
                                row,
                                [-row["rear_offset_mm"], 0, 0],
                                moving=[part + "_nut", part + "_lower_washer"],
                                absent=absent
                                + [part + "_screw", part + "_upper_washer"],
                                stage=stage,
                            )
                variants.append(
                    dict(
                        name="Rear opening · front then rear"
                        if staged
                        else "Closed head · failed routes",
                        version=3,
                        status="partial"
                        if staged and r["staged_service_status"] == "pass"
                        else "fail",
                        inherited_status=r["inherited_assembly_status"],
                        source=str(folder / "report.json"),
                        tools=tools,
                        samples=samples,
                        checks={
                            k: r[k]
                            for k in [
                                "bearing_status",
                                "material_retention_status",
                                "wrench_engagement_status",
                                "drivers_status",
                                "withdrawal_status",
                                *(
                                    ["staged_service_status"]
                                    if staged
                                    else ["wrenches_status", "nut_insertion_status"]
                                ),
                                "physical_status",
                            ]
                        },
                        default_case=tools[0]["screw"],
                    )
                )
                summary.append(dict(robot=name, staged=staged, samples=len(samples)))
            robot["variants"] = variants
        encoded = base64.b64encode(
            gzip.compress(json.dumps(data, separators=(",", ":")).encode())
        ).decode()
        html = (
            Path("ui/head-bridge-service.fragment.html")
            .read_text()
            .replace("__INTEGRAL_HEAD_SERVICE_DATA__", encoded)
        )
        (output / "index.html").write_text(wrapper(html))
        (output / "review.css").write_text(Path("ui/product-review.css").read_text())
        report.update(
            status="pass",
            source_bindings=bindings,
            cases=summary,
            presentation_files=["index.html", "review.css"],
            limits=[
                "Simplified mesh display of discrete CAD samples. Two simultaneous-driver/wrench overlap checks per robot are reported but not animated.",
                "Nominal socket/allocated wrench geometry, no physical tool/hand/retention/torque qualification.",
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
