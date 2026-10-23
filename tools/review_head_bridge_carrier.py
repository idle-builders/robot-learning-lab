"""Present the printed head-bridge carrier's staged tool and hardware path."""

import argparse
import base64
import gzip
import json
from pathlib import Path
import traceback

from build123d import Pos, import_step

from tools.build_product_review import wrapper
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.review_integral_head_service import add_asset, embedded_dataset


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__)
    try:
        source_files = [
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/review_integral_head_service.py"),
            Path("tools/build_product_review.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/cap_mesh.py"),
            Path("ui/head-bridge-carrier.fragment.html"),
            Path("ui/product-review.css"),
        ]
        report["source_hashes"] = copy_source(
            output, {str(path): path for path in source_files}
        )
        bindings = {}

        def bind(path):
            bindings[str(path)] = sha(path)
            return path

        context = Path("artifacts/head-bridge-service-review-v3")
        context_report = verify_report(context)
        bind(context / "report.json")
        context_data = embedded_dataset(bind(context / "index.html"))

        summaries = []
        for robot in context_data["robots"]:
            name = robot["name"].split(" · ", 1)[0].lower()
            carrier_dir = Path(f"artifacts/{name}-head-bridge-carrier-v1")
            carrier_report = verify_report(carrier_dir)
            bind(carrier_dir / "report.json")
            bind(carrier_dir / "carrier.step")
            for path, digest in carrier_report["source_bindings"].items():
                if sha(Path(path)) != digest:
                    raise ValueError(f"carrier source binding changed: {path}")
                bindings[path] = digest

            # Carrier producers must agree with the source-bound context on the
            # assembly, head wings, bridge hardware and resolved configuration.
            for path, digest in carrier_report["source_bindings"].items():
                if any(
                    path.endswith(suffix)
                    for suffix in (
                        "assembly.step",
                        "head_shell.step",
                        "head_rear_cover.step",
                        "resolved-config.json",
                        "head_bridge_bolt_-5_screw.step",
                        "head_bridge_bolt_5_screw.step",
                    )
                ):
                    if context_report["source_bindings"].get(path) != digest:
                        raise ValueError(f"carrier/context geometry mismatch: {path}")

            service_dir = Path(f"artifacts/{name}-head-bridge-service-v3")
            verify_report(service_dir)
            carrier_assets = {}
            carrier_shape = import_step(bind(carrier_dir / "carrier.step"))
            for sign in (-5, 5):
                carrier_assets[sign] = add_asset(
                    context_data,
                    Pos(sign, 0, 0) * carrier_shape,
                    output / f"{name}-carrier-{sign}.stl",
                )[0]

            samples = []
            tools = []
            for row in carrier_report["samples"]:
                part = row["part"]
                screw = f"{part}_carrier"
                if not any(tool["screw"] == screw for tool in tools):
                    sign = int(part.rsplit("_", 1)[1])
                    tools.append(
                        dict(
                            screw=screw,
                            label=f"{part} carrier service tool",
                            asset=carrier_assets[sign],
                        )
                    )
                sign = int(part.rsplit("_", 1)[1])
                absent = list(row["absent_parts"])
                if row["kind"] == "loaded_insertion":
                    absent += [f"{part}_screw", f"{part}_upper_washer"]
                    moving = [f"{part}_nut", f"{part}_lower_washer"]
                else:
                    moving = []
                stage_label = (
                    "Rear cover/Pi removed · front stack first; rear stack absent"
                    if row["stage"] == "front_first"
                    else "Rear cover/Pi removed · rear stack second; front stack installed"
                )
                samples.append(
                    dict(
                        screw=screw,
                        kind=row["kind"],
                        stage="carrier_service",
                        stage_label=stage_label,
                        axis="x",
                        center_head_mm=[0, 0, 0],
                        status=row["status"],
                        hits=row["hits"],
                        distance_mm=row["rear_offset_mm"],
                        angle_deg=0,
                        profile="printed carrier; level-held positioning aid",
                        shift_mm=[-row["rear_offset_mm"], 0, 0],
                        moving_parts=moving,
                        absent_parts=sorted(set(absent)),
                        show_tool=True,
                    )
                )

            robot["variants"] = [
                dict(
                    name="Carrier staged service · front then rear",
                    version=1,
                    status="partial",
                    inherited_status=carrier_report["inherited_assembly_status"],
                    source=str(carrier_dir / "report.json"),
                    tools=tools,
                    samples=samples,
                    checks={
                        "carrier_fits": carrier_report["fits_status"],
                        "carrier_supports": carrier_report["supports_status"],
                        "sampled_service": carrier_report["samples_status"],
                        "physical": carrier_report["physical_status"],
                        "strength": carrier_report["strength_status"],
                        "harness": carrier_report["harness_status"],
                    },
                    default_case=tools[0]["screw"],
                )
            ]
            summaries.append(
                dict(
                    robot=name,
                    samples=len(samples),
                    loaded=sum(s["kind"] == "loaded_insertion" for s in samples),
                    unloaded=sum(s["kind"] == "unloaded_withdrawal" for s in samples),
                )
            )

        for filename in ["carrier-print.stl", "carrier.step"]:
            source = bind(Path("artifacts/pip-head-bridge-carrier-v1") / filename)
            (output / filename).write_bytes(source.read_bytes())
        encoded = base64.b64encode(
            gzip.compress(json.dumps(context_data, separators=(",", ":")).encode())
        ).decode()
        html = (
            Path("ui/head-bridge-carrier.fragment.html")
            .read_text()
            .replace("__HEAD_BRIDGE_CARRIER_DATA__", encoded)
        )
        (output / "index.html").write_text(wrapper(html))
        (output / "review.css").write_text(Path("ui/product-review.css").read_text())
        report.update(
            status="pass",
            source_bindings=bindings,
            cases=summaries,
            presentation_files=[
                "index.html",
                "review.css",
                "carrier-print.stl",
                "carrier.step",
            ],
            limits=[
                "Full neutral robot context is inherited from the source-bound bridge-service review; the carrier STEP is placed at x=+/-5 in the viewer tool assets.",
                "Loaded insertion hides the selected screw and upper washer and moves the placed carrier/tool plus nut and lower washer together. Unloaded withdrawal moves only the placed carrier/tool while retaining the complete engaged stack.",
                "Discrete nominal translations only. Printed material, strength, gravity capture, hand clearance, torque, thread engagement, harness routing, and physical qualification remain unqualified or not_run.",
            ],
        )
        report["output_hashes"] = {
            str(path.relative_to(output)): sha(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "report.json"
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
