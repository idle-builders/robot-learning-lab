"""One local command produces artifacts, measurements and explicit coverage gaps."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import yaml
from .artifact_store import snapshot_file, plan_migration, apply_migration, verify_manifest
from .config import load_config, RobotConfig
from .robot import make_robot
from .export import export_all
from .control import build_supervisor
from .cost import estimate


def source_hashes():
    paths = []
    for folder in (
        "lab",
        "robot",
        "variants",
        "firmware",
        "electronics",
        "tools",
        "tests",
        "experiments",
    ):
        paths.extend(
            p
            for p in Path(folder).rglob("*")
            if p.is_file() and p.suffix in (".py", ".yaml", ".c", ".h", ".ato")
        )
    paths.extend(Path(".").glob("requirements*.lock"))
    paths.append(Path("pyproject.toml"))
    paths.append(Path("tools/lab"))
    paths.append(Path("AGENTS.md"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def run(variant, output, with_cad=True, with_render=True, with_ato=True, overrides=()):
    from .simulation import tracking, render
    from .electrical import power_transient, atopile_contract, find_tool
    from .pcb import pcb_check

    config = load_config(Path(variant)).model_dump()
    for assignment in overrides:
        key, value = assignment.split("=", 1)
        keys = key.split(".")
        node = config
        for part in keys[:-1]:
            node = node[part]
        if keys[-1] not in node:
            raise ValueError(f"unknown parameter: {key}")
        node[keys[-1]] = yaml.safe_load(value)
    robot = make_robot(RobotConfig.model_validate(config))
    output = Path(output)
    if output.exists():
        raise ValueError(f"output already exists; choose a new run directory: {output}")
    output.mkdir(parents=True)
    export_all(robot, output)
    (output / "manifest.schema.json").write_text(
        json.dumps(RobotConfig.model_json_schema(), indent=2)
    )
    report = {
        "schema_version": 1,
        "robot": robot.config.name,
        "variant_path": str(Path(variant)),
        "overrides": [
            {
                "parameter": item.split("=", 1)[0],
                "value": yaml.safe_load(item.split("=", 1)[1]),
            }
            for item in overrides
        ],
        "resolved_config_sha256": hashlib.sha256(
            (output / "resolved-config.json").read_bytes()
        ).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "source_hashes": source_hashes(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("build123d", "mujoco", "pydantic", "PyYAML")
        },
        "checks": {},
        "cost": estimate(robot),
        "coverage": {
            "locomotion": "not_run",
            "grasping": "not_run",
            "measured_actuator_identification": "not_run",
            "component_level_pcb": "not_run",
            "physical_hardware": "not_run",
            "industrial_design": "concept_only",
        },
    }
    if robot.config.head_service.enabled:
        from .head_service import interface as head_service_interface

        report["head_service"] = head_service_interface(
            robot.config.dimensions, robot.config.head_service
        )
    if robot.config.face.visor_mount_enabled:
        from .visor_mount import interface as visor_interface

        report["visor_mount"] = visor_interface(robot.config.dimensions)
    if robot.config.camera.part_id:
        from .vision import load_vision

        report["checks"]["vision_power_allocation"] = load_vision().power.budget_check
        report["coverage"]["physical_vision_mount"] = "not_run"
        report["coverage"]["physical_vision_stream"] = "not_run"
    if robot.config.camera.support_enabled:
        from .camera_support import specification, interface

        support = specification(robot.config.camera.support_profile)
        report["camera_support"] = interface(support)
        dependencies = [Path(support["source_drawing"])]
        if support["profile"] == "vendor_step":
            dependencies.append(Path(support["source_vendor"]))
        for path in dependencies:
            report["source_hashes"][str(path)] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    for path in report["source_hashes"]:
        target = output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, target)
    external = {}
    for tool, args in [("kicad-cli", ["version"]), ("ngspice", ["--version"])]:
        exe = find_tool(tool)
        if exe:
            result = subprocess.run(
                [exe, *args], capture_output=True, text=True, timeout=15
            )
            external[tool] = {
                "executable": exe,
                "version_output": (result.stdout + result.stderr).strip(),
            }
    ato_python = Path(".venv-ato/bin/python")
    if ato_python.exists():
        result = subprocess.run(
            [
                str(ato_python),
                "-c",
                'import importlib.metadata; print(importlib.metadata.version("atopile"))',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        external["atopile"] = {"version": result.stdout.strip()}
    report["external_tools"] = external

    def check(name, fn):
        try:
            report["checks"][name] = fn()
        except Exception as exc:
            (output / f"{name}-error.log").write_text(traceback.format_exc())
            report["checks"][name] = {
                "status": "fail",
                "error": str(exc),
                "log": f"{name}-error.log",
            }

    from .components import check_components

    check("component_constraints", lambda: check_components(robot))

    if with_cad:
        from .mechanical import build_cad

        check("cad", lambda: build_cad(robot, output))
        export_all(robot, output)
    else:
        report["checks"]["cad"] = {"status": "not_run"}
    libraries = []

    def firmware_check():
        libraries.append(build_supervisor(output))
        return {"status": "pass", "scope": "host compile, not target-board firmware"}

    check("native_firmware", firmware_check)
    trace = []

    def sim_check():
        result, items = tracking(robot, output, libraries[0])
        trace.extend(items)
        return result

    if libraries:
        check("simulation", sim_check)
    else:
        report["checks"]["simulation"] = {
            "status": "not_run",
            "reason": "native compile failed",
        }
    if trace:
        check("power", lambda: power_transient(robot, trace, output))
        check(
            "power_fault_detection", lambda: power_transient(robot, trace, output, True)
        )
    else:
        report["checks"]["power"] = {
            "status": "not_run",
            "reason": "simulation produced no load trace",
        }
    if with_render:
        check("render", lambda: render(output))
    else:
        report["checks"]["render"] = {"status": "not_run"}
    if with_ato:
        check("atopile_interfaces", lambda: atopile_contract(robot, output))
    else:
        report["checks"]["atopile_interfaces"] = {"status": "not_run"}
    check("pcb_toolchain", lambda: pcb_check(robot, output))
    report["status"] = (
        "fail"
        if any(x["status"] == "fail" for x in report["checks"].values())
        else "partial"
        if any(x["status"] == "not_run" for x in report["checks"].values())
        else "pass"
    )
    report["status_scope"] = "workbench checks only; does not certify product readiness"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_review(report, output)
    # Finalize only closed local artifact runs. Private test/export directories
    # remain independently writable and portable.
    repository = Path(__file__).resolve().parents[1]
    if output.absolute().is_relative_to(repository / "artifacts"):
        manifest = plan_migration(repository, [output.absolute()])
        apply_migration(repository, manifest)
        integrity = verify_manifest(repository, manifest)
        if integrity["status"] != "pass":
            raise ValueError(f"completed artifact integrity failed: {manifest}")
    return report


def write_review(report, output):
    import html

    cards = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{html.escape(value['status'])}</td><td><pre>{html.escape(json.dumps(value, indent=2))}</pre></td></tr>"
        for name, value in report["checks"].items()
    )
    robot_image = (
        '<img src="robot.png" alt="Robot geometry from shared manifest">'
        if (output / "robot.png").exists()
        else "<p>Render omitted in this run.</p>"
    )
    body = f"""<!doctype html><meta charset="utf-8"><title>Robot Learning Lab</title>
<style>body{{font:16px system-ui;background:#f4f2eb;color:#172c32;margin:40px auto;max-width:1100px;padding:0 24px}}h1{{font-size:40px}}img{{width:100%;border-radius:16px}}table{{width:100%;border-collapse:collapse}}td{{border-bottom:1px solid #b7c5c2;padding:16px;vertical-align:top}}pre{{white-space:pre-wrap;font-size:12px}}a{{color:#007966}}.tag{{color:#506765}}</style>
<p class="tag">LOCAL DESIGN EXPERIMENT · {html.escape(report["created_at"])}</p>
<h1>{html.escape(report["robot"])}</h1><p>Engineering envelope, not the final industrial design. Workbench result: <b>{report["status"]}</b>.</p>
{robot_image}
<p><a href="robot-envelope.step">CAD STEP</a> · <a href="robot.urdf">URDF</a> · <a href="robot.xml">MuJoCo</a> · <a href="interfaces.json">Interfaces</a> · <a href="report.json">Full report</a></p>
<h2>Checks</h2><table>{cards}</table><h2>Production planning estimate</h2><pre>{html.escape(json.dumps(report["cost"], indent=2))}</pre>
<h2>Unvalidated areas</h2><pre>{html.escape(json.dumps(report["coverage"], indent=2))}</pre>"""
    (output / "review.html").write_text(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("run")
    build.add_argument("variant")
    build.add_argument("--output")
    build.add_argument("--no-cad", action="store_true")
    build.add_argument("--no-render", action="store_true")
    build.add_argument("--no-atopile", action="store_true")
    build.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="PARAMETER=VALUE",
    )
    sweep = sub.add_parser("sweep")
    sweep.add_argument("variant")
    sweep.add_argument("--parameter", required=True)
    sweep.add_argument("--values", required=True)
    sweep.add_argument("--output")
    study = sub.add_parser("study")
    study.add_argument("specification")
    study.add_argument("--output", required=True)
    study.add_argument(
        "--quick",
        action="store_true",
        help="skip CAD/render/atopile, retain partial status",
    )
    review = sub.add_parser(
        "review", help="join retained per-candidate evidence locally"
    )
    review.add_argument("specification")
    review.add_argument("--output", required=True)
    sub.add_parser("schema")
    args = parser.parse_args()
    if args.command == "schema":
        print(json.dumps(RobotConfig.model_json_schema(), indent=2))
        return
    if args.command == "review":
        from .candidate_review import run as review_candidate

        report = review_candidate(args.specification, args.output)
        print(
            json.dumps(
                {k: report[k] for k in ("status", "consistency", "variant")}, indent=2
            )
        )
        if report["status"] == "fail":
            sys.exit(1)
        return
    if args.command == "study":
        from .experiments import run_study

        comparison = run_study(args.specification, args.output, args.quick)
        print(
            json.dumps(
                {
                    "output": args.output,
                    "cases": {c["case"]: c["status"] for c in comparison["cases"]},
                },
                indent=2,
            )
        )
        if any(c["status"] == "fail" for c in comparison["cases"]):
            sys.exit(1)
        return
    output = (
        args.output
        or "artifacts/"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + Path(args.variant).stem
    )
    if args.command == "sweep":
        parent = Path(output)
        if parent.exists():
            raise ValueError("sweep output already exists")
        parent.mkdir(parents=True)
        results = []
        for i, value in enumerate(args.values.split(",")):
            r = run(
                args.variant,
                parent / f"case-{i}",
                overrides=[args.parameter + "=" + value],
            )
            results.append(
                {
                    "parameter": args.parameter,
                    "value": yaml.safe_load(value),
                    "status": r["status"],
                    "resolved_config_sha256": r["resolved_config_sha256"],
                    "checks": r["checks"],
                    "cost": r["cost"],
                    "review": f"case-{i}/review.html",
                }
            )
        (parent / "comparison.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    "output": str(parent),
                    "runs": [
                        {"value": r["value"], "status": r["status"]} for r in results
                    ],
                },
                indent=2,
            )
        )
        if any(r["status"] == "fail" for r in results):
            sys.exit(1)
        return
    report = run(
        args.variant,
        output,
        not args.no_cad,
        not args.no_render,
        not args.no_atopile,
        args.overrides,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": output,
                "checks": {n: v["status"] for n, v in report["checks"].items()},
            },
            indent=2,
        )
    )
    if report["status"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
