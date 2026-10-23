#!/usr/bin/env python3
"""Compile the populated Atopile carrier and retain each local design iteration."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.carrier import check_carrier


def generate_parts(root):
    parts = json.loads((root / "parts.json").read_text())
    lines = [
        "# Generated from parts.json; circuit connectivity is in main.ato.",
        "import is_atomic_part",
        "",
    ]
    for name, p in parts.items():
        lines += [
            f"component {name}:",
            f'    trait is_atomic_part<manufacturer="{p["manufacturer"]}", partnumber="{p["mpn"]}", footprint="{p["footprint"]}", symbol="">',
        ]
        lines += [f"    signal {sig} ~ pin {pin}" for sig, pin in p["pins"].items()]
        lines += [""]
    (root / "parts/parts.ato").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError("output must be new")
    shutil.copytree("electronics/carrier", output)
    shutil.copyfile("electronics/controller.yaml", output / "controller.yaml")
    generate_parts(output)
    report = {
        "status": "not_run",
        "physical_hardware": "not_run",
        "scope": "logic carrier; external power board and IMU remain separate",
        "steps": [],
    }
    report["source_hashes"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            *Path("electronics/carrier").rglob("*"),
            Path("electronics/controller.yaml"),
            Path("tools/build_carrier.py"),
            Path("tools/carrier_layout.py"),
            Path("tools/carrier_finish.py"),
            Path("lab/carrier.py"),
        ]
        if p.is_file()
    }
    for path in report["source_hashes"]:
        dest = output / "source" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    env = dict(os.environ, FBRK_TELEMETRY="0", ATO_NON_INTERACTIVE="1")
    commands = [
        (
            [str(Path(".venv-ato/bin/ato").resolve()), "build", str(output)],
            "atopile.log",
        ),
        (
            [
                str(
                    Path(
                        ".tools/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3"
                    ).resolve()
                ),
                "tools/carrier_layout.py",
                str(output),
            ],
            "layout.log",
        ),
    ]
    kicad = str(Path(".tools/KiCad/KiCad.app/Contents/MacOS/kicad-cli").resolve())
    kipython = commands[1][0][0]
    router = Path(".tools/freerouting-2.4.1.jar").resolve()
    report["tool_versions"] = {
        "atopile": "0.15.8",
        "kicad": "10.0.6",
        "freerouting": "2.4.1",
    }
    report["router_sha256"] = hashlib.sha256(router.read_bytes()).hexdigest()
    if (
        report["router_sha256"]
        != "251101c3eeac22d7e7dfcf6796603279e5d1000283eb82d8f093780f7afc6aa9"
    ):
        raise ValueError("router archive hash differs from release")
    commands += [
        (
            [
                "/opt/homebrew/opt/openjdk/bin/java",
                "-Xmx3g",
                "-Djava.awt.headless=true",
                "-jar",
                str(router),
                "-de",
                str(output / "carrier.dsn"),
                "-do",
                str(output / "carrier.ses"),
                "-da",
                "--gui.enabled=false",
                "--api_server.enabled=false",
                f"--user_data_path={output}/router-state",
                "--router.automatic_neckdown=false",
                "--router.fanout.enabled=false",
                "-mp",
                "20",
                "-mt",
                "1",
            ],
            "router.log",
        ),
        ([kipython, "tools/carrier_finish.py", str(output)], "import.log"),
        (
            [
                kicad,
                "pcb",
                "drc",
                "--format",
                "json",
                "-o",
                str(output / "routed-drc.json"),
                str(output / "carrier.kicad_pcb"),
            ],
            "drc.log",
        ),
        (
            [
                kicad,
                "pcb",
                "export",
                "step",
                "--subst-models",
                "-o",
                str(output / "carrier.step"),
                str(output / "carrier.kicad_pcb"),
            ],
            "step.log",
        ),
        (
            [
                kicad,
                "pcb",
                "render",
                "--width",
                "1000",
                "--height",
                "1200",
                "--background",
                "opaque",
                "-o",
                str(output / "carrier.png"),
                str(output / "carrier.kicad_pcb"),
            ],
            "render.log",
        ),
    ]
    try:
        for command, log in commands:
            p = subprocess.run(
                command, env=env, capture_output=True, text=True, timeout=180
            )
            (output / log).write_text(p.stdout + p.stderr)
            report["steps"].append(
                {"command": command, "exit_code": p.returncode, "log": log}
            )
            p.check_returncode()
        report["interface_check"] = check_carrier(output)
        drc = json.loads((output / "routed-drc.json").read_text())
        errors = [v for v in drc["violations"] if v["severity"] == "error"]
        warnings = [v for v in drc["violations"] if v["severity"] == "warning"]
        report["drc"] = {
            "errors": len(errors),
            "warnings": len(warnings),
            "warning_types": sorted({v["type"] for v in warnings}),
            "unconnected": len(drc["unconnected_items"]),
            "schematic_parity": "not_run; no KiCad schematic exists",
        }
        negative = subprocess.run(
            [
                kicad,
                "pcb",
                "drc",
                "--format",
                "json",
                "--exit-code-violations",
                "-o",
                str(output / "shorted-drc.json"),
                str(output / "carrier-shorted.kicad_pcb"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        (output / "shorted-drc.log").write_text(negative.stdout + negative.stderr)
        short_report = json.loads((output / "shorted-drc.json").read_text())
        detected = negative.returncode == 5 and any(
            v["type"] in {"shorting_items", "tracks_crossing"}
            and v["severity"] == "error"
            for v in short_report["violations"]
        )
        report["negative_drc"] = {
            "status": "pass" if detected else "fail",
            "exit_code": negative.returncode,
            "scope": "deliberate GND to 5 V short",
        }
        report["status"] = (
            "pass"
            if detected
            and not errors
            and not drc["unconnected_items"]
            and report["interface_check"]["status"] == "pass"
            else "fail"
        )
    except Exception as exc:
        report.update(status="fail", error=str(exc))
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(report["status"] != "pass")


if __name__ == "__main__":
    sys.exit(main())
