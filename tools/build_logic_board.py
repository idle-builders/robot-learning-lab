#!/usr/bin/env python3
"""Compile and route the logic supply into a new retained local artifact."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.build_carrier import generate_parts
from lab.logic_supply import compiled_circuit, validate_config, verify
import yaml


def run(output):
    if output.exists():
        raise FileExistsError(output)
    source = Path("electronics/logic-supply")
    shutil.copytree(source, output)
    report = {
        "status": "fail",
        "scope": "candidate 5 V PCB layout; electrical, thermal and physical qualification remain separate",
        "steps": [],
    }
    paths = [
        *source.rglob("*"),
        Path("tools/build_logic_board.py"),
        Path("tools/logic_board_layout.py"),
        Path("tools/build_carrier.py"),
        Path("lab/logic_supply.py"),
    ]
    report["source_hashes"] = {}
    for p in paths:
        if p.is_file():
            dest = output / "source" / p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
            report["source_hashes"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    generate_parts(output)
    # Conservative maximum package body, deliberately distinct from vendor CAD.
    from build123d import Box, Pos, Color, export_step

    parts = json.loads((output / "parts.json").read_text())
    sx, sy, sz = parts["Inductor10u"]["model"]["size_mm"]
    body = Pos(0, 0, sz / 2) * Box(sx, sy, sz)
    body.label = "7447714100_maximum_envelope"
    body.color = Color(0.16, 0.16, 0.18)
    export_step(body, output / "parts/L_Wuerth_7447714100-envelope.step")
    ki = str(Path(".tools/KiCad/KiCad.app/Contents/MacOS/kicad-cli").resolve())
    py = str(
        Path(
            ".tools/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/3.9/bin/python3"
        ).resolve()
    )
    commands = [
        (
            [str(Path(".venv-ato/bin/ato").resolve()), "build", str(output.resolve())],
            "atopile.log",
        ),
        ([py, "tools/logic_board_layout.py", str(output)], "layout.log"),
    ]
    for name in ("logic-supply", "logic-supply-shorted"):
        commands.append(
            (
                [
                    ki,
                    "pcb",
                    "drc",
                    "--format",
                    "json",
                    "-o",
                    str(output / (name + "-drc.json")),
                    str(output / (name + ".kicad_pcb")),
                ],
                name + "-drc.log",
            )
        )
    commands += [
        (
            [
                ki,
                "pcb",
                "export",
                "step",
                "--subst-models",
                "-o",
                str(output / "logic-supply.step"),
                str(output / "logic-supply.kicad_pcb"),
            ],
            "step.log",
        ),
        (
            [
                ki,
                "pcb",
                "render",
                "--width",
                "1200",
                "--height",
                "1000",
                "--background",
                "opaque",
                "-o",
                str(output / "logic-supply.png"),
                str(output / "logic-supply.kicad_pcb"),
            ],
            "render.log",
        ),
        (
            [
                ki,
                "pcb",
                "export",
                "svg",
                "--mode-multi",
                "--layers",
                "F.Cu,B.Cu,Edge.Cuts,F.Fab",
                "--output",
                str(output / "layers") + "/",
                str(output / "logic-supply.kicad_pcb"),
            ],
            "svg.log",
        ),
    ]
    try:
        for cmd, log in commands:
            p = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=dict(os.environ, FBRK_TELEMETRY="0", ATO_NON_INTERACTIVE="1"),
                timeout=180,
            )
            (output / log).write_text(p.stdout + p.stderr)
            report["steps"].append(
                {"command": cmd, "exit_code": p.returncode, "log": log}
            )
            p.check_returncode()
        circuit = compiled_circuit(output)
        report["circuit_constraints"] = verify(
            circuit,
            validate_config(yaml.safe_load((output / "verification.yaml").read_text())),
        )
        interface = json.loads((output / "board-interface.json").read_text())
        expected = {
            ref + "." + n: net
            for ref, part in circuit.items()
            for n, net in part["nets"].items()
        }
        report["pad_net_parity"] = expected == {
            n: p["net"] for n, p in interface["pads"].items()
        }
        d = json.loads((output / "logic-supply-drc.json").read_text())
        report["drc"] = {
            "errors": [v for v in d["violations"] if v["severity"] == "error"],
            "warnings": [v for v in d["violations"] if v["severity"] == "warning"],
            "unconnected": d["unconnected_items"],
        }
        d = json.loads((output / "logic-supply-shorted-drc.json").read_text())
        report["negative_short_detected"] = any(
            v["type"] in ("shorting_items", "tracks_crossing")
            and v["severity"] == "error"
            for v in d["violations"]
        )
        report["status"] = (
            "pass"
            if report["pad_net_parity"]
            and report["negative_short_detected"]
            and not report["drc"]["errors"]
            and not report["drc"]["unconnected"]
            and report["circuit_constraints"]["status"] == "pass"
            else "fail"
        )
        report["missing_3d_models"] = interface["missing_3d_models"]
    except Exception as exc:
        report["error"] = str(exc)
    (output / "build-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "error"] if k in report}))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    raise SystemExit(int(run(args.output)["status"] != "pass"))
