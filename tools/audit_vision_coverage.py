"""Reconcile baseline, retained waveform and new full-battery coverage.

Pass/fail applies to the stated regulator case; system power stays partial.
"""

import argparse
import json
import html
import os
import subprocess
from pathlib import Path
import traceback
import numpy as np
import yaml
from lab.logic_supply import compiled_circuit, measure, spice, validate_config, verify
from lab.vision import load_vision
from tools.review_vision_power import checked_report, metrics
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def present(output, report):
    """Plot retained traces; no solver result is inferred from presentation."""
    write(output / "plot-input.json", {"cases": report["cases"]})
    subprocess.run(
        [
            str(Path.cwd() / ".venv-plot/bin/python"),
            "tools/plot_vision_coverage.py",
            str(output),
        ],
        check=True,
    )

    def href(path):
        return html.escape(
            os.path.relpath(Path(path).resolve(), output.resolve()), quote=True
        )

    rows = []
    for row in report["cases"]:
        m, c = row["measurement"], row["case"]
        band = (
            f"{m['minimum_output_v']:.4f}–{m['maximum_output_v']:.4f} V"
            if "minimum_output_v" in m
            else "No complete result"
        )
        peak = (
            f"{m['whole_trace_peak_output_v']:.4f} V"
            if "whole_trace_peak_output_v" in m
            else "—"
        )
        result = html.escape(m["status"])
        mode = (
            "Warm · after 0.4 ms"
            if c["steady_state"]
            else "Cold · after 8 ms + whole peak"
        )
        case_path = Path(row["source"]) / row["name"]
        rows.append(
            f"<tr><td>{html.escape(row['name'])}</td><td>{c['input_v']} V<br>{mode}</td>"
            f"<td>{band}</td><td>{peak}</td><td>{result.upper()}</td>"
            f'<td><a href="{href(case_path / "circuit.cir")}">Netlist</a> · '
            f'<a href="{href(Path(row["source"]) / "report.json")}">Producer report</a>'
            + (
                f' · <a href="{href(case_path / "trace.dat")}">Raw waveform</a>'
                if "end_time_s" in m
                else ""
            )
            + "</td></tr>"
        )
    missing = "".join(f"<li>{html.escape(x)}</li>" for x in report["not_run"])
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Camera power · simulation and bench</title><style>"
        "body{font:16px/1.55 system-ui;background:#f5f2ea;color:#243c37;margin:0}main{max-width:1160px;margin:auto;padding:40px 24px}"
        "h1{font-size:clamp(30px,5vw,52px);line-height:1.1}h2{margin-top:40px}a{color:#187161}"
        "table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:12px;text-align:left;border-bottom:1px solid #c8d3cd}"
        ".table{overflow:auto}.note{background:#fff8df;border-left:4px solid #bd8743;padding:16px 22px}img{width:100%;height:auto}"
        ".scope{color:#53655c}.links{display:flex;gap:24px;flex-wrap:wrap}</style><main>"
        "<p>ROBOT LEARNING LAB / ELECTRONICS</p><h1>Camera power, from simulation to the bench.</h1>"
        f'<p class="scope">Regulator evidence: {report["status"].upper()} · Physical tests: NOT RUN</p>'
        "<p>The 5 V candidate serves the proposed Pi and camera. These cases test an assumed 1.6 A running / 2.6 A startup allocation. "
        "A passing voltage case does not establish actual computer startup, PCB temperature or harness performance.</p>"
        '<div class="note"><strong>Two different evidence scopes</strong><p>The ordinary workbench still compares the load against its original 1 A probe and fails that baseline check. '
        "The five explicit cases below are the broader regulator evidence. Warm initialization overshoot remains visible; its settled-window pass is not a cold-start result.</p></div>"
        '<h2>Voltage results</h2><div class="table"><table><thead><tr><th>Case</th><th>Input / evaluated window</th><th>Evaluated voltage</th>'
        "<th>Whole peak</th><th>Result</th><th>Retained files</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        "<h2>Inspect the waveforms</h2><p>Each plotted bin preserves minimum and maximum samples. Gray shading marks time outside the regulation window. "
        "Cold cases additionally check the whole trace for overvoltage. Inductor current uses an ideal linear component model.</p>"
        '<img src="waveforms.png" alt="Five regulator voltage and inductor-current traces, with full startup and evaluated windows visible">'
        "<h2>Build the first comparison bench</h2><p>Use one XH candidate board, a controlled source, a resistor/load-step fixture and simultaneous voltage/current capture. "
        "Then replace the fixture with the real computer and camera. Record actual source ramp and load behavior so the next simulation uses measured inputs.</p>"
        f'<div class="links"><a href="{href("docs/design/logic-rail-bench.md")}">Bench procedure</a>'
        f'<a href="{href("docs/design/logic-rail-bench-template.json")}">Blank measurement record</a>'
        f'<a href="{href("artifacts/logic-board-xh-v1/logic-supply.kicad_pcb")}">Candidate PCB</a><a href="report.json">Coverage audit</a></div>'
        "<h2>Still to evaluate</h2><ul>" + missing + "</ul>"
        '<p class="scope">The archived PH-board waveforms transfer only through byte-identical lumped analog netlists to the XH candidate. '
        "The original runs were inventoried after completion; this is retained-byte provenance, not original signing or physical qualification.</p></main></html>"
    )


def run(high, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "fail", "scope": __doc__, "cases": []}
    try:
        files = [
            Path(__file__).relative_to(Path.cwd()),
            Path("lab/logic_supply.py"),
            Path("lab/vision.py"),
            Path("tools/review_vision_power.py"),
            Path("tools/seal_power_inputs.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("electronics/vision.yaml"),
            Path("tools/plot_vision_coverage.py"),
            Path("requirements-plot-macos-arm64.lock"),
            Path("docs/design/logic-rail-bench.md"),
            Path("docs/design/logic-rail-bench-template.json"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        bindings = {}

        def bind(p):
            bindings[str(p)] = sha(p)
            return p

        review = Path("artifacts/vision-power-validation-v2/report.json")
        prior = json.loads(bind(review).read_text())
        seal = Path(prior["external_snapshot"]["path"])
        bind(seal)
        if sha(seal) != prior["external_snapshot"]["sha256"]:
            raise ValueError("Legacy input snapshot differs")
        manifest = json.loads(seal.read_text())
        board = Path("artifacts/logic-board-xh-v1")
        checked_report(board, "build-report.json", manifest)
        for p in [
            board / "build-report.json",
            board / "parts.json",
            board / "components.json",
            *sorted((board / "build/builds/default/pinout").glob("*.json")),
        ]:
            bind(p)
        if json.loads((board / "build-report.json").read_text())["status"] != "pass":
            raise ValueError("XH candidate board build failed")
        circuit = compiled_circuit(board)
        wanted = {
            "vision-high-step",
            "vision-low-step",
            "vision-half-capacitance",
            "vision-cold-low",
        }
        model_digests = set()
        init_digests = set()
        for folder in [
            Path("artifacts/vision-power-warm-v1"),
            Path("artifacts/vision-power-cold-v1"),
        ]:
            producer = checked_report(folder, "report.json", manifest)
            bind(folder / "report.json")
            config = validate_config(
                yaml.safe_load(bind(folder / "verification.yaml").read_text())
            )
            expected = {c["name"]: c for c in config["cases"]}
            if len(producer["cases"]) != len(expected) or {
                c["name"] for c in producer["cases"]
            } != set(expected):
                raise ValueError("Historical producer case inventory differs")
            for item in producer["cases"]:
                if item["status"] == "not_run":
                    continue
                case = item["case"]
                name = case["name"]
                if case != expected[item["name"]]:
                    raise ValueError("Historical case differs from retained config")
                if name not in wanted:
                    raise ValueError("Unexpected historical case")
                wanted.remove(name)
                net = bind(folder / name / "circuit.cir")
                trace = bind(folder / name / "trace.dat")
                model_digests.add(sha(bind(folder / name / "vendor-ngspice.lib")))
                init_digests.add(sha(bind(folder / name / ".spiceinit")))
                if net.read_text() != spice(circuit, config, case):
                    raise ValueError(
                        "Historical analog netlist differs from XH candidate"
                    )
                values = np.loadtxt(trace, skiprows=1)
                measured = measure(values, config, case)
                for k, v in measured.items():
                    if item.get(k) != v:
                        raise ValueError("Historical measurement differs")
                report["cases"].append(
                    dict(
                        name=name,
                        source=str(folder),
                        case=case,
                        measurement=measured,
                        metrics=metrics(values, case),
                        analog_equivalence="pass",
                    )
                )
        if wanted:
            raise ValueError("Missing historical waveform")
        fresh = verify_report(high)
        bind(high / "report.json")
        config = validate_config(
            yaml.safe_load(bind(high / "verification.yaml").read_text())
        )
        if len(fresh["cases"]) != 1 or fresh["cases"][0]["name"] != "vision-cold-high":
            raise ValueError("Full-battery case inventory differs")
        item = fresh["cases"][0]
        case = item["case"]
        if config["cases"] != [case]:
            raise ValueError("New case differs from retained config")
        if (
            case["input_v"] != 12.6
            or case["load_a"] != load_vision().power.budget_check["startup_total_a"]
            or case["steady_state"]
        ):
            raise ValueError(
                "Full-battery cold case does not match intended allocation"
            )
        net = bind(high / "vision-cold-high/circuit.cir")
        if net.read_text() != spice(circuit, config, case):
            raise ValueError("New analog netlist differs from XH candidate")
        model_digests.add(sha(bind(high / "vision-cold-high/vendor-ngspice.lib")))
        init_digests.add(sha(bind(high / "vision-cold-high/.spiceinit")))
        if len(model_digests) != 1 or len(init_digests) != 1:
            raise ValueError(
                "Cases use different device models or simulator initialization"
            )
        constraints = verify(circuit, config)
        if "end_time_s" in item:
            trace = bind(high / "vision-cold-high/trace.dat")
            values = np.loadtxt(trace, skiprows=1)
            measured = measure(values, config, case)
            for k, v in measured.items():
                if item.get(k) != v:
                    raise ValueError("New measurement differs")
            extra = metrics(values, case)
        else:
            measured = {k: item[k] for k in ["status", "error", "reason"] if k in item}
            extra = {}
        report["cases"].append(
            dict(
                name=case["name"],
                source=str(high),
                case=case,
                measurement=measured,
                metrics=extra,
                analog_equivalence="pass",
            )
        )
        report.update(
            status="partial"
            if constraints["status"] == "pass"
            and all(r["measurement"]["status"] == "pass" for r in report["cases"])
            else "fail",
            baseline_comparison=load_vision().power.budget_check,
            replacement_constraints=constraints,
            shared_vendor_model_sha256=next(iter(model_digests)),
            shared_spiceinit_sha256=next(iter(init_digests)),
            source_bindings=bindings,
            system_power_status="partial",
            physical_status="not_run",
            not_run=[
                "measured computer/camera inrush",
                "constant-power startup",
                "PCB/harness parasitics",
                "effective capacitance and temperature corners",
                "thermal steady state",
                "fault-current survival",
            ],
            limits=[
                "Historical PH-board waveforms transfer only through byte-identical lumped analog netlists to the retained XH candidate; connector/PCB behavior is not simulated.",
                "Warm cases exclude artificial initialization from their settled-voltage result, with whole-trace overshoot retained. Cold cases use resistor loads, not measured computer startup.",
                "The baseline 1 A comparison remains failed; it is not a complete inventory of the five explicit regulator cases.",
            ],
        )
        present(output, report)
    except Exception:
        report["error"] = traceback.format_exc()
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    print({k: report[k] for k in ["status", "error"] if k in report})
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--high", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.high, a.output)
