"""Join retained SPICE probes and a replacement PCB, without promoting rail qualification."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml

from lab.logic_supply import compiled_circuit, measure, spice, validate_config, verify
from tools.seal_power_inputs import inventory


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_report(folder, name, manifest):
    key = folder.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if inventory(folder) != manifest["folders"].get(key):
        raise ValueError(f"input inventory changed after external snapshot: {key}")
    report = json.loads((folder / name).read_text())
    for relative, expected in report["source_hashes"].items():
        if sha(folder / "source" / relative) != expected:
            raise ValueError(f"retained source changed: {relative}")
    return report


def metrics(trace, case):
    t, v, vin, current, source = trace.T
    start = case.get("evaluation_start_s", 0.0004 if case["steady_state"] else 0.008)
    s = t >= start
    ts = t[s]

    def mean(values):
        return float(np.trapezoid(values, ts) / (ts[-1] - ts[0]))

    result = dict(
        evaluated_peak_inductor_a=float(current[s].max()),
        evaluated_rms_inductor_a=mean(current[s] ** 2) ** 0.5,
        evaluated_minimum_input_v=float(vin[s].min()),
        evaluated_rms_input_source_a=mean(source[s] ** 2) ** 0.5,
        evaluated_peak_input_source_a=float((-source[s]).max()),
    )
    load = v * case["load_a"] / 5
    if "step_load_a" in case:
        rise, fall = case["step_start_s"], case["step_end_s"]
        load += np.interp(
            t,
            [0, rise, rise + 1e-5, fall, fall + 1e-5],
            [
                0,
                0,
                case["step_load_a"] - case["load_a"],
                case["step_load_a"] - case["load_a"],
                0,
            ],
        )
        plateau = (t >= rise + 1e-5) & (t <= fall)
        result["pulse_plateau_load_range_a"] = [
            float(load[plateau].min()),
            float(load[plateau].max()),
        ]
    result["evaluated_load_range_a"] = [float(load[s].min()), float(load[s].max())]
    return result


def run(warm, cold, board, output, seal):
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = seal / "manifest.json"
    if sha(manifest_path) != (seal / "manifest.sha256").read_text().strip():
        raise ValueError("external manifest digest mismatch")
    manifest = json.loads(manifest_path.read_text())
    board_report = checked_report(board, "build-report.json", manifest)
    if board_report["status"] != "pass":
        raise ValueError("replacement board did not pass")
    circuit = compiled_circuit(board)
    config = validate_config(yaml.safe_load((warm / "verification.yaml").read_text()))
    if config != yaml.safe_load((cold / "verification.yaml").read_text()):
        raise ValueError("probe runs use different configurations")
    rows, bindings = [], {}
    for folder in (warm, cold):
        report = checked_report(folder, "report.json", manifest)
        expected_cases = {case["name"]: case for case in config["cases"]}
        if len(report["cases"]) != len(expected_cases) or {
            item["name"] for item in report["cases"]
        } != set(expected_cases):
            raise ValueError("run case inventory differs from configuration")
        for item in report["cases"]:
            if item["status"] not in ("pass", "fail", "not_run"):
                raise ValueError("invalid probe status")
            if (
                item["status"] != "not_run"
                and item.get("case") != expected_cases[item["name"]]
            ):
                raise ValueError("run case definition differs from configuration")
        bindings[str(folder / "report.json")] = sha(folder / "report.json")
        for item in report["cases"]:
            if item["status"] == "not_run":
                continue
            case = item["case"]
            case_folder = folder / case["name"]
            row = dict(name=case["name"], original=item, source=str(case_folder))
            netlist = case_folder / "circuit.cir"
            equivalent = netlist.read_text() == spice(circuit, config, case)
            if not equivalent:
                raise ValueError(
                    "new PCB compiled circuit changes the simulated netlist"
                )
            row["replacement_spice_byte_equivalent"] = True
            bindings[str(netlist)] = sha(netlist)
            trace_path = case_folder / "trace.dat"
            if trace_path.is_file():
                bindings[str(trace_path)] = sha(trace_path)
            if trace_path.is_file() and "end_time_s" in item:
                trace = np.loadtxt(trace_path, skiprows=1)
                measured = measure(trace, config, case)
                for key, value in measured.items():
                    if item.get(key) != value:
                        raise ValueError(f"measurement differs: {case['name']} {key}")
                row["additional_metrics"] = metrics(trace, case)
                bindings[str(trace_path)] = sha(trace_path)
                # Preserve every extremum in plot bins; raw full traces stay linked.
                envelope = []
                for indices in np.array_split(
                    np.arange(len(trace)), min(1400, len(trace))
                ):
                    block = trace[indices]
                    envelope.append(
                        [
                            float(block[:, 0].mean() * 1000),
                            float(block[:, 1].min()),
                            float(block[:, 1].max()),
                            float(block[:, 3].min()),
                            float(block[:, 3].max()),
                        ]
                    )
                row["plot_envelope"] = envelope
            rows.append(row)
    if len(rows) != len(config["cases"]) or {row["name"] for row in rows} != {
        case["name"] for case in config["cases"]
    }:
        raise ValueError("missing or duplicate selected probe coverage")
    result = dict(
        status="fail"
        if any(row["original"]["status"] == "fail" for row in rows)
        or verify(circuit, config)["status"] == "fail"
        else "partial",
        scope="regulator waveform review and PCB connector replacement; not system power qualification",
        cases=rows,
        replacement_constraints=verify(circuit, config),
        pcb_status=board_report["status"],
        source_bindings=bindings,
        external_snapshot=dict(
            path=str(manifest_path), sha256=sha(manifest_path), scope=manifest["scope"]
        ),
        unchanged_probe_coverage="electronics/vision.yaml retains its earlier 1 A gate; these probes do not qualify startup inrush, thermal or the harness",
        not_run=[
            "cold high-input probe",
            "measured Pi/camera startup profile",
            "constant-power load",
            "PCB/harness parasitics",
            "temperature and effective capacitance corners",
            "fault-current survival",
        ],
        comparison_scope="identical lumped regulator netlist; connectors and PCB parasitics are not simulated",
    )
    for filename in (
        "build-report.json",
        "board-interface.json",
        "logic-supply.kicad_pcb",
        "logic-supply.step",
    ):
        result["source_bindings"][str(board / filename)] = sha(board / filename)
    for name in (
        "tools/review_vision_power.py",
        "tools/plot_vision_power.py",
        "tools/seal_power_inputs.py",
        "lab/logic_supply.py",
    ):
        dest = output / "source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, dest)
        result["source_bindings"][name] = sha(name)
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    subprocess.run(
        [
            str(Path(".venv-plot/bin/python").absolute()),
            "tools/plot_vision_power.py",
            str(output),
        ],
        check=True,
    )
    body = []
    for row in rows:
        item = row["original"]
        summary = (
            f"{item['minimum_output_v']:.3f}–{item['maximum_output_v']:.3f} V"
            if "minimum_output_v" in item
            else html.escape(item.get("error", "no completed waveform"))
        )
        body.append(
            f"<tr><td>{row['name']}</td><td>{item['status']}</td><td>{summary}</td><td>{'Warm / settled only' if item['case']['steady_state'] else 'Cold / resistor load'}</td></tr>"
        )
    board_relative = html.escape(os.path.relpath(board, output), quote=True)
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Camera-inclusive power study</title>'
        "<style>body{font:17px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;background:#eef1eb;color:#19332e}table{width:100%;border-collapse:collapse}td,th{padding:12px;text-align:left;border-bottom:1px solid #bbc}img{width:100%}a{color:#176751}</style>"
        "<h1>Camera-inclusive 5 V power candidate</h1><p>1.6 A nominal / 2.6 A peak planning allocation. Scope remains partial: no full rail, startup-inrush or thermal qualification.</p>"
        "<table><tr><th>Probe</th><th>Voltage result</th><th>Evaluated output</th><th>Scope</th></tr>"
        + "".join(body)
        + "</table>"
        "<p>Warm initialization overshoot remains visible and is excluded only from the stated settled-window result. The cold test uses a resistor; its current rises with voltage, unlike an unknown real Pi startup profile.</p>"
        '<img src="waveforms.png" alt="Output-voltage and inductor-current envelopes across the retained probes">'
        "<p>J2 changes from 2 A JST PH to a 3 A JST XH candidate requiring AWG22, XHP-2 housing and two SXH-001T-P0.6 contacts. Mating harness, crimp quality, temperature rise and board copper remain unqualified.</p>"
        f'<p><a href="{board_relative}/logic-supply.png">Board render</a> · <a href="{board_relative}/logic-supply.step">PCB STEP</a> · <a href="report.json">Measurements, provenance and limits</a></p>'
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "replacement_constraints": result["replacement_constraints"]["status"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("warm", "cold", "board", "output", "seal"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.warm, args.cold, args.board, args.output, args.seal)
