"""Exercise hold analysis with labeled synthetic faults, never physical readings."""

import argparse
import csv
import json
from pathlib import Path

from lab.joint_bench import COLUMNS
from tools.analyze_joint_bench import run, sha


def example_manifest():
    def bounded(value, error):
        return {"value": value, "absolute_error": error}

    return {
        "schema_version": 1,
        "source_kind": "synthetic",
        "actuator_identity": "NO HARDWARE — analytical test fixture",
        "firmware_and_settings": "NO FIRMWARE — constant synthetic target",
        "fixture_identity": "NO PHYSICAL FIXTURE — 100 mm lever example",
        "acquisition_clock": "synthetic synchronous seconds",
        "instrumentation_and_calibration": "NO INSTRUMENTS — generated values and assumed errors",
        "trace": "trace.csv",
        "fixture": {
            "payload_mass_kg": bounded(0.4, 0.001),
            "payload_radius_m": bounded(0.1, 0.0005),
            "lever_mass_kg": bounded(0.08, 0.001),
            "lever_com_radius_m": bounded(0.05, 0.001),
            "gravity_m_s2": bounded(9.80665, 0.001),
            "angle_absolute_error_rad": 0.002,
            "geometry": "horizontal_shaft_direct_cantilever",
            "positive_angle": "upward_from_horizontal",
            "notes": "No bearing or support bypass; collinear lever and load COM; illustrative assumptions only",
        },
        "limits": {
            "start_s": 0,
            "end_s": 60,
            "maximum_sample_gap_s": 1.1,
            "maximum_position_error_rad": 0.035,
            "maximum_speed_rad_s": 0.02,
            "minimum_voltage_v": 11.4,
            "maximum_voltage_v": 12.6,
            "maximum_current_a": 1.5,
            "maximum_case_temperature_c": 50,
            "maximum_temperature_rise_c": 20,
            "maximum_fixture_torque_nm": 0.65,
            "model_limit_nm": 0.4,
            "rationale": "Synthetic parser/checker demonstration only; these are not approved physical rig limits",
        },
        "observations": [],
    }


def example_trace():
    return [[float(t), 0, 0.005, 12.0, 0.6, 24 + 0.1 * t, 23] for t in range(61)]


def write_input(folder, manifest, rows):
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (folder / "trace.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows(rows)


def demo(output):
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for name in (
        "steady",
        "drift",
        "hot",
        "brownout",
        "missing_samples",
        "operator_slip",
    ):
        manifest, rows = example_manifest(), example_trace()
        if name == "drift":
            for row in rows:
                row[2] += row[0] * 0.002
        elif name == "hot":
            rows[-1][5] = 55
        elif name == "brownout":
            rows[20][3] = 8.5
        elif name == "missing_samples":
            del rows[20:30]
        elif name == "operator_slip":
            manifest["observations"] = ["SYNTHETIC: fixture moved at 30 s"]
        folder = output / "inputs" / name
        write_input(folder, manifest, rows)
        report = run(folder / "manifest.json", output / name)
        expected = "pass" if name == "steady" else "fail"
        results.append(
            {
                "name": name,
                "expected": expected,
                "actual": report["status"],
                "detected": report["status"] == expected,
            }
        )
    passed = all(r["detected"] for r in results)
    report = {
        "status": "pass" if passed else "fail",
        "scope": "Synthetic analyzer exercise; no actuator test or model identification",
        "physical_status": "not_run",
        "cases": results,
    }
    links = "".join(
        f'<li><a href="{r["name"]}/index.html">{r["name"]}</a>: {r["actual"].upper()} (expected {r["expected"]})</li>'
        for r in results
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Actuator bench analysis examples</title><style>body{font:17px/1.6 system-ui;max-width:900px;margin:40px auto;padding:20px;background:#f5f2ea;color:#243c37}a{color:#187161}</style><h1>From joint measurements to model evidence</h1><p><strong>SYNTHETIC DATA ONLY · Physical tests NOT RUN</strong></p><p>The analyzer retains raw readings, computes the gravity load and its declared error bounds, and checks a predeclared hold window. These examples prove fault detection, not actuator capability.</p><ul>'
        + links
        + '</ul><p><a href="../../docs/design/actuator-hold-bench.md">Physical setup and data contract</a> · <a href="../../docs/research/actuator-bench-interface.md">Adapter and wiring research</a></p><p>No changes to robot torque limits or manufacturing readiness follow from these examples.</p></html>'
    )
    sources = [
        Path(p)
        for p in (
            "lab/joint_bench.py",
            "tools/analyze_joint_bench.py",
            "tools/demo_joint_bench.py",
            "docs/design/actuator-hold-bench.md",
            "docs/design/actuator-hold-template.json",
            "docs/research/actuator-bench-interface.md",
        )
    ]
    for path in sources:
        target = output / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    report["source_hashes"] = {str(path): sha(path) for path in sources}
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(demo(args.output)))
