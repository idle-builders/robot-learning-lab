"""Retain raw inputs and analyze one predeclared actuator static hold."""

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

import numpy
import pydantic

from lab.joint_bench import BenchManifest, analyze, read_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(manifest_path, output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    report = {
        "status": "fail",
        "physical_status": "not_run",
        "scope": "Input validation failed; no hold evaluated",
    }
    try:
        shutil.copy2(manifest_path, source / "manifest.json")
        for name in ("lab/joint_bench.py", "tools/analyze_joint_bench.py"):
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, target)
        manifest = BenchManifest.model_validate_json(
            (source / "manifest.json").read_text()
        )
        from lab.artifact_paths import checked_path
        try:
            trace = checked_path(manifest_path.parent, manifest.trace)
        except ValueError as error:
            raise ValueError("Trace must remain inside its manifest directory") from error
        shutil.copy2(trace, source / "trace.csv")
        replay = manifest.model_dump(mode="json")
        replay["trace"] = "trace.csv"
        (source / "replay-manifest.json").write_text(
            json.dumps(replay, indent=2) + "\n"
        )
        report = analyze(manifest, read_trace(source / "trace.csv"))
    except (ValueError, TypeError, OSError, csv.Error) as error:
        report["error"] = str(error)
    report["source_hashes"] = {
        str(p.relative_to(source)): sha(p) for p in source.rglob("*") if p.is_file()
    }
    report["runtime"] = {
        "python": sys.version,
        "numpy": numpy.__version__,
        "pydantic": pydantic.__version__,
    }
    content = html.escape(json.dumps(report, indent=2, allow_nan=False))
    checks = "".join(
        f"<tr><td>{html.escape(key.replace('_', ' '))}</td><td>{'PASS' if value else 'FAIL'}</td></tr>"
        for key, value in report.get("checks", {}).items()
    )
    summary = f"<p>{html.escape(report.get('error', report['scope']))}</p>"
    if "gravity_torque_nm" in report:
        torque = report["gravity_torque_nm"]
        summary += (
            f"<p><strong>Gravity load: {torque['interval_min']:.3f}–{torque['interval_max']:.3f} Nm</strong> including declared load/angle bounds.</p>"
            f"<p>Compared with the configured limit: {html.escape(report['model_limit_relation'].replace('_', ' '))}.</p>"
            f"<p>Declared hold: {report['window_s'][1] - report['window_s'][0]:g} seconds · {report['samples']} evaluated samples.</p>"
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Actuator hold analysis</title><style>body{background:#f5f2ea;color:#243c37;font:16px/1.5 system-ui;max-width:1000px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:20px}a{color:#187161}</style>"
        "<h1>Actuator hold analysis</h1>"
        f"<p><strong>{report.get('source_kind', 'invalid input').upper()} · {report['status'].upper()}</strong> · Physical qualification: {report['physical_status'].upper()}</p>"
        "<p>A finite-duration hold is not a continuous motor rating. Synthetic traces verify the analyzer only. No robot parameters are changed.</p>"
        '<p><a href="report.json">Analysis report</a> · <a href="source/manifest.json">Declared setup</a></p>'
        + summary
        + f"<table><tbody>{checks}</tbody></table><details><summary>Full measurements, limits and provenance</summary><pre>{content}</pre></details></html>"
    )
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.manifest, args.output)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    raise SystemExit(0 if result["status"] == "pass" else 1)
