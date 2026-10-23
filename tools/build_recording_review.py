"""Retain an eight-trial Pip/Scout recording and Rerun verification review."""

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.recording import export_recording, sha256
from lab.rerun_recording import export_rerun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    cli = Path(sys.executable).parent / "rerun"
    results = []
    for robot in ("pip", "scout"):
        cases = [
            ("task", "task", Path(f"artifacts/{robot}-neck-collar-task-v1")),
            ("open", "task", Path(f"artifacts/{robot}-neck-collar-open-v1")),
        ]
        cases += [
            (name, "balance", Path(f"artifacts/{robot}-neck-collar-balance-v1/{name}"))
            for name in ("drive-return", "disabled-controller")
        ]
        for case, kind, source in cases:
            name = f"{robot}-{case}"
            record = output / name / "recording"
            view = output / name / "rerun"
            result = export_recording(source, record, kind)
            rerun = export_rerun(record, view)
            check = subprocess.run(
                [str(cli), "rrd", "verify", str(view / "episode.rrd")],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (view / "verify.log").write_text(check.stdout + check.stderr)
            results.append(
                {
                    "case": name,
                    "recording": result,
                    "rerun": rerun,
                    "rrd_verify_exit_code": check.returncode,
                }
            )
            print(
                name,
                result["source_outcome"]["status"],
                f"rrd_verify={check.returncode}",
                flush=True,
            )
    sources = [
        Path("lab/recording.py"),
        Path("lab/rerun_recording.py"),
        Path(__file__),
        Path("requirements-observe.txt"),
        Path("requirements-observe-macos-arm64.lock"),
    ]
    (output / "source").mkdir()
    for source in sources:
        (output / "source" / source.name).write_bytes(source.read_bytes())
    passed = all(r["rrd_verify_exit_code"] == 0 for r in results)
    report = {
        "status": "pass" if passed else "fail",
        "cases": results,
        "source_hashes": {p.name: sha256(p) for p in sources},
        "scope": "Import/schema/integrity and RRD decoding, not new simulation or visual UI inspection",
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = "".join(
        f"<tr><td>{html.escape(r['case'])}</td><td>{r['recording']['source_outcome']['status']}</td>"
        f"<td>{r['recording']['rows']}</td><td>{r['rrd_verify_exit_code']}</td>"
        f'<td><a href="{r["case"]}/rerun/episode.rrd">Rerun file</a> · '
        f'<a href="{r["case"]}/recording/manifest.json">Manifest</a> · '
        f'<a href="{r["case"]}/rerun/report.json">Export report</a></td></tr>'
        for r in results
    )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Robot telemetry review</title>'
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;background:#f5f3ed;color:#213334}"
        "td,th{padding:12px;text-align:left;border-bottom:1px solid #ccc}code{display:block;padding:14px;background:#fff}</style>"
        "<h1>Pip & Scout · recorded telemetry</h1><p>Four successful trials and four negative controls. "
        "Original robot outcomes remain separate from recording validation. Rerun file verification exit 0 means it decodes.</p>"
        "<p>Replay includes joint state, mixed-unit targets, IMU/estimator diagnostics and outcomes. "
        "Full base pose and camera video are unavailable. No new simulation, learned policy or hardware transfer is claimed. "
        "Viewer layout has not been visually inspected.</p><table><tr><th>Trial</th><th>Robot result</th>"
        "<th>Rows</th><th>RRD verification</th><th>Evidence</th></tr>"
        + rows
        + "</table>"
        "<h2>Open locally from the repo</h2><code>.venv-observe/bin/rerun "
        + html.escape(str(output / "pip-task/rerun/episode.rrd"))
        + "</code>"
        '<p><a href="validation.json">Full validation report</a></p>'
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
