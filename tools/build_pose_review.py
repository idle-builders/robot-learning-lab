"""Run fresh, source-bound Pip/Scout trials and export their full-pose replay.

Run with .venv/bin/python; uses .venv-observe only for Rerun export/decoding.
Outputs are immutable. This is an offline simulation check, not a GPU worker.
"""

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.balance import run_suite
from lab.manipulation import run
from lab.recording import export_recording, sha256
from lab.rerun_scene import export_scene
from lab.tasks import PickPlaceTask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for robot in ("pip", "scout"):
        variant = Path(f"variants/{robot}-neck-collar.yaml")
        model = Path(f"artifacts/{robot}-neck-collar-mass-v1/nominal")
        mass = model / "mass-properties.json"
        cases = []
        for name, controller in (("task", "feedback"), ("open", "open")):
            old = Path(f"artifacts/{robot}-neck-collar-{name}-v1")
            task = PickPlaceTask.model_validate_json((old / "task.json").read_text())
            trial = output / "trials" / f"{robot}-{name}"
            run(
                variant,
                trial,
                task,
                model_run=model,
                controller=controller,
                render=False,
                release_clearance_m=0.01,
                mass_model=mass,
            )
            cases.append((name, "task", trial, old / "report.json"))
        suite = output / "trials" / f"{robot}-balance"
        run_suite(
            variant,
            model,
            Path("experiments/balance-recording.yaml"),
            suite,
            mass_model=mass,
        )
        for name in ("drive-return", "delayed-command", "disabled-controller"):
            cases.append(
                (
                    name,
                    "balance",
                    suite / name,
                    Path(
                        f"artifacts/{robot}-neck-collar-balance-v1/{name}/result.json"
                    ),
                )
            )
        for name, kind, trial, old in cases:
            case = f"{robot}-{name}"
            target = output / case
            target.mkdir()
            scene = target / "scene.json"
            export_scene(
                trial / ("scene.xml" if kind == "task" else "model.xml"), scene
            )
            validation = export_recording(trial, target / "recording", kind)
            export = subprocess.run(
                [
                    ".venv-observe/bin/python",
                    "-m",
                    "lab.rerun_recording",
                    str(target / "recording"),
                    "--output",
                    str(target / "rerun"),
                    "--scene",
                    str(scene),
                ],
                capture_output=True,
                text=True,
                timeout=180,
            )
            (target / "export.log").write_text(export.stdout + export.stderr)
            if export.returncode:
                raise RuntimeError(f"Rerun export failed; see {target / 'export.log'}")
            check = subprocess.run(
                [
                    ".venv-observe/bin/rerun",
                    "rrd",
                    "verify",
                    str(target / "rerun/episode.rrd"),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (target / "verify.log").write_text(check.stdout + check.stderr)
            baseline = json.loads(old.read_text())
            result = dict(
                case=case,
                recording=validation,
                baseline_report=str(old),
                baseline_sha256=sha256(old),
                baseline_status=baseline["status"],
                same_outcome=baseline["status"]
                == validation["source_outcome"]["status"],
                rrd_verify_exit_code=check.returncode,
                rerun=json.loads((target / "rerun/report.json").read_text()),
            )
            results.append(result)
            print(
                case,
                validation["source_outcome"]["status"],
                f"decode={check.returncode}",
                flush=True,
            )
    passed = all(r["same_outcome"] and r["rrd_verify_exit_code"] == 0 for r in results)
    report = dict(
        status="pass" if passed else "fail",
        cases=results,
        scope="fresh simulation poses/timing, canonical validation and RRD decode; no GPU or physical run",
        visual_inspection="not_run",
    )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "build_pose_review.py").write_bytes(Path(__file__).read_bytes())
    rows = "".join(
        f"<tr><td>{r['case']}</td><td>{r['recording']['source_outcome']['status']}</td>"
        f"<td>{r['rerun']['scene_frames']}</td><td>{r['same_outcome']}</td>"
        f'<td><a href="{r["case"]}/rerun/episode.rrd">Replay</a> · '
        f'<a href="{r["case"]}/recording/manifest.json">Manifest</a></td></tr>'
        for r in results
    )
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Pip & Scout full-pose replay</title>'
        "<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f5f3ed;color:#213334}"
        "td,th{padding:12px;text-align:left;border-bottom:1px solid #ccc}</style>"
        "<h1>Pip & Scout · full-pose replay</h1><p>Fresh simulation with full body poses, source-bound geometry, "
        "command issue/receipt/first control tick, and IMU diagnostics. Failed controls remain failed. "
        "No learned policy, camera video or physical motion. Decoder verification is not GUI inspection.</p>"
        "<table><tr><th>Trial</th><th>Robot outcome</th><th>Scene frames</th><th>Matches baseline</th><th>Evidence</th></tr>"
        + rows
        + "</table><p>From the repository:</p><code>.venv-observe/bin/rerun "
        + html.escape(str(output / "pip-task/rerun/episode.rrd"))
        + '</code><p><a href="validation.json">Validation report</a></p>'
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
