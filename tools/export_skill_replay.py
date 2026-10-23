"""Project one verified skill evaluation probe into an offline Rerun scene.

Run with .venv; only the viewer-export subprocess uses .venv-observe.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.recording import read, sha256
from lab.rerun_scene import export_scene
from lab.skills import verify_result


def export(evaluation, probe, output):
    evaluation, output = Path(evaluation), Path(output)
    verification = verify_result(evaluation)
    report = read(evaluation / "report.json")
    matches = [c for c in report["cases"] if c["name"] == probe and c.get("recording")]
    if len(matches) != 1:
        raise ValueError("probe does not identify one retained recording")
    recording = evaluation / matches[0]["recording"]
    manifest = read(recording / "manifest.json")
    origin = Path(manifest["source_location"])
    model = origin / ("scene.xml" if manifest["adapter"] == "task" else "model.xml")
    if sha256(model) != manifest["source_hashes"]["scene.xml"]:
        raise ValueError("original simulator scene changed since recording")
    output.mkdir(parents=True, exist_ok=False)
    export_scene(model, output / "scene.json")
    command = [
        ".venv-observe/bin/python",
        "-m",
        "lab.rerun_recording",
        str(recording),
        "--scene",
        str(output / "scene.json"),
        "--output",
        str(output / "rerun"),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    (output / "export.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    check = subprocess.run(
        [".venv-observe/bin/rerun", "rrd", "verify", str(output / "rerun/episode.rrd")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (output / "verify.log").write_text(check.stdout + check.stderr)
    check.check_returncode()
    metadata = dict(
        evaluation_manifest_sha256=sha256(evaluation / "manifest.json"),
        candidate_id=report["candidate_id"],
        probe=probe,
        evaluation_integrity=verification,
        robot_outcome=matches[0]["outcome"],
        rrd_verify_exit_code=check.returncode,
        exporter_sha256=sha256(__file__),
    )
    (output / "projection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evaluation", type=Path)
    p.add_argument("--probe", required=True)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(export(args.evaluation, args.probe, args.output), indent=2))


if __name__ == "__main__":
    main()
