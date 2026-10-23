"""Build Pip/Scout bundles and verify each after moving outside the repository."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.recording import encoded
from lab.skills import generate_candidates
from lab.worker_bundle import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    definition = Path("skills/definitions/wheel-drive-hold.yaml")
    proposals = output / "candidates"
    ids = generate_candidates(
        definition,
        "pitch_gain_m_s",
        [4.0],
        "Balance, roll forward and return.",
        proposals,
    )
    entries = []
    for name in ("pip", "scout"):
        folder = output / name
        entry = build(
            definition,
            proposals / f"{ids[0]}.json",
            f"variants/{name}-neck-collar.yaml",
            f"artifacts/{name}-neck-collar-mass-v1/nominal",
            folder,
        )
        with tempfile.TemporaryDirectory(
            prefix="robot-worker-relocation-"
        ) as temporary:
            moved = Path(temporary) / "bundle"
            shutil.copytree(folder, moved)
            result = Path(temporary) / "result.json"
            process = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(moved / "asset_probe.py"),
                    str(moved),
                    "--expected-sha256",
                    entry["bundle_sha256"],
                    "--output",
                    str(result),
                ],
                cwd=temporary,
                capture_output=True,
                text=True,
                timeout=120,
            )
            (output / f"{name}-relocation.log").write_text(
                process.stdout + process.stderr
            )
            if result.exists():
                shutil.copyfile(result, output / f"{name}-relocation.json")
                entry["relocation"] = json.loads(result.read_text())
            else:
                entry["relocation"] = {"status": "fail", "reason": "no result"}
            entry["relocation_exit_code"] = process.returncode
        entries.append(entry)
    report = dict(
        schema_version=1,
        status="pass"
        if all(
            e["relocation"]["status"] == "pass" and e["relocation_exit_code"] == 0
            for e in entries
        )
        else "fail",
        scope="local asset portability; external NVIDIA execution not_run",
        entries=entries,
    )
    (output / "report.json").write_text(encoded(report) + "\n")
    print(encoded(report))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
