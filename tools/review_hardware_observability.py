"""Inspectable raw-to-canonical-to-Rerun evidence, with honest capture origins."""

import argparse
import json
from pathlib import Path
import shutil
from lab.recording import read, sha256, verify_recording


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    cases, bindings = [], {}
    for name in ("pip", "scout", "pip-partial"):
        recording = Path(f"artifacts/{name}-hardware-recording-v2")
        view = Path(f"artifacts/{name}-hardware-rerun-v3")
        verification = verify_recording(recording)
        manifest, report = read(recording / "manifest.json"), read(view / "report.json")
        if (
            report["source_recording_id"] != manifest["recording_id"]
            or report["source_manifest_sha256"] != sha256(recording / "manifest.json")
            or report["rrd_sha256"] != sha256(view / "episode.rrd")
        ):
            raise ValueError("Rerun/source identity mismatch")
        if (
            manifest["source_kind"] != "synthetic_transport"
            or report["source_kind"] != manifest["source_kind"]
        ):
            raise ValueError("unexpected demonstration origin")
        if report["exporter_sha256"] != sha256(view / "rerun_recording.py"):
            raise ValueError("Rerun exporter archive changed")
        for path in [
            recording / "manifest.json",
            recording / "events.jsonl",
            view / "report.json",
            view / "episode.rrd",
            view / "rerun_recording.py",
        ]:
            bindings[str(path)] = sha256(path)
        cases.append(
            dict(
                name=name,
                manifest=manifest,
                verification=verification,
                rerun=report,
                events=[
                    json.loads(x)
                    for x in (recording / "events.jsonl").read_text().splitlines()
                ],
                recording="../" + recording.name,
                view="../" + view.name,
            )
        )
    screenshot = Path("artifacts/hardware-recording-validation-v1/rerun-pip-final.png")
    shutil.copyfile(screenshot, output / "rerun-pip.png")
    bindings[str(screenshot)] = sha256(screenshot)
    template = Path("ui/hardware-observability.html")
    (output / "index.html").write_text(
        template.read_text().replace(
            "__DATA__", json.dumps(cases).replace("<", "\\u003c")
        )
    )
    source_hashes = {}
    for path in [
        Path(__file__).relative_to(Path.cwd()),
        template,
        Path("docs/design/hardware-observability.md"),
    ]:
        destination = output / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(path)] = sha256(path)
    report = dict(
        status="pass",
        scope="Presentation/source integrity; synthetic offline recordings only",
        physical_status="not_run",
        cases=[
            dict(
                name=c["name"],
                source_status=c["manifest"]["outcome"]["status"],
                events=c["manifest"]["events"],
                source_kind=c["manifest"]["source_kind"],
            )
            for c in cases
        ],
        source_hashes=source_hashes,
        source_bindings=bindings,
        output_hashes={n: sha256(output / n) for n in ["index.html", "rerun-pip.png"]},
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("source_hashes", "source_bindings", "output_hashes")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
