"""Attach explicit mass-input attestations to immutable composed geometry.

Copies the already verified composition without rerunning or changing CAD.
The power manifest is a newly declared planning input, not inherited evidence.
"""

import argparse
from pathlib import Path
import shutil
from lab.artifact_store import snapshot_file, snapshot_tree, retain_dependency
from tools.compose_forearm_enclosure import verify_report
from tools.check_camera_support import sha, write


def run(source, output):
    report = verify_report(source)
    if (
        report.get("geometry_build_status") != "pass"
        or report.get("assembly_step_roundtrip") != "pass"
    ):
        raise ValueError("Geometry and STEP identity required before attestation")
    output.mkdir(parents=True, exist_ok=False)
    inherited = retain_dependency(source)
    report.setdefault("dependency_manifests", {})["composition:" + inherited["sha256"]] = inherited
    report["source_hashes"] = {}
    for p in source.iterdir():
        if p.name in {"report.json", "report.sha256", "source"}:
            continue
        if p.is_dir():
            snapshot_tree(p, output / p.name)
        else:
            snapshot_file(p, output / p.name)
    for p in [
        Path(__file__).absolute().relative_to(Path.cwd()),
        Path("electronics/power-system.yaml"),
    ]:
        target = output / "source" / p
        if target.exists():
            raise ValueError("Attestation must not overwrite producer source")
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(p, target)
        report["source_hashes"][str(p)] = sha(p)
    report["source_bindings"][str(source / "report.json")] = sha(source / "report.json")
    report["resolved_config_sha256"] = sha(output / "resolved-config.json")
    report["mass_input_attestation"] = {
        "scope": __doc__,
        "source_composition": str(source),
        "geometry_changed": False,
        "power_input": "electronics/power-system.yaml",
        "power_input_status": "unmeasured_planning_allocation",
    }
    report["output_hashes"] = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
    write(output / "report.json", report)
    verify_report(output)
    print(output, report["status"], "unchanged geometry, attested mass inputs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.source, a.output)
