"""Audit retained composition identities and pose coverage without rerunning CAD."""

import argparse
import json
from pathlib import Path

from tools.cad_identity import geometry_agrees
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import task_samples, verify_report


def run(assemblies, output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    bindings = {}
    for folder in assemblies:
        report = verify_report(folder)
        bindings[str(folder / "report.json")] = sha(folder / "report.json")
        source = folder / "source"
        base = json.loads((source / "assembly/report.json").read_text())
        preview = json.loads((source / "preview/report.json").read_text())
        cover = json.loads((source / "cover/report.json").read_text())
        frames = json.loads((source / "preview/part-frames.json").read_text())
        assert base["status"] == "pass" and preview["status"] == "rendered"
        assert len(frames["step_roundtrip"]) == 22
        assert all(
            geometry_agrees(c["actual"], c["expected"])
            for c in frames["step_roundtrip"]
        )
        assert report["part_count"] == len(base["parts"]) + 20
        assert (
            len(report["parts"])
            == len({p["name"] for p in report["parts"]})
            == report["part_count"]
        )
        assert report["assembly_step_roundtrip"] == "pass"
        identity = json.loads((folder / "step-roundtrip.json").read_text())
        assert len(identity) == report["part_count"] and all(
            geometry_agrees(c["actual"], c["expected"]) for c in identity
        )
        samples = report["samples"]
        design = [s for s in samples if s["kind"] == "design"]
        tasks = [s for s in samples if s["kind"] == "task"]
        assert [s["name"] for s in design] == cover["poses"]
        rows = [
            json.loads(line)
            for line in (source / "episode/trajectory.jsonl").read_text().splitlines()
        ]
        contract = json.loads((source / "assembly/interfaces.json").read_text())
        expected = task_samples(rows, contract)
        assert [
            {k: s[k] for k in e} for s, e in zip(tasks, expected, strict=True)
        ] == expected
        assert report["cover_producer_status"] == cover["status"]
        assert report["status"] == (
            "pass"
            if cover["status"] == "pass" and all(s["status"] == "pass" for s in samples)
            else "fail"
        )
        assert all(
            s["status"] == ("fail" if s["failures"] else "pass") for s in samples
        )
        for s in samples:
            for f in s["failures"]:
                assert f["hits"] or f["distance_mm"] < 0.5 - 1e-7
                assert all(h["overlap_mm3"] > 0.02 for h in f["hits"])
        records.append(
            dict(
                assembly=str(folder),
                parts=report["part_count"],
                candidate_status=report["status"],
                design_pass=sum(s["status"] == "pass" for s in design),
                design_total=len(design),
                task_pass=sum(s["status"] == "pass" for s in tasks),
                task_total=len(tasks),
                failed_poses=[s["name"] for s in samples if s["status"] == "fail"],
                minimum_passing_gap_mm=min(
                    s["minimum_distance_mm"] for s in samples if s["status"] == "pass"
                ),
                task_minimum_gap_mm=min(s["minimum_distance_mm"] for s in tasks),
                task_joint_limit_excursion_samples=sum(
                    bool(s["measured_joint_limit_excursions"]) for s in tasks
                ),
                replay_source_matches_current=sha(
                    Path("tools/compose_forearm_enclosure.py")
                )
                == report["source_hashes"]["tools/compose_forearm_enclosure.py"],
            )
        )
    hashes = copy_source(
        output,
        {
            str(p): p
            for p in [
                Path("tools/audit_forearm_integration.py"),
                Path("tools/compose_forearm_enclosure.py"),
                Path("tools/cad_identity.py"),
                Path("tools/check_camera_support.py"),
            ]
        },
    )
    result = dict(
        status="pass",
        scope="Retained identities, status semantics and discrete pose coverage; no new CAD or dynamics evaluation",
        records=records,
        source_bindings=bindings,
        source_hashes=hashes,
    )
    write(output / "report.json", result)
    print(json.dumps(result["records"], indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assemblies", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(**vars(p.parse_args()))
