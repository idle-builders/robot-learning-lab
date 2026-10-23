"""Inspectable competing yoke/spacer candidates and retained assembly samples."""

import argparse
import json
from pathlib import Path
import trimesh
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report


def run(study, output):
    r = verify_report(study)
    if r["status"] == "running" or "routes" not in r:
        raise ValueError("Study not complete")
    output.mkdir(parents=True, exist_ok=False)
    bindings = {str(study / "report.json"): sha(study / "report.json")}
    meshes = {}
    for file in sorted(study.glob("*.stl")):
        bindings[str(file)] = sha(file)
        m = trimesh.load(file, force="mesh")
        if m.is_empty:
            raise ValueError("Empty viewer asset")
        meshes[file.stem] = dict(
            vertices=m.vertices.round(5).flatten().tolist(),
            faces=m.faces.flatten().tolist(),
        )
    data = dict(report=r, meshes=meshes, study="../" + study.name)
    template = Path("ui/yoke-spacers.html")
    bindings[str(template)] = sha(template)
    bindings[str(Path(__file__).relative_to(Path.cwd()))] = sha(__file__)
    (output / "index.html").write_text(
        template.read_text().replace(
            "__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")
        )
    )
    write(
        output / "report.json",
        dict(
            status="partial",
            scope=__doc__,
            source_bindings=bindings,
            output_hashes={"index.html": sha(output / "index.html")},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.study, a.output)
