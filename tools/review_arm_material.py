"""Build the inspectable clamp/load/cost comparison without promoting the parts."""

import argparse
import json
from pathlib import Path
import trimesh
import yaml

from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.print_cost import PrintAssumptions, calculate


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bind(p):
        bindings[str(p)] = sha(p)
        return p

    def read(p):
        return json.loads(bind(p).read_text())

    study = Path("artifacts/clamp-lightening-v1")
    loads = Path("artifacts/arm-material-load-v2")
    geometry, load_report = verify_report(study), verify_report(loads)
    bind(study / "report.json")
    bind(loads / "report.json")
    meshes = {}
    for p in study.glob("*.stl"):
        m = trimesh.load(bind(p), force="mesh")
        if m.is_empty:
            raise ValueError("Empty display mesh")
        meshes[p.stem] = dict(
            vertices=m.vertices.flatten().tolist(), faces=m.faces.flatten().tolist()
        )
    printing = PrintAssumptions.model_validate(
        yaml.safe_load(bind(Path("robot/printing-cost.yaml")).read_text())
    )
    integrated = yaml.safe_load(bind(Path("robot/integrated-cost.yaml")).read_text())
    inventory_path = Path("artifacts/composed-print-inventory-v3")
    inventory = verify_report(inventory_path)
    bind(inventory_path / "report.json")
    costs = []
    for row in load_report["rows"]:
        name = row["robot"]
        old_volume = next(g for g in inventory["groups"] if g["robot"] == name)[
            "total_volume_mm3"
        ]
        folder = Path(f"artifacts/{name}-composed-cost-v1")
        cost = verify_report(folder)
        bind(folder / "report.json")
        for scenario in cost["scenarios"]:
            old = calculate(
                old_volume,
                printing,
                scenario,
                integrated["contingency_fraction"],
                cost["target_usd"],
            )
            new = calculate(
                old_volume - row["bilateral_removed_volume_mm3"],
                printing,
                scenario,
                integrated["contingency_fraction"],
                cost["target_usd"],
            )
            costs.append(
                dict(
                    robot=name,
                    units=scenario["production_units"],
                    hypothetical_print_saving_usd=old["replacement_structure_cost_usd"]
                    - new["replacement_structure_cost_usd"],
                    hypothetical_total_usd=new["example_total_usd"],
                )
            )
    template = bind(Path("ui/arm-material.html"))
    source = Path(__file__).relative_to(Path.cwd())
    bind(source)
    data = dict(
        geometry=geometry, loads=load_report["rows"], costs=costs, meshes=meshes
    )
    (output / "index.html").write_text(
        template.read_text().replace(
            "__DATA__", json.dumps(data).replace("<", "\\u003c")
        )
    )
    (output / "builder.py").write_bytes(source.read_bytes())
    write(
        output / "report.json",
        dict(
            status="pass",
            scope="Presentation integrity; clamp study remains partial and uninstalled",
            source_bindings=bindings,
            geometry_status=geometry["status"],
            physical_status="not_run",
            candidate_dynamics_status="not_run",
            cost_status="unquoted_planning_estimate",
            cost_rows=costs,
            output_hashes={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
