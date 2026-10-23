"""Bind current assemblies, actual solid inventory and print-process sensitivity."""

import argparse
import json
from pathlib import Path
import yaml
from tools.check_camera_support import sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.print_cost import PrintAssumptions, calculate


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {}

    def bind(path):
        p = Path(path)
        sources[str(p)] = sha(p)
        return p

    inventory_path = Path("artifacts/composed-print-inventory-v3")
    inventory = verify_report(inventory_path)
    bind(inventory_path / "report.json")
    assumptions = PrintAssumptions.model_validate(
        yaml.safe_load(bind("robot/printing-cost.yaml").read_text())
    )
    cost_assumptions = yaml.safe_load(bind("robot/integrated-cost.yaml").read_text())
    robots = []
    for robot in inventory["groups"]:
        name = robot["robot"]
        solids = [s for g in robot["groups"] for s in g["solids"]]
        ids = {(s["part_id"], s["solid_index"]) for s in solids}
        if (
            len(ids) != len(solids)
            or len(solids) != robot["printed_solid_count"]
            or len({s["part_id"] for s in solids}) != robot["printed_group_count"]
            or abs(sum(s["volume_mm3"] for s in solids) - robot["total_volume_mm3"])
            > 0.01
        ):
            raise ValueError("Solid inventory counts or volume do not reconcile")
        folder = Path(f"artifacts/{name}-composed-cost-v1")
        cost = verify_report(folder)
        bind(folder / "report.json")
        assembly = Path(f"artifacts/{name}-integral-head-composed-v2")
        report = verify_report(assembly)
        bind(assembly / "report.json")
        if cost["assembly_step_sha256"] != sha(assembly / "assembly.step"):
            raise ValueError("Cost / CAD identity mismatch")
        source_inventory = json.loads(
            (inventory_path / "source" / assembly / "report.json").read_text()
        )
        if source_inventory != report:
            raise ValueError("Print inventory / CAD identity mismatch")
        samples = [
            calculate(
                robot["total_volume_mm3"],
                assumptions,
                s,
                cost_assumptions["contingency_fraction"],
                cost["target_usd"],
            )
            for s in cost["scenarios"]
        ]
        robots.append(dict(name=name, inventory=robot, cost=cost, samples=samples))
    template = bind("ui/production-cost.html").read_text()
    for p in ("tools/review_production_cost.py", "tools/print_cost.py"):
        bind(p)
    data = dict(
        robots=robots,
        assumptions=assumptions.model_dump(),
        contingency=cost_assumptions["contingency_fraction"],
        target_usd=cost_assumptions["target_usd"],
    )
    (output / "index.html").write_text(
        template.replace(
            "__DATA__", json.dumps(data, allow_nan=False).replace("<", "\\u003c")
        )
    )
    write(
        output / "report.json",
        dict(
            status="partial",
            price_qualified=False,
            scope="Current CAD inventory and illustrative print-process cost sensitivity; no production qualification",
            source_bindings=sources,
            robots=robots,
            assumptions=assumptions.model_dump(),
            output_hashes={"index.html": sha(output / "index.html")},
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
