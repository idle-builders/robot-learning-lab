"""Bounded print-cost delta with unchanged, unmeasured handwork assumptions."""

import argparse
import json
from pathlib import Path
import yaml
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.print_cost import PrintAssumptions, calculate


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    bindings = {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    assumptions = PrintAssumptions.model_validate(
        yaml.safe_load(bind(Path("robot/printing-cost.yaml")).read_text())
    )
    full = yaml.safe_load(bind(Path("robot/integrated-cost.yaml")).read_text())
    original_folder = Path("artifacts/composed-print-inventory-v3")
    original = verify_report(original_folder)
    bind(original_folder / "report.json")
    rows = []
    for name in ("pip", "scout"):
        folder = Path(f"artifacts/{name}-carrier-composed-v1")
        candidate = verify_report(folder)
        bind(folder / "report.json")
        inventory = json.loads(bind(folder / "printed-inventory.json").read_text())
        baseline = next(r for r in original["groups"] if r["robot"] == name)
        base_report = Path(candidate["source_assembly"]) / "report.json"
        if (
            candidate["source_bindings"][str(base_report)]
            != original["source_hashes"][str(base_report)]
        ):
            raise ValueError("Print inventory and candidate baseline differ")
        if candidate["geometry_build_status"] != "pass":
            raise ValueError("Candidate geometry failed")
        old_volume = baseline["total_volume_mm3"]
        new_volume = sum(r["volume_mm3"] for r in inventory)
        if abs(new_volume - old_volume - candidate["added_volume_mm3"]) > 0.01:
            raise ValueError("Print-volume delta does not match replacement stock")
        cost_folder = Path(f"artifacts/{name}-composed-cost-v1")
        costs = verify_report(cost_folder)
        bind(cost_folder / "report.json")
        scenarios = []
        for scenario in costs["scenarios"]:
            old = calculate(
                old_volume,
                assumptions,
                scenario,
                full["contingency_fraction"],
                costs["target_usd"],
            )
            new = calculate(
                new_volume,
                assumptions,
                scenario,
                full["contingency_fraction"],
                costs["target_usd"],
            )
            scenarios.append(
                dict(
                    production_units=scenario["production_units"],
                    baseline=old,
                    candidate=new,
                    print_cost_delta_usd=new["replacement_structure_cost_usd"]
                    - old["replacement_structure_cost_usd"],
                )
            )
        rows.append(
            dict(
                robot=name,
                piece_counts=candidate["piece_counts"],
                baseline_volume_mm3=old_volume,
                candidate_volume_mm3=new_volume,
                scenarios=scenarios,
            )
        )
    paths = [Path(__file__).relative_to(Path.cwd()), Path("tools/print_cost.py")]
    write(
        output / "report.json",
        dict(
            status="planning_estimate",
            price_qualified=False,
            scope=__doc__,
            assumptions=assumptions.model_dump(),
            robots=rows,
            source_bindings=bindings,
            source_hashes=copy_source(output, {str(p): p for p in paths}),
            limits=[
                "No time-per-piece saving has been measured or assumed. Handwork remains one illustrative hour per robot.",
                "Only print volume changes; all other production cost ranges and contingency remain unchanged.",
                "No quote, slicing, new mass/inertia or manufacturing qualification. Candidate full robot remains failed.",
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
