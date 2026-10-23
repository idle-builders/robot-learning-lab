"""Reconcile the covered robot BOM, exact printed solids and production cost."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import traceback

from build123d import import_step
import yaml

from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import verify_report
from tools.cost_integrated import Assumptions, calculate as allocate, verify_assembly
from tools.print_cost import PrintAssumptions, calculate as print_cost


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        price_qualified=False,
        physical_status="not_run",
        source_bindings={},
        robots=[],
    )
    files = [
        Path(__file__).relative_to(Path.cwd()),
        Path("tools/cost_integrated.py"),
        Path("tools/print_cost.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/check_neck_collar.py"),
        Path("tools/check_camera_support.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("robot/covered-production-study.yaml"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in files})

    def bind(p):
        p = Path(p)
        report["source_bindings"][str(p)] = sha(p)
        return p

    try:
        study = yaml.safe_load(bind("robot/covered-production-study.yaml").read_text())
        assumptions = Assumptions.model_validate(
            yaml.safe_load(bind(study["cost_assumptions"]).read_text())
        )
        # Separate candidate model; never mutate the legacy cost source.
        structure = next(
            line for line in assumptions.lines if line.id == "structure_finish"
        )
        structure.roles.append(study["added_print_role"])
        printing = PrintAssumptions.model_validate(
            yaml.safe_load(bind(study["print_assumptions"]).read_text())
        )
        original_bom = json.loads(bind(study["baseline_bom"]).read_text())
        baseline_folder = Path(study["baseline_print_inventory"])
        baseline = verify_report(baseline_folder)
        bind(baseline_folder / "report.json")
        bom = deepcopy(original_bom)
        bom.update(
            scope="Covered-shin candidate planning BOM, 234/240 groups; not an order release.",
            source_reports={},
            inventory_counts={},
        )
        bom.pop("source_report", None)
        new_entries = [
            dict(
                id="print-shin-covers",
                label="Printed shin covers",
                make_or_buy="make",
                stage="later",
                specification="Two removable front covers with open rear; conditional nominal fit, no physical print qualification.",
                quantity_initial_bench=0,
                quantity_per_robot={"pip": 2, "scout": 2},
                part_ids_by_robot={},
            )
        ]
        for role in ["screw", "insert"]:
            h = study["hardware"][role]
            new_entries.append(
                dict(
                    id=f"shin-cover-{role}s",
                    label=f"Shin cover {role}s",
                    make_or_buy="buy",
                    stage="later",
                    specification=h["specification"],
                    candidate=h["candidate"],
                    source=h["source"],
                    quantity_initial_bench=0,
                    quantity_per_robot={
                        "pip": h["count_per_robot"],
                        "scout": h["count_per_robot"],
                    },
                    part_ids_by_robot={},
                )
            )
        for name, assembly_name in study["assemblies"].items():
            assembly = Path(assembly_name)
            retained = verify_assembly(assembly, allow_unqualified_assembly=True)
            for filename in [
                "report.json",
                "assembly.step",
                "interfaces.json",
                "resolved-config.json",
            ]:
                bind(assembly / filename)
            interface = json.loads((assembly / "interfaces.json").read_text())
            config = json.loads((assembly / "resolved-config.json").read_text())
            if any(
                a["part_id"] != assumptions.actuator_part_id
                for a in config["actuators"].values()
            ):
                raise ValueError("Actuator part mismatch")
            print("Importing", name, flush=True)
            parts = parts_from_step(
                list(import_step(assembly / "assembly.step").children),
                retained["parts"],
            )
            by_name = {p.name: p for p in parts}
            if len(by_name) != len(parts):
                raise ValueError("Duplicate part identity")
            for entry, role in zip(
                new_entries, ["shin_cover", "screw", "insert"], strict=True
            ):
                ids = [
                    p.name
                    for p in parts
                    if p.role == role
                    and (role == "shin_cover" or "_shin_cover_" in p.name)
                ]
                if len(ids) != entry["quantity_per_robot"][name]:
                    raise ValueError("Added hardware or cover count differs")
                entry["part_ids_by_robot"][name] = ids
            entries = [*bom["entries"], *new_entries]
            mapped = Counter(
                p
                for e in entries
                for p in e.get("part_ids_by_robot", {}).get(name, e.get("part_ids", []))
            )
            if set(mapped) != set(by_name) or any(n != 1 for n in mapped.values()):
                raise ValueError("BOM does not cover assembly exactly once")
            print_entries = [
                e
                for e in entries
                if e["make_or_buy"] == "make"
                and e.get("part_ids_by_robot", {}).get(name)
            ]
            group_rows = []
            for entry in print_entries:
                solids = []
                ids = entry["part_ids_by_robot"][name]
                for key in ids:
                    part = by_name[key]
                    actual = list(part.shape.solids())
                    if not actual or any(
                        not s.is_valid or s.volume <= 0 for s in actual
                    ):
                        raise ValueError(f"Invalid printable solid: {key}")
                    if abs(sum(s.volume for s in actual) - part.shape.volume) > max(
                        0.01, part.shape.volume * 1e-6
                    ):
                        raise ValueError(f"Unreconciled solid volume: {key}")
                    for i, solid in enumerate(actual):
                        box = ExactQueries().bounds(solid)
                        bounds = [
                            [box.min.X, box.min.Y, box.min.Z],
                            [box.max.X, box.max.Y, box.max.Z],
                        ]
                        solids.append(
                            dict(
                                part_id=key,
                                solid_index=i,
                                owner=part.owner,
                                role=part.role,
                                volume_mm3=float(solid.volume),
                                bounds_mm=bounds,
                            )
                        )
                group_rows.append(
                    dict(
                        group_id=entry["id"],
                        label=entry["label"],
                        group_count=len(ids),
                        solid_count=len(solids),
                        solids=solids,
                        total_volume_mm3=sum(s["volume_mm3"] for s in solids),
                    )
                )
            inventory = dict(
                robot=name,
                groups=group_rows,
                printed_group_count=sum(g["group_count"] for g in group_rows),
                printed_solid_count=sum(g["solid_count"] for g in group_rows),
                total_volume_mm3=sum(g["total_volume_mm3"] for g in group_rows),
            )
            for group in group_rows:
                entry = next(e for e in entries if e["id"] == group["group_id"])
                entry.setdefault("cad_group_count", {})[name] = group["group_count"]
                entry.setdefault("geometric_solid_count", {})[name] = group[
                    "solid_count"
                ]
                entry.setdefault("nominal_cad_volume_mm3", {})[name] = group[
                    "total_volume_mm3"
                ]
                entry["quantity_per_robot"][name] = group["solid_count"]
                entry["quantity_unit"] = (
                    "geometric solids; printing and physical fit unqualified"
                )
                if "prior_reference_evidence_paths" not in entry:
                    entry["prior_reference_evidence_paths"] = entry.pop(
                        "evidence_paths", []
                    )
                    entry["evidence_paths"] = []
                entry["evidence_paths"].append(str(assembly / "report.json"))
            old = next(r for r in baseline["groups"] if r["robot"] == name)
            previous = Path(retained["source_assembly"]) / "report.json"
            if (
                retained["source_bindings"][str(previous)]
                != baseline["source_hashes"][str(previous)]
            ):
                raise ValueError("Baseline print identity differs")
            old_by_part = {s["part_id"]: [] for g in old["groups"] for s in g["solids"]}
            for g in old["groups"]:
                for s in g["solids"]:
                    old_by_part[s["part_id"]].append(s["volume_mm3"])
            new_by_part = {s["part_id"]: [] for g in group_rows for s in g["solids"]}
            for g in group_rows:
                for s in g["solids"]:
                    new_by_part[s["part_id"]].append(s["volume_mm3"])
            for key, values in old_by_part.items():
                if key in ["left_shin_spine", "right_shin_spine"]:
                    continue
                if (
                    len(values) != len(new_by_part[key])
                    or abs(sum(values) - sum(new_by_part[key])) > 0.01
                ):
                    raise ValueError(f"Unexpected old-part print change: {key}")
            cost = allocate(
                retained["parts"], len(interface["joint_order"]), assumptions
            )
            samples = []
            for scenario in cost["scenarios"]:
                calculated = print_cost(
                    inventory["total_volume_mm3"],
                    printing,
                    scenario,
                    assumptions.contingency_fraction,
                    assumptions.target_usd,
                )
                actuator = next(
                    line for line in scenario["lines"] if line["id"] == "actuators"
                )
                # Replace structure before solving actuator budget; don't reuse flat-line ceilings.
                base = [
                    calculated["example_total_usd"][i]
                    / (1 + assumptions.contingency_fraction)
                    - actuator["cost_usd"][i]
                    for i in range(2)
                ]
                calculated["actuator_unit_ceiling_usd"] = [
                    (
                        assumptions.target_usd / (1 + assumptions.contingency_fraction)
                        - b
                    )
                    / cost["actuator_count"]
                    for b in base
                ]
                h = study["hardware"]["screw"]
                quantity = h["count_per_robot"] * scenario["production_units"]
                ordered = (
                    math.ceil(quantity / h["order_multiple"]) * h["order_multiple"]
                )
                tier = max(
                    (t for t in h["tiers"] if t[0] <= ordered), key=lambda t: t[0]
                )
                calculated["added_screw_reference"] = dict(
                    installed_quantity=quantity,
                    order_quantity=ordered,
                    unit_gbp=tier[1],
                    order_gbp=ordered * tier[1],
                    per_robot_gbp=ordered * tier[1] / scenario["production_units"],
                    scope=h["price_scope"],
                )
                calculated["added_insert_order_quantity"] = (
                    study["hardware"]["insert"]["count_per_robot"]
                    * scenario["production_units"]
                )
                calculated["added_insert_unit_price"] = None
                samples.append(calculated)
            bom["source_reports"][name] = dict(
                path=str(assembly / "report.json"),
                part_count=len(parts),
                status=retained["status"],
            )
            bom["inventory_counts"][name] = len(parts)
            report["robots"].append(
                dict(
                    name=name,
                    inventory=inventory,
                    baseline_print_volume_mm3=old["total_volume_mm3"],
                    baseline_print_solids=old["printed_solid_count"],
                    print_delta_mm3=inventory["total_volume_mm3"]
                    - old["total_volume_mm3"],
                    cost=cost,
                    samples=samples,
                    inherited_assembly_status=retained["status"],
                )
            )
            print(
                name,
                inventory["printed_group_count"],
                inventory["printed_solid_count"],
                inventory["total_volume_mm3"],
                flush=True,
            )
            write(output / "progress.json", report)
        bom["entries"].extend(new_entries)
        bom["limitations"].extend(
            [
                study["hardware_allowance_treatment"],
                study["handwork_treatment"],
                "Baseline existing quantities retain their prior unresolved specifications. The new exact candidate quantities are not an order release.",
            ]
        )
        write(output / "prototype-bom.json", bom)
        report.update(
            status="planning_estimate",
            assumptions=printing.model_dump(),
            cost_assumptions=assumptions.model_dump(),
            study=study,
            limitations=[
                "No quote, slice or measured production time/yield.",
                "Print-cost replacement removes the old structure allowance before adding explicit printing.",
                "Added screws/inserts remain inside the existing hardware allowance, which is unqualified. Added insert labor remains within the unmeasured handwork allowance; it is not claimed free.",
                "Source assembly failures and unmapped physical hardware specifications remain.",
                "Two shin spines replace the originals, never two extra spines. Linked spacer and lighter-clamp studies are separate and not installed here.",
            ],
        )
    except Exception:
        report["error"] = traceback.format_exc()
        print(report["error"], flush=True)
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    result = run(parser.parse_args().output)
    if result["status"] == "fail":
        raise SystemExit(1)
