"""Compose two-carrier spacer candidates without promoting whole-robot readiness.

Retains all other groups, interfaces and source failures. Export identity and
material retention are separate from added-stock clearance and physical fit.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy
import traceback

import yaml

from build123d import Color, Compound, export_step, import_step
from lab.artifact_store import retain_dependency, snapshot_file
from lab.config import RobotConfig, load_config
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.cad_identity import geometry_record, geometry_agrees
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.build_carrier_option import require_spacer_change


def volume(shape):
    return 0.0 if shape is None else sum(s.volume for s in shape.solids())


def run(name, output, covered=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        physical_status="not_run",
        dynamics_status="not_run",
        added_stock_clearance_status="not_run",
    )
    bindings = {}

    def bind(path):
        bindings[str(path)] = sha(path)
        return path

    try:
        paths = [
            Path(__file__).relative_to(Path.cwd()),
            Path("lab/config.py"),
            Path("lab/robot.py"),
            Path("lab/arm_mechanism.py"),
            Path("lab/leg_mechanism.py"),
            Path("tools/cad_identity.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
        ]
        if covered:
            paths += [Path("lab/artifact_store.py"), Path("tools/build_carrier_option.py"), Path("robot/covered-production-study.yaml"), Path("robot/base.yaml"), Path(f"variants/{name}-carriers.yaml")]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        source = Path(f"artifacts/{name}-integral-head-composed-v2")
        delta_folder = None
        if covered:
            selection = bind(Path("robot/covered-production-study.yaml"))
            source = Path(yaml.safe_load(selection.read_text())["assemblies"][name])
            delta_folder = Path(f"artifacts/{name}-covered-carriers-v2")
            delta = verify_report(delta_folder)
            bind(delta_folder / "report.json")
            if delta["geometry_and_mass_status"] != "pass":
                raise ValueError("Configured carrier geometry and mass did not pass")
            source_mass = Path(f"artifacts/{name}-shin-covered-mass-v1")
            mass_source = verify_report(source_mass)
            bind(source_mass / "report.json")
            assembly_path = str(source / "assembly.step")
            if mass_source["source_hashes"].get(assembly_path) != sha(Path(assembly_path)):
                raise ValueError("Covered geometry differs from mass derivation")
            if delta["source_hashes"].get(str(source_mass / "report.json")) != sha(source_mass / "report.json"):
                raise ValueError("Carrier derivation uses different covered mass")
            report["matched_mass_source"] = str(delta_folder)
            report["mass_profiles"] = delta["profiles"]
        study = Path("artifacts/yoke-spacer-study-v2")
        registration = Path("artifacts/carrier-registration-v1")
        base, module, registered = [
            verify_report(p) for p in (source, study, registration)
        ]
        for p in (source, study, registration):
            bind(p / "report.json")
        if base["assembly_step_roundtrip"] != "pass" or registered["status"] != "pass":
            raise ValueError("Source export or module orientation is unverified")
        module_ids = sorted(
            r["module"] for r in registered["rows"] if r["robot"] == name
        )
        if len(module_ids) != 12 or len(set(module_ids)) != 12:
            raise ValueError("Expected twelve distinct generic spacer groups")
        mapper_path = bind(Path(registered["checked_mapping_source"]))
        if sha(mapper_path) != registered["source_hashes"][str(mapper_path)]:
            raise ValueError(
                "Carrier orientation mapping differs from registration audit"
            )
        mapper = runpy.run_path(str(mapper_path))["oriented_web"]
        cfg_path = bind(source / "resolved-config.json")
        cfg = RobotConfig.model_validate_json(cfg_path.read_text())
        if covered:
            target = bind(Path(f"variants/{name}-carriers.yaml"))
            next_cfg = load_config(target)
            require_spacer_change(cfg, next_cfg)
            derived_cfg = RobotConfig.model_validate_json(bind(delta_folder / "nominal/resolved-config.json").read_text())
            if next_cfg != derived_cfg:
                raise ValueError("Configured carrier mass and assembly configuration differ")
            cfg = next_cfg
        robot = make_robot(cfg)
        frames = robot.forward_kinematics()
        interface = json.loads(bind(source / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != interface:
            raise ValueError("Kinematic or command interface changed")
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2) + "\n")
        (output / "interfaces.json").write_bytes(
            (source / "interfaces.json").read_bytes()
        )
        print(name, "importing source assembly", flush=True)
        raw = parts_from_step(
            list(import_step(bind(source / "assembly.step")).children), base["parts"]
        )
        manifest = {r["name"]: r for r in base["parts"]}
        raw = [replace(p, color=tuple(manifest[p.name]["color"])) for p in raw]
        originals = {p.name: p for p in raw}
        old_identity = {
            r["name"]: r["actual"]
            for r in json.loads(bind(source / "step-roundtrip.json").read_text())
        }
        carriers = Compound(
            [
                s
                for side in ("rear", "front")
                for s in import_step(bind(study / f"carrier_{side}.step")).solids()
            ]
        )
        rings = import_step(bind(study / "loose_rings.step"))
        if len(carriers.solids()) != 2 or len(rings.solids()) != 8:
            raise ValueError("Module piece count changed")
        replacements, checks, expected_records = {}, [], {}
        for key in module_ids:
            old = originals[key]
            local = mapper(key, carriers)
            if covered:
                local = import_step(bind(delta_folder / f"{key}.step"))
                generated = next(p for p in delta["parts"] if p["name"] == key)
                if generated["owner"] != old.owner:
                    raise ValueError("Configured carrier owner differs from assembly")
            if covered:
                # Geometry is already independently exported by the configured
                # producer. Preserve those immutable bytes instead of re-exporting
                # an imported assembly node (which OCCT may reject).
                snapshot_file(delta_folder / f"{key}.step", output / f"{key}.step")
            else:
                export_step(local, output / f"{key}.step")
            independent = import_step(output / f"{key}.step")
            world = Compound(transform_shape(independent, frames[old.owner]).solids())
            expected_old = transform_shape(mapper(key, rings), frames[old.owner])
            source_missing = volume(expected_old - old.shape)
            source_extra = volume(old.shape - expected_old)
            removed = volume(old.shape - world)
            added = volume(world - old.shape)
            row = dict(
                name=key,
                owner=old.owner,
                old_solids=len(old.shape.solids()),
                new_solids=len(world.solids()),
                source_missing_mm3=source_missing,
                source_extra_mm3=source_extra,
                removed_mm3=removed,
                added_mm3=added,
                expected_added_mm3=module["added_carrier_volume_mm3"],
            )
            row["status"] = (
                "pass"
                if (
                    world.is_valid
                    and row["old_solids"] == 8
                    and row["new_solids"] == 2
                    and max(source_missing, source_extra, removed) < 0.0001
                    and abs(added - row["expected_added_mm3"]) < 0.0001
                )
                else "fail"
            )
            checks.append(row)
            replacements[key] = replace(old, shape=world)
            expected_records[key] = geometry_record(world)
            write(output / "composition-progress.json", checks)
            print(name, key, row["status"], flush=True)
        combined = [replacements.get(p.name, p) for p in raw]
        # Unchanged groups retain imported world geometry; no inverse/forward roundtrip.
        for p in combined:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        print(name, "exporting complete candidate", flush=True)
        export_step(
            Compound(children=[p.shape for p in combined]), output / "assembly.step"
        )
        inventory = [
            dict(
                name=p.name,
                owner=p.owner,
                role=p.role,
                module=p.module,
                color=p.color,
                volume_mm3=p.shape.volume,
            )
            for p in combined
        ]
        restored = parts_from_step(
            list(import_step(output / "assembly.step").children), inventory
        )
        identities = []
        for p in restored:
            expected = (
                expected_records[p.name]
                if p.name in replacements
                else old_identity[p.name]
            )
            actual = geometry_record(p.shape)
            identities.append(
                dict(
                    name=p.name,
                    replaced=p.name in replacements,
                    expected=expected,
                    actual=actual,
                    status="pass" if geometry_agrees(actual, expected) else "fail",
                )
            )
            if len(identities) % 25 == 0:
                write(output / "step-roundtrip-progress.json", identities)
                print(
                    name, "STEP checks", len(identities), "/", len(restored), flush=True
                )
        write(output / "step-roundtrip.json", identities)
        bom_path = Path("artifacts/covered-carriers-cost-v2/prototype-bom.json") if covered else Path("robot/prototype-bom.json")
        if covered:
            cost = verify_report(bom_path.parent)
            bind(bom_path.parent / "report.json")
            cost_robot = next(r for r in cost["robots"] if r["name"] == name)
            if cost["source_bindings"].get(str(delta_folder / "report.json")) != sha(delta_folder / "report.json"):
                raise ValueError("Candidate BOM/cost uses different carrier delta")
        bom = json.loads(bind(bom_path).read_text())
        printed = [
            p
            for e in bom["entries"]
            if e["make_or_buy"] == "make"
            for p in e.get("part_ids_by_robot", {}).get(name, [])
        ]
        if len(printed) != len(set(printed)):
            raise ValueError("Duplicate printed group mapping")
        actual_parts = {p.name: p for p in restored}
        counts = dict(
            printed_group_count=len(printed),
            source_printed_solid_count=sum(
                len(originals[p].shape.solids()) for p in printed
            ),
            candidate_printed_solid_count=sum(
                len(actual_parts[p].shape.solids()) for p in printed
            ),
            gripper_ring_count=sum(
                len(actual_parts[f"{side}_gripper_spacer"].shape.solids())
                for side in ("left", "right")
            ),
        )
        if (
            counts["source_printed_solid_count"]
            - counts["candidate_printed_solid_count"]
            != 72
            or counts["gripper_ring_count"] != 16
        ):
            raise ValueError("Unexpected installed piece-count delta")
        print_inventory = [
            dict(
                name=p,
                solids=len(actual_parts[p].shape.solids()),
                volume_mm3=actual_parts[p].shape.volume,
            )
            for p in printed
        ]
        write(output / "printed-inventory.json", print_inventory)
        if covered:
            expected_count = cost_robot["inventory"]["printed_solid_count"]
            expected_volume = cost_robot["inventory"]["total_volume_mm3"]
            actual_volume = sum(r["volume_mm3"] for r in print_inventory)
            if counts["candidate_printed_solid_count"] != expected_count or abs(actual_volume - expected_volume) > .02:
                raise ValueError("Composed printed geometry differs from retained cost inventory")
            report["cost_inventory_agreement"] = dict(status="pass", source=str(bom_path.parent), printed_solids=expected_count, volume_mm3=actual_volume, volume_difference_mm3=actual_volume-expected_volume)
            report["dependency_manifests"] = {
                "covered_source": retain_dependency(source),
                "configured_carriers_and_mass": retain_dependency(delta_folder),
                "cost_inventory": retain_dependency(bom_path.parent),
            }
        passed = all(r["status"] == "pass" for r in checks + identities)
        report.update(
            source_assembly=str(source),
            source_assembly_status=base["status"],
            parts=inventory,
            part_count=len(inventory),
            replacements=module_ids,
            composition_checks=checks,
            piece_counts=counts,
            added_volume_mm3=sum(r["added_mm3"] for r in checks),
            geometry_build_status="pass" if passed else "fail",
            assembly_step_roundtrip="pass"
            if all(r["status"] == "pass" for r in identities)
            else "fail",
            material_retention_status="pass"
            if all(r["status"] == "pass" for r in checks)
            else "fail",
            status="fail" if base["status"] == "fail" or not passed else "partial",
            not_run=[
                "complete installation/tool access",
                "physical fit/retention/creep",
                "candidate mass/inertia and control validation",
                "full motion/whole-product qualification",
            ],
            limits=[
                "Candidate composition only; installed/procurement quantities are unchanged.",
                "STEP metrics check volume, bounds, validity and solid count, not independent boundary equivalence.",
                "Added-stock scan is separate; no missing or in-progress report is treated as clearance.",
                "All inherited whole-robot failures remain failed.",
            ],
        )
    except Exception:
        report["error"] = traceback.format_exc()
    if covered and report.get("geometry_build_status") == "pass":
        report["scope"] = "Complete covered assembly with configuration-generated linked carriers; export, material retention and existing mass/cost identity. Not a full-robot clearance or manufacturing qualification."
        report["not_run"] = ["new whole-robot motion/contact evaluation", "complete installation/tool and harness access", "physical fit/retention/strength", "manufacturing and production-price qualification"]
    report["source_bindings"] = bindings
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    print(
        name,
        {
            k: report[k]
            for k in ["status", "geometry_build_status", "piece_counts", "error"]
            if k in report
        },
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=["pip", "scout"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--covered", action="store_true", help="Compose existing shin-covered bodies with configured carriers and bind matching mass/cost evidence")
    a = p.parse_args()
    result = run(a.name, a.output, a.covered)
    if result.get("geometry_build_status") != "pass":
        raise SystemExit(1)
