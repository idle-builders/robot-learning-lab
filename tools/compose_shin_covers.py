"""Compose the retained robot assembly with the checked shin-cover STEP parts.

This is an identity-preserving composition only.  The retained source status,
physical status, and continuous-clearance limitations remain in force.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import Color, Compound, export_step, import_step

from lab.config import RobotConfig
from lab.leg_mechanism import transform_shape
from lab.robot import make_robot
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def _local_part(part, frame, color):
    return replace(
        part,
        shape=transform_shape(part.shape, np.linalg.inv(frame)),
        color=tuple(color),
    )


def _step_shape(path):
    imported = import_step(path)
    solids = imported.solids()
    if not solids:
        raise ValueError(f"STEP has no solids: {path}")
    return Compound(solids)


def run(name, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": __doc__.strip(),
        "physical_status": "not_run",
        "dynamics_status": "not_run",
    }
    try:
        source = Path(f"artifacts/{name}-integral-head-composed-v2")
        study = Path("artifacts/shin-cover-study-v1")
        qualification = Path("artifacts/shin-cover-qualification-v1")
        retained = verify_report(source)
        study_report = verify_report(study)
        qualification_report = verify_report(qualification)
        for filename in (
            "report.json",
            "assembly.step",
            "resolved-config.json",
            "interfaces.json",
        ):
            path = source / filename
            if study_report["source_bindings"].get(str(path)) != sha(path):
                raise ValueError(f"Study source assembly mismatch: {path}")
        if qualification_report["source_bindings"].get(
            str(study / "report.json")
        ) != sha(study / "report.json"):
            raise ValueError("Supplemental qualification refers to another study")
        if name not in study_report.get("robots", {}):
            raise ValueError(f"shin-cover study has no {name} result")

        source_files = [
            *Path("lab").glob("*.py"),
            *Path("tools").glob("*.py"),
            Path("robot/shin-cover-study.yaml"),
            Path("robot/printing-cost.yaml"),
            Path("electronics/power-system.yaml"),
        ]
        report["source_hashes"] = copy_source(
            output, {str(path): path for path in source_files}
        )

        bindings = {}

        def bind(path):
            path = Path(path)
            bindings[str(path)] = sha(path)
            return path

        for folder in (source, study, qualification):
            bind(folder / "report.json")
        for filename in ("assembly.step", "resolved-config.json", "interfaces.json"):
            bind(source / filename)

        cfg = RobotConfig.model_validate_json(
            (source / "resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        frames = robot.forward_kinematics()
        contract = json.loads((source / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != json.loads(
            json.dumps(contract)
        ):
            raise ValueError("retained kinematic interface changed")
        write(output / "resolved-config.json", cfg.model_dump(mode="json"))
        write(output / "interfaces.json", contract)

        world = parts_from_step(
            list(import_step(source / "assembly.step").children), retained["parts"]
        )
        records = {entry["name"]: entry for entry in retained["parts"]}
        local = [
            _local_part(p, frames[p.owner], records[p.name]["color"]) for p in world
        ]
        local_by_name = {p.name: p for p in local}

        robot_report = study_report["robots"][name]
        candidate_names = [
            f"{side}_shin_{suffix}"
            for side in ("left", "right")
            for suffix in (
                "cover",
                "cover_screw_0_0",
                "cover_screw_0_1",
                "cover_screw_1_0",
                "cover_screw_1_1",
                "cover_insert_0_0",
                "cover_insert_0_1",
                "cover_insert_1_0",
                "cover_insert_1_1",
            )
        ]
        for candidate_name in candidate_names:
            path = bind(study / f"{name}-{candidate_name}.step")
            digest = study_report.get("output_hashes", {}).get(path.name)
            if digest != sha(path):
                raise ValueError(f"candidate STEP is not bound by study: {path}")
        if qualification_report.get("source_bindings", {}).get(
            str(source / "assembly.step")
        ) != sha(source / "assembly.step"):
            raise ValueError("qualification source assembly binding differs")

        replacements = {}
        additions = []
        for side in ("left", "right"):
            spine_name = f"{side}_shin_spine"
            if spine_name not in local_by_name:
                raise ValueError(f"missing retained part: {spine_name}")
            spine = local_by_name[spine_name]
            modified_spine_path = bind(study / f"{name}-{side}_shin_spine.step")
            if study_report.get("output_hashes", {}).get(
                modified_spine_path.name
            ) != sha(modified_spine_path):
                raise ValueError(
                    f"modified spine STEP is not bound by study: {modified_spine_path}"
                )
            modified_spine = _step_shape(modified_spine_path)
            cover = _step_shape(study / f"{name}-{side}_shin_cover.step")
            replacements[spine_name] = replace(spine, shape=modified_spine)
            color = next(
                p["color"]
                for p in robot_report["parts"]
                if p["name"] == f"{side}_shin_cover"
            )
            additions.append(
                replace(
                    spine,
                    name=f"{side}_shin_cover",
                    shape=cover,
                    role="shin_cover",
                    module=spine.owner,
                    color=tuple(color),
                )
            )
            for suffix in candidate_names:
                if not suffix.startswith(f"{side}_shin_cover_"):
                    continue
                path = study / f"{name}-{suffix}.step"
                role = "screw" if "_screw_" in suffix else "insert"
                color = next(
                    p["color"] for p in robot_report["parts"] if p["name"] == suffix
                )
                additions.append(
                    replace(
                        spine,
                        name=suffix,
                        shape=_step_shape(path),
                        role=role,
                        module=spine.owner,
                        color=tuple(color),
                    )
                )

        combined_local = [replacements.get(p.name, p) for p in local] + additions
        expected_count = len(local) + 18
        if len(combined_local) != expected_count:
            raise ValueError(
                f"unexpected inventory count {len(combined_local)} != {expected_count}"
            )

        if len({p.name for p in combined_local}) != len(combined_local):
            raise ValueError("Duplicate part identity")
        for p in [*replacements.values(), *additions]:
            record = next(row for row in robot_report["parts"] if row["name"] == p.name)
            if record["owner"] != p.owner or record["role"] != p.role:
                raise ValueError(f"Candidate owner/role differs: {p.name}")
        print(name, "exporting composed assembly", flush=True)
        world_combined = [
            replace(
                p, shape=Compound(transform_shape(p.shape, frames[p.owner]).solids())
            )
            for p in combined_local
        ]
        for part in world_combined:
            part.shape.label = part.name
            part.shape.color = Color(*part.color[:3])
        export_step(
            Compound(children=[p.shape for p in world_combined]),
            output / "assembly.step",
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
            for p in world_combined
        ]
        write(output / "parts.json", inventory)
        restored = {
            p.name: p.shape
            for p in parts_from_step(
                list(import_step(output / "assembly.step").children), inventory
            )
        }
        identity = []
        for part in world_combined:
            expected = geometry_record(part.shape)
            actual = geometry_record(restored[part.name])
            identity.append(
                {
                    "name": part.name,
                    "expected": expected,
                    "actual": actual,
                    "status": "pass" if geometry_agrees(actual, expected) else "fail",
                }
            )
            if len(identity) % 25 == 0:
                print(
                    name,
                    "STEP roundtrip",
                    len(identity),
                    "/",
                    len(inventory),
                    flush=True,
                )
        write(output / "step-roundtrip.json", identity)
        roundtrip = all(row["status"] == "pass" for row in identity)

        report.update(
            source_assembly_status=retained["status"],
            source_assembly=str(source),
            source_bindings=bindings,
            qualification_status=qualification_report.get("status"),
            resolved_config_sha256=sha(output / "resolved-config.json"),
            part_count=len(inventory),
            replaced_parts=["left_shin_spine", "right_shin_spine"],
            added_groups=[p["name"] for p in inventory if "shin_cover" in p["name"]],
            nominal_composition_status="not_run",
            composition_scope="Identity-preserving composition of previously sampled candidate; no fresh full-body collision qualification",
            geometry_build_status="pass" if roundtrip else "fail",
            assembly_step_roundtrip="pass" if roundtrip else "fail",
            parts=inventory,
            status=(
                "partial" if roundtrip and retained["status"] == "pass" else "fail"
            ),
            limitations=[
                "Retained source whole-robot status remains unchanged and failed.",
                "Shin-cover study's candidate fit and wheel-sweep scope remain separate evidence.",
                "No candidate-to-candidate hardware, continuous motion, dynamics, mass, or physical qualification is inferred.",
            ],
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["output_hashes"] = {
        str(path.relative_to(output)): sha(path)
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "report.json"
        and "source" not in path.relative_to(output).parts
    }
    write(output / "report.json", report)
    print(
        name, {k: report.get(k) for k in ("status", "part_count", "error")}, flush=True
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["pip", "scout"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.name, args.output)
