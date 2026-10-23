"""Bounded CAD volume and neutral arm check for longer finger hypotheses.

This study is intentionally separate from the shared variants.  It snapshots
the source inputs before constructing any robot, then evaluates the existing
arm recipe at 50, 60 and 70 mm with a 20 mm pad root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from build123d import Compound, export_step

from lab.artifact_store import snapshot_file
from lab.arm_mechanism import arm_parts
from lab.config import load_config
from lab.leg_mechanism import check_pairs, located_parts
from lab.robot import make_robot


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("variants/pip-head-service.yaml", "variants/scout-head-service.yaml")
SOURCE_FILES = (
    *VARIANTS,
    "robot/base.yaml",
    "robot/actuator-catalog.yaml",
    "robot/mass-assumptions.yaml",
    "lab/arm_mechanism.py",
    "lab/config.py",
    "lab/leg_mechanism.py",
    "lab/joint_module.py",
    "lab/robot.py",
    "docs/design/references/waveshare-st3215/ST3215.step",
    "tools/study_finger_length_mechanics.py",
)
DENSITY_KG_M3 = (1150.0, 1270.0, 1350.0)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(output: Path) -> dict[str, str]:
    """Copy and hash all inputs before loading or constructing any geometry."""
    hashes = {}
    for rel in SOURCE_FILES:
        source = ROOT / rel
        target = output / "source" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(source, target)
        hashes[rel] = sha256(source)
    return hashes


def robot_for(variant: str, length_mm: float, pad_root_mm: float):
    config = load_config(ROOT / variant)
    dims = config.dimensions.model_copy(
        update={
            "gripper_finger_length_m": length_mm / 1000.0,
            "gripper_pad_root_m": pad_root_mm / 1000.0,
        }
    )
    return make_robot(config.model_copy(update={"dimensions": dims}))


def finger_rows(robot) -> tuple[list[dict], list]:
    parts, _ = arm_parts(robot)
    selected = [
        p
        for p in parts
        if p.name.endswith("_forearm_spine") or p.name.endswith("_gripper_finger")
    ]
    rows = []
    for part in selected:
        solids = part.shape.solids()
        rows.append(
            {
                "part": part.name,
                "kind": "fixed_forearm_finger_stock"
                if part.name.endswith("_forearm_spine")
                else "moving_gripper_finger",
                "owner": part.owner,
                "role": part.role,
                "valid": bool(part.shape.is_valid),
                "solid_count": len(solids),
                "connected_single_solid": bool(part.shape.is_valid and len(solids) == 1),
                "volume_mm3": float(part.shape.volume),
            }
        )
    return rows, selected


def neutral_check(robot, parts) -> dict:
    located = located_parts(robot, parts)
    failures, exact_checks = check_pairs(
        parts,
        located,
        robot.forward_kinematics({}),
        {},
        max_collisions=None,
    )
    relevant = [
        row
        for row in failures
        if any(token in " ".join(row["parts"]) for token in ("finger", "forearm_spine"))
    ]
    return {
        "status": "pass" if not failures else "fail",
        "collision_count": len(failures),
        "finger_collision_count": len(relevant),
        "finger_collisions": relevant,
        "exact_checks": exact_checks,
        "scope": "neutral arm-core exact Boolean overlaps after bounds culling and designed-pair filters; head-service/body access not modeled",
    }


def run(
    output: Path,
    *,
    baseline_length_mm: float = 50.0,
    baseline_pad_root_mm: float = 20.0,
    candidate_length_mm: float = 70.0,
    candidate_pad_root_mm: float = 40.0,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = snapshot(output)
    cases_spec = (
        ("baseline", baseline_length_mm, baseline_pad_root_mm),
        ("candidate", candidate_length_mm, candidate_pad_root_mm),
    )
    cases = []
    for variant in VARIANTS:
        robot_name = Path(variant).name.split("-")[0]
        baseline_total = None
        for case_name, length_mm, pad_root_mm in cases_spec:
            robot = robot_for(variant, length_mm, pad_root_mm)
            rows, selected = finger_rows(robot)
            by_kind = {}
            for row in rows:
                # The mirrored recipe produces one left and one right arm.  Keep
                # the requested result explicitly per arm, while retaining both
                # sides' rows for the connectedness check.
                if row["part"].startswith("left_"):
                    by_kind[row["kind"]] = by_kind.get(row["kind"], 0.0) + row["volume_mm3"]
            total = sum(by_kind.values())
            if baseline_total is None:
                baseline_total = total
            density_mass = {
                str(d): total * 1e-9 * d for d in DENSITY_KG_M3
            }
            case = {
                "robot": robot_name,
                "variant": variant,
                "case": case_name,
                "finger_length_mm": length_mm,
                "pad_root_mm": pad_root_mm,
                "parts": rows,
                "per_arm_volume_mm3": total,
                "per_arm_volume_delta_mm3": total - baseline_total,
                "bilateral_volume_mm3": 2.0 * total,
                "bilateral_volume_delta_mm3": 2.0 * (total - baseline_total),
                "per_arm_mass_kg_by_density": density_mass,
                "per_arm_mass_delta_kg_by_density": {
                    str(d): (total - baseline_total) * 1e-9 * d
                    for d in DENSITY_KG_M3
                },
                "bilateral_mass_kg_by_density": {
                    str(d): 2.0 * total * 1e-9 * d for d in DENSITY_KG_M3
                },
                "bilateral_mass_delta_kg_by_density": {
                    str(d): 2.0 * (total - baseline_total) * 1e-9 * d
                    for d in DENSITY_KG_M3
                },
                "connected_single_solid": all(r["connected_single_solid"] for r in rows),
                "neutral_arm_check": neutral_check(robot, arm_parts(robot)[0]),
            }
            # Export only the inspected finger solids to make the artifact reviewable.
            solids = []
            for part in selected:
                shape = part.shape
                shape.label = part.name
                solids.append(shape)
            export_step(
                Compound(children=solids),
                output / f"{robot_name}-{case_name}-finger-{int(length_mm)}-{int(pad_root_mm)}mm.step",
            )
            cases.append(case)
    output_hashes = {
        str(path.relative_to(output)): sha256(path)
        for path in output.rglob("*")
        if path.is_file() and path.name != "report.json"
    }
    report = {
        "status": "pass" if all(c["connected_single_solid"] for c in cases) else "fail",
        "scope": "supporting CAD mechanics study; not manufacturing, strength, service or dynamics qualification",
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "baseline": {"finger_length_mm": baseline_length_mm, "pad_root_mm": baseline_pad_root_mm},
        "candidate": {"finger_length_mm": candidate_length_mm, "pad_root_mm": candidate_pad_root_mm},
        "polymer_density_kg_m3": list(DENSITY_KG_M3),
        "cases": cases,
        "limits": [
            "The fixed stock is the forearm_spine part's appended distal geometry; it is not a separately manufactured finger bill of materials.",
            "Volumes are exact nominal BRep integrals from the existing recipe, with no print infill, wall, pad, fastener, retention or cable model.",
            "Mass uses the existing unselected effective polymer planning range and is not measured.",
            "Neutral arm-core pair checks omit head-service shell removal, tool access, continuous swept clearance, contact dynamics, strength and actuator torque margin.",
            "A passing connectedness result does not establish a buildable or supported gripper.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-length-mm", type=float, default=50.0)
    parser.add_argument("--baseline-pad-root-mm", type=float, default=20.0)
    parser.add_argument("--candidate-length-mm", type=float, default=70.0)
    parser.add_argument("--candidate-pad-root-mm", type=float, default=40.0)
    args = parser.parse_args()
    result = run(
        args.output,
        baseline_length_mm=args.baseline_length_mm,
        baseline_pad_root_mm=args.baseline_pad_root_mm,
        candidate_length_mm=args.candidate_length_mm,
        candidate_pad_root_mm=args.candidate_pad_root_mm,
    )
    print(json.dumps({"status": result["status"], "cases": len(result["cases"])}))
