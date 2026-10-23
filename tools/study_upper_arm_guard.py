"""Bounded upper-arm guard checks against exact retained Pip/Scout assemblies.

Only added material is qualified here. Full-robot failures remain failures.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
import yaml
from build123d import Pos, export_step, import_step

from lab.artifact_store import retain_dependency
from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.robot import make_robot
from lab.upper_arm_guard import candidate
from tools.cad_identity import geometry_record, geometry_agrees
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def volume(shape):
    return 0 if shape is None else sum(s.volume for s in shape.solids())


def poses(robot):
    limits = {j.name: j.limits for j in robot.joints}
    yield "neutral", {}
    for suffix in ("shoulder_roll", "upper_arm", "forearm"):
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            pose = {}
            for side in ("left", "right"):
                key = f"{side}_{suffix}_joint"
                lo, hi = limits[key]
                f = (
                    1 - fraction
                    if side == "right" and suffix == "shoulder_roll"
                    else fraction
                )
                pose[key] = lo + f * (hi - lo)
            yield f"{suffix}_{fraction:g}", pose
    for roll in (0, 1):
        for upper in (0, 1):
            for elbow in (0, 1):
                pose = {}
                for side in ("left", "right"):
                    for suffix, index in (
                        ("shoulder_roll", roll),
                        ("upper_arm", upper),
                        ("forearm", elbow),
                    ):
                        key = f"{side}_{suffix}_joint"
                        pose[key] = limits[key][
                            1 - index
                            if side == "right" and suffix == "shoulder_roll"
                            else index
                        ]
                yield f"coupled_{roll}_{upper}_{elbow}", pose


def run(output, spec_path, clearance_screen_only=False, full_pose_grid=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        robots={},
        source_bindings={},
        dependency_manifests={},
        physical_status="not_run",
        dynamics_status="not_run",
        distance_metric=(
            "capped at the required clearance; passing value is a lower bound, not the global minimum"
            if clearance_screen_only
            else "global nearest surface distance"
        ),
    )
    files = [
        *Path("lab").glob("*.py"),
        Path(__file__).relative_to(Path.cwd()),
        spec_path,
        *[
            Path("tools") / p
            for p in (
                "cad_identity.py",
                "check_arm_episode.py",
                "check_camera_support.py",
                "check_neck_collar.py",
                "compose_forearm_enclosure.py",
            )
        ],
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in files})
    try:
        c = yaml.safe_load(spec_path.read_text())
        for name, source_name in c["assemblies"].items():
            source = Path(source_name)
            print(name, "loading complete covered-carrier assembly", flush=True)
            retained = verify_report(source)
            report["dependency_manifests"][name] = retain_dependency(source)
            for filename in (
                "report.json",
                "assembly.step",
                "resolved-config.json",
                "interfaces.json",
            ):
                report["source_bindings"][str(source / filename)] = sha(
                    source / filename
                )
            robot = make_robot(
                RobotConfig.model_validate_json(
                    (source / "resolved-config.json").read_text()
                )
            )
            if json.loads(json.dumps(robot.interface())) != json.loads(
                (source / "interfaces.json").read_text()
            ):
                raise ValueError("Source interface mismatch")
            neutral = robot.forward_kinematics()
            raw = parts_from_step(
                list(import_step(source / "assembly.step").children), retained["parts"]
            )
            local = [
                replace(
                    p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner]))
                )
                for p in raw
            ]
            changed, additions, recipes, mounts = [], [], {}, []
            polymer, hardware = 0, 0
            for side in ("left", "right"):
                owner = f"{side}_upper_arm"
                spine = next(p for p in local if p.name == f"{side}_upper_spine")
                cand = candidate(spine.shape, robot.config.dimensions, c, name, side)
                recipes[side] = cand
                color = (0.59, 0.76, 0.69) if name == "pip" else (0.74, 0.66, 0.48)
                definitions = [
                    (spine.name, cand["spine"], "spine", (0.36, 0.64, 0.57)),
                    (f"{owner}_guard", cand["cover"], "upper_arm_guard", color),
                ]
                for kind in ("screws", "inserts"):
                    definitions += [
                        (f"{owner}_guard_{kind}_{k}", v, kind, (0.5, 0.5, 0.5))
                        for k, v in cand[kind].items()
                    ]
                for label, shape, role, rgb in definitions:
                    path = output / f"{name}-{label}.step"
                    expected = geometry_record(shape)
                    export_step(shape, path)
                    actual = geometry_record(import_step(path))
                    if not geometry_agrees(actual, expected):
                        raise ValueError(f"STEP readback differs: {label}")
                    changed.append(
                        dict(
                            name=label,
                            owner=owner,
                            role=role,
                            color=rgb,
                            step=path.name,
                            replaces_original=label == spine.name,
                            geometry=actual,
                        )
                    )
                    if label != spine.name:
                        additions.append(Part(label, owner, shape, role, owner, rgb))
                additions.append(
                    Part(
                        f"{owner}_added_stock",
                        owner,
                        cand["added_spine"],
                        "added_stock",
                        owner,
                        color,
                    )
                )
                polymer += (
                    cand["cover"].volume + cand["spine"].volume - spine.shape.volume
                )
                hardware += sum(
                    p.volume * 0.00786 for p in cand["screws"].values()
                ) + sum(p.volume * 0.00852 for p in cand["inserts"].values())
                seating = volume(cand["cover"] & cand["spine"])
                sunk = volume((Pos(-1, 0, 0) * cand["cover"]) & cand["spine"])
                mounts.append(
                    dict(
                        side=side,
                        mounts=cand["mounts"],
                        seating_overlap_mm3=seating,
                        sunk_negative_overlap_mm3=sunk,
                        status="pass" if seating < 0.02 and sunk > 0.02 else "fail",
                    )
                )
            result = dict(
                status="fail",
                parts=changed,
                mounts=mounts,
                poses=[],
                service=[],
                polymer_added_g=polymer * c["density_g_cm3"] / 1000,
                nominal_hardware_added_g=hardware,
                inherited_status=retained["status"],
                source=source_name,
            )
            report["robots"][name] = result
            selected = (
                list(poses(robot))
                if full_pose_grid or "poses" not in c
                else list(c["poses"].items())
            )
            result["pose_scope"] = (
                "full 24-pose sampled grid"
                if full_pose_grid or "poses" not in c
                else "explicit bounded design poses; full range remains unqualified"
            )
            for label, pose in selected:
                for key, value in pose.items():
                    joint = next(j for j in robot.joints if j.name == key)
                    if not joint.limits[0] <= value <= joint.limits[1]:
                        raise ValueError(f"Pose outside joint limit: {label}/{key}")
                frames = robot.forward_kinematics(pose)
                targets = [
                    replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                    for p in local
                ]
                placed = [
                    replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                    for p in additions
                ]
                queries = CameraQueries()
                rows = []
                for p in placed:
                    hits = queries.collide(p.shape, targets)
                    nearest = (
                        queries.nearest(
                            p.shape,
                            targets,
                            initial=(c["clearance"], "no part below clearance screen")
                            if clearance_screen_only
                            else None,
                        )
                        if p.role == "upper_arm_guard"
                        else None
                    )
                    rows.append(
                        dict(
                            part=p.name,
                            hits=hits,
                            nearest=nearest,
                            status="fail"
                            if hits or (nearest and nearest[0] < c["clearance"] - 1e-5)
                            else "pass",
                        )
                    )
                # All opposite-owner additions are tested; no same-owner mounting exemption is needed here.
                for p in placed:
                    others = [
                        q for q in placed if q.owner != p.owner and q.name > p.name
                    ]
                    found = queries.collide(p.shape, others)
                    if found:
                        rows.append(
                            dict(
                                part=p.name,
                                hits=found,
                                nearest=None,
                                status="fail",
                                scope="opposite arm additions",
                            )
                        )
                row = dict(
                    name=label,
                    pose_rad=pose,
                    rows=rows,
                    status="pass"
                    if all(r["status"] == "pass" for r in rows)
                    else "fail",
                )
                result["poses"].append(row)
                print(name, label, row["status"], flush=True)
                write(output / "progress.json", report)
            for side, cand in recipes.items():
                owner = f"{side}_upper_arm"
                targets = [p for p in local if p.name != f"{side}_upper_spine"]
                targets += [
                    Part(
                        f"{side}_modified_spine",
                        owner,
                        cand["spine"],
                        "spine",
                        owner,
                        (0.5, 0.5, 0.5),
                    )
                ]
                targets = [
                    replace(p, shape=transform_shape(p.shape, neutral[p.owner]))
                    for p in targets
                ]
                queries = CameraQueries()
                for offset in (0, 2, 10, 40, 70):
                    shape = transform_shape(
                        Pos(offset, 0, 0) * cand["cover"], neutral[owner]
                    )
                    found = queries.collide(shape, targets)
                    result["service"].append(
                        dict(
                            side=side,
                            kind="front_extraction",
                            offset_mm=offset,
                            hits=found,
                            status="fail" if found else "pass",
                        )
                    )
                for key, tool in cand["tools"].items():
                    found = queries.collide(
                        transform_shape(tool, neutral[owner]), targets
                    )
                    result["service"].append(
                        dict(
                            side=side,
                            kind="driver",
                            mount=key,
                            hits=found,
                            status="fail" if found else "pass",
                        )
                    )
            result["status"] = (
                "pass"
                if all(
                    r["status"] == "pass"
                    for r in result["poses"] + result["mounts"] + result["service"]
                )
                else "fail"
            )
        report["guard_checks_status"] = (
            "pass"
            if all(r["status"] == "pass" for r in report["robots"].values())
            else "fail"
        )
        report["not_run"] = [
            "continuous motion",
            "insert retention and print tolerances",
            "strength/drop tests",
            "wiring and pinch protection",
            "assembly mass/dynamics update",
            "production costing",
            "same-owner hardware interface qualification",
        ]
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["output_hashes"] = {
            p.name: sha(p)
            for p in output.iterdir()
            if p.is_file() and p.name not in ("report.json", "progress.json")
        }
        write(output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--spec", type=Path, default=Path("robot/upper-arm-guard-study.yaml")
    )
    parser.add_argument("--clearance-screen-only", action="store_true")
    parser.add_argument("--full-pose-grid", action="store_true")
    args = parser.parse_args()
    run(args.output, args.spec, args.clearance_screen_only, args.full_pose_grid)
