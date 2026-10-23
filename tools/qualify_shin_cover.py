"""Supplement shin-cover study with mutual interfaces and asymmetric poses."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback
import numpy as np
from build123d import Pos, import_step
from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.robot import make_robot
from lab.shin_cover import cylinder_x
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.check_neck_collar import ExactQueries
from tools.compose_forearm_enclosure import verify_report
from tools.study_shin_cover import hits

POSES = [
    ("knees_negative_limit", dict(left_shin_joint=-0.05, right_shin_joint=-0.05)),
    (
        "left_forward_fold_right_back",
        dict(
            left_thigh_joint=1.2,
            left_shin_joint=1.8,
            right_thigh_joint=-1.2,
            right_shin_joint=-0.05,
        ),
    ),
    (
        "right_forward_fold_left_back",
        dict(
            right_thigh_joint=1.2,
            right_shin_joint=1.8,
            left_thigh_joint=-1.2,
            left_shin_joint=-0.05,
        ),
    ),
    ("one_knee_folded", dict(left_shin_joint=1.8, right_shin_joint=0)),
]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        physical_status="not_run",
        robots={},
        source_bindings={},
    )
    paths = [
        Path(__file__).relative_to(Path.cwd()),
        Path("tools/study_shin_cover.py"),
        Path("tools/check_neck_collar.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/compose_forearm_enclosure.py"),
        Path("tools/check_camera_support.py"),
        *Path("lab").glob("*.py"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in paths})

    def bind(p):
        report["source_bindings"][str(p)] = sha(p)
        return p

    study = Path("artifacts/shin-cover-study-v1")
    source = verify_report(study)
    bind(study / "report.json")
    try:
        for name, result in source["robots"].items():
            assembly = Path(f"artifacts/{name}-integral-head-composed-v2")
            baseline = verify_report(assembly)
            for filename in (
                "report.json",
                "assembly.step",
                "resolved-config.json",
                "interfaces.json",
            ):
                p = bind(assembly / filename)
                if sha(p) != source["source_bindings"][str(p)]:
                    raise ValueError("Assembly input identity changed")
            robot = make_robot(
                RobotConfig.model_validate_json(
                    (assembly / "resolved-config.json").read_text()
                )
            )
            neutral = robot.forward_kinematics({})
            raw = parts_from_step(
                list(import_step(assembly / "assembly.step").children),
                baseline["parts"],
            )
            original = [
                replace(
                    p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner]))
                )
                for p in raw
            ]
            proposed = [
                Part(
                    p["name"],
                    p["owner"],
                    import_step(bind(study / p["step"])),
                    p["role"],
                    p["owner"],
                    tuple(p["color"]),
                )
                for p in result["parts"]
            ]
            lookup = {p.name: p for p in proposed}
            old = {p.name: p for p in original}
            additions = [
                replace(
                    p,
                    name=p.name.replace("_spine", "_added_spine_stock"),
                    shape=p.shape - old[p.name].shape,
                )
                if p.role == "spine"
                else p
                for p in proposed
            ]
            rows = []
            for label, pose in POSES:
                frames = robot.forward_kinematics(pose)
                targets = [
                    replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                    for p in original
                ]
                checks = []
                query = ExactQueries()
                for p in additions:
                    found, n = hits(
                        transform_shape(p.shape, frames[p.owner]), targets, query
                    )
                    checks.append(
                        dict(
                            part=p.name,
                            hits=found,
                            pair_count=len(targets),
                            exact_pairs=n,
                        )
                    )
                rows.append(
                    dict(
                        name=label,
                        pose_rad=pose,
                        rows=checks,
                        status="fail" if any(r["hits"] for r in checks) else "pass",
                    )
                )
                print(name, label, rows[-1]["status"], flush=True)
            mutual = []
            expected = []
            # All explicit poses are also checked for candidate-to-candidate fit.
            for pose in [*result["poses"], *rows]:
                frames = robot.forward_kinematics(pose["pose_rad"])
                query = ExactQueries()
                parts = [
                    replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                    for p in proposed
                ]
                for i, a in enumerate(parts):
                    for b in parts[i + 1 :]:
                        intentional = a.owner == b.owner and {a.role, b.role} == {
                            "insert",
                            "spine",
                        }
                        found, n = hits(a.shape, [b], query)
                        record = dict(
                            pose=pose["name"],
                            part_a=a.name,
                            part_b=b.name,
                            hits=found,
                            exact_pairs=n,
                        )
                        if intentional:
                            record["contact_present"] = bool(found)
                            expected.append(record)
                        else:
                            mutual.append(record)
            service = []
            installed = [p for p in original if p.name not in lookup] + proposed
            query = ExactQueries()
            for side in ("left", "right"):
                owner = f"{side}_shin"
                cover = lookup[f"{owner}_cover"]
                remaining = [
                    p
                    for p in installed
                    if p.name != cover.name
                    and not (
                        p.owner == owner and p.name.startswith(f"{owner}_cover_screw_")
                    )
                ]
                targets = [
                    replace(p, shape=transform_shape(p.shape, neutral[p.owner]))
                    for p in remaining
                ]
                for dx in (0, 5, 10, 20, 35, 50, 70):
                    found, n = hits(
                        transform_shape(Pos(dx, 0, 0) * cover.shape, neutral[owner]),
                        targets,
                        query,
                    )
                    service.append(
                        dict(
                            side=side,
                            kind="cover_removal_after_own_screws_removed",
                            distance_mm=dx,
                            hits=found,
                            exact_pairs=n,
                        )
                    )
                targets = [
                    replace(p, shape=transform_shape(p.shape, neutral[p.owner]))
                    for p in installed
                ]
                mounts = next(m for m in result["mount_checks"] if m["side"] == side)[
                    "mounts"
                ]
                for mount in mounts:
                    # Same declared tool envelope as the source recipe (not a vendor tool).
                    for dx in (0, 15, 35):
                        tool = cylinder_x(
                            32 + 2.5 + 15 + dx, mount["y_mm"], mount["z_mm"], 2.5, 30
                        )
                        found, n = hits(
                            transform_shape(tool, neutral[owner]), targets, query
                        )
                        service.append(
                            dict(
                                side=side,
                                kind="driver_withdrawal",
                                mount=mount,
                                distance_mm=dx,
                                hits=found,
                                exact_pairs=n,
                            )
                        )
            pass_checks = (
                all(p["status"] == "pass" for p in rows)
                and all(not p["hits"] for p in [*mutual, *service])
                and all(p["contact_present"] for p in expected)
            )
            report["robots"][name] = dict(
                status="partial" if pass_checks else "fail",
                poses=rows,
                mutual=mutual,
                intentional_insert_pilot_interference=expected,
                service=service,
                interfaces="Heat-insert envelope intentionally overlaps polymer pilot before installation. This is not rigid-part fit or retention evidence.",
            )
            write(
                output / "progress.json",
                dict(robot=name, status=report["robots"][name]["status"]),
            )
        report.update(
            status="partial"
            if all(r["status"] == "partial" for r in report["robots"].values())
            else "fail",
            checked_geometry_status="pass"
            if all(r["status"] == "partial" for r in report["robots"].values())
            else "fail",
            whole_robot_status="fail",
            limitations=[
                "Ten finite poses, not continuous or complete Cartesian joint grid",
                "Finite neutral removal/tool samples; no hands, cable or original M3 service qualification",
                "No strength, print fit, thermal or dynamics qualification",
            ],
        )
    except Exception as error:
        report.update(error=str(error), traceback=traceback.format_exc())
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file()
    }
    write(output / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(a.output)
    print(json.dumps(dict(status=r["status"], error=r.get("error"))))
