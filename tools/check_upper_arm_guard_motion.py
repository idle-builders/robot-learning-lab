"""Sample recorded joint motion against retained upper-arm guard CAD.

Self-clearance only: root pose, table/object contact, continuous trajectories and
new-guard dynamics remain outside this test. No recorded joint excursion is clipped.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import import_step
from lab.artifact_store import retain_dependency
from lab.config import RobotConfig
from lab.leg_mechanism import Part, transform_shape, overlap
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.check_head_cap_fasteners import transformed_bounds, bounds_gap
from tools.compose_forearm_enclosure import (
    verify_report,
    kinematic_contract,
    task_samples,
)


def match_contract(a, b):
    if kinematic_contract(a) != kinematic_contract(b):
        raise ValueError("Recording and CAD kinematic contracts differ")


def samples_from_task(folder, contract):
    current = json.loads((folder / "interfaces.json").read_text())
    match_contract(contract, current)
    if json.loads((folder / "task.json").read_text())["fixture"] != "supported":
        raise ValueError("Only supported task self-clearance is implemented")
    rows = [
        json.loads(s) for s in (folder / "trajectory.jsonl").read_text().splitlines()
    ]
    return rows, task_samples(rows, contract, interval=0.4)


def evaluate_pose(probes, targets, frames, query, static_cache):
    """Existing conservative-box rejection and native OCCT narrow phase.

    Same-owner pairs are invariant and cached without rounding joint positions.
    Every other pair is recomputed from its exact recorded rigid frames.
    """
    failures = []
    counts = dict(
        pairs=0, containing_bounds_clear=0, exact_queries=0, invariant_cached=0
    )
    for p in probes:
        pbox = query.bounds(p.shape)
        bounds = (np.array(list(pbox.min)), np.array(list(pbox.max)))
        inverse = np.linalg.inv(frames[p.owner])
        margin = 0.5 if p.role == "upper_arm_guard" else 0.0
        selected = list(targets) + [
            t for t in probes if t.owner != p.owner and t.name > p.name
        ]
        for t in selected:
            counts["pairs"] += 1
            key = (p.name, t.name)
            fixed = p.owner == t.owner
            if fixed and key in static_cache:
                counts["invariant_cached"] += 1
                finding = static_cache[key]
            else:
                relative = np.eye(4) if fixed else inverse @ frames[t.owner]
                tbox = transformed_bounds(query.bounds(t.shape), relative)
                if bounds_gap(bounds, tbox) > margin + 1e-7:
                    counts["containing_bounds_clear"] += 1
                    finding = None
                else:
                    counts["exact_queries"] += 1
                    located = t.shape if fixed else transform_shape(t.shape, relative)
                    distance = p.shape.distance_to(located) if margin else None
                    intersection = (
                        overlap(p.shape, located)
                        if not margin or distance < margin
                        else 0.0
                    )
                    finding = (
                        dict(
                            part=p.name,
                            target=t.name,
                            distance_mm=distance,
                            intersection_volume_mm3=intersection,
                            required_gap_mm=margin,
                        )
                        if (
                            intersection > 0.02 or (margin and distance < margin - 1e-5)
                        )
                        else None
                    )
                if fixed:
                    static_cache[key] = finding
            if finding:
                failures.append(finding)
    assert counts["pairs"] == sum(
        counts[k]
        for k in ("containing_bounds_clear", "exact_queries", "invariant_cached")
    )
    return dict(status="fail" if failures else "pass", failures=failures, counts=counts)


def run(name, study, tasks, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        robot=name,
        scope=__doc__,
        source_bindings={},
        dependency_manifests={},
        cases=[],
        physical_status="not_run",
        dynamics_with_guards="not_run",
        clearance_mm=0.5,
    )
    paths = [
        Path(__file__).relative_to(Path.cwd()),
        *Path("lab").glob("*.py"),
        *[
            Path("tools") / p
            for p in (
                "check_arm_episode.py",
                "check_camera_support.py",
                "check_neck_collar.py",
                "check_head_cap_fasteners.py",
                "compose_forearm_enclosure.py",
            )
        ],
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in paths})

    def bind(path):
        report["source_bindings"][str(path)] = sha(path)
        return path

    try:
        guard = verify_report(study)
        report["dependency_manifests"]["guards"] = retain_dependency(study)
        bind(study / "report.json")
        candidate = guard["robots"][name]
        source = Path(candidate["source"])
        original = verify_report(source)
        report["dependency_manifests"]["assembly"] = retain_dependency(source)
        if guard["source_bindings"][str(source / "report.json")] != sha(
            bind(source / "report.json")
        ):
            raise ValueError("Guard uses different source assembly")
        robot = make_robot(
            RobotConfig.model_validate_json(
                bind(source / "resolved-config.json").read_text()
            )
        )
        contract = json.loads(bind(source / "interfaces.json").read_text())
        match_contract(json.loads(json.dumps(robot.interface())), contract)
        neutral = robot.forward_kinematics()
        raw = parts_from_step(
            list(import_step(bind(source / "assembly.step")).children),
            original["parts"],
        )
        local = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner])))
            for p in raw
        ]
        probes = []
        for p in candidate["parts"]:
            shape = import_step(bind(study / p["step"]))
            label, role = p["name"], p["role"]
            if p["replaces_original"]:
                spine = next(q for q in local if q.name == label)
                if overlap(spine.shape, shape) < spine.shape.volume - 0.02:
                    raise ValueError("Guard spine removes original stock")
                shape = shape - spine.shape
                label = label.replace("_upper_spine", "_upper_arm_added_stock")
                role = "added_stock"
            probes.append(
                Part(label, p["owner"], shape, role, p["owner"], tuple(p["color"]))
            )
        report["probes"] = [
            dict(name=p.name, owner=p.owner, role=p.role) for p in probes
        ]
        report["retained_target_count"] = len(local)
        report["inherited_assembly_status"] = original["status"]
        query, cache = CameraQueries(), {}
        balance = Path(f"artifacts/{name}-covered-carriers-balance-v2")
        b = json.loads(bind(balance / "balance-report.json").read_text())
        match_contract(
            contract, json.loads(bind(balance / "interfaces.json").read_text())
        )
        rest = b["controller"]["arm_rest_targets_rad"]
        report["rest_pose"] = dict(
            pose=rest,
            **evaluate_pose(
                probes, local, robot.forward_kinematics(rest), query, cache
            ),
        )
        # Retain recorded task states, including failed trials, at 0.4 s intervals.
        requested = []
        for label in ("lower", "nominal", "upper", "open"):
            folder = tasks / f"{name}-{label}"
            for f in (
                "report.json",
                "interfaces.json",
                "task.json",
                "trajectory.jsonl",
            ):
                bind(folder / f)
            report["dependency_manifests"][f"task-{label}"] = retain_dependency(folder)
            rows, samples = samples_from_task(folder, contract)
            result = json.loads((folder / "report.json").read_text())
            requested.append((label, folder, samples, result["status"], len(rows)))
        for label, folder, samples, status, count in requested:
            case = dict(
                name=label,
                source=str(folder),
                source_status=status,
                source_rows=count,
                sampling_interval_s=0.4,
                selected_rows=[p["source_row"] for p in samples],
                poses=[],
            )
            report["cases"].append(case)
            for sample in samples:
                frames = robot.forward_kinematics(sample["pose"], enforce_limits=False)
                checked = evaluate_pose(probes, local, frames, query, cache)
                excursions = [
                    j.name
                    for j in robot.joints
                    if not j.continuous
                    and not j.limits[0] <= sample["pose"].get(j.name, 0) <= j.limits[1]
                ]
                case["poses"].append(
                    dict(**sample, **checked, joint_limit_excursions=excursions)
                )
                write(output / "progress.json", report)
                print(
                    name,
                    label,
                    sample["name"],
                    checked["status"],
                    len(checked["failures"]),
                    flush=True,
                )
            case["status"] = (
                "fail" if any(p["status"] == "fail" for p in case["poses"]) else "pass"
            )
        report["added_guard_status"] = (
            "pass"
            if report["rest_pose"]["status"] == "pass"
            and all(c["status"] == "pass" for c in report["cases"])
            else "fail"
        )
        report["status"] = (
            "partial" if report["added_guard_status"] == "pass" else "fail"
        )
    except BaseException as exc:
        report["error"] = repr(exc)
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        write(output / "report.json", report)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=("pip", "scout"))
    p.add_argument("--study", type=Path, default=Path("artifacts/upper-arm-guard-v2"))
    p.add_argument(
        "--tasks", type=Path, default=Path("artifacts/covered-carriers-placement-v1")
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.name, a.study, a.tasks, a.output)
