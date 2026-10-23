"""Reuse a completed distal-grip STEP for a new matching task recording.

Reports complete robot-to-scene checks and added-finger-material self checks.
Inherited assembly failures and other untested self-collisions remain distinct.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

from build123d import import_step
import numpy as np
import yaml

from lab.config import RobotConfig
from lab.leg_mechanism import transform_shape, VENDOR
from lab.robot import make_robot
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_distal_grip import (
    compatible_config,
    evaluate_motion,
    replacement_parts,
)
from tools.compose_forearm_enclosure import verify_report


def run(assembly, baseline, episode, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__.strip())
    try:
        paths = [
            *Path("lab").glob("*.py"),
            *Path("tools").glob("*.py"),
            Path("robot/forearm-cover.yaml"),
            Path("robot/actuator-catalog.yaml"),
            VENDOR,
        ]
        paths += [
            assembly / n
            for n in (
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "assembly.step",
            )
        ]
        paths += [baseline / n for n in ("report.json", "resolved-config.json")]
        paths += [
            episode / n
            for n in (
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "trajectory.jsonl",
                "scene.xml",
                "task.json",
            )
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        full = verify_report(assembly)
        if full.get("assembly_step_roundtrip") != "pass" or full["output_hashes"].get(
            "assembly.step"
        ) != sha(assembly / "assembly.step"):
            raise ValueError("complete bound STEP required")
        for n in ("report.json", "resolved-config.json"):
            if full["source_hashes"].get(str(baseline / n)) != sha(baseline / n):
                raise ValueError("baseline identity differs")
        cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        episode_cfg = RobotConfig.model_validate_json(
            (episode / "resolved-config.json").read_text()
        )
        if cfg != episode_cfg:
            raise ValueError("episode geometry differs")
        robot = make_robot(cfg)
        contract = json.loads(json.dumps(robot.interface()))
        if contract != json.loads(
            (episode / "interfaces.json").read_text()
        ) or contract != json.loads((assembly / "interfaces.json").read_text()):
            raise ValueError("interface/mass differs")
        if json.loads((episode / "task.json").read_text())["fixture"] != "supported":
            raise ValueError("only supported task replay implemented")
        oldcfg = RobotConfig.model_validate_json(
            (baseline / "resolved-config.json").read_text()
        )
        compatible_config(oldcfg, cfg)
        old = make_robot(oldcfg)
        spec = yaml.safe_load(Path("robot/forearm-cover.yaml").read_text())
        old_parts, new_parts = (
            replacement_parts(old, spec),
            replacement_parts(robot, spec),
        )
        print("importing complete CAD for new motion", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), full["parts"]
        )
        if len(parts) != full["part_count"]:
            raise ValueError("inventory differs")
        neutral = robot.forward_kinematics()
        local = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner])))
            for p in parts
        ]
        by_name = {p.name: p for p in local}
        comparisons = [
            dict(
                name=n,
                expected=geometry_record(p.shape),
                actual=geometry_record(by_name[n].shape),
            )
            for n, p in new_parts.items()
        ]
        write(output / "replacement-identity.json", comparisons)
        if not all(geometry_agrees(c["actual"], c["expected"]) for c in comparisons):
            raise ValueError("replacement recipe differs from retained CAD")
        added = [
            replace(p, shape=p.shape - old_parts[n].shape) for n, p in new_parts.items()
        ]
        if any(not p.shape.is_valid or p.shape.volume <= 0 for p in added):
            raise ValueError("invalid added-material probe")
        report.update(evaluate_motion(robot, local, added, episode, output))
        report.update(
            status="pass"
            if report["scene_status"] == "pass"
            and report["added_material_self_status"] == "pass"
            else "fail",
            source_assembly_status=full["status"],
            assembly_part_count=len(parts),
            part_names=[p.name for p in local],
            episode_source_status=json.loads((episode / "report.json").read_text())[
                "status"
            ],
            not_run=[
                "old-old robot self-clearance",
                "continuous/swept clearance",
                "updated mass/dynamics",
                "physical qualification",
            ],
        )
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name not in ("report.json", "report.sha256")
    }
    write(output / "report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "error",
                    "scene_status",
                    "added_material_self_status",
                )
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assembly", "baseline", "episode", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
