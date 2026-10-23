"""Recompose four changed finger parts into a retained complete robot.

Checks every robot part against the recorded task environment, plus only added
finger material against the other robot parts. Existing source failures remain
visible. This does not qualify old-old self-clearance or a new mass model.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

from build123d import Box, Color, Compound, export_step, import_step
import mujoco
import numpy as np
import yaml

from lab.arm_mechanism import arm_parts
from lab.config import RobotConfig
from lab.forearm_cover import candidate
from lab.leg_mechanism import Part, VENDOR, transform_shape
from lab.robot import make_robot
from tools.cad_identity import geometry_agrees, geometry_record
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.check_forearm_task_scene import frame, scene_clearance
from tools.compose_forearm_enclosure import task_samples, verify_report


def compatible_config(old, new):
    expected = old.model_dump(mode="json")
    actual = new.model_dump(mode="json")
    for key in ("gripper_finger_length_m", "gripper_pad_root_m"):
        expected["dimensions"][key] = actual["dimensions"][key]
    if expected != actual:
        raise ValueError("configuration change exceeds finger length/pad root")


def replacement_parts(robot, spec):
    parts, _ = arm_parts(robot)
    selected = {}
    for p in parts:
        if p.name.endswith("_gripper_finger"):
            selected[p.name] = p
        elif p.name.endswith("_forearm_spine"):
            shape = candidate(p.shape, spec, right=p.name.startswith("right_"))["spine"]
            selected[p.name] = replace(p, shape=shape)
    assert len(selected) == 4
    return selected


def evaluate_motion(robot, combined, added, episode, output):
    """Replay complete scene and added-material checks without exporting CAD."""
    contract = json.loads(json.dumps(robot.interface()))
    report = {}
    model = mujoco.MjModel.from_xml_path(str(episode / "scene.xml"))
    if model.nmesh:
        raise ValueError("this exploration requires mesh-free recorded physics")
    targets = []
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name in ("task_table", "task_block_geom") or name.startswith("tray_"):
            assert model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
            t = frame(model.geom_pos[i], model.geom_quat[i], scalar_first=True)
            targets.append(
                Part(
                    name,
                    model.body(int(model.geom_bodyid[i])).name,
                    transform_shape(Box(*(2000 * model.geom_size[i])), t),
                    "scene",
                    "scene",
                    (0.5,) * 3,
                )
            )
    assert len(targets) == 7
    rows = [
        json.loads(line)
        for line in (episode / "trajectory.jsonl").read_text().splitlines()
    ]
    results = []
    for sample in task_samples(rows, contract):
        scene = rows[sample["source_row"]]["scene_state"]
        if abs(scene["time_s"] - sample["time_s"]) > 1e-9:
            raise ValueError("recorded clocks differ")
        frames = {
            n: frame(b["position_m"], b["quaternion_xyzw"])
            for n, b in scene["bodies"].items()
        }
        fk = robot.forward_kinematics(sample["pose"], enforce_limits=False)
        root = frames["torso"] @ np.linalg.inv(fk["torso"])
        error = max(float(np.max(np.abs(frames[n] - root @ t))) for n, t in fk.items())
        if error > 1e-8:
            raise ValueError("recorded frames differ from FK")
        placed = [
            replace(p, shape=transform_shape(p.shape, frames[p.owner]))
            for p in combined
        ]
        environment = [
            replace(p, shape=transform_shape(p.shape, frames[p.owner])) for p in targets
        ]
        minimum, failures = scene_clearance(placed, environment)
        self_rows = []
        for p in added:
            probe = replace(p, shape=transform_shape(p.shape, frames[p.owner]))
            m, f = scene_clearance(
                [probe], [other for other in placed if other.name != p.name]
            )
            self_rows.append(
                dict(
                    part=p.name,
                    minimum_distance_mm=m[0],
                    closest_pair=m[1:],
                    failures=f,
                    status="fail" if f else "pass",
                )
            )
        results.append(
            dict(
                **sample,
                scene_status="fail" if failures else "pass",
                failures=failures,
                minimum_distance_mm=minimum[0],
                closest_pair=minimum[1:],
                fk_max_matrix_error=error,
                added_material_self_checks=self_rows,
            )
        )
        write(output / "motion-progress.json", results)
        print(
            sample["name"],
            "scene flags",
            len(failures),
            "new-material flags",
            sum(r["status"] == "fail" for r in self_rows),
            flush=True,
        )
    report["samples"] = results
    report["scene_status"] = "fail" if any(s["failures"] for s in results) else "pass"
    report["added_material_self_status"] = (
        "fail"
        if any(
            r["status"] == "fail"
            for s in results
            for r in s["added_material_self_checks"]
        )
        else "pass"
    )
    return report


def run(assembly, episode, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail", scope=__doc__.strip(), builder="distal_grip_composition"
    )
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
        paths += [
            episode / n
            for n in (
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "task.json",
                "scene.xml",
                "trajectory.jsonl",
            )
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        base = verify_report(assembly)
        task_report = verify_report(episode)
        if base.get("assembly_step_roundtrip") != "pass" or base["output_hashes"].get(
            "assembly.step"
        ) != sha(assembly / "assembly.step"):
            raise ValueError("source lacks bound STEP identity")
        old_config = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        new_config = RobotConfig.model_validate_json(
            (episode / "resolved-config.json").read_text()
        )
        compatible_config(old_config, new_config)
        old_robot, robot = make_robot(old_config), make_robot(new_config)
        contract = json.loads((episode / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != contract:
            raise ValueError("episode must use matching explicit legacy allocation")
        if json.loads(json.dumps(old_robot.interface())) != json.loads(
            (assembly / "interfaces.json").read_text()
        ):
            raise ValueError("source interface differs")
        if json.loads((episode / "task.json").read_text())["fixture"] != "supported":
            raise ValueError("supported replay required")
        write(output / "resolved-config.json", new_config.model_dump(mode="json"))
        write(output / "interfaces.json", contract)
        print("importing retained complete STEP", flush=True)
        original = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        if len(original) != base["part_count"]:
            raise ValueError("part count differs")
        neutral = robot.forward_kinematics()
        old_frames = old_robot.forward_kinematics()
        for name in neutral:
            np.testing.assert_allclose(
                neutral[name], old_frames[name], atol=1e-12, rtol=0
            )
        records = {p["name"]: p for p in base["parts"]}
        original_local = {
            p.name: replace(
                p,
                shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner])),
                color=tuple(records[p.name]["color"]),
            )
            for p in original
        }
        spec = yaml.safe_load(Path("robot/forearm-cover.yaml").read_text())
        old_parts = replacement_parts(old_robot, spec)
        new_parts = replacement_parts(robot, spec)
        checks = []
        for name, part in old_parts.items():
            expected, actual = (
                geometry_record(part.shape),
                geometry_record(original_local[name].shape),
            )
            checks.append(
                dict(
                    name=name,
                    expected=expected,
                    actual=actual,
                    status="pass" if geometry_agrees(expected, actual) else "fail",
                )
            )
        write(output / "replacement-source-identity.json", checks)
        if any(c["status"] != "pass" for c in checks):
            raise ValueError(
                "reconstructed original finger/cover stock differs from retained CAD"
            )
        combined = [new_parts.get(p.name, original_local[p.name]) for p in original]
        added = [
            replace(p, shape=p.shape - old_parts[p.name].shape)
            for p in new_parts.values()
        ]
        if any(not p.shape.is_valid or p.shape.volume <= 0 for p in added):
            raise ValueError("invalid added-material probe")
        report.update(
            source_assembly_status=base["status"],
            source_failed_poses=base.get("cover_producer_failed_poses", []),
            episode_source_status=task_report["status"],
            part_count=len(combined),
            replacements=list(new_parts),
            added_material_mm3={p.name: p.shape.volume for p in added},
            net_replacement_volume_delta_mm3=sum(
                p.shape.volume - old_parts[n].shape.volume for n, p in new_parts.items()
            ),
        )
        # Replay before serialization so interrupted CAD exports retain the useful motion result.
        report.update(evaluate_motion(robot, combined, added, episode, output))
        write(output / "motion-report.json", report)
        print("exporting recomposed complete assembly", flush=True)
        world = [
            replace(p, shape=transform_shape(p.shape, neutral[p.owner]))
            for p in combined
        ]
        for p in world:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in world]), output / "assembly.step"
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
            for p in world
        ]
        print("reimporting complete assembly for identity", flush=True)
        restored = {
            p.name: p.shape
            for p in parts_from_step(
                list(import_step(output / "assembly.step").children), inventory
            )
        }
        identity = [
            dict(
                name=p.name,
                expected=geometry_record(p.shape),
                actual=geometry_record(restored[p.name]),
            )
            for p in world
        ]
        write(output / "step-roundtrip.json", identity)
        passed = all(geometry_agrees(r["actual"], r["expected"]) for r in identity)
        report.update(
            parts=inventory,
            assembly_step_roundtrip="pass" if passed else "fail",
            geometry_build_status="pass" if passed else "fail",
            status="pass"
            if passed
            and base["status"] == "pass"
            and report["scene_status"] == "pass"
            and report["added_material_self_status"] == "pass"
            else "fail",
            not_run=[
                "old-old robot self-clearance",
                "new design-pose sweep",
                "continuous/swept clearance",
                "updated mass/inertia/dynamics",
                "strength/harness/physical qualification",
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
                    "geometry_build_status",
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
    for name in ("assembly", "episode", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
