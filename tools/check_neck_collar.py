"""Compose a fixed lower-neck collar; sample added-part fit and conditional service."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from build123d import BoundBox, Color, Compound, Cylinder, Pos, export_step, import_step
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig, load_config
from lab.leg_mechanism import (
    Part,
    bounds_overlap,
    overlap,
    render_core,
    transform_shape,
)
from lab.mechanical import service_parts
from lab.neck_collar import collar_parts, centers, hardware, interface, TAB_THICKNESS_MM
from lab.robot import make_robot
from lab.logic_mount import mount_bom, SCREW_HEAD_HEIGHT_MM
from lab.power_pack import pack_shapes
from lab.power_system import load_system
from lab.styling import PALETTES
from tools.check_arm_episode import parts_from_step

THRESHOLD = 0.02


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class ExactQueries:
    """Cache unchanged shape bounds; retain references to prevent ID reuse.

    Shapes must not be mutated after querying. Every posed/translated shape is
    a new object. Narrow-phase geometry and overlap threshold remain unchanged.
    """

    def __init__(self):
        self._bounds = {}

    def bounds(self, shape):
        key = id(shape)
        if key not in self._bounds:
            # Broad-phase bounds must enclose the surface. AddOptimal can
            # underbound scaled spline ellipsoids; cached mesh extrema also
            # cannot prove enclosure. A looser conservative box only increases
            # narrow-phase work, never rejects a potentially colliding pair.
            box = Bnd_Box()
            BRepBndLib.Add_s(shape.wrapped, box, False)
            if box.IsVoid():
                raise ValueError("cannot bound an empty collision shape")
            self._bounds[key] = (shape, BoundBox(box))
        return self._bounds[key][1]

    def collide(self, shape, targets):
        hits = []
        box = self.bounds(shape)
        for p in targets:
            if bounds_overlap(box, self.bounds(p.shape)):
                volume = overlap(shape, p.shape)
                if volume > THRESHOLD:
                    hits.append({"part": p.name, "overlap_mm3": volume})
        return hits


def validate_source(assembly, report, robot):
    if report.get("status") != "pass":
        raise ValueError("requires passed source assembly")
    for name, expected in report["source_hashes"].items():
        if sha(assembly / "source" / name) != expected:
            raise ValueError(f"changed retained source snapshot: {name}")
    for name in ["assembly.step", "front-shell.step", "interfaces.json"]:
        if sha(assembly / name) != report["output_hashes"][name]:
            raise ValueError(f"changed retained assembly output: {name}")
    if sha(assembly / "resolved-config.json") != report["resolved_config_sha256"]:
        raise ValueError("changed assembly configuration attestation")
    old = make_robot(
        RobotConfig.model_validate_json((assembly / "resolved-config.json").read_text())
    )
    if (
        not robot.config.dimensions.neck_collar_enabled
        or old.config.dimensions.neck_collar_enabled
    ):
        raise ValueError("requires enabled collar variant and disabled source")
    a, b = robot.config.model_dump(), old.config.model_dump()
    for config in [a, b]:
        config.pop("name")
        for key in [
            "neck_collar_enabled",
            "neck_collar_radius_m",
            "neck_collar_height_m",
            "neck_collar_wall_m",
        ]:
            config["dimensions"].pop(key)
    if a != b:
        raise ValueError("variant changes parameters outside collar configuration")
    if json.loads(json.dumps(old.interface())) != json.loads(
        (assembly / "interfaces.json").read_text()
    ):
        raise ValueError("retained source interface differs from its generated tree")
    return old


def episode_poses(episode, robot):
    contract = json.loads((episode / "interfaces.json").read_text())
    current = robot.interface()
    # Reference poses need identical joint order, frames and actuator/joint contracts;
    # different mass and collar visuals do not change these kinematic coordinates.
    if contract["joint_order"] != current["joint_order"] or contract[
        "joints"
    ] != json.loads(json.dumps(current["joints"])):
        raise ValueError("episode joint interface differs")

    def frames(links):
        return [{k: p[k] for k in ["name", "parent", "pos", "joint"]} for p in links]

    if frames(contract["links"]) != json.loads(json.dumps(frames(current["links"]))):
        raise ValueError("episode link frames differ")
    rows = {
        row["phase"]: row
        for row in map(
            json.loads, (episode / "trajectory.jsonl").read_text().splitlines()
        )
    }
    result = []
    for phase in ["approach", "lift", "lower", "retreat"]:
        state = rows[phase]["joint_state"]
        if state["names"] != current["joint_order"]:
            raise ValueError("trajectory joint ordering differs")
        pose = dict(zip(state["names"], state["position_rad"], strict=True))
        robot.forward_kinematics(pose, enforce_limits=False)
        result.append((phase, pose))
    return result


def run(variant, assembly, output, episode=None):
    variant, assembly, output = map(Path, (variant, assembly, output))
    episode = Path(episode) if episode else None
    queries = ExactQueries()
    collide = queries.collide
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "builder": "neck_collar_composition",
        "scope": "added collar/posts/hardware vs retained body; sampled motion and conditional removal, not full-body or pinch qualification",
    }

    def checkpoint(message):
        write(output / "progress.json", report)
        print(message, flush=True)

    try:
        source = json.loads((assembly / "report.json").read_text())
        robot = make_robot(load_config(variant))
        d = robot.config.dimensions
        old = validate_source(assembly, source, robot)
        sources = [
            variant,
            Path("robot/base.yaml"),
            Path("robot/fastener-catalog.yaml"),
            Path("electronics/power-system.yaml"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_neck_collar.py"),
            *Path("lab").glob("*.py"),
        ]
        sources += [
            assembly / n
            for n in [
                "report.json",
                "assembly.step",
                "front-shell.step",
                "interfaces.json",
                "resolved-config.json",
            ]
        ]
        if episode:
            sources += [
                episode / n
                for n in [
                    "trajectory.jsonl",
                    "interfaces.json",
                    "task.json",
                    "report.json",
                ]
            ]
        report["source_hashes"] = {}
        for path in sources:
            rel = path.absolute().relative_to(Path.cwd().resolve())
            destination = output / "source" / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, destination)
            report["source_hashes"][str(rel)] = sha(destination)
        (output / "resolved-config.json").write_text(
            robot.config.model_dump_json(indent=2)
        )
        report["resolved_config_sha256"] = sha(output / "resolved-config.json")
        write(output / "interfaces.json", robot.interface())
        write(output / "neck-collar-interface.json", interface(d))
        report["mount_interface"] = interface(d)
        report["mount_bom"] = [mount_bom(n, per_robot=4) for n in [1000, 5000]]
        checkpoint("Importing source body")
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), source["parts"]
        )
        if len(parts) != 152:
            raise ValueError("expected 152 source CAD groups")
        frames = robot.forward_kinematics()
        neutral = old.forward_kinematics()
        for name, frame in frames.items():
            np.testing.assert_allclose(frame, neutral[name], atol=1e-12)

        def world(shape):
            return transform_shape(shape, frames["torso"])

        g = robot.links[0].geometry[0]
        retained = next(p.shape for p in parts if p.name == "torso_front")
        regen = world(
            Pos(0, 0, g.pos[2] * 1000) * service_parts(g, old.config.dimensions)[0]
        )
        a, b = Compound(retained.solids()), Compound(regen.solids())
        report["retained_front_symmetric_difference_mm3"] = sum(
            max(0, s.volume) if s is not None else 0 for s in [a - b, b - a]
        )
        if report["retained_front_symmetric_difference_mm3"] > THRESHOLD:
            raise ValueError("retained front differs geometrically from source recipe")
        front = world(Pos(0, 0, g.pos[2] * 1000) * service_parts(g, d)[0])
        added_stock = front - retained
        report["post_added_volume_mm3"] = added_stock.volume
        parts = [
            replace(p, shape=front) if p.name == "torso_front" else p for p in parts
        ]
        halves = collar_parts(d)
        screws, inserts = hardware(d)
        accent = PALETTES[robot.config.appearance][1]
        extras = [
            Part(
                "neck_collar_" + name,
                "torso",
                world(shape),
                "shell",
                "neck_collar",
                accent,
            )
            for name, shape in halves.items()
        ]
        for kind, shapes, role, color in [
            ("screw", screws, "screw", (0.4, 0.42, 0.44)),
            ("insert", inserts, "insert", (0.8, 0.6, 0.2)),
        ]:
            extras += [
                Part(
                    f"neck_collar_{kind}_{i}",
                    "torso",
                    world(shape),
                    role,
                    "neck_collar",
                    color,
                )
                for i, shape in enumerate(shapes)
            ]
        without_front = [p for p in parts if p.name != "torso_front"]
        report["post_collisions"] = collide(added_stock, without_front)
        report["installed_collisions"] = {
            p.name: collide(p.shape, without_front if p.role == "insert" else parts)
            for p in extras
        }
        report["insert_front_interference_mm3"] = [
            overlap(p.shape, front) for p in extras if p.role == "insert"
        ]
        report["extra_part_collisions"] = []
        for i, a in enumerate(extras):
            hits = collide(a.shape, extras[i + 1 :])
            report["extra_part_collisions"] += [
                {"part": a.name, "other": r["part"], "overlap_mm3": r["overlap_mm3"]}
                for r in hits
            ]
        pack, _ = pack_shapes(load_system())
        if (
            sha("electronics/power-system.yaml")
            != source["source_hashes"]["electronics/power-system.yaml"]
        ):
            raise ValueError("power allocation differs from retained body")
        report["battery_collisions"] = collide(
            world(pack),
            extras
            + [Part("new_posts", "torso", added_stock, "stock", "neck_collar", accent)],
        )
        checkpoint("Nominal mounting checked")
        probes = extras + [
            Part("new_posts", "torso", added_stock, "stock", "neck_collar", accent)
        ]
        poses = [
            (
                f"neck-{y:.3f}-{p:.3f}",
                {"neck_yaw_joint": float(y), "head_joint": float(p)},
            )
            for y in np.linspace(-1.2, 1.2, 9)
            for p in np.linspace(-0.5, 0.5, 5)
        ]
        if episode:
            episode_report = json.loads((episode / "report.json").read_text())
            report["reference_episode"] = {
                "path": str(episode),
                "status": episode_report.get("status"),
                "fixture": json.loads((episode / "task.json").read_text()).get(
                    "fixture"
                ),
                "scope": "kinematic phase endpoints only; no task or environment-contact qualification",
            }
            poses += episode_poses(episode, robot)
        cache = {}
        located_cache = {}
        motion = []
        for pose_name, pose in poses:
            posed = robot.forward_kinematics(pose, enforce_limits=False)
            failures = []
            checks = 0
            for p in parts:
                delta = posed[p.owner] @ np.linalg.inv(neutral[p.owner])
                key = (p.name, tuple(delta.round(10).flatten()))
                if key not in located_cache:
                    located_cache[key] = transform_shape(p.shape, delta)
                shape = located_cache[key]
                bb = queries.bounds(shape)
                for probe in probes:
                    if p.name == "torso_front" and probe.role in ["insert", "stock"]:
                        continue
                    pair = (probe.name, *key)
                    if pair not in cache:
                        cache[pair] = (
                            overlap(probe.shape, shape)
                            if bounds_overlap(queries.bounds(probe.shape), bb)
                            else 0
                        )
                        checks += 1
                    if cache[pair] > THRESHOLD:
                        failures.append(
                            {"parts": [probe.name, p.name], "overlap_mm3": cache[pair]}
                        )
            motion.append(
                {
                    "name": pose_name,
                    "pose_rad": pose,
                    "status": "fail" if failures else "pass",
                    "collisions": failures,
                    "new_pair_checks": checks,
                    "joint_limit_excursions": [
                        {
                            "joint": j.name,
                            "position_rad": pose[j.name],
                            "limits_rad": j.limits,
                        }
                        for j in robot.joints
                        if j.name in pose
                        and not j.continuous
                        and not j.limits[0] <= pose[j.name] <= j.limits[1]
                    ],
                }
            )
            if len(motion) % 9 == 0 or len(motion) == len(poses):
                report["motion"] = motion
                checkpoint(f"Motion samples {len(motion)}/{len(poses)}")
        report["motion"] = motion
        removed = [
            p.name
            for p in parts
            if p.name == "head_shell" or p.name.startswith("head_bridge_bolt_")
        ]
        service = [p for p in parts if p.name not in removed]
        report["service_removed_parts"] = removed
        report["service_prerequisites"] = [
            "Power isolated; head shell/face/camera unplugged and removed through head-bridge fasteners",
            "Pitch horn/yoke and both servo clamps stay installed",
            "Head-shell removal path, harness and tool socket engagement are not proved",
        ]
        report["tool_envelope"] = {
            "radius_mm": 2,
            "length_mm": 80,
            "scope": "straight driver-shaft allocation, not selected tool/handle or engagement model",
        }
        withdrawals = []
        for i, (x, y, z) in enumerate(centers(d)):
            own = f"neck_collar_screw_{i}"
            targets = service + [p for p in extras if p.name != own]
            for dist in np.linspace(0, 7, 8):
                shifted = world(Pos(0, 0, float(dist)) * screws[i])
                tool = world(
                    Pos(
                        x,
                        y,
                        z + TAB_THICKNESS_MM + SCREW_HEAD_HEIGHT_MM + float(dist) + 40,
                    )
                    * Cylinder(2, 80)
                )
                hits = collide(shifted, targets) + collide(tool, targets)
                withdrawals.append(
                    {"screw": i, "withdrawal_mm": float(dist), "collisions": hits}
                )
        report["screw_withdrawal"] = withdrawals
        removals = []
        for index, sign in [(0, 1), (1, -1)]:
            targets = (
                service
                + [extras[1 - index]]
                + [p for p in extras if p.role == "insert"]
            )
            path = [(0.0, float(z)) for z in np.linspace(0, 3, 7)] + [
                (sign * float(x), 3.0) for x in np.linspace(5, 65, 13)
            ]
            for x, z in path:
                shape = Pos(x, 0, z) * extras[index].shape
                removals.append(
                    {
                        "half": extras[index].name,
                        "translation_x_mm": x,
                        "translation_z_mm": z,
                        "collisions": collide(shape, targets),
                    }
                )
        report["half_removal"] = removals
        fixed_case = next(p for p in parts if p.name == "neck_yaw_housing")
        negative = []
        for dist in [20, 30, 40, 45, 50]:
            volume = overlap(Pos(-dist, 0, 0) * extras[0].shape, fixed_case.shape)
            negative.append(
                {
                    "translation_x_mm": -dist,
                    "target": fixed_case.name,
                    "overlap_mm3": volume,
                }
            )
            if volume > THRESHOLD:
                break
        report["negative_control"] = negative
        nominal_ok = (
            not report["post_collisions"]
            and not any(report["installed_collisions"].values())
            and not report["extra_part_collisions"]
            and not report["battery_collisions"]
            and all(v > THRESHOLD for v in report["insert_front_interference_mm3"])
        )
        motion_ok = all(row["status"] == "pass" for row in motion)
        service_ok = all(not row["collisions"] for row in withdrawals + removals)
        negative_ok = any(row["overlap_mm3"] > THRESHOLD for row in negative)
        report.update(
            installation_status="pass" if nominal_ok else "fail",
            motion_status="pass" if motion_ok else "fail",
            service_status="pass" if service_ok else "fail",
            status="pass"
            if nominal_ok and motion_ok and service_ok and negative_ok
            else "fail",
        )
        report["not_run"] = [
            "complete continuous/coupled clearance",
            "old-old part pairs",
            "pinch/finger protection",
            "actual harness and upper-neck enclosure",
            "material/process/strength/heat insertion/tolerances",
            "head-shell prerequisite disassembly",
            "task dynamics with new mass",
            "complete production cost",
        ]
        report["parts"] = [
            {
                "name": p.name,
                "owner": p.owner,
                "role": p.role,
                "module": p.module,
                "volume_mm3": p.shape.volume,
            }
            for p in parts + extras
        ]
        report["part_count"] = len(report["parts"])
        checkpoint("Exporting 162-group collar assembly")
        for p in parts + extras:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in parts + extras]),
            output / "assembly.step",
        )
        # Export a standalone topology, detached from the assembly tree above.
        export_step(Compound(front.solids()), output / "front-shell.step")
        shown = parts + extras
        render_core(shown, {p.name: p.shape for p in shown}, output, robot)
        detail = output / "collar-detail"
        detail.mkdir()
        shown = [
            replace(p, color=(0.8, 0.8, 0.75, 0.2))
            for p in parts
            if p.name == "torso_front"
        ] + [
            replace(p, color=(*accent[:3], 0.35)) if p.role == "shell" else p
            for p in extras
        ]
        render_core(shown, {p.name: p.shape for p in shown}, detail, robot)
        report["output_hashes"] = {
            name: sha(output / name)
            for name in [
                "assembly.step",
                "front-shell.step",
                "interfaces.json",
                "neck-collar-interface.json",
                "mechanism.png",
                "collar-detail/mechanism.png",
            ]
        }
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    for name in ["report.json", "arm-report.json"]:
        write(output / name, report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "installation_status",
                    "motion_status",
                    "service_status",
                    "error",
                ]
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", type=Path)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args()
    raise SystemExit(
        0
        if run(args.variant, args.assembly, args.output, args.episode)["status"]
        == "pass"
        else 1
    )
