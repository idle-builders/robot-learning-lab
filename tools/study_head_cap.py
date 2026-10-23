"""Source-bound feasibility check for recovered, unmounted lower-head wings."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback

import numpy as np
from build123d import Pos, export_step, import_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.head_cap import shell_wings
from lab.leg_mechanism import Part, overlap, transform_shape
from lab.head_service import pi_reservation
from tools.study_neck_skirt import recorded_poses
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.cad_identity import geometry_record, geometry_agrees


def run(name, output, depth_mm=None, apply_relief=False):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope="Unmounted recovered lower-head shell stock; sampled fit feasibility, not attachment, service or physical qualification",
    )
    try:
        report["source_hashes"] = copy_source(
            output,
            {
                str(p): p
                for p in [
                    *Path("lab").glob("*.py"),
                    *Path("tools").glob("*.py"),
                    Path("robot/base.yaml"),
                ]
            },
        )
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        episode = Path(f"artifacts/{name}-distal-grip-park-task-v1")
        source = verify_report(assembly)
        verify_report(episode)
        report["source_bindings"] = {
            str(p): sha(p)
            for p in [
                assembly / "report.json",
                assembly / "assembly.step",
                assembly / "resolved-config.json",
                episode / "report.json",
                episode / "trajectory.jsonl",
                episode / "interfaces.json",
            ]
        }
        report["inherited_assembly_status"] = source["status"]
        cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        print(name, "importing complete assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), source["parts"]
        )
        head_targets = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral["head"])))
            for p in parts
            if p.owner == "head"
        ]
        if cfg.head_service.pi_mount_enabled:
            head_targets.append(
                Part(
                    "pi_component_reservation",
                    "head",
                    pi_reservation(cfg.head_service),
                    "reservation",
                    "head_service",
                    (1, 0, 0),
                )
            )
        report["parts"] = []
        covers = shell_wings(robot, depth_mm=depth_mm)
        envelopes = shell_wings(robot, depth_mm=depth_mm, margin_mm=1.5)
        if apply_relief:
            if depth_mm != 8:
                raise ValueError("Local relief currently requires the 8 mm seed")
            from tools.head_cap_relief import relieve

            covers, envelopes, report["relief"] = relieve(
                covers, envelopes, robot, parts, name
            )
        for key, shape in covers.items():
            if not shape.solids() or not shape.is_valid:
                raise ValueError(f"{key} relief consumed or invalidated the cap")
            if not envelopes[key].solids() or not envelopes[key].is_valid:
                raise ValueError(f"{key} clearance envelope empty or invalid")
            record = geometry_record(shape)
            if (
                not record["valid"]
                or record["solids"] != 1
                or record["volume_mm3"] <= 0
            ):
                raise ValueError(f"{key} disconnected, empty or invalid")
            export_step(shape, output / f"{key}.step")
            if not geometry_agrees(
                record, geometry_record(import_step(output / f"{key}.step"))
            ):
                raise ValueError(f"{key} STEP round trip differs")
            residual = shape.volume - overlap(shape, envelopes[key])
            if residual > 0.02:
                raise ValueError(f"{key} clearance envelope omits stock")
            hits = CameraQueries().collide(shape, head_targets)
            report["parts"].append(
                dict(
                    name=key,
                    owner="head",
                    geometry=record,
                    geometry_status="pass",
                    clearance_envelope_missing_mm3=max(0, residual),
                    clearance_envelope_valid=True,
                    clearance_envelope_solids=len(envelopes[key].solids()),
                    nominal_head_status="fail" if hits else "pass",
                    nominal_head_collisions=hits,
                )
            )
        poses = [
            (
                f"neck-{y:.2f}-{p:.2f}",
                {"neck_yaw_joint": float(y), "head_joint": float(p)},
            )
            for y in np.linspace(-1.2, 1.2, 5)
            for p in np.linspace(-0.5, 0.5, 5)
        ] + recorded_poses(episode, robot)
        report.update(
            depth_mm=depth_mm,
            seam_mm=0.3,
            moving_clearance_mm=1.5,
            motion=[],
            requested_pose_count=len(poses),
            mounting_status="not_run",
            removal_status="not_run",
            physical_status="not_run",
            mass_and_cost_status="not_run",
            robot_geometry_changed=False,
        )
        nonhead = [p for p in parts if p.owner != "head"]
        for index, (label, pose) in enumerate(poses):
            fk = robot.forward_kinematics(pose, enforce_limits=False)
            located = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape,
                        np.linalg.inv(fk["head"])
                        @ fk[p.owner]
                        @ np.linalg.inv(neutral[p.owner]),
                    ),
                )
                for p in nonhead
            ]
            q = CameraQueries()
            hits, margins = [], []
            for key, shape in covers.items():
                hits += [dict(cover=key, **h) for h in q.collide(shape, located)]
                margins += [
                    dict(cover=key, **h) for h in q.collide(envelopes[key], located)
                ]
            report["motion"].append(
                dict(
                    name=label,
                    pose_rad=pose,
                    status="fail" if hits or margins else "pass",
                    collisions=hits,
                    conservative_clearance_collisions=margins,
                )
            )
            write(output / "progress.json", report)
            print(
                name,
                index + 1,
                label,
                "hits",
                len(hits),
                "margin",
                len(margins),
                flush=True,
            )
        # A small misplaced wing must enter adjacent retained stock. The old
        # 20 mm displacement jumped past the shell in three retained cases.
        probe = Pos(0, 0, 1) * covers["head_lower_front_left"]
        bad = CameraQueries().collide(
            probe, [p for p in head_targets if p.name == "head_shell"]
        )
        report["negative_control"] = dict(
            status="pass" if bad else "fail",
            displacement_head_mm=[0, 0, 1],
            collisions=bad,
        )
        fit_pass = (
            all(p["nominal_head_status"] == "pass" for p in report["parts"])
            and all(p["status"] == "pass" for p in report["motion"])
            and bool(bad)
        )
        report["fit_status"] = "pass" if fit_pass else "fail"
        report["status"] = "partial" if fit_pass else "fail"
        report["output_hashes"] = {p.name: sha(p) for p in output.glob("*.step")}
        report["output_hashes"]["resolved-config.json"] = sha(
            output / "resolved-config.json"
        )
        write(output / "report.json", report)
        return report
    except Exception as exc:
        report.update(status="fail", error=str(exc), traceback=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=["pip", "scout"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-mm", type=float)
    parser.add_argument("--relieve", action="store_true")
    args = parser.parse_args()
    result = run(args.name, args.output, args.depth_mm, args.relieve)
    print(result["status"], flush=True)
    raise SystemExit(0 if result["status"] == "partial" else 1)
