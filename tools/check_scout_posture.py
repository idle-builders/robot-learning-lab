"""Check sampled changed-arm/body clearance for Scout's covered rest posture.

The same-side arm subtree and upper-arm/shoulder mating owner are excluded.
This is a bounded changed-posture screen, not full robot collision qualification.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback
import numpy as np
from build123d import import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(output, samples=None):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        physical_status="not_run",
        rows=[],
        source_bindings={},
    )
    files = [
        Path(__file__).relative_to(Path.cwd()),
        *Path("lab").glob("*.py"),
        Path("tools/check_camera_support.py"),
        Path("tools/check_neck_collar.py"),
        Path("tools/check_arm_episode.py"),
        Path("tools/compose_forearm_enclosure.py"),
    ]
    report["source_hashes"] = copy_source(output, {str(p): p for p in files})
    try:
        assembly = Path("artifacts/scout-shin-cover-composed-v2")
        selection = Path("artifacts/scout-posture-selection-v1")
        base = verify_report(assembly)
        selected = verify_report(selection)
        for p in [
            assembly / "report.json",
            assembly / "assembly.step",
            assembly / "resolved-config.json",
            assembly / "interfaces.json",
            selection / "report.json",
        ]:
            report["source_bindings"][str(p)] = sha(p)
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        if json.loads(json.dumps(robot.interface())) != json.loads(
            (assembly / "interfaces.json").read_text()
        ):
            raise ValueError("Assembly frame mismatch")
        neutral = robot.forward_kinematics()
        print("Importing covered Scout", flush=True)
        raw = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        local = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral[p.owner])))
            for p in raw
        ]
        poses = []
        if samples is None:
            for a in selected["selection_grid_rad"]:
                q = dict(
                    selected["selected_pose_rad"],
                    left_upper_arm_joint=a,
                    right_upper_arm_joint=a,
                )
                poses.append((f"command_{a:.2f}", q))
        else:
            manifest = json.loads(samples.read_text())
            report["source_bindings"][str(samples)] = sha(samples)
            for p, expected in manifest["source_bindings"].items():
                if sha(p) != expected:
                    raise ValueError(f"Observed source changed: {p}")
                report["source_bindings"][p] = expected
            if manifest["joint_order"] != robot.interface()["joint_order"]:
                raise ValueError("Observed joint order differs")
            poses = [(p["name"], p["pose_rad"]) for p in manifest["samples"]]
            report["observed_samples"] = manifest["samples"]
        for label, pose in poses:
            frames = robot.forward_kinematics(pose, enforce_limits=samples is None)
            placed = [
                replace(p, shape=transform_shape(p.shape, frames[p.owner]))
                for p in local
            ]
            query = CameraQueries()
            rows = []
            near = []
            pair_count = 0
            for side in ("left", "right"):
                owners = {f"{side}_{s}" for s in ("upper_arm", "forearm", "gripper")}
                moving = [p for p in placed if p.owner in owners]
                for p in moving:
                    targets = [
                        b
                        for b in placed
                        if b.owner not in owners
                        and not (
                            p.owner == f"{side}_upper_arm"
                            and b.owner == f"{side}_shoulder_roll"
                        )
                    ]
                    pair_count += len(targets)
                    hits = query.collide(p.shape, targets)
                    if hits:
                        rows.append(dict(part=p.name, hits=hits))
                    if p.owner in {f"{side}_forearm", f"{side}_gripper"}:
                        lower = [
                            b
                            for b in targets
                            if b.owner
                            in {f"{side}_thigh", f"{side}_shin", f"{side}_wheel"}
                        ]
                        distance, target = query.nearest(p.shape, lower)
                        near.append(
                            dict(part=p.name, target=target, gap_mm=float(distance))
                        )
            row = dict(
                name=label,
                pose_rad=pose,
                status="fail" if rows else "pass",
                pair_count=pair_count,
                hits=rows,
                nearest_leg_parts=near,
                minimum_leg_gap_mm=min(n["gap_mm"] for n in near),
            )
            report["rows"].append(row)
            write(output / "progress.json", report)
            print(label, row["status"], row["minimum_leg_gap_mm"], flush=True)
        report.update(
            status="partial",
            checked_geometry_status="fail"
            if any(r["status"] == "fail" for r in report["rows"])
            else "pass",
            source_assembly_status=base["status"],
            part_count=len(raw),
            limitations=[
                "Finite commanded or observed joint poses only; no continuous-path or tolerance proof.",
                "Within each same-side arm subtree, pairs are excluded: commanded relative joints are unchanged; observed relative joints can differ and are NOT qualified by this screen.",
                "Upper-arm versus its same-side shoulder-roll owner is a mating interface excluded from this screen; no joint-internal clearance claim.",
                "Opposite-arm pairs may appear twice; report pair counts are query counts, not unique unordered pairs.",
                "No world-ground or scene-object clearance, harness or physical qualification.",
            ],
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    print(report.get("checked_geometry_status"), report.get("error", ""), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=Path)
    args = p.parse_args()
    run(args.output, args.samples)
