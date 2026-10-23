"""Bounded exact-distance witnesses for rejected mount-clearance samples."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from build123d import Compound, import_step, export_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def pair(folder):
    a = import_step(folder / "stock.step")
    b = import_step(folder / "target.step")
    distance, p, q = a.distance_to_with_closest_points(b)
    write(
        folder / "distance.json",
        dict(
            distance_mm=distance,
            stock_point_head_mm=list(p),
            target_point_head_mm=list(q),
        ),
    )


def run(study, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="partial", scope=__doc__, cases=[])
    try:
        paths = [
            *Path("lab").glob("*.py"),
            Path("tools/measure_head_cap_mount_gap.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        prior = verify_report(study)
        cfg = RobotConfig.model_validate_json(
            (study / "resolved-config.json").read_text()
        )
        assembly_key = next(
            k for k in prior["source_bindings"] if k.endswith("assembly.step")
        )
        assembly = Path(assembly_key).parent
        source = verify_report(assembly)
        report["source_bindings"] = {
            str(p): sha(p)
            for p in [
                study / "report.json",
                study / "resolved-config.json",
                assembly / "report.json",
                assembly / "assembly.step",
            ]
        }
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        print("importing witness assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), source["parts"]
        )
        lookup = {p.name: p for p in parts}
        cases = {c["cover"]: c for c in prior["selected_cases"]}
        shapes = {}
        for key, case in cases.items():
            paths = [Path(p) for p in case["input_paths"].values()]
            shapes[key] = Compound(
                [s for path in paths for s in import_step(path).solids()]
            )
            report["source_bindings"].update({str(p): sha(p) for p in paths})
        for pose in prior["poses"]:
            fk = robot.forward_kinematics(pose["pose_rad"], enforce_limits=False)
            for hit in pose["conservative_clearance_collisions"]:
                target = lookup[hit["part"]]
                shape = transform_shape(
                    target.shape,
                    np.linalg.inv(fk["head"])
                    @ fk[target.owner]
                    @ np.linalg.inv(neutral[target.owner]),
                )
                folder = output / f"pair-{len(report['cases']):02d}"
                folder.mkdir()
                export_step(shapes[hit["cover"]], folder / "stock.step")
                export_step(shape, folder / "target.step")
                row = dict(
                    pose=pose["name"],
                    pose_rad=pose["pose_rad"],
                    cover=hit["cover"],
                    target=hit["part"],
                    status="not_run",
                    pair_directory=str(folder.relative_to(output)),
                )
                report["cases"].append(row)
                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "tools.measure_head_cap_mount_gap",
                            "--pair-directory",
                            str(folder),
                        ],
                        check=True,
                        timeout=60,
                        capture_output=True,
                        text=True,
                    )
                    metrics = json.loads((folder / "distance.json").read_text())
                    separation = np.linalg.norm(
                        np.array(metrics["stock_point_head_mm"])
                        - np.array(metrics["target_point_head_mm"])
                    )
                    if abs(separation - metrics["distance_mm"]) > 1e-5:
                        raise ValueError(
                            "Closest point separation differs from distance"
                        )
                    row.update(
                        **metrics,
                        status="pass"
                        if metrics["distance_mm"] >= prior["margin_mm"] - 1e-5
                        else "fail",
                    )
                except subprocess.TimeoutExpired:
                    row["reason"] = (
                        "exact-distance query exceeded 60 seconds; no gap result"
                    )
                print(
                    row["pose"],
                    row["cover"],
                    row["target"],
                    row.get("distance_mm"),
                    row["status"],
                    flush=True,
                )
                write(output / "progress.json", report)
        report["status"] = (
            "fail"
            if any(r["status"] == "fail" for r in report["cases"])
            else "partial"
            if any(r["status"] == "not_run" for r in report["cases"])
            or not report["cases"]
            else "pass"
        )
        report["limits"] = [
            "Only clearance-envelope flagged pairs are measured; this is not continuous or global minimum clearance.",
            "CAD distances and witnesses are nominal geometry, not manufactured measurements.",
            "Unmounted and unbored stock remains unqualified for fasteners, service, strength or hardware use.",
        ]
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file()
            and not p.is_relative_to(output / "source")
            and p.name != "report.json"
        }
        write(output / "report.json", report)
        return report
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pair-directory", type=Path)
    args = parser.parse_args()
    if args.pair_directory:
        pair(args.pair_directory)
    else:
        outcome = run(args.study, args.output)
        raise SystemExit(1 if outcome["status"] == "fail" else 0)
