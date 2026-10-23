"""Independently inspect retained shell unions after an overlap diagnostic fails."""

import argparse
from pathlib import Path
import traceback
import numpy as np
from build123d import Compound, import_step, export_step
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from lab.leg_mechanism import transform_shape
from lab.config import RobotConfig
from lab.robot import make_robot
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def volume(shape):
    if shape is None or not shape.solids():
        return 0.0
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape.wrapped, props, 1e-7, True, False)
    return abs(props.Mass())


def run(name, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__, cases=[])
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        study = Path(f"artifacts/{name}-integral-head-wings-v1")
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        base = verify_report(assembly)
        producer = verify_report(study)
        expected = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        actual = RobotConfig.model_validate_json(
            (study / "resolved-config.json").read_text()
        )
        if expected != actual:
            raise ValueError("Candidate and assembly configurations differ")
        report["producer_status"] = producer["status"]
        report["specification"] = producer["specification"]
        # Future audits require a completed or explicitly stopped producer report.
        # The first retained audit bound STEP directly while its producer ran.
        bindings = [
            assembly / "report.json",
            assembly / "assembly.step",
            assembly / "resolved-config.json",
            study / "report.json",
            study / "resolved-config.json",
        ] + [study / f"{key}.step" for key in ("head_shell", "head_rear_cover")]
        report["source_bindings"] = {str(p): sha(p) for p in bindings}
        cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        robot = make_robot(cfg)
        print(name, "importing original shell halves", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        for key in ("head_shell", "head_rear_cover"):
            original = transform_shape(
                next(p.shape for p in parts if p.name == key),
                np.linalg.inv(robot.forward_kinematics()["head"]),
            )
            candidate = import_step(study / f"{key}.step")
            export_step(original, output / f"{key}-original.step")
            a, b = Compound(original.solids()), Compound(candidate.solids())
            missing = a - b
            added = b - a
            common = a & b
            metrics = dict(
                name=key,
                original_volume_mm3=volume(a),
                candidate_volume_mm3=volume(b),
                omitted_original_volume_mm3=volume(missing),
                added_volume_mm3=volume(added),
                common_volume_mm3=volume(common),
                missing_solids=len(missing.solids()),
                candidate_valid=b.is_valid,
                candidate_solids=len(b.solids()),
            )
            metrics["original_partition_error_mm3"] = abs(
                metrics["original_volume_mm3"]
                - metrics["common_volume_mm3"]
                - metrics["omitted_original_volume_mm3"]
            )
            metrics["candidate_partition_error_mm3"] = abs(
                metrics["candidate_volume_mm3"]
                - metrics["common_volume_mm3"]
                - metrics["added_volume_mm3"]
            )
            metrics["status"] = (
                "pass"
                if metrics["omitted_original_volume_mm3"] <= 0.02
                and metrics["original_partition_error_mm3"] <= 0.02
                and metrics["candidate_partition_error_mm3"] <= 0.02
                and metrics["candidate_valid"]
                and metrics["candidate_solids"] == 1
                else "fail"
            )
            for tag, shape in [
                ("missing", missing),
                ("added", added),
                ("common", common),
            ]:
                if shape.solids():
                    export_step(shape, output / f"{key}-{tag}.step")
            report["cases"].append(metrics)
            write(output / "progress.json", report)
            print(name, key, metrics, flush=True)
        report["status"] = (
            "pass" if all(r["status"] == "pass" for r in report["cases"]) else "fail"
        )
        report["limits"] = [
            "Boolean partition and adaptive volume audit only; no override of producer nominal/motion/service status.",
            "Original whole-assembly failure and unqualified physical strength remain.",
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
    parser.add_argument("robot", choices=["pip", "scout"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.robot, args.output)
    raise SystemExit(1 if outcome["status"] == "fail" else 0)
