"""Fast neck-only camera motion probe using an attested retained head STEP."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from build123d import import_step
from lab.camera_mount import cassette, screws
from lab.config import load_config, RobotConfig
from lab.leg_mechanism import Part, transform_shape
from lab.neck_collar import collar_parts
from lab.robot import make_robot
from tools.check_neck_collar import ExactQueries
from tools.check_arm_episode import parts_from_step
from tools.review_camera_mount import verify_study


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(variant, study, output):
    output.mkdir(parents=True, exist_ok=False)
    report = verify_study(study)
    source_step = study / "head-allocations.step"
    if sha(source_step) != report["output_hashes"]["head-allocations.step"]:
        raise ValueError("retained head STEP changed")
    assembly_key = next(
        name for name in report["source_hashes"] if name.endswith("/assembly.step")
    )
    metadata_key = str(Path(assembly_key).parent / "report.json")
    metadata = study / "source" / metadata_key
    if sha(metadata) != report["source_hashes"][metadata_key]:
        raise ValueError("retained assembly metadata changed")
    if sha("lab/neck_collar.py") != report["source_hashes"]["lab/neck_collar.py"]:
        raise ValueError("collar implementation changed; regenerate reduced input")
    robot = make_robot(load_config(variant))
    a = robot.config.model_dump()
    b = RobotConfig.model_validate_json(
        (study / "resolved-config.json").read_text()
    ).model_dump()
    for config in (a, b):
        config.pop("name")
        config.pop("camera")
    if a != b:
        raise ValueError("neck-only probe permits only camera changes")
    records = [
        p
        for p in json.loads(metadata.read_text())["parts"]
        if p["name"] == "head_shell" or p["owner"] in {"head", "neck_yaw"}
    ]
    if "camera_mount" not in report:
        raise ValueError("reduced probe requires a cassette study")
    records += [
        dict(name=name, owner="head", role="allocation", module="vision")
        for name in [
            "camera_full_envelope",
            "pi_board_height_allocation",
            "camera_cassette",
            "camera_shell_screw_0",
            "camera_shell_screw_1",
        ]
    ]
    neutral = robot.forward_kinematics()
    imported = parts_from_step(list(import_step(source_step).children), records)
    mapping = {p.shape.label: p.name for p in imported}
    targets = [p for p in imported if p.owner != "head"]
    for name, shape in collar_parts(robot.config.dimensions).items():
        targets.append(
            Part(
                "neck_collar_" + name,
                "torso",
                transform_shape(shape, neutral["torso"]),
                "cover",
                "neck",
                (0.5, 0.5, 0.5),
            )
        )
    required = {
        "neck_bridge_bolt_16.5",
        "neck_bridge_bolt_-16.5",
        "neck_pitch_support",
        "neck_collar_front",
        "neck_collar_rear",
    }
    if not required.issubset({p.name for p in targets}):
        raise ValueError("missing required neck targets")
    query = ExactQueries()
    candidates = [("cassette", cassette(robot))] + [
        (f"shell_screw_{i}", shape) for i, shape in enumerate(screws(robot))
    ]
    rows = []
    for yaw in [-1.2, -0.6, 0, 0.6, 1.2]:
        for pitch in [-0.5, 0, 0.5]:
            posed = robot.forward_kinematics(
                {"neck_yaw_joint": yaw, "head_joint": pitch}
            )
            located = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape, posed[p.owner] @ np.linalg.inv(neutral[p.owner])
                    ),
                )
                for p in targets
            ]
            for name, candidate in candidates:
                rows.append(
                    dict(
                        yaw_rad=yaw,
                        pitch_rad=pitch,
                        moving_part=name,
                        conflicts=query.collide(
                            transform_shape(candidate, posed["head"]), located
                        ),
                    )
                )
    result = dict(
        status="fail" if any(row["conflicts"] for row in rows) else "pass",
        scope="neck-only sampled geometry; full assembly study still required",
        targets=[p.name for p in targets],
        imported_label_mapping=mapping,
        cases=rows,
        input_hashes={
            str(p): sha(p)
            for p in [
                variant,
                study / "report.json",
                source_step,
                metadata,
                Path("lab/camera_mount.py"),
                Path("lab/neck_collar.py"),
                Path(__file__),
            ]
        },
    )
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("variant", "study", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.variant, args.study, args.output)
    sys.exit(result["status"] != "pass")
