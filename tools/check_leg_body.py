"""Check the retained detailed leg core against the populated torso assembly."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build123d import Compound, Pos, export_step, import_step
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.leg_mechanism import (
    bounds_overlap,
    transform_shape,
    motion_poses,
    Part,
    render_core,
)
from lab.packaging import board_location, mounted_shell, overlap
from lab.robot import make_robot


def run(core, carrier, output, sweep=False):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "detailed core versus torso and carrier; nominal geometry",
    }
    try:
        inputs = [
            core / "resolved-config.json",
            core / "interfaces.json",
            core / "leg-core.step",
            core / "mechanism-report.json",
            carrier / "carrier.step",
            carrier / "board-interface.json",
            carrier / "build-report.json",
        ]
        for path in inputs + [Path(__file__), *Path("lab").glob("*.py")]:
            dest = (
                output / "source" / path.relative_to(Path.cwd())
                if path.is_absolute()
                else output / "source" / path
            )
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, dest)
        report["source_hashes"] = {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        if json.loads((carrier / "build-report.json").read_text())["status"] != "pass":
            raise ValueError("carrier build did not pass")
        config = RobotConfig.model_validate_json(
            (core / "resolved-config.json").read_text()
        )
        robot = make_robot(config)
        # JSON normalizes tuple fields to lists.
        if json.loads(json.dumps(robot.interface())) != json.loads(
            (core / "interfaces.json").read_text()
        ):
            raise ValueError(
                "retained core does not match current shared robot interfaces"
            )
        contract = json.loads((carrier / "board-interface.json").read_text())
        if contract["schema_version"] != 2 or contract["outline_mm"] != [
            config.dimensions.board_width_m * 1000,
            config.dimensions.board_length_m * 1000,
        ]:
            raise ValueError("board frame version or dimensions disagree")
        loc = Pos(0, 0, robot.base_height * 1000)
        front, rear, _ = mounted_shell(robot, contract)
        targets = {
            "torso_front": loc * front,
            "torso_rear": loc * rear,
            "carrier": loc
            * board_location(robot, contract["outline_mm"])
            * import_step(carrier / "carrier.step"),
        }
        raw = import_step(core / "leg-core.step")
        # build123d reparents children when composing the export assembly.
        # Keep a stable list for checks and rendering after export.
        core_parts = list(raw.children)
        expected = {
            p["name"]: p
            for p in json.loads((core / "mechanism-report.json").read_text())["parts"]
        }
        if {p.label for p in core_parts} != set(expected):
            raise ValueError("STEP part names do not match core manifest")
        checks, cache = [], {}
        neutral_frames = robot.forward_kinematics()
        target_bounds = {n: p.bounding_box() for n, p in targets.items()}
        samples = [("neutral", {})] + (list(motion_poses(robot)) if sweep else [])
        rows = []
        for sample, pose in samples:
            frames = robot.forward_kinematics(pose)
            local_checks = []
            for part in core_parts:
                owner = expected[part.label]["owner"]
                change = frames[owner] @ np.linalg.inv(neutral_frames[owner])
                located = transform_shape(part, change)
                bounds = located.bounding_box()
                for name, shape in targets.items():
                    if not bounds_overlap(bounds, target_bounds[name]):
                        continue
                    key = (part.label, name, tuple(change.round(10).flatten()))
                    if key not in cache:
                        cache[key] = overlap(located, shape)
                    volume = cache[key]
                    local_checks.append(
                        dict(
                            pose=sample,
                            part=part.label,
                            against=name,
                            overlap_mm3=volume,
                            status="pass" if volume <= 0.02 else "fail",
                        )
                    )
            checks.extend(local_checks)
            rows.append(
                dict(
                    name=sample,
                    pose_rad=pose,
                    status="fail"
                    if any(c["status"] == "fail" for c in local_checks)
                    else "pass",
                )
            )
            (output / "motion-progress.json").write_text(json.dumps(rows, indent=2))
            print(f"{sample}: {rows[-1]['status']}", flush=True)
        failures = [c for c in checks if c["status"] == "fail"]
        if config.dimensions.hip_socket_width_m:
            from lab.pelvis import rear_tool_spaces

            access = []
            for i, space in enumerate(rear_tool_spaces(config.dimensions)):
                world_space = loc * space
                bound = world_space.bounding_box()
                for part_name, part in [
                    ("torso_front", targets["torso_front"]),
                    *[(p.label, p) for p in core_parts],
                ]:
                    if not bounds_overlap(bound, part.bounding_box()):
                        continue
                    volume = overlap(world_space, part)
                    access.append(
                        dict(
                            tool=i,
                            against=part_name,
                            overlap_mm3=volume,
                            status="pass" if volume <= 0.02 else "fail",
                        )
                    )
            report["rear_tool_access"] = dict(
                scope="6 mm shaft approach to screw-head faces; rear cap/board removed; drive sockets omitted",
                checks=access,
                status="fail" if any(r["status"] == "fail" for r in access) else "pass",
            )
            failures.extend(r for r in access if r["status"] == "fail")
        for name, shape in targets.items():
            shape.label = name
        export_step(
            Compound(children=[*core_parts, *targets.values()]),
            output / "body-core.step",
        )
        report.update(
            status="fail" if failures else "pass",
            checks=checks,
            collisions=failures,
            poses=rows,
        )
        parts = [
            Part(p.label, expected[p.label]["owner"], p, "core", "", (0.5, 0.65, 0.6))
            for p in core_parts
        ]
        parts += [
            Part(
                name,
                "torso",
                shape,
                "body",
                "",
                (0.85, 0.8, 0.65, 0.22) if name != "carrier" else (0.12, 0.5, 0.25),
            )
            for name, shape in targets.items()
        ]
        if len(parts) != len(expected) + len(targets):
            raise ValueError("assembly export lost parts before rendering")
        render_core(parts, {p.name: p.shape for p in parts}, output, robot)
        report["rendered_part_count"] = len(parts)
        report["limitations"] = [
            "only listed poses sampled; upper limbs and leg skins excluded; not continuous clearance proof",
            "pelvis load path and clamp fasteners are candidates; strength, harness, battery and remaining fasteners incomplete",
            "STEP proxies and nominal geometry do not establish tolerance, strength or interchange",
        ]
    except Exception as exc:
        import traceback

        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    (output / "body-core-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "source_hashes"}, indent=2)
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", type=Path, required=True)
    p.add_argument("--carrier", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sweep", action="store_true")
    a = p.parse_args()
    sys.exit(int(run(a.core, a.carrier, a.output, a.sweep)["status"] != "pass"))
