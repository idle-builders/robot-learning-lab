"""Replay selected supported-task poses against a retained detailed assembly.

This checks nominal robot self-collision at phase endpoints. It does not replay
physics, include the table/object, or prove continuous trajectory clearance.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from build123d import import_step
from lab.artifact_store import snapshot_file
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, check_pairs, transform_shape


def parts_from_step(shapes, records):
    # Match pinned build123d 0.11.1 import_step.get_name, preserving original IDs.
    def imported_label(name):
        clean = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
        return clean.translate(str.maketrans(" .()", "____"))

    metadata = {imported_label(record["name"]): record for record in records}
    if len(metadata) != len(records):
        raise ValueError("ambiguous assembly names after STEP label normalization")
    labels = [shape.label for shape in shapes]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate imported STEP labels")
    if set(metadata) != set(labels):
        raise ValueError("STEP labels differ from assembly manifest")
    return [
        Part(
            metadata[shape.label]["name"],
            metadata[shape.label]["owner"],
            shape,
            metadata[shape.label]["role"],
            metadata[shape.label]["module"],
            (0.5, 0.5, 0.5),
        )
        for shape in shapes
    ]


def split_installation_contacts(collisions, manifest):
    """Preserve measured heat-insert interference as a separate contact result.

    Only the four source-attested, torso-fixed insert/front-shell pairs qualify.
    A changed volume or owner is a failure, not a general collision exemption.
    """
    if manifest.get("builder") != "logic_mount_composition":
        return collisions, []
    volumes = manifest.get("insert_installation_interference_mm3", [])
    if manifest.get("status") != "pass" or len(volumes) != 4:
        raise ValueError("mounted source lacks passed installation-contact evidence")
    records = {r["name"]: r for r in manifest["parts"]}
    front = records["torso_front"]
    expected = {}
    for i, volume in enumerate(volumes):
        name = f"logic_insert_{i}"
        insert = records[name]
        if (
            not np.isfinite(volume)
            or volume <= 0.02
            or front["owner"] != "torso"
            or insert["owner"] != "torso"
            or insert["role"] != "hardware"
            or insert["module"] != "logic_mount"
        ):
            raise ValueError("invalid fixed installation-contact contract")
        expected[frozenset(("torso_front", name))] = volume
    failures, contacts = [], []
    for collision in collisions:
        volume = expected.get(frozenset(collision["parts"]))
        if volume is not None and abs(collision["volume_mm3"] - volume) <= 0.02:
            contacts.append(
                {
                    **collision,
                    "source_volume_mm3": volume,
                    "reason": "heat-insert envelope in undersized pilot; material process unqualified",
                }
            )
        else:
            failures.append(collision)
    # If collision enumeration was capped, missing pairs cannot be interpreted.
    # There are only four allowed pairs, so a capped 25-pair result still fails.
    if len(collisions) < 25 and len(contacts) != 4 and not failures:
        raise ValueError("expected installation contacts absent from retained geometry")
    return failures, contacts


def run(assembly, episode, output):
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "fail", "scope": __doc__.strip()}
    try:
        for prefix, root, names in [
            (
                "assembly",
                assembly,
                [
                    "arm-report.json",
                    "assembly.step",
                    "interfaces.json",
                    "resolved-config.json",
                ],
            ),
            (
                "episode",
                episode,
                ["interfaces.json", "task.json", "report.json", "trajectory.jsonl"],
            ),
        ]:
            for name in names:
                dest = output / "source" / prefix / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                snapshot_file(root / name, dest)
        for source in [Path(__file__), *Path("lab").glob("*.py")]:
            relative = (
                source.relative_to(Path.cwd()) if source.is_absolute() else source
            )
            dest = output / "source" / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(source, dest)
        result["source_hashes"] = {
            str(p.relative_to(output / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        contract = json.loads((assembly / "interfaces.json").read_text())
        if contract != json.loads((episode / "interfaces.json").read_text()):
            raise ValueError("assembly and episode joint/frame interfaces differ")
        if json.loads((episode / "task.json").read_text())["fixture"] != "supported":
            raise ValueError(
                "replay requires a supported torso; free root motion is not implemented"
            )
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        if json.loads(json.dumps(robot.interface())) != contract:
            raise ValueError("current kinematics disagree with retained assembly")
        rows = [
            json.loads(line)
            for line in (episode / "trajectory.jsonl").read_text().splitlines()
        ]
        endpoints = {r["phase"]: r for r in rows}
        selected = [
            endpoints[name] for name in ["approach", "lift", "lower", "retreat"]
        ]
        poses = []
        for row in selected:
            state = row["joint_state"]
            if state["names"] != contract["joint_order"]:
                raise ValueError("trajectory joint ordering differs")
            pose = dict(zip(state["names"], state["position_rad"], strict=True))
            poses.append(
                (
                    row["phase"],
                    state["time_s"],
                    pose,
                    robot.forward_kinematics(pose, enforce_limits=False),
                )
            )
        manifest = json.loads((assembly / "arm-report.json").read_text())
        shapes = list(import_step(assembly / "assembly.step").children)
        parts = parts_from_step(shapes, manifest["parts"])
        neutral = {
            name: np.linalg.inv(frame)
            for name, frame in robot.forward_kinematics().items()
        }
        checks, cache = [], {}
        for phase, time_s, pose, frames in poses:
            excursions = [
                {"joint": j.name, "position_rad": pose[j.name], "limits_rad": j.limits}
                for j in robot.joints
                if not j.continuous and not j.limits[0] <= pose[j.name] <= j.limits[1]
            ]
            placed = {
                p.name: transform_shape(p.shape, frames[p.owner] @ neutral[p.owner])
                for p in parts
            }
            failures, exact = check_pairs(
                parts, placed, frames, cache, max_collisions=25
            )
            truncated = len(failures) >= 25
            failures, contacts = split_installation_contacts(failures, manifest)
            checks.append(
                {
                    "phase": phase,
                    "time_s": time_s,
                    "pose_rad": pose,
                    "status": "fail" if failures else "pass",
                    "collisions": failures,
                    "installation_contacts": contacts,
                    "exact_checks": exact,
                    "collision_list_may_be_truncated": truncated,
                    "measured_joint_limit_excursions": excursions,
                }
            )
            (output / "progress.json").write_text(json.dumps(checks, indent=2))
            print(phase, checks[-1]["status"], len(failures), flush=True)
        result.update(
            status="fail" if any(c["status"] == "fail" for c in checks) else "pass",
            samples=checks,
            episode_task_status=json.loads((episode / "report.json").read_text())[
                "status"
            ],
            coverage="four measured phase endpoints; no continuous clearance, environment contacts or strength proof",
        )
    except Exception as exc:
        result["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    (output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ["status", "error"] if k in result}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(
        int(run(args.assembly, args.episode, args.output)["status"] != "pass")
    )
