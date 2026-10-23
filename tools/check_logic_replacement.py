"""Check a replacement logic PCB against current detailed collar assemblies.

This is a neutral torso fit check.  It consumes the variant-owned board pose,
removes only the prior logic PCB from the retained assembly, and keeps the
existing logic screws/inserts as obstacles.  It does not qualify service,
harness, dynamic clearance, or mounting strength.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

WORKSPACE = Path.cwd().resolve()

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from build123d import Box, Compound, Pos, import_step  # noqa: E402

from lab.artifact_store import snapshot_file
from lab.config import RobotConfig  # noqa: E402
from lab.leg_mechanism import transform_shape  # noqa: E402
from lab.logic_packaging import placement  # noqa: E402
from lab.power_pack import collisions as allocation_collisions, overlap, pack_shapes, torso_containment  # noqa: E402
from lab.power_system import load_system  # noqa: E402
from lab.robot import make_robot  # noqa: E402
from tools.check_arm_episode import parts_from_step  # noqa: E402


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def torso_world_shape(robot, shape):
    """Map a torso-local millimetre allocation into world coordinates."""
    return transform_shape(shape, robot.forward_kinematics()["torso"])


def workspace_path(path):
    from lab.artifact_paths import checked_path
    path = Path(path).absolute()
    if path.is_file():
        checked_path(WORKSPACE, path.relative_to(WORKSPACE))
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise ValueError(f"input is outside workspace: {path}") from exc
    return path


def run(assembly, board, output):
    assembly, board, output = map(workspace_path, (assembly, board, output))
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "fail",
        "scope": "replacement populated board vs current detailed collar neutral assembly; no service, harness, strength, dynamic clearance or whole-body requalification",
    }
    try:
        body_report = json.loads((assembly / "arm-report.json").read_text())
        board_report = json.loads((board / "build-report.json").read_text())
        contract = json.loads((board / "board-interface.json").read_text())
        if body_report.get("status") != "pass" or body_report.get("builder") != "neck_collar_composition":
            raise ValueError("requires passed current neck-collar composition")
        if board_report.get("status") != "pass" or board_report.get("drc", {}).get("errors"):
            raise ValueError("requires passed replacement board build")
        if contract["outline_mm"] != [38, 34] or sorted(contract["mounting_holes_mm"]) != sorted([[3, 3], [35, 3], [3, 31], [35, 31]]):
            raise ValueError("replacement board differs from logic mount pattern")
        raw = import_step(board / "logic-supply.step")
        if len(raw.children) != 16 or contract.get("missing_3d_models"):
            raise ValueError("replacement board must contain 15 component models and substrate")
        resolved = json.loads((assembly / "resolved-config.json").read_text())
        robot = make_robot(RobotConfig.model_validate(resolved))
        if not robot.config.dimensions.logic_mount_enabled:
            raise ValueError("variant-owned logic mount is not enabled")
        if json.loads(json.dumps(robot.interface())) != json.loads((assembly / "interfaces.json").read_text()):
            raise ValueError("kinematics differ from retained collar assembly")
        # Use the variant-owned pose, while retaining the established board frame.
        pose = {
            "schema_version": 1,
            "normal": "rearward",
            "position_torso_mm": [
                robot.config.dimensions.logic_board_x_m * 1000,
                robot.config.dimensions.logic_board_y_m * 1000,
                robot.config.dimensions.logic_board_z_m * 1000,
            ],
        }
        loc = placement(contract, pose)
        board_local = loc * Compound(raw.solids())
        body_shapes = list(import_step(assembly / "assembly.step").children)
        body_parts = parts_from_step(body_shapes, body_report["parts"])
        old = [p.name for p in body_parts if p.name == "logic_board"]
        retained = [p for p in body_parts if p.name != "logic_board"]
        expected_mount = {
            "logic_screw_0", "logic_screw_1", "logic_screw_2", "logic_screw_3",
            "logic_insert_0", "logic_insert_1", "logic_insert_2", "logic_insert_3",
        }
        retained_mount = {p.name for p in retained if p.module == "logic_mount"}
        if old != ["logic_board"]:
            raise ValueError("retained assembly does not contain exactly one prior logic_board")
        if retained_mount != expected_mount:
            raise ValueError("retained assembly logic mount inventory differs from expected screws/inserts")
        board_world = transform_shape(board_local, robot.forward_kinematics()["torso"])
        # Only replacement-board-to-retained-part intersections are in scope;
        # rechecking every old-old pair would falsely turn this into a body
        # requalification and is needlessly expensive on imported STEP.
        pair_failures = allocation_collisions(board_world, retained)
        tested = len(retained)
        pack, _ = pack_shapes(load_system())
        pack_world = torso_world_shape(robot, pack)
        report = {
            "status": "pass" if not pair_failures else "fail",
            "scope": result["scope"],
            "assembly_builder": body_report["builder"],
            "assembly_group_count": len(body_parts),
            "retained_group_count": len(retained),
            "excluded_parts": old,
            "retained_logic_mount_parts": sorted(retained_mount),
            "board_source": str((board / "logic-supply.step").absolute()),
            "board_hash": sha256(board / "logic-supply.step"),
            "board_build_hash": sha256(board / "build-report.json"),
            "board_pose_torso_mm": pose["position_torso_mm"],
            "board_frame": {"normal": "rearward", "pcb_y_to_step_y": -1},
            "body_pair_collisions": pair_failures,
            "body_pairs_tested": tested,
            "battery_overlap_mm3": overlap(board_world, pack_world),
            "battery_status": "pass" if overlap(board_world, pack_world) < 0.02 else "fail",
            "board_containment": torso_containment(robot, board_local),
            "connector_approach_status": "not_run",
            "not_run": ["actual mated connector and harness bend/retention", "service/removal", "dynamic clearance", "mount strength", "electrical/thermal qualification"],
        }
        # Preserve the same unplugged connector reservations used by the existing checker.
        packaging_path = WORKSPACE / "electronics/logic-supply/packaging.yaml"
        approach = yaml.safe_load(packaging_path.read_text())["connector_approach"]
        allocations = {}
        for ref in ["J1", "J2", "J3"]:
            xy = np.mean([p["xy_mm"] for n, p in contract["pads"].items() if n.startswith(ref + ".")], axis=0)
            box = loc * Pos(xy[0], -xy[1], approach["normal_center_mm"]) * Box(approach["width_mm"], approach["height_mm"], approach["normal_depth_mm"])
            world_box = transform_shape(box, robot.forward_kinematics()["torso"])
            allocations[ref] = {"body_collisions": allocation_collisions(world_box, retained), "battery_overlap_mm3": overlap(world_box, pack_world)}
        report["connector_approach"] = allocations
        report["connector_approach_status"] = "pass" if all(not v["body_collisions"] and v["battery_overlap_mm3"] < 0.02 for v in allocations.values()) else "fail"
        report["status"] = "pass" if report["status"] == "pass" and report["battery_status"] == "pass" and report["board_containment"]["status"] == "pass" and report["connector_approach_status"] == "pass" else "fail"
        result = report
        dependencies = [
            assembly / "arm-report.json", assembly / "assembly.step", assembly / "interfaces.json", assembly / "resolved-config.json",
            board / "board-interface.json", board / "build-report.json", board / "logic-supply.step",
            packaging_path, WORKSPACE / "electronics/power-system.yaml",
            WORKSPACE / "lab/config.py", WORKSPACE / "lab/robot.py", WORKSPACE / "lab/leg_mechanism.py",
            WORKSPACE / "lab/logic_packaging.py", WORKSPACE / "lab/power_pack.py", WORKSPACE / "lab/power_system.py",
            WORKSPACE / "tools/check_arm_episode.py", WORKSPACE / "tools/check_logic_replacement.py",
        ]
        for src in dependencies:
            src = workspace_path(src)
            dest = output / "source" / src.relative_to(WORKSPACE)
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(src, dest)
        result["source_hashes"] = {str(src.relative_to(WORKSPACE)): sha256(src) for src in dependencies}
    except Exception as exc:
        result["error"] = str(exc)
    digest_input = dict(result)
    digest_input.pop("report_digest_sha256", None)
    result["report_digest_sha256"] = hashlib.sha256(json.dumps(digest_input, sort_keys=True, indent=2).encode()).hexdigest()
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "error", "body_pair_collisions", "battery_overlap_mm3") if k in result}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--board", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(int(run(args.assembly, args.board, args.output)["status"] != "pass"))
