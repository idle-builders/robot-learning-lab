"""Check a populated logic-supply board against detailed bodies and battery space."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np
import yaml
from build123d import Box, Color, Compound, Pos, Rot, export_step, import_step

from .artifact_store import snapshot_file
from .config import RobotConfig
from .robot import make_robot
from .packaging import BOARD_ROTATION
from .power_pack import collisions, pack_shapes, torso_containment
from .power_system import load_system
from .leg_mechanism import Part, overlap, render_core, transform_shape
from tools.check_arm_episode import parts_from_step


def placement(contract, config):
    if config["schema_version"] != 1 or config["normal"] != "rearward":
        raise ValueError("unsupported logic board frame")
    width, height = contract["outline_mm"]
    return (
        Pos(*config["position_torso_mm"])
        * BOARD_ROTATION
        * Rot(0, 180, 0)
        * Pos(-width / 2, height / 2, 0)
    )


def run(
    assembly, board, output, config_path=Path("electronics/logic-supply/packaging.yaml")
):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "populated package-model board and connector-space allocation vs neutral detailed body and battery; mounting structure, harness, removal and dynamic clearance remain unqualified",
    }
    try:
        paths = [
            assembly / n
            for n in (
                "assembly.step",
                "arm-report.json",
                "interfaces.json",
                "resolved-config.json",
            )
        ]
        paths += [
            board / n
            for n in ("logic-supply.step", "board-interface.json", "build-report.json")
        ]
        paths += [
            config_path,
            Path("electronics/power-system.yaml"),
            *Path("lab").glob("*.py"),
            Path("tools/check_arm_episode.py"),
        ]
        report["source_hashes"] = {}
        for p in paths:
            rel = p.absolute().relative_to(Path.cwd())
            dest = output / "source" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(p, dest)
            report["source_hashes"][str(rel)] = hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
        board_report = json.loads((board / "build-report.json").read_text())
        contract = json.loads((board / "board-interface.json").read_text())
        if board_report["status"] != "pass" or contract["missing_3d_models"]:
            raise ValueError(
                "requires passed populated board with all model allocations present"
            )
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        if json.loads(json.dumps(robot.interface())) != json.loads(
            (assembly / "interfaces.json").read_text()
        ):
            raise ValueError("kinematics differ from retained body")
        config = yaml.safe_load(config_path.read_text())
        loc = placement(contract, config)
        raw = import_step(board / "logic-supply.step")
        if len(raw.children) != 16:
            raise ValueError("expected 15 component models and one substrate")
        model = loc * Compound(raw.solids())
        model.label = "logic_supply_populated"
        bounds = model.bounding_box()
        report["populated_bounds_torso_mm"] = {
            "minimum": list(bounds.min),
            "maximum": list(bounds.max),
        }
        # Test enclosure containment against a full envelope, including board underside pins.
        envelope = Pos(
            *((np.array(list(bounds.min)) + np.array(list(bounds.max))) / 2)
        ) * Box(*list(bounds.size))
        report["board_containment"] = torso_containment(robot, envelope)
        pack, _ = pack_shapes(load_system())
        report["battery_overlap_mm3"] = overlap(model, pack)
        frame = robot.forward_kinematics()["torso"]
        world = transform_shape(model, frame)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children),
            json.loads((assembly / "arm-report.json").read_text())["parts"],
        )
        report["body_collisions"] = collisions(world, parts)
        report["assembly_groups_checked"] = len(parts)
        report["mounting_points_torso_mm"] = [
            list((loc * Pos(x, -y, 0)).position)
            for x, y in contract["mounting_holes_mm"]
        ]
        report["board_frame"] = {
            "center_substrate_bottom_torso_mm": config["position_torso_mm"],
            "rotation_matrix": [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
            "pcb_y_to_step_y": -1,
        }
        reservations = []
        approach = config["connector_approach"]
        report["connector_allocations"] = {}
        for ref in ["J1", "J2", "J3"]:
            xy = np.mean(
                [
                    p["xy_mm"]
                    for n, p in contract["pads"].items()
                    if n.startswith(ref + ".")
                ],
                axis=0,
            )
            box = (
                loc
                * Pos(xy[0], -xy[1], approach["normal_center_mm"])
                * Box(
                    approach["width_mm"],
                    approach["height_mm"],
                    approach["normal_depth_mm"],
                )
            )
            box.label = ref + "_approach"
            world_box = transform_shape(box, frame)
            results = {
                "containment": torso_containment(robot, box),
                "battery_overlap_mm3": overlap(box, pack),
                "body_collisions": collisions(world_box, parts),
            }
            results["status"] = (
                "pass"
                if results["containment"]["status"] == "pass"
                and results["battery_overlap_mm3"] < 0.02
                and not results["body_collisions"]
                else "fail"
            )
            report["connector_allocations"][ref] = results
            reservations.append(
                Part(
                    box.label,
                    "torso",
                    world_box,
                    "allocation",
                    "logic-board",
                    (1, 0.5, 0.1, 0.3),
                )
            )
        report["status"] = (
            "pass"
            if report["board_containment"]["status"] == "pass"
            and report["battery_overlap_mm3"] < 0.02
            and not report["body_collisions"]
            and all(
                v["status"] == "pass" for v in report["connector_allocations"].values()
            )
            else "fail"
        )
        world.color = Color(0.15, 0.45, 0.3)
        world.label = "logic_supply_populated"
        export_step(
            Compound(children=[world, *[p.shape for p in reservations]]),
            output / "logic-board-installed.step",
        )
        # Inspect torso cutaway with existing rear board, neck supports and allocated pack.
        shown = [
            p
            for p in parts
            if p.owner == "torso" and p.name not in ("torso_front", "torso_rear")
        ]
        wp = transform_shape(pack, frame)
        shown += [
            Part(
                "logic_supply_populated",
                "torso",
                world,
                "board",
                "logic-board",
                (0.15, 0.45, 0.3),
            ),
            Part(
                "protected_pack_allocation",
                "torso",
                wp,
                "allocation",
                "battery",
                (0.9, 0.6, 0.1, 0.35),
            ),
            *reservations,
        ]
        render_core(shown, {p.name: p.shape for p in shown}, output, robot)
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("status", "error", "body_collisions", "battery_overlap_mm3")
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assembly", required=True, type=Path)
    p.add_argument("--board", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    raise SystemExit(int(run(a.assembly, a.board, a.output)["status"] != "pass"))
