"""Screen proposed underside cap mount stock; no assembly or motion release."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback

import numpy as np
import yaml
from build123d import Box, Cylinder, Pos, Compound, import_step, export_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.mechanical import primitive
from lab.head_service import pi_reservation
from lab.leg_mechanism import Part, overlap, transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def run(name, specification, output):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="fail",
        scope=__doc__,
        motion_status="not_run",
        attachment_status="not_run",
        physical_status="not_run",
        cases=[],
    )
    try:
        result["source_hashes"] = copy_source(
            output,
            {
                str(p): p
                for p in [
                    *Path("lab").glob("*.py"),
                    Path(__file__).relative_to(Path.cwd()),
                    specification,
                    Path("tools/check_arm_episode.py"),
                    Path("tools/check_camera_support.py"),
                    Path("tools/check_neck_collar.py"),
                    Path("tools/compose_forearm_enclosure.py"),
                ]
            },
        )
        spec = yaml.safe_load(specification.read_text())
        result["specification"] = spec
        source = Path(f"artifacts/{name}-head-cap-relief-v3")
        prior = verify_report(source)
        if prior["fit_status"] != "pass":
            raise ValueError("Requires passing sampled cap study")
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        base = verify_report(assembly)
        if prior["source_bindings"][str(assembly / "assembly.step")] != sha(
            assembly / "assembly.step"
        ):
            raise ValueError("Source assembly changed")
        bindings = [
            source / "report.json",
            source / "resolved-config.json",
            assembly / "report.json",
            assembly / "assembly.step",
        ]
        bindings += [source / f"{p['name']}.step" for p in prior["parts"]]
        result["source_bindings"] = {str(p): sha(p) for p in bindings}
        cfg = RobotConfig.model_validate_json(
            (source / "resolved-config.json").read_text()
        )
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        base_cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        if cfg != base_cfg:
            raise ValueError("Relief and source assembly configurations differ")
        result["source_bindings"][str(assembly / "resolved-config.json")] = sha(
            assembly / "resolved-config.json"
        )
        robot = make_robot(cfg)
        frame = np.linalg.inv(robot.forward_kinematics()["head"])
        print(name, "importing source assembly", flush=True)
        all_parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        targets = [
            replace(p, shape=transform_shape(p.shape, frame))
            for p in all_parts
            if p.owner == "head"
        ]
        targets.append(
            Part(
                "pi_component_reservation",
                "head",
                pi_reservation(cfg.head_service),
                "reservation",
                "head_service",
                (1, 0, 0),
            )
        )
        lookup = {p.name: p.shape for p in targets}
        outer = primitive(
            next(link for link in robot.links if link.name == "head").geometry[0]
        )
        z = cfg.dimensions.neck_lower_port_top_m * 1000
        radius, depth, seat = (
            spec["boss_radius_mm"],
            spec["boss_depth_mm"],
            spec["seat_thickness_mm"],
        )
        result["mount_plane_head_z_mm"] = z
        result["inherited_assembly_status"] = base["status"]
        eligible = {}
        for record in prior["parts"]:
            key = record["name"]
            cap = import_step(source / f"{key}.step")
            end = "front" if "_front_" in key else "rear"
            sign = 1 if key.endswith("left") else -1
            mate_name = "head_shell" if end == "front" else "head_rear_cover"
            mate = lookup[mate_name]
            other = [p for p in targets if p.name != mate_name]
            for x in spec[f"{end}_x_mm"]:
                for yabs in spec["absolute_y_mm"]:
                    y = sign * yabs
                    row = dict(cover=key, center_head_mm=[x, y, z], status="fail")
                    result["cases"].append(row)
                    boss = Pos(x, y, z + depth / 2) * Cylinder(radius, depth)
                    tab = Pos(x, y, z - seat / 2) * Cylinder(radius, seat)
                    if spec.get("contour_tab", False):
                        tab &= outer
                        core = Pos(x, y, z - seat / 2) * Cylinder(
                            spec["bearing_core_radius_mm"], seat
                        )
                        row["missing_bearing_core_mm3"] = max(
                            0, core.volume - overlap(core, tab)
                        )
                        if row["missing_bearing_core_mm3"] > 0.02:
                            row["reason"] = (
                                "contoured tab omits required screw-head bearing core"
                            )
                            continue
                    if spec.get("rib_height_mm", 0):
                        direction = 1 if end == "front" else -1
                        tip = direction * (cfg.dimensions.head_depth_m * 500 + 1)
                        rib = Pos(
                            (x + tip) / 2, y, z + depth - spec["rib_height_mm"] / 2
                        ) * Box(
                            abs(tip - x), spec["rib_width_mm"], spec["rib_height_mm"]
                        )
                        boss += rib & outer
                    if (
                        not boss.is_valid
                        or not tab.is_valid
                        or not boss.solids()
                        or not tab.solids()
                    ):
                        row["reason"] = "invalid or empty stock"
                        continue
                    stock = Compound([boss, tab])
                    if spec.get("preserve_central_corridor", False):
                        corridor = Box(300, 58, 300)
                        row["corridor_overlap_mm3"] = overlap(stock, corridor)
                        if row["corridor_overlap_mm3"] > 0.02:
                            row["reason"] = (
                                "stock intrudes into the reserved 58 mm corridor"
                            )
                            continue
                    outside = stock.volume - overlap(stock, outer)
                    row["exterior_excess_mm3"] = max(0, outside)
                    if outside > 0.02:
                        row["reason"] = (
                            "full boss/seat envelope protrudes outside existing head exterior"
                        )
                        continue
                    root = overlap(boss, mate)
                    joining = overlap(tab, cap)
                    row.update(boss_shell_overlap_mm3=root, tab_cap_overlap_mm3=joining)
                    if (
                        root < spec["minimum_connection_overlap_mm3"]
                        or joining < spec["minimum_connection_overlap_mm3"]
                    ):
                        row["reason"] = "insufficient nominal connection overlap"
                        continue
                    row["tab_mate_interference_mm3"] = overlap(tab, mate)
                    if row["tab_mate_interference_mm3"] > 0.02:
                        row["reason"] = (
                            "separate cap tab intersects retained mating shell"
                        )
                        continue
                    hits = CameraQueries().collide(stock, other)
                    row["head_conflicts"] = hits
                    if hits:
                        row["reason"] = (
                            "new stock intersects a retained head part or Pi reservation"
                        )
                        continue
                    modified_mate = mate + boss
                    modified_cap = cap + tab
                    row["mate_solids"] = len(modified_mate.solids())
                    row["cap_solids"] = len(modified_cap.solids())
                    if (
                        not modified_mate.is_valid
                        or not modified_cap.is_valid
                        or row["mate_solids"] != 1
                        or row["cap_solids"] != 1
                    ):
                        row["reason"] = "invalid or disconnected added stock"
                        continue
                    row.update(
                        status="pass",
                        reason="nominal unbored stock screen only; fastening and mechanism motion not run",
                    )
                    eligible.setdefault(key, []).append(row)
                    folder = output / key / f"x{x:g}-y{y:g}"
                    folder.mkdir(parents=True)
                    export_step(boss, folder / "boss.step")
                    export_step(tab, folder / "tab.step")
            print(name, key, "eligible", len(eligible.get(key, [])), flush=True)
            write(output / "progress.json", result)
        result["eligible_by_cover"] = {
            p["name"]: len(eligible.get(p["name"], [])) for p in prior["parts"]
        }
        result["status"] = (
            "partial" if all(result["eligible_by_cover"].values()) else "fail"
        )
        result["limits"] = [
            "Grid search is bounded; rejection is not a proof that no mount exists.",
            "Connection overlap threshold is a design screen, not a strength or load calculation.",
            "Unbored stock only: insert engagement, bearing, screwdriver and removal paths untested.",
            "Full non-head mechanism motion and 1.5 mm moving-clearance envelope not run.",
            "Current robot geometry, BOM, mass model and control remain unchanged.",
        ]
        result["output_hashes"] = {
            str(p.relative_to(output)): sha(p) for p in output.rglob("*.step")
        }
        result["output_hashes"]["resolved-config.json"] = sha(
            output / "resolved-config.json"
        )
        write(output / "report.json", result)
        return result
    except Exception:
        result["error"] = traceback.format_exc()
        write(output / "report.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=["pip", "scout"])
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("experiments/head-cap-mount-stock.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.robot, args.specification, args.output)
    print(outcome["status"])
    raise SystemExit(1 if outcome["status"] == "fail" else 0)
