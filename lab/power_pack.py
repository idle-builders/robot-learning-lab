"""Check a battery space allocation against a retained detailed neutral assembly."""

import argparse
import hashlib
import json
from pathlib import Path
from math import degrees
import shutil
import traceback

from build123d import Box, Color, Compound, Cylinder, Pos, Rot, export_step, import_step

from .artifact_store import snapshot_file
from .config import RobotConfig
from .leg_mechanism import Part, bounding_overlap, overlap, render_core, transform_shape
from .power_system import load_system
from .mechanical import primitive, hollow_shell
from .robot import make_robot
from tools.check_arm_episode import parts_from_step


def pack_shapes(system):
    """Torso-local mm. No holders, insulation details, BMS, leads or mounts yet."""
    c, p = system.cell, system.pack
    origin = Pos(*(v * 1000 for v in p.position_torso_m))
    envelope = origin * Box(*(v * 1000 for v in p.envelope_m))
    cells = []
    for x in range(p.series):
        for z in range(p.parallel):
            center_x = (x - (p.series - 1) / 2) * (c.diameter_m + p.cell_gap_m)
            center_z = (z - (p.parallel - 1) / 2) * (c.diameter_m + p.cell_gap_m)
            cell = (
                origin
                * Pos(center_x * 1000, 0, center_z * 1000)
                * Rot(90, 0, 0)
                * Cylinder(c.diameter_m * 500, c.length_m * 1000)
            )
            cell.label = f"cell_{x}_{z}"
            cell.color = Color(0.20, 0.55, 0.65)
            cells.append(cell)
    envelope.label = "protected_pack_allocation"
    envelope.color = Color(0.94, 0.60, 0.13, 0.25)
    return envelope, cells


def collisions(envelope, parts):
    findings = []
    for part in parts:
        if bounding_overlap(envelope, part.shape):
            volume = overlap(envelope, part.shape)
            if volume > 0.02:
                findings.append({"part": part.name, "overlap_mm3": volume})
    return findings


def torso_containment(robot, envelope):
    """Reject an allocation outside the body even when it hits no retained part."""
    torso = next(link for link in robot.links if link.name == "torso")
    geometry = next(g for g in torso.geometry if g.role == "service_shell")
    interior = primitive(geometry) - hollow_shell(
        geometry, robot.config.dimensions.shell_wall_m
    )
    interior = (
        Pos(*(v * 1000 for v in geometry.pos))
        * Rot(*(degrees(v) for v in geometry.rpy))
        * interior
    )
    outside = max(0.0, envelope.volume - overlap(envelope, interior))
    return {
        "status": "pass" if outside < 0.02 else "fail",
        "outside_interior_mm3": outside,
    }


def run(assembly, output, system_path=Path("electronics/power-system.yaml")):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": "battery space allocation vs detailed neutral robot; no harness, mount, removal, mass update or dynamic clearance",
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
            system_path,
            Path("tools/check_arm_episode.py"),
            *Path("lab").glob("*.py"),
        ]
        for path in paths:
            dest = output / "source" / path.absolute().relative_to(Path.cwd())
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, dest)
        report["source_hashes"] = {
            str(p.relative_to(output / "source")): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (output / "source").rglob("*")
            if p.is_file()
        }
        robot = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        contract = json.loads((assembly / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != contract:
            raise ValueError("current kinematics differ from retained assembly")
        system = load_system(system_path)
        envelope, cells = pack_shapes(system)
        report["torso_containment"] = torso_containment(robot, envelope)
        report["cells_contained"] = all(
            abs(overlap(envelope, c) - c.volume) < 0.02 for c in cells
        )
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children),
            json.loads((assembly / "arm-report.json").read_text())["parts"],
        )
        frame = robot.forward_kinematics()["torso"]
        world_envelope = transform_shape(envelope, frame)
        report["collisions"] = collisions(world_envelope, parts)
        report["pack_envelope_m"] = system.pack.envelope_m
        report["pack_position_torso_m"] = system.pack.position_torso_m
        report["assembly_groups_checked"] = len(parts)
        report["status"] = (
            "pass"
            if report["cells_contained"]
            and not report["collisions"]
            and report["torso_containment"]["status"] == "pass"
            else "fail"
        )
        export_step(
            Compound(children=[envelope, *cells]), output / "pack-allocation.step"
        )
        # Existing imported bodies are in world mm. Render only the torso and its
        # attachments so the allocation stays legible; collision checks used all.
        view = []
        placed = {}
        for part in parts:
            if part.owner != "torso":
                continue
            color = (
                (0.60, 0.65, 0.67, 0.18)
                if "torso" in part.name
                else (0.3, 0.35, 0.4, 1)
            )
            view.append(
                Part(part.name, part.owner, part.shape, part.role, part.module, color)
            )
            placed[part.name] = part.shape
        view.append(
            Part(
                "pack_allocation",
                "torso",
                envelope,
                "allocation",
                "battery",
                (0.94, 0.60, 0.13, 0.25),
            )
        )
        placed["pack_allocation"] = world_envelope
        for c in cells:
            view.append(
                Part(c.label, "torso", c, "cell", "battery", (0.20, 0.55, 0.65, 1))
            )
            placed[c.label] = transform_shape(c, frame)
        render_core(view, placed, output, robot)
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            {k: report[k] for k in ("status", "collisions", "error") if k in report}
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--system", type=Path, default=Path("electronics/power-system.yaml")
    )
    args = parser.parse_args()
    raise SystemExit(
        int(run(args.assembly, args.output, args.system)["status"] != "pass")
    )
