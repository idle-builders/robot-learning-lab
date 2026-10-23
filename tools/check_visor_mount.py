"""Scout visor attachment, bearing stock, service and sampled body clearances."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from build123d import (
    Circle,
    Compound,
    Cylinder,
    Pos,
    RegularPolygon,
    Rot,
    export_step,
    extrude,
    import_step,
)

from lab.config import RobotConfig, load_config
from lab.leg_mechanism import Part, render_core, transform_shape
from lab.mechanical import part_shape, primitive
from lab.robot import make_robot
from lab.visor_mount import centers, interface, SEAT_THICKNESS_MM
from lab.camera_vendor import assembly_overlap as overlap
from lab.vision import load_vision, pi_geometry
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, POSES, copy_source, sha, write
from tools.check_vision import allocation
from tools.review_camera_mount import verify_study


def ring(x, y, z, outer, inner, depth):
    return Pos(x, y, z) * (Cylinder(outer, depth) - Cylinder(inner, depth + 1))


def planar_bearing(shape, x, y, z, normal_z, inner, outer):
    witness = Pos(x, y, z) * (Circle(outer) - Circle(inner))
    area = 0.0
    for face in shape.faces():
        if (
            str(face.geom_type).endswith("PLANE")
            and abs(face.center().Z - z) < 1e-6
            and abs(face.normal_at().Z - normal_z) < 1e-6
        ):
            common = face & witness
            if common is not None:
                area += common.area
    fraction = area / witness.area
    return dict(
        plane_z_mm=z,
        area_mm2=area,
        coverage=fraction,
        status="pass" if fraction >= 0.999 else "fail",
    )


def compatible(current, baseline):
    a, b = current.model_dump(mode="json"), baseline.model_dump(mode="json")
    for c in (a, b):
        c.pop("name")
        c["face"]["visor_mount_enabled"] = False
    if a != b:
        raise ValueError("visor study permits only name and visor mount enable changes")


def run(variant, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__, source_hashes={}, output_hashes={})
    try:
        old_report = verify_study(baseline)
        if old_report["status"] != "pass":
            raise ValueError("baseline camera study must pass")
        config = load_config(variant)
        if not config.face.visor_mount_enabled:
            raise ValueError("requires enabled visor mount")
        old_config = RobotConfig.model_validate_json(
            (baseline / "resolved-config.json").read_text()
        )
        compatible(config, old_config)
        robot, old = make_robot(config), make_robot(old_config)
        report["interface"] = interface(config.dimensions)
        sources = {str(p): p for p in Path("lab").glob("*.py")}
        sources.update(
            {
                str(p): p
                for p in [
                    Path("tools/check_visor_mount.py"),
                    Path("tools/check_camera_support.py"),
                    Path("tools/check_arm_episode.py"),
                    Path("tools/check_neck_collar.py"),
                    Path("tools/check_vision.py"),
                    Path("tools/review_camera_mount.py"),
                    Path("robot/base.yaml"),
                    Path("robot/fastener-catalog.yaml"),
                    Path("electronics/vision.yaml"),
                    Path("electronics/camera-support-vendor.yaml"),
                ]
            }
        )
        sources.update(
            {
                "variant.yaml": variant,
                "baseline/report.json": baseline / "report.json",
                "baseline/resolved-config.json": baseline / "resolved-config.json",
                "baseline/assembly.step": baseline / "source/retained/assembly.step",
                "baseline/assembly-report.json": baseline
                / "source/retained/assembly-report.json",
                "baseline/face-report.json": baseline / "source/study/report.json",
            }
        )
        for key in old_report["source_hashes"]:
            if key.startswith("docs/design/references/"):
                sources[key] = baseline / "source" / key
        report["source_hashes"] = copy_source(output, sources)
        (output / "resolved-config.json").write_text(
            config.model_dump_json(indent=2) + "\n"
        )
        head = next(link for link in robot.links if link.name == "head")
        old_head = next(link for link in old.links if link.name == "head")
        local = {}
        colors = {}
        for i, g in enumerate(head.geometry):
            name = f"{g.role}_{i}" if g.role.startswith("eye_") else g.role
            local[name] = (
                Pos(*(v * 1000 for v in g.pos))
                * Rot(*np.degrees(g.rpy))
                * part_shape(robot, head, i)
            )
            colors[name] = g.color
        report["geometry"] = {
            name: dict(valid=s.is_valid, solids=len(s.solids()), volume_mm3=s.volume)
            for name, s in local.items()
        }
        if any(
            not r["valid"]
            or r["solids"] != (631 if name == "support_camera_vendor_module" else 1)
            for name, r in report["geometry"].items()
        ):
            raise ValueError("invalid or disconnected part")
        print("head generated", flush=True)
        shell, visor = local["head_shell"], local["brow"]
        old_shell = part_shape(old, old_head, 0)
        old_brow_g = next(g for g in old_head.geometry if g.role == "brow")
        old_brow = Pos(*(v * 1000 for v in old_brow_g.pos)) * primitive(old_brow_g)
        added_shell = shell - old_shell
        report["changes"] = dict(
            added_shell_mm3=added_shell.volume,
            visor_mm3=visor.volume,
            old_solid_visor_mm3=old_brow.volume,
            visor_outside_prior_envelope_mm3=overlap(visor, visor - old_brow),
        )
        moving = {
            "brow": visor,
            "added_shell_bosses": added_shell,
            **{n: s for n, s in local.items() if n.startswith("visor_")},
        }
        targets = [
            Part(n, "head", s, "head", "head", colors[n])
            for n, s in local.items()
            if n not in moving and n != "head_shell"
        ]
        pi_position = json.loads(sources["baseline/face-report.json"].read_text())[
            "selected_computer_reservation"
        ]["position_head_mm"]
        report["selected_computer_position_mm"] = pi_position
        pi = allocation(
            pi_geometry(load_vision()),
            np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
            pi_position,
        )
        targets.append(
            Part(
                "pi_reservation",
                "head",
                pi,
                "allocation",
                "vision",
                (0.2, 0.7, 0.4, 0.3),
            )
        )
        q = CameraQueries()
        report["head_conflicts"] = [
            {"moving": n, **hit}
            for n, s in moving.items()
            for hit in q.collide(s, targets)
        ]
        report["mating"] = dict(
            visor_shell_mm3=overlap(visor, shell),
            screw_shell_mm3=sum(
                overlap(s, shell)
                for n, s in moving.items()
                if n.startswith("visor_screw")
            ),
            screw_visor_mm3=sum(
                overlap(s, visor)
                for n, s in moving.items()
                if n.startswith("visor_screw")
            ),
            inserts_visor_mm3=sum(
                overlap(s, visor)
                for n, s in moving.items()
                if n.startswith("visor_insert")
            ),
        )
        report["bearing"] = []
        report["tools"] = []
        tools_local = []
        report["screw_removal"] = []
        for i, (x, y, z) in enumerate(centers(config.dimensions)):
            contact = []
            for name, shape, height, inner, outer in [
                ("shell_seat", shell, z - 0.025, 2.35, 3.9),
                ("visor_seat", visor, z + 0.025, 2.35, 3.9),
                ("head_seat", visor, z + SEAT_THICKNESS_MM - 0.025, 1.35, 2.2),
            ]:
                probe = ring(x, y, height, outer, inner, 0.05)
                fraction = overlap(probe, shape) / probe.volume
                contact.append(
                    dict(
                        surface=name,
                        coverage=fraction,
                        status="pass" if fraction >= 0.999 else "fail",
                    )
                )
            sleeve = ring(x, y, z - 2.85, 3.9, 2.31, 5.7)
            fraction = overlap(sleeve, shell) / sleeve.volume
            contact.append(
                dict(
                    surface="insert_surround",
                    coverage=fraction,
                    status="pass" if fraction >= 0.999 else "fail",
                )
            )
            report["bearing"].append(dict(index=i, witnesses=contact))
            for label, shape, height, normal, inner, outer in [
                ("shell_mating_face", shell, z, 1, 2.35, 3.9),
                ("visor_mating_face", visor, z, -1, 2.35, 3.9),
                ("screw_bearing_face", visor, z + SEAT_THICKNESS_MM, 1, 1.35, 2.2),
            ]:
                contact.append(
                    dict(
                        surface=label,
                        **planar_bearing(shape, x, y, height, normal, inner, outer),
                    )
                )
            # 1.9 mm AF tool engages 1 mm into the allocated socket; round shank
            # begins above the head. Physical driver/socket depth remains unmeasured.
            bit = Pos(x, y, z + SEAT_THICKNESS_MM + 1.5) * extrude(
                RegularPolygon(0.95 / (3**0.5 / 2), 6), amount=2
            )
            bit += Pos(x, y, z + SEAT_THICKNESS_MM + 17.5) * Cylinder(1.15, 30)
            tools_local.append(bit)
            all_targets = [
                Part(n, "head", s, "head", "head", colors[n]) for n, s in local.items()
            ]
            hits = q.collide(bit, all_targets)
            screw = local[f"visor_screw_{i}"]
            negative = overlap(Pos(1.5, 0, 0) * bit, screw)
            report["tools"].append(
                dict(
                    index=i,
                    hits=hits,
                    shifted_tool_interference_mm3=negative,
                    status="pass" if not hits and negative > 0.02 else "fail",
                )
            )
            removal_targets = [p for p in all_targets if p.name != f"visor_screw_{i}"]
            for distance in (0, 3, 8, 16):
                report["screw_removal"].append(
                    dict(
                        index=i,
                        up_mm=distance,
                        hits=q.collide(Pos(0, 0, distance) * screw, removal_targets),
                    )
                )
        # Deliberate downward visor displacement must reveal a seating collision.
        report["negative_control"] = dict(
            downward_visor_interference_mm3=overlap(Pos(0, 0, -0.5) * visor, shell)
        )
        fixed = [
            Part("head_shell", "head", shell, "head", "head", colors["head_shell"]),
            *targets,
        ]
        fixed += [
            Part(n, "head", s, "hardware", "head", colors[n])
            for n, s in moving.items()
            if n.startswith("visor_insert")
        ]
        report["removal"] = [
            dict(up_mm=z, hits=q.collide(Pos(0, 0, z) * visor, fixed))
            for z in (0, 2, 5, 15, 30)
        ]
        print("seating, tools and head fit checked", flush=True)
        assembly_report = json.loads(
            sources["baseline/assembly-report.json"].read_text()
        )
        body = parts_from_step(
            list(import_step(sources["baseline/assembly.step"]).children),
            assembly_report["parts"],
        )
        body = [p for p in body if p.name != "head_shell"]
        frames = robot.forward_kinematics()
        head_targets = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(frames["head"])))
            for p in body
            if p.owner == "head"
        ]
        all_body_local = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(frames["head"])))
            for p in body
        ]
        report["tools_body"] = [
            dict(index=i, hits=q.collide(bit, all_body_local))
            for i, bit in enumerate(tools_local)
        ]
        # Service is evaluated at neutral, with the opposing screw removed.
        # Head-part collisions above and detailed-body collisions here are both
        # necessary; a clear installed pose does not prove a removal path.
        report["service_body"] = []
        for name, shape, distances in [
            ("brow", visor, (0, 2, 5, 15, 30)),
            *[
                (n, s, (0, 3, 8, 16))
                for n, s in moving.items()
                if n.startswith("visor_screw")
            ],
        ]:
            for distance in distances:
                report["service_body"].append(
                    dict(
                        part=name,
                        up_mm=distance,
                        hits=q.collide(Pos(0, 0, distance) * shape, all_body_local),
                    )
                )
        fixed_hits = {n: q.collide(s, head_targets) for n, s in moving.items()}
        report["body_motion"] = []
        for yaw, pitch in POSES:
            posed = robot.forward_kinematics(
                {"neck_yaw_joint": yaw, "head_joint": pitch}
            )
            body_targets = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape, posed[p.owner] @ np.linalg.inv(frames[p.owner])
                    ),
                )
                for p in body
                if p.owner != "head"
            ]
            hits = []
            for n, s in moving.items():
                shape = transform_shape(s, posed["head"])
                hits += [
                    dict(moving=n, **hit)
                    for hit in fixed_hits[n] + q.collide(shape, body_targets)
                ]
            report["body_motion"].append(dict(yaw_rad=yaw, pitch_rad=pitch, hits=hits))
        print("15 body poses checked", flush=True)
        shown = [
            Part(n, "head", s, "head", "head", colors[n]) for n, s in local.items()
        ]
        for name, shift in [("assembled", 0), ("exploded", 25)]:
            folder = output / name
            folder.mkdir()
            located = {
                p.name: Pos(
                    0,
                    0,
                    shift
                    if p.name == "brow"
                    else 2 * shift
                    if p.name.startswith("visor_screw")
                    else 0,
                )
                * p.shape
                for p in shown
            }
            render_core(shown, located, folder, robot, azimuth=30, elevation=-25)
        for n, s in local.items():
            s.label = n
        export_step(Compound(children=list(local.values())), output / "head.step")
        report["not_run"] = [
            "continuous motion / tolerances",
            "insert installation and retention loads",
            "heat-set installation tool envelope and process",
            "physical socket depth / tool fit",
            "Pi mounting and cable route",
            "new mass/dynamics and complete production BOM",
        ]
        report["status"] = (
            "pass"
            if (
                not report["head_conflicts"]
                and max(report["mating"].values()) < 0.02
                and all(
                    w["status"] == "pass"
                    for b in report["bearing"]
                    for w in b["witnesses"]
                )
                and all(t["status"] == "pass" for t in report["tools"])
                and report["negative_control"]["downward_visor_interference_mm3"] > 0.02
                and not any(
                    r["hits"]
                    for r in report["removal"]
                    + report["body_motion"]
                    + report["screw_removal"]
                    + report["tools_body"]
                    + report["service_body"]
                )
                and report["changes"]["visor_outside_prior_envelope_mm3"] < 0.02
            )
            else "fail"
        )
    except Exception as exc:
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    report["output_hashes"] = {
        str(p.relative_to(output)): sha(p)
        for p in output.rglob("*")
        if p.is_file()
        and "source" not in p.relative_to(output).parts
        and p.name not in {"report.json", "report.sha256"}
    }
    write(output / "report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.variant, args.baseline, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "error": result.get("error"),
                "output": str(args.output),
            }
        )
    )
    raise SystemExit(0 if result["status"] == "pass" else 1)
