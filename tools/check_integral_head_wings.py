"""Fuse relieved lower wings into existing service halves; bounded CAD study."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback
import numpy as np
import yaml
from build123d import Pos, Box, Compound, import_step, export_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.mechanical import primitive
from lab.leg_mechanism import Part, transform_shape, overlap
from lab.head_service import pi_reservation
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.check_head_cap_fasteners import transformed_bounds, bounds_gap
from tools.audit_integral_head_wings import volume
from tools.head_wing_bounds import strip_bounds


def run(name, specification, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        nominal_status="not_run",
        motion_status="not_run",
        service_status="not_run",
        physical_status="not_run",
        parts=[],
        poses=[],
        service=[],
    )
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            specification,
            Path("tools/check_head_cap_fasteners.py"),
            Path("tools/audit_integral_head_wings.py"),
            Path("tools/head_wing_bounds.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/compose_forearm_enclosure.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        spec = yaml.safe_load(specification.read_text())
        report["specification"] = spec
        lift = spec["seam_bridge_translation_mm"]
        margin = spec["moving_clearance_mm"]
        if not 0.3 < lift <= 1 or margin != 1.5:
            raise ValueError("Unsupported seam bridge/margin")
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        relief = Path(f"artifacts/{name}-head-cap-relief-v3")
        base, prior = [verify_report(p) for p in (assembly, relief)]
        cfgs = [
            RobotConfig.model_validate_json((p / "resolved-config.json").read_text())
            for p in (assembly, relief)
        ]
        if (
            cfgs[0] != cfgs[1]
            or prior["fit_status"] != "pass"
            or len(prior["motion"]) != 37
        ):
            raise ValueError("Incompatible source")
        cfg = cfgs[0]
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        q = CameraQueries()
        bindings = [
            p / f
            for p in (assembly, relief)
            for f in ("report.json", "resolved-config.json")
        ] + [assembly / "assembly.step"]
        if prior["source_bindings"][str(assembly / "assembly.step")] != sha(
            assembly / "assembly.step"
        ):
            raise ValueError("Assembly changed")
        bindings += [relief / f"{p['name']}.step" for p in prior["parts"]]
        report["source_bindings"] = {str(p): sha(p) for p in bindings}
        report["inherited_assembly_status"] = base["status"]
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        print(name, "importing complete source assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        head = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral["head"])))
            for p in parts
            if p.owner == "head"
        ]
        reserve = Part(
            "pi_component_reservation",
            "head",
            pi_reservation(cfg.head_service),
            "reservation",
            "head_service",
            (1, 0, 0),
        )
        head.append(reserve)
        lookup = {p.name: p for p in head}
        outer = primitive(
            next(link for link in robot.links if link.name == "head").geometry[0]
        )
        wing_shapes = {}
        motion_shapes = {}
        modified = {key: lookup[key].shape for key in ("head_shell", "head_rear_cover")}
        for p in prior["parts"]:
            key = p["name"]
            old = import_step(relief / f"{key}.step")
            extension = Pos(0, 0, lift) * old
            if spec.get("bridge_mode") == "top_strip":
                seam_overlap = spec["seam_overlap_mm"]
                if not 0 < seam_overlap < 0.1:
                    raise ValueError("Unsupported strip overlap")
                top = cfg.dimensions.neck_lower_port_top_m * 1000 - prior["seam_mm"]
                extension &= Pos(0, 0, top + (lift - seam_overlap) / 2) * Box(
                    300, 300, lift + seam_overlap
                )
            wing = old + extension
            motion_shapes[key] = (
                extension if spec.get("bridge_mode") == "top_strip" else wing
            )
            export_step(extension, output / f"{key}-extension.step")
            mate = "head_shell" if "_front_" in key else "head_rear_cover"
            row = dict(
                name=key,
                mate=mate,
                valid=wing.is_valid,
                solids=len(wing.solids()),
                source_volume_mm3=old.volume,
                wing_volume_mm3=wing.volume,
                mate_overlap_mm3=overlap(wing, lookup[mate].shape),
                exterior_excess_mm3=max(0, wing.volume - overlap(wing, outer)),
                corridor_overlap_mm3=overlap(wing, Box(300, 58, 300)),
            )
            row["head_collisions"] = q.collide(
                wing, [p for p in head if p.name != mate]
            )
            row["status"] = (
                "pass"
                if row["valid"]
                and row["solids"] == 1
                and row["mate_overlap_mm3"] > 0.02
                and row["exterior_excess_mm3"] <= 0.02
                and row["corridor_overlap_mm3"] <= 0.02
                and not row["head_collisions"]
                else "fail"
            )
            report["parts"].append(row)
            wing_shapes[key] = wing
            modified[mate] += wing
            export_step(wing, output / f"{key}.step")
        report["modified_halves"] = []
        for key, shape in modified.items():
            original = lookup[key].shape
            # Separate serialized topology before coincident Boolean diagnostics.
            export_step(shape, output / f"{key}.step")
            shape = import_step(output / f"{key}.step")
            modified[key] = shape
            row = dict(
                name=key,
                valid=shape.is_valid,
                solids=len(shape.solids()),
                net_added_volume_mm3=volume(shape) - volume(original),
                volume_method="adaptive surface integration, tolerance 1e-7",
                omitted_original_volume_mm3=volume(
                    Compound(original.solids()) - Compound(shape.solids())
                ),
            )
            row["status"] = (
                "pass"
                if row["valid"]
                and row["solids"] == 1
                and row["omitted_original_volume_mm3"] <= 0.02
                else "fail"
            )
            report["modified_halves"].append(row)
        report["cross_half_overlap_mm3"] = overlap(
            modified["head_shell"], modified["head_rear_cover"]
        )
        report["nominal_status"] = (
            "pass"
            if all(
                p["status"] == "pass"
                for p in report["parts"] + report["modified_halves"]
            )
            and report["cross_half_overlap_mm3"] <= 0.02
            else "fail"
        )
        write(output / "progress.json", report)
        print(name, "nominal", report["nominal_status"], flush=True)
        source_boxes = {p.name: q.bounds(p.shape) for p in parts if p.owner != "head"}
        wing_boxes = {
            key: (np.array(list(q.bounds(s).min)), np.array(list(q.bounds(s).max)))
            for key, s in motion_shapes.items()
        }
        report["seam_construction_bounds"] = []
        if spec.get("bridge_mode") == "top_strip":
            for key, extension in motion_shapes.items():
                low, high = strip_bounds(
                    robot, key, spec, prior["seam_mm"], q.bounds(extension)
                )
                bound = Pos(*((low + high) / 2)) * Box(*(high - low))
                missing = volume(Compound(extension.solids()) - bound)
                report["seam_construction_bounds"].append(
                    dict(
                        wing=key,
                        lower_mm=low.tolist(),
                        upper_mm=high.tolist(),
                        missing_extension_mm3=missing,
                    )
                )
                if missing > 0.02:
                    raise ValueError("Construction bounds omit new seam material")
                wing_boxes[key] = low, high
        for pose in prior["motion"]:
            fk = robot.forward_kinematics(pose["pose_rad"], enforce_limits=False)
            failures = []
            near_count = 0
            for target in (p for p in parts if p.owner != "head"):
                tf = (
                    np.linalg.inv(fk["head"])
                    @ fk[target.owner]
                    @ np.linalg.inv(neutral[target.owner])
                )
                bound = transformed_bounds(source_boxes[target.name], tf)
                near = [
                    key
                    for key, b in wing_boxes.items()
                    if bounds_gap(b, bound) <= margin + 1e-7
                ]
                if not near:
                    continue
                located = transform_shape(target.shape, tf)
                for key in near:
                    near_count += 1
                    gap, a, b = motion_shapes[key].distance_to_with_closest_points(
                        located
                    )
                    if (
                        abs(np.linalg.norm(np.array(list(a)) - np.array(list(b))) - gap)
                        > 1e-5
                    ):
                        raise ValueError("Witness inconsistency")
                    if gap < margin - 1e-5:
                        failures.append(
                            dict(
                                wing=key,
                                target=target.name,
                                distance_mm=gap,
                                wing_point_head_mm=list(a),
                                target_point_head_mm=list(b),
                            )
                        )
            report["poses"].append(
                dict(
                    name=pose["name"],
                    pose_rad=pose["pose_rad"],
                    status="fail" if failures else "pass",
                    failures=failures,
                    exact_near_pairs=near_count,
                )
            )
            write(output / "progress.json", report)
            print(name, pose["name"], len(failures), "gap failures", flush=True)
        report["motion_status"] = (
            "pass" if all(p["status"] == "pass" for p in report["poses"]) else "fail"
        )
        # Rear cover includes its Pi parts/reservation; cover screws must first be removed.
        carried = {
            p.name: p.shape
            for p in head
            if p.name == "head_rear_cover" or p.name.startswith("pi_")
        }
        carried["head_rear_cover"] = modified["head_rear_cover"]
        remaining = [
            replace(p, shape=modified[p.name]) if p.name == "head_shell" else p
            for p in head
            if p.name not in carried and not p.name.startswith("head_cover_screw_")
        ]
        remaining += [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(neutral["head"])))
            for p in parts
            if p.owner != "head"
        ]
        for offset in spec["rearward_service_offsets_mm"]:
            failures = []
            for key, shape in carried.items():
                failures += [
                    dict(moving=key, **hit)
                    for hit in q.collide(Pos(-offset, 0, 0) * shape, remaining)
                ]
            report["service"].append(
                dict(
                    kind="rear_cover_with_pi",
                    offset_mm=offset,
                    status="fail" if failures else "pass",
                    failures=failures,
                )
            )
            write(output / "progress.json", report)
            print(name, "rear service", offset, len(failures), flush=True)
        # On the detached cover, Pi screws are removed before board extraction.
        pi_targets = [
            replace(lookup["head_rear_cover"], shape=modified["head_rear_cover"])
        ] + [p for p in head if p.name.startswith("pi_insert_")]
        for offset in spec["pi_service_offsets_mm"]:
            failures = q.collide(
                Pos(0, -offset, 0) * lookup["pi_substrate"].shape, pi_targets
            )
            report["service"].append(
                dict(
                    kind="detached_pi_substrate",
                    offset_mm=offset,
                    status="fail" if failures else "pass",
                    failures=failures,
                )
            )
        report["service_status"] = (
            "pass" if all(p["status"] == "pass" for p in report["service"]) else "fail"
        )
        report["status"] = (
            "partial"
            if all(
                report[k] == "pass"
                for k in ("nominal_status", "motion_status", "service_status")
            )
            else "fail"
        )
        report["limits"] = [
            "Study overlay only: source robot geometry, part count, mass, dynamics, purchasing and print release are unchanged.",
            "Seam bridge construction is specified by bridge_mode; original cap STEP is retained exactly in each union. No inferred strength.",
            "For top_strip, source-bound original wing clearance is reused only at the same 37 poses; new extension alone receives exact gap checks. Other modes screen the full changed wing.",
            "37 discrete poses use all retained non-head parts, conservative transformed bounds and exact gaps near the 1.5 mm allowance; no continuous sweep.",
            "Rear-cover removal assumes four existing cover screws removed and absent/disconnected harness; driver engagement/withdrawal and screw removal not rechecked here.",
            "Detached Pi check covers substrate only; populated hardware, connectors, cable disconnection, bridge bolt/tool/nut access and printed strength remain unqualified.",
            "Original whole-assembly failures remain; adding integral wings does not prove complete neck protection or independent cap replaceability.",
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
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("experiments/head-integral-wings.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    outcome = run(args.robot, args.specification, args.output)
    raise SystemExit(1 if outcome["status"] == "fail" else 0)
