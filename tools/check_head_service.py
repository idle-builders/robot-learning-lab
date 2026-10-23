"""Nominal rear-cover/Pi retention, service and sampled detailed-body fit."""

import argparse
from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
import sys
import traceback
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from build123d import (
    Compound,
    Pos,
    Rot,
    Cylinder,
    RegularPolygon,
    extrude,
    export_step,
    import_step,
)
from lab import head_service as hs
from lab.config import RobotConfig, HeadService, load_config
from lab.robot import make_robot
from lab.mechanical import part_shape, primitive, unsplit_head_shell
from lab.leg_mechanism import Part, render_core, transform_shape
from lab.camera_vendor import assembly_overlap as overlap
from tools.check_camera_support import CameraQueries, POSES, copy_source, sha, write
from tools.check_arm_episode import parts_from_step
from tools.check_visor_mount import planar_bearing
from tools.review_camera_mount import verify_study
from tools.cad_identity import geometry_record, geometry_agrees


def head_part_names(geometry):
    """Return deterministic, collision-free labels for head geometry entries.

    Unique non-eye roles retain their historical names. Eye roles keep the
    existing indexed form, and every repeated role receives its geometry index
    so no physical part is silently dropped from the local assembly map.
    """
    roles = [g.role for g in geometry]
    counts = {role: roles.count(role) for role in roles}
    names = [
        f"{role}_{index}" if counts[role] > 1 or role.startswith("eye_") else role
        for index, role in enumerate(roles)
    ]
    if len(set(names)) != len(names):
        raise ValueError("head geometry labels are not unique")
    return names


def compatible(current, baseline):
    a, b = current.model_dump(mode="json"), baseline.model_dump(mode="json")
    for item in (a, b):
        item.pop("name")
        item["head_service"] = HeadService().model_dump(mode="json")
        item["face"]["visor_mount_enabled"] = False
    if a != b:
        raise ValueError(
            "head service permits only name, head_service and visor mount changes"
        )


def bearing(shape, center, axis, normal, inner, outer):
    rotation = Rot(0, -90, 0) if axis == "x" else Rot(90, 0, 0)
    x, y, z = center
    point = (-z, y, x) if axis == "x" else (x, -z, y)
    result = planar_bearing(rotation * shape, *point, normal, inner, outer)
    if result["coverage"] > 1.001:
        result["status"] = "fail"
    return result


def tool(center, axis):
    # Canonical tool enters +Z, with 1 mm engagement at z=0 in a 1.25 mm
    # allocated socket. 1.9 AF bit + 2.3 mm diameter shank.
    bit = Pos(0, 0, -1) * extrude(RegularPolygon(0.95 / (3**0.5 / 2), 6), amount=2)
    bit += Pos(0, 0, -16) * Cylinder(1.15, 30)
    rotation = Rot(0, 90, 0) if axis == "x" else Rot(-90, 0, 0)
    return Pos(*center) * rotation * bit


def procurement_snapshot(config):
    catalog = yaml.safe_load(Path("robot/fastener-catalog.yaml").read_text())
    counts = {"screw_m2p5_8": 4, "screw_m2p5_6": 4, "insert_m2p5_5p7": 8}
    rows = []
    for units in (1000, 5000):
        items = []
        subtotal = 0.0
        for key, count in counts.items():
            part = catalog["parts"][key]
            quantity = units * count
            tier = max((t for t in part["tiers_gbp"] if t[0] <= quantity), default=None)
            amount = count * tier[1] if tier else None
            subtotal += amount or 0
            items.append(
                dict(
                    part=key,
                    candidate=part["procurement_candidate"],
                    count_per_robot=count,
                    order_quantity=quantity,
                    applied_published_tier=tier,
                    per_robot_gbp_ex_vat=amount,
                    source=part["source"],
                    status="published_snapshot_not_quote" if tier else "unquoted",
                )
            )
        rows.append(
            dict(
                production_units=units,
                items=items,
                priced_screw_subtotal_gbp_ex_vat=round(subtotal, 4),
                subtotal_scope="cover and Pi screws only; separate SKU quantities; highest qualifying retained tier, no extrapolated discount",
            )
        )
    return dict(
        status="partial_not_product_price",
        catalog_retrieved=catalog["retrieved"],
        variant=config.name,
        quantities=rows,
        unpriced=[
            "8 inserts",
            "modified front shell and rear cap with integral carrier, materials/finish/manufacturing",
            "heat-setting, assembly/test, yield, tooling and logistics",
        ],
        excluded_existing_components=[
            "Pi Zero 2 W and camera",
            "camera and visor hardware",
        ],
        full_1000_usd_sell_price_feasibility="unverified",
    )


def run(variant, baseline, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="fail", scope=__doc__, source_hashes={}, output_hashes={})
    try:
        prior = verify_study(baseline)
        if prior["status"] != "pass":
            raise ValueError("requires passing sealed camera baseline")
        cfg = load_config(variant)
        if not (cfg.head_service.enabled and cfg.head_service.pi_mount_enabled):
            raise ValueError("requires cover and Pi mount")
        old = RobotConfig.model_validate_json(
            (baseline / "resolved-config.json").read_text()
        )
        compatible(cfg, old)
        robot = make_robot(cfg)
        settings, d = cfg.head_service, cfg.dimensions
        sources = {str(p): p for p in Path("lab").glob("*.py")}
        for p in [
            Path(__file__).relative_to(Path.cwd()),
            Path("tools/check_camera_support.py"),
            Path("tools/check_visor_mount.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/review_camera_mount.py"),
            Path("tools/cad_identity.py"),
            Path("robot/base.yaml"),
            Path("robot/fastener-catalog.yaml"),
            Path("electronics/vision.yaml"),
            Path("electronics/camera-support-vendor.yaml"),
        ]:
            sources[str(p)] = p
        sources.update(
            {
                "variant.yaml": variant,
                "baseline/report.json": baseline / "report.json",
                "baseline/resolved-config.json": baseline / "resolved-config.json",
                "baseline/assembly.step": baseline / "source/retained/assembly.step",
                "baseline/assembly-report.json": baseline
                / "source/retained/assembly-report.json",
            }
        )
        for key in prior["source_hashes"]:
            if key.startswith("docs/design/references/"):
                sources[key] = baseline / "source" / key
        report["source_hashes"] = copy_source(output, sources)
        (output / "resolved-config.json").write_text(
            cfg.model_dump_json(indent=2) + "\n"
        )
        report["interface"] = hs.interface(d, settings)
        report["procurement"] = procurement_snapshot(cfg)
        head = next(link for link in robot.links if link.name == "head")
        unsplit = unsplit_head_shell(robot)
        outer = primitive(head.geometry[0])
        split = hs.split_parts(unsplit, outer, d, settings, head.geometry[0])
        local = {
            "head_shell": split["front"],
            "head_rear_cover": split["rear"],
            **hs.hardware(d, settings),
        }
        colors = {}
        for i, (g, name) in enumerate(
            zip(head.geometry, head_part_names(head.geometry), strict=True)
        ):
            colors[name] = g.color
            if name not in local:
                local[name] = (
                    Pos(*(v * 1000 for v in g.pos))
                    * Rot(*np.degrees(g.rpy))
                    * part_shape(robot, head, i)
                )
        report["geometry"] = {n: geometry_record(s) for n, s in local.items()}
        if any(
            not r["valid"]
            or r["solids"] != (631 if n == "support_camera_vendor_module" else 1)
            for n, r in report["geometry"].items()
        ):
            raise ValueError("invalid or disconnected head part")
        front, rear = split["front"], split["rear"]
        # Separate physical parts meet at nominal bearing planes. Do not fuse
        # those coincident faces merely to measure added material.
        added = Compound(
            children=[
                deepcopy(front) - deepcopy(unsplit),
                deepcopy(rear) - deepcopy(unsplit),
            ]
        )
        if not added.is_valid or not 0 < added.volume < front.volume + rear.volume:
            raise ValueError("added-stock Boolean returned invalid material")
        reserve = hs.pi_reservation(settings)
        bb = reserve.bounding_box()
        report["component_reservation_bounds_mm"] = dict(
            min=list(bb.min), max=list(bb.max)
        )
        # Diagnostic cuts use independent topology: OCCT may update coincident
        # input tolerances in a Boolean operation. Do not corrupt later queries.
        outside = deepcopy(added) - deepcopy(outer)
        report["exterior_excess_mm3"] = outside.volume
        if outside.solids() and (
            not outside.is_valid or outside.volume > added.volume + 0.02
        ):
            raise ValueError("exterior Boolean returned invalid material")
        report["new_stock_mm3"] = added.volume

        def parts(mapping):
            return [
                Part(n, "head", s, "head", "head", colors.get(n, (0.2, 0.7, 0.4, 0.25)))
                for n, s in mapping.items()
            ]

        q = CameraQueries()
        new_names = {"head_shell", "head_rear_cover", *hs.hardware(d, settings)}
        fixed = {n: s for n, s in local.items() if n not in new_names}
        report["head_conflicts"] = [
            dict(moving=n, **hit)
            for n, s in {
                "new_stock": added,
                **{
                    n: s
                    for n, s in local.items()
                    if n in new_names and n not in {"head_shell", "head_rear_cover"}
                },
                "component_reservation": reserve,
            }.items()
            for hit in q.collide(s, parts(fixed))
        ]
        # Check all pairs involving the new pieces, except prescribed insert/plastic
        # interference. Reservation is occupied components, excluding assumed pads.
        candidates = {n: s for n, s in local.items() if n in new_names}
        candidates["component_reservation"] = reserve
        report["new_pair_conflicts"] = []
        report["insert_interference_mm3"] = {}
        for i, (n, s) in enumerate(candidates.items()):
            for m, t in list(candidates.items())[i + 1 :]:
                if "component_reservation" in (n, m) and "pi_substrate" in (n, m):
                    continue
                host = (
                    "head_shell" in (n, m)
                    and any(v.startswith("head_cover_insert") for v in (n, m))
                ) or (
                    "head_rear_cover" in (n, m)
                    and any(v.startswith("pi_insert") for v in (n, m))
                )
                hits = q.collide(s, parts({m: t}))
                if host:
                    report["insert_interference_mm3"][n + ":" + m] = overlap(s, t)
                else:
                    report["new_pair_conflicts"] += [dict(moving=n, **h) for h in hits]
        report["bearing"] = []
        report["tools"] = []
        report["screw_removal"] = []
        report["tool_withdrawal"] = []
        tools_local = []
        screw_paths = []
        end = settings.split_x_m * 1000 - hs.SPLIT_GAP_MM / 2
        seat = end - hs.REAR_SEAT_MM
        for i, (y, z) in enumerate(hs.centers(d)):
            for name, shape, x, normal, inner, outer_r in [
                ("front_mating", front, end, -1, 2.35, 3.9),
                ("rear_mating", rear, end, 1, 2.35, 3.9),
                ("cover_screw_seat", rear, seat, -1, 1.35, 2.2),
            ]:
                report["bearing"].append(
                    dict(
                        index=i,
                        surface=name,
                        **bearing(shape, (x, y, z), "x", normal, inner, outer_r),
                    )
                )
            screw_paths.append((f"head_cover_screw_{i}", "x", (seat - 2.5, y, z)))
            sleeve = hs.cylinder_x(end + 2.85, y, z, 3.9, 5.7) - hs.cylinder_x(
                end + 2.85, y, z, 2.31, 6.7
            )
            fraction = overlap(sleeve, front) / sleeve.volume
            report["bearing"].append(
                dict(
                    index=i,
                    surface="cover_insert_surround",
                    coverage=fraction,
                    status="pass" if fraction >= 0.999 else "fail",
                )
            )
        px, py, pz = [v * 1000 for v in settings.pi_position_head_m]
        for i, (x, y, z) in enumerate(hs.pi_centers(settings)):
            for name, shape, plane, normal, inner, outer_r in [
                ("pi_post", rear, y, -1, 2.35, 3.4),
                ("pi_underside", local["pi_substrate"], y, 1, 2.35, 3.4),
                ("pi_screw_seat", local["pi_substrate"], py, -1, 1.4, 2.2),
            ]:
                report["bearing"].append(
                    dict(
                        index=i,
                        surface=name,
                        **bearing(shape, (x, plane, z), "y", normal, inner, outer_r),
                    )
                )
            screw_paths.append((f"pi_screw_{i}", "y", (x, py - 2.5, z)))
            sleeve = hs.cylinder_y(x, y + 2.85, z, 3.9, 5.7) - hs.cylinder_y(
                x, y + 2.85, z, 2.31, 6.7
            )
            fraction = overlap(sleeve, rear) / sleeve.volume
            report["bearing"].append(
                dict(
                    index=i,
                    surface="pi_insert_surround",
                    coverage=fraction,
                    status="pass" if fraction >= 0.999 else "fail",
                )
            )
        detached = {
            n: s
            for n, s in local.items()
            if n == "head_rear_cover" or n.startswith("pi_")
        }
        for n, axis, center in screw_paths:
            targets = local if axis == "x" else detached
            bit = tool(center, axis)
            hits = q.collide(bit, parts(targets))
            negative = overlap(
                Pos(0, 1.5, 0) * bit if axis == "x" else Pos(1.5, 0, 0) * bit, local[n]
            )
            report["tools"].append(
                dict(
                    part=n,
                    stage="assembled" if axis == "x" else "detached_cap",
                    hits=hits,
                    negative_mm3=negative,
                )
            )
            if axis == "x":
                tools_local.append((n, bit))
            for distance in (0, 3, 8, 20):
                shift = Pos(-distance, 0, 0) if axis == "x" else Pos(0, -distance, 0)
                report["tool_withdrawal"].append(
                    dict(
                        part=n,
                        distance_mm=distance,
                        stage="assembled" if axis == "x" else "detached_cap",
                        hits=q.collide(
                            shift * bit,
                            parts({k: s for k, s in targets.items() if k != n}),
                        ),
                    )
                )
                report["screw_removal"].append(
                    dict(
                        part=n,
                        distance_mm=distance,
                        hits=q.collide(
                            shift * local[n],
                            parts({k: s for k, s in targets.items() if k != n}),
                        ),
                    )
                )
        report["negative_control"] = dict(
            cap_forward_overlap_mm3=overlap(Pos(0.5, 0, 0) * rear, front)
        )
        carried = {
            "head_rear_cover": rear,
            **{n: s for n, s in local.items() if n.startswith("pi_")},
            "component_reservation": reserve,
        }
        remaining = {
            n: s
            for n, s in local.items()
            if n not in carried and not n.startswith("head_cover_screw")
        }
        report["cap_removal"] = [
            dict(
                rearward_mm=distance,
                moving=n,
                hits=q.collide(Pos(-distance, 0, 0) * s, parts(remaining)),
            )
            for distance in (0, 2, 5, 15, 35, 70, 110)
            for n, s in carried.items()
        ]
        report["pi_removal"] = [
            dict(
                distance_mm=distance,
                hits=q.collide(
                    Pos(0, -distance, 0) * local["pi_substrate"],
                    parts(
                        {
                            n: s
                            for n, s in detached.items()
                            if n == "head_rear_cover" or n.startswith("pi_insert")
                        }
                    ),
                ),
            )
            for distance in (0, 2, 5, 15, 35)
        ]
        write(output / "head-checkpoint.json", report)
        print(
            "head fit, seating and service checked; importing detailed body", flush=True
        )
        table = json.loads(sources["baseline/assembly-report.json"].read_text())
        body = parts_from_step(
            list(import_step(sources["baseline/assembly.step"]).children),
            table["parts"],
        )
        body = [p for p in body if p.name != "head_shell"]
        frames = robot.forward_kinematics()
        body_local = [
            replace(p, shape=transform_shape(p.shape, np.linalg.inv(frames["head"])))
            for p in body
        ]
        report["tools_body"] = [
            dict(
                part=n,
                rearward_mm=distance,
                hits=q.collide(Pos(-distance, 0, 0) * bit, body_local),
            )
            for n, bit in tools_local
            for distance in (0, 3, 8, 20)
        ]
        report["service_body"] = []
        for n, s in carried.items():
            for distance in (0, 2, 5, 15, 35, 70, 110):
                report["service_body"].append(
                    dict(
                        part=n,
                        rearward_mm=distance,
                        hits=q.collide(Pos(-distance, 0, 0) * s, body_local),
                    )
                )
        for n, _, _ in screw_paths:
            if not n.startswith("head_cover"):
                continue
            for distance in (0, 3, 8, 20):
                report["service_body"].append(
                    dict(
                        part=n,
                        rearward_mm=distance,
                        hits=q.collide(Pos(-distance, 0, 0) * local[n], body_local),
                    )
                )
        moving = {
            "new_stock": added,
            **{
                n: s
                for n, s in candidates.items()
                if n not in {"head_shell", "head_rear_cover"}
            },
        }
        attached = [p for p in body_local if p.owner == "head"]
        fixed_hits = {n: q.collide(s, attached) for n, s in moving.items()}
        report["body_motion"] = []
        for yaw, pitch in POSES:
            posed = robot.forward_kinematics(
                {"neck_yaw_joint": yaw, "head_joint": pitch}
            )
            targets = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape, posed[p.owner] @ np.linalg.inv(frames[p.owner])
                    ),
                )
                for p in body
                if p.owner != "head"
            ]
            hits = [
                dict(moving=n, **h)
                for n, s in moving.items()
                for h in fixed_hits[n]
                + q.collide(transform_shape(s, posed["head"]), targets)
            ]
            report["body_motion"].append(dict(yaw_rad=yaw, pitch_rad=pitch, hits=hits))
        print("detailed body checks complete; rendering", flush=True)
        for label, shift in [("assembled", 0), ("exploded", 75)]:
            folder = output / label
            folder.mkdir()
            located = {
                n: Pos(
                    -shift if n in carried or n.startswith("head_cover_screw") else 0,
                    0,
                    0,
                )
                * s
                for n, s in local.items()
            }
            render_core(parts(local), located, folder, robot, azimuth=30, elevation=-25)
        folder = output / "carrier"
        folder.mkdir()
        render_core(
            parts(detached), detached, folder, robot, azimuth=155, elevation=-25
        )
        for n, s in local.items():
            s.label = n
        export_step(Compound(children=list(local.values())), output / "head.step")
        imported = list(import_step(output / "head.step").children)
        if len(imported) != len(local) or {s.label for s in imported} != set(local):
            raise ValueError("exported head inventory differs")
        report["step_roundtrip"] = [
            dict(
                name=s.label,
                actual=geometry_record(s),
                expected=report["geometry"][s.label],
            )
            for s in imported
        ]
        changed = [
            r["name"]
            for r in report["step_roundtrip"]
            if not geometry_agrees(r["actual"], r["expected"])
        ]
        if changed:
            raise ValueError(f"exported head geometry differs: {changed}")
        report["not_run"] = [
            "actual populated Pi CAD, antenna/connector/cable/thermal keepouts",
            "assumed mounting-pad keepouts and substrate thickness qualification",
            "continuous sweeps, tolerances, insert installation, physical socket/tool fit and retention loads",
            "new mass/dynamics, complete manufacturing BOM and volume quotes",
        ]
        report["status"] = (
            "pass"
            if (
                not report["head_conflicts"]
                and not report["new_pair_conflicts"]
                and report["exterior_excess_mm3"] < 0.02
                and all(r["status"] == "pass" for r in report["bearing"])
                and all(
                    not r["hits"] and r["negative_mm3"] > 0.02 for r in report["tools"]
                )
                and report["negative_control"]["cap_forward_overlap_mm3"] > 0.02
                and not any(
                    r["hits"]
                    for key in [
                        "screw_removal",
                        "tool_withdrawal",
                        "cap_removal",
                        "pi_removal",
                        "tools_body",
                        "service_body",
                        "body_motion",
                    ]
                    for r in report[key]
                )
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
    for name in ("variant", "baseline", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.variant, args.baseline, args.output)
    print(
        json.dumps(
            dict(
                status=result["status"],
                error=result.get("error"),
                output=str(args.output),
            )
        )
    )
    raise SystemExit(0 if result["status"] == "pass" else 1)
