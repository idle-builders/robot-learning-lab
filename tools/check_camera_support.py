"""Bounded nominal study for the split camera carrier and populated module."""

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from build123d import Compound, Pos, Rot, export_step, import_step

from lab.artifact_store import snapshot_file
from lab.camera_mount import datum
from lab.camera_support import (
    carrier_parts,
    hardware_parts,
    interface,
    module_parts,
    centers,
    cylinder,
    aperture_check,
    specification,
)
from lab.config import RobotConfig, load_config
from lab.face_layout import check_face
from lab.leg_mechanism import Part, render_core, transform_shape
from lab.camera_vendor import assembly_overlap as overlap
from lab.mechanical import part_shape
from lab.robot import make_robot
from lab.vision import load_vision, pi_geometry
from tools.check_arm_episode import parts_from_step
from tools.check_vision import allocation
from tools.check_neck_collar import ExactQueries
from tools.review_camera_mount import verify_study
from tools.check_camera_tools import check as check_tools


THRESHOLD = 0.02
POSES = [(y, p) for y in (-1.2, -0.6, 0, 0.6, 1.2) for p in (-0.5, 0, 0.5)]


class CameraQueries(ExactQueries):
    def nearest(self, shape, targets, initial=None):
        box = self.bounds(shape)
        candidates = []
        for index, p in enumerate(targets):
            b = self.bounds(p.shape)
            gap = [
                max(lo - hi2, lo2 - hi, 0)
                for lo, hi, lo2, hi2 in zip(box.min, box.max, b.min, b.max)
            ]
            candidates.append((float(np.linalg.norm(gap)), index, p))
        best = initial or (float("inf"), "")
        for lower_bound, _, p in sorted(candidates):
            # An AABB distance is a lower bound on the exact shape distance.
            # Keep a small numerical allowance and evaluate ties explicitly.
            if lower_bound > best[0] + 1e-7:
                break
            best = min(best, (shape.distance_to(p.shape), p.name))
        if not np.isfinite(best[0]):
            raise ValueError("nearest query requires nonempty valid targets")
        return best

    def collide(self, shape, targets):
        hits = []
        box = self.bounds(shape)
        for p in targets:
            b = self.bounds(p.shape)
            if any(
                min(x, y) < max(u, v)
                for x, y, u, v in zip(box.max, b.max, box.min, b.min)
            ):
                continue
            volume = overlap(shape, p.shape)
            if volume > THRESHOLD:
                hits.append({"part": p.name, "overlap_mm3": volume})
        return hits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def copy_source(output, names):
    root = output / "source"
    hashes = {}
    for name, path in names.items():
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        snapshot_file(path, dest)
        hashes[name] = sha(dest)
    return hashes


def locate(shape, frame, offset=None):
    if offset is not None:
        shape = Pos(*offset) * shape
    return transform_shape(shape, frame)


def config_compatible(current, old):
    a, b = current.model_dump(mode="json"), old.model_dump(mode="json")
    a.pop("name")
    b.pop("name")
    for config in (a, b):
        config["camera"]["support_enabled"] = False
        config["camera"]["support_profile"] = "drawing"
        config["face"]["brow_rear_x_m"] = None
    if a != b:
        raise ValueError(
            "support variant changes parameters outside support_enabled/support_profile/brow_rear_x_m/name"
        )


def run(variant, study, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "fail",
        "scope": __doc__,
        "source_hashes": {},
        "output_hashes": {},
    }
    try:
        retained = verify_study(study)
        assembly_key = next(
            k for k in retained["source_hashes"] if k.endswith("/assembly.step")
        )
        assembly = study / "source" / assembly_key
        assembly_report_path = (
            study
            / "source"
            / assembly_key.removesuffix("/assembly.step")
            / "report.json"
        )
        if not assembly_report_path.exists():
            assembly_report_path = (
                study
                / "source"
                / assembly_key.replace("/assembly.step", "/report.json")
            )
        assembly_report = json.loads(assembly_report_path.read_text())
        current = make_robot(load_config(variant))
        if not current.config.camera.support_enabled:
            raise ValueError("requires enabled camera support")
        old_cfg = RobotConfig.model_validate_json(
            (study / "resolved-config.json").read_text()
        )
        old = make_robot(old_cfg)
        config_compatible(current.config, old.config)
        if current.interface()["joint_order"] != old.interface()["joint_order"]:
            raise ValueError("support variant changes joint order")
        spec = specification(current.config.camera.support_profile)
        source_names = {
            "study/report.json": study / "report.json",
            "retained/assembly.step": assembly,
            "retained/assembly-report.json": assembly_report_path,
            "variant.yaml": variant,
            "electronics/camera-support.yaml": Path("electronics/camera-support.yaml"),
            "electronics/vision.yaml": Path("electronics/vision.yaml"),
            "docs/design/references/vision/camera-module-3-wide-mechanical-drawing.pdf": Path(
                "docs/design/references/vision/camera-module-3-wide-mechanical-drawing.pdf"
            ),
        }
        source_names[spec["manifest_path"]] = Path(spec["manifest_path"])
        if spec["profile"] == "vendor_step":
            from lab.camera_vendor import model

            source_names[spec["source_vendor"]] = Path(spec["source_vendor"])
            source_names["lab/camera_vendor.py"] = Path("lab/camera_vendor.py")
            report["vendor_geometry"] = model(spec)[1]
        source_names.update(
            {
                f"lab/{p.name}": p
                for p in [
                    Path("lab/camera_support.py"),
                    Path("lab/camera_mount.py"),
                    Path("lab/mechanical.py"),
                    Path("lab/robot.py"),
                    Path("lab/styling.py"),
                ]
            }
        )
        for path in [
            Path("lab/config.py"),
            Path("lab/vision.py"),
            Path("lab/leg_mechanism.py"),
            Path("lab/packaging.py"),
            Path("lab/face_layout.py"),
            Path("lab/camera_vendor.py"),
            Path("tools/check_camera_tools.py"),
            Path("tools/check_vision.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/review_camera_mount.py"),
            Path("robot/base.yaml"),
            Path("electronics/power-system.yaml"),
            Path("electronics/logic-supply/verification.yaml"),
        ]:
            source_names[str(path)] = path
        source_names["retained/resolved-config.json"] = study / "resolved-config.json"
        source_names["tools/check_camera_support.py"] = Path(__file__)
        report["source_hashes"] = copy_source(output, source_names)
        (output / "resolved-config.json").write_text(
            current.config.model_dump_json(indent=2) + "\n"
        )
        shapes = list(import_step(assembly).children)
        parts = parts_from_step(shapes, assembly_report["parts"])
        frames = current.forward_kinematics()
        head = next(link for link in current.links if link.name == "head")
        parts = [
            replace(
                p, shape=transform_shape(part_shape(current, head, 0), frames["head"])
            )
            if p.name == "head_shell"
            else p
            for p in parts
        ]
        body = [p for p in parts if p.name != "head_shell"]
        shell = next(p for p in parts if p.name == "head_shell")

        print("body imported; building camera parts", flush=True)
        support_local = {}
        support_local.update(carrier_parts(spec))
        support_local.update(module_parts(spec))
        support_local.update(hardware_parts(spec))
        head_frame = frames["head"]
        off = datum(current)
        support = [
            Part(
                name,
                "head",
                locate(shape, head_frame, off),
                "camera_support",
                "vision",
                (0.3, 0.55, 0.65, 1),
            )
            for name, shape in support_local.items()
        ]
        face = []
        for i, g in enumerate(head.geometry[1:], start=1):
            if g.role.startswith("eye_") or g.role == "brow":
                local = (
                    Pos(*(v * 1000 for v in g.pos))
                    * Rot(*np.degrees(g.rpy))
                    * part_shape(current, head, i)
                )
                face.append(
                    Part(
                        f"visual_{g.role}_{i}",
                        "head",
                        transform_shape(local, head_frame),
                        "visual",
                        "head",
                        g.color,
                    )
                )
        q = CameraQueries()
        support_pairs = []
        for i, left in enumerate(support):
            if not left.shape.is_valid or (
                left.name != "camera_vendor_module" and len(left.shape.solids()) != 1
            ):
                support_pairs.append(
                    {
                        "parts": [left.name],
                        "status": "fail",
                        "reason": "invalid/disconnected",
                    }
                )
            for right in support[i + 1 :]:
                v = overlap(left.shape, right.shape)
                if v > THRESHOLD:
                    support_pairs.append(
                        {
                            "parts": [left.name, right.name],
                            "overlap_mm3": v,
                            "status": "fail",
                        }
                    )
        report["support_self_interference"] = support_pairs
        print("support self-interference checked", flush=True)

        module = [
            p
            for p in support
            if p.name.startswith("camera_")
            and p.name not in {"camera_rear_carrier", "camera_front_bezel"}
        ]
        carrier = next(p for p in support if p.name == "camera_rear_carrier")
        bezel = next(p for p in support if p.name == "camera_front_bezel")
        # Reuse the selected Pi reservation from the sealed face study.
        pi_record = retained.get("selected_computer_reservation", {})
        pi_position = pi_record.get("position_head_mm")
        if pi_position is None:
            raise ValueError("retained face study lacks selected Pi position")
        pi_shape = allocation(
            pi_geometry(load_vision()),
            np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
            pi_position,
        )
        pi = Part(
            "selected_pi_reservation",
            "head",
            transform_shape(pi_shape, head_frame),
            "allocation",
            "vision",
            (0.2, 0.65, 0.4, 0.3),
        )
        fixed_targets = [shell, *body, *face, pi]
        report["installed_conflicts"] = {
            "carrier_body": q.collide(carrier.shape, fixed_targets),
            "bezel_body": q.collide(bezel.shape, fixed_targets),
            "module_body": [
                dict(part=p.name, hits=q.collide(p.shape, fixed_targets))
                for p in module
            ],
        }
        report["insertion"] = []
        for distance in (0, 5, 15, 30):
            moving = [
                replace(p, shape=Pos(-distance, 0, 0) * p.shape)
                for p in [carrier, *module]
            ]
            hits = []
            for p in moving:
                hits += [
                    {"moving_part": p.name, **h} for h in q.collide(p.shape, [bezel])
                ]
            report["insertion"].append({"rearward_mm": distance, "conflicts": hits})
        report["cassette_removal"] = []
        for distance in (0, 5, 15, 30):
            hits = []
            for p in [carrier, bezel, *module]:
                moved = Pos(distance, 0, 0) * p.shape
                hits += [
                    {"moving_part": p.name, **h}
                    for h in q.collide(moved, fixed_targets)
                ]
            report["cassette_removal"].append(
                {"forward_mm": distance, "conflicts": hits}
            )

        front_tools, rear_tools = [], []
        for i, (y, z) in enumerate(centers()):
            front_tools.append(
                Part(
                    f"front_driver_{i}",
                    "head",
                    locate(cylinder(9.5, y, z, 1.5, 32), head_frame, off),
                    "tool",
                    "check",
                    (0.8, 0.3, 0.2, 1),
                )
            )
            rear_tools.append(
                Part(
                    f"rear_driver_{i}",
                    "head",
                    locate(cylinder(-20, y, z, 3, 10), head_frame, off),
                    "tool",
                    "check",
                    (0.8, 0.3, 0.2, 1),
                )
            )
        report["driver_access"] = []
        for i, (front, rear) in enumerate(zip(front_tools, rear_tools, strict=True)):
            # The front driver check is explicitly before bezel installation;
            # the rear driver sees the installed carrier and nut pocket.
            targets = [
                p
                for p in [carrier, *module]
                if p.name not in {f"camera_board_screw_{i}", f"camera_board_nut_{i}"}
            ]
            report["driver_access"].append(
                {
                    "fastener": i,
                    "front_hits": q.collide(front.shape, targets),
                    "rear_hits": q.collide(rear.shape, targets),
                }
            )

        print("installed fit and assembly tools checked", flush=True)
        report["motion"] = []
        report["brow_clearance"] = []
        head_targets = [p for p in body if p.owner == "head"]
        fixed_hits = {
            p.name: q.collide(p.shape, head_targets) for p in [*support, *face]
        }
        fixed_brow_distance = {
            p.name: q.nearest(p.shape, head_targets) for p in face if "brow" in p.name
        }
        report["motion_query_scope"] = (
            "head-owned pairs share one rigid transform, so neutral exact results are reused; all other pairs evaluated at each pose"
        )
        for yaw, pitch in POSES:
            posed = current.forward_kinematics(
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
            hits = []
            for p in [*support, *face]:
                moved = replace(
                    p,
                    shape=transform_shape(
                        p.shape, posed["head"] @ np.linalg.inv(frames["head"])
                    ),
                )
                hits += [
                    {"moving_part": p.name, **h}
                    for h in fixed_hits[p.name] + q.collide(moved.shape, targets)
                ]
                if "brow" in p.name:
                    nearest = q.nearest(
                        moved.shape, targets, initial=fixed_brow_distance[p.name]
                    )
                    report["brow_clearance"].append(
                        dict(
                            yaw_rad=yaw,
                            pitch_rad=pitch,
                            nearest_part=nearest[1],
                            minimum_gap_mm=nearest[0],
                            status="pass" if nearest[0] >= 1 else "fail",
                        )
                    )
            report["motion"].append(
                {"yaw_rad": yaw, "pitch_rad": pitch, "conflicts": hits}
            )

        print("motion checked", flush=True)
        report["interface"] = interface(spec)
        margin = report["interface"]["nut_full_engagement_margin_mm"]
        report["contact_stack"] = {
            "pcb_front_x_mm": report["interface"]["pcb_front_x_mm"],
            "pcb_rear_x_mm": report["interface"]["pcb_rear_x_mm"],
            "carrier_split_x_mm": report["interface"]["carrier_split_x_mm"],
            "support_length_mm": report["interface"]["support_length_mm"],
            "nut_full_engagement_margin_mm": margin,
            "status": "pass"
            if margin > 0 and report["interface"]["support_length_mm"] > 0
            else "fail",
            "scope": "selected nominal support planes and candidate fastener envelopes",
        }
        local_pcb = support_local[
            "camera_vendor_module" if spec["profile"] == "vendor_step" else "camera_pcb"
        ]
        local_carrier = support_local["camera_rear_carrier"]
        contacts = []
        from build123d import Box

        for i, (y, z) in enumerate(centers()):
            rear = report["interface"]["pcb_rear_x_mm"]
            support_witness = Pos(rear - 0.025, y + 1.5, z) * Box(0.05, 0.2, 0.2)
            pcb_witness = Pos(rear + 0.025, y + 1.5, z) * Box(0.05, 0.2, 0.2)
            contacts.append(
                dict(
                    index=i,
                    support_stock_mm3=overlap(support_witness, local_carrier),
                    pcb_stock_mm3=overlap(pcb_witness, local_pcb),
                    nut_to_plate_gap_mm=support_local[
                        f"camera_board_nut_{i}"
                    ].distance_to(local_carrier),
                    screw_to_pcb_gap_mm=support_local[
                        f"camera_board_screw_{i}"
                    ].distance_to(local_pcb),
                )
            )
        report["contact_witnesses"] = contacts
        rings = []
        for i, (y, z) in enumerate(centers()):
            for side, plane, direction, radius in [
                (
                    "rear",
                    report["interface"]["pcb_rear_x_mm"],
                    1,
                    spec["support_diameter_mm"] / 2,
                ),
                (
                    "front",
                    report["interface"]["pcb_front_x_mm"],
                    -1,
                    spec["fastener"]["screw_nominal_head_diameter_mm"] / 2,
                ),
            ]:
                probe = cylinder(plane + direction * 0.025, y, z, radius, 0.05)
                probe -= cylinder(plane + direction * 0.025, y, z, 1.1, 0.1)
                expected = probe.volume
                stock = overlap(probe, local_pcb)
                shifted = overlap(probe, Pos(0, 0.5, 0) * local_pcb)
                rings.append(
                    dict(
                        index=i,
                        side=side,
                        expected_stock_mm3=expected,
                        actual_stock_mm3=stock,
                        coverage_fraction=stock / expected,
                        displaced_board_coverage_fraction=shifted / expected,
                        status="pass"
                        if stock / expected >= 0.999 and shifted / expected < 0.95
                        else "fail",
                    )
                )
        report["annular_bearing_witnesses"] = dict(
            status="pass" if all(r["status"] == "pass" for r in rings) else "fail",
            rows=rings,
            scope="50 micrometre-deep full bearing-ring stock in selected CAD, plus 0.5 mm lateral board-displacement negative control; not physical preload or tolerance qualification",
        )
        if any(
            c["support_stock_mm3"] < 0.0019
            or c["pcb_stock_mm3"] < 0.0019
            or c["nut_to_plate_gap_mm"] > 1e-5
            or c["screw_to_pcb_gap_mm"] > 1e-5
            for c in contacts
        ):
            report["contact_stack"]["status"] = "fail"
        report["tool_witnesses"] = check_tools(spec)
        report["aperture_envelope"] = aperture_check(
            support_local["camera_front_bezel"]
        )
        report["face_layout"] = check_face(current)
        shown = [shell, *face, carrier, bezel, *module]
        (output / "assembled").mkdir()
        (output / "exploded").mkdir()
        render_core(
            shown, {p.name: p.shape for p in shown}, output / "assembled", current
        )
        exploded = [
            replace(p, shape=Pos(-25 if p is carrier else 25, 0, 0) * p.shape)
            for p in [carrier, *module]
        ]
        render_core(
            exploded,
            {p.name: p.shape for p in exploded},
            output / "exploded",
            current,
            azimuth=25,
            elevation=-12,
        )
        for p in shown:
            p.shape.label = p.name
        export_step(
            Compound(children=[p.shape for p in shown]), output / "camera-support.step"
        )
        report["not_run"] = [
            "actual flex mating/bend",
            "tolerances",
            "mass and dynamics",
            "thread torque/material strength",
        ]
        report["status"] = (
            "pass"
            if not report["support_self_interference"]
            and all(row["status"] == "pass" for row in report["brow_clearance"])
            and report["face_layout"]["status"] == "pass"
            and report["contact_stack"]["status"] == "pass"
            and report["tool_witnesses"]["status"] == "pass"
            and report["annular_bearing_witnesses"]["status"] == "pass"
            and not (
                report["installed_conflicts"]["carrier_body"]
                or report["installed_conflicts"]["bezel_body"]
                or any(
                    row["hits"] for row in report["installed_conflicts"]["module_body"]
                )
            )
            and report["aperture_envelope"]["status"] == "pass"
            and not any(
                r["conflicts"]
                for r in report["insertion"]
                + report["cassette_removal"]
                + report["motion"]
            )
            and not any(
                r["front_hits"] or r["rear_hits"] for r in report["driver_access"]
            )
            else "fail"
        )
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    write(output / "report.json", report)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.variant, args.study, args.output)
    print(
        json.dumps({"status": result["status"], "output": str(args.output)}, indent=2)
    )
    raise SystemExit(0 if result["status"] in {"pass", "partial"} else 1)


if __name__ == "__main__":
    main()
