"""Nominal fit, gravity support and sampled service of a removable printed aid."""

import argparse
from dataclasses import replace
from pathlib import Path
import traceback
import numpy as np
import yaml
from build123d import Pos, Compound, import_step, export_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape, overlap
from lab.head_service import pi_reservation
from lab.head_bridge_carrier import carrier
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.cap_mesh import export_checked_stl


def support_area(upper, lower):
    # With nominal horizontal contact, a20micrometre downward witness gives
    # the supporting area. This is geometry, not a stiffness/gravity solver.
    return overlap(Pos(0, 0, -0.02) * upper, lower) / 0.02


def run(name, specification, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        fits=[],
        supports=[],
        samples=[],
        physical_status="not_run",
        strength_status="not_run",
        harness_status="not_run",
    )
    try:
        files = [
            *Path("lab").glob("*.py"),
            Path(__file__).relative_to(Path.cwd()),
            specification,
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/compose_forearm_enclosure.py"),
            Path("tools/cap_mesh.py"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in files})
        spec = yaml.safe_load(specification.read_text())
        report["specification"] = spec
        bridge = Path(f"artifacts/{name}-head-bridge-service-v3")
        r = verify_report(bridge)
        if (
            r["staged_service_status"] != "pass"
            or r["material_retention_status"] != "pass"
        ):
            raise ValueError("Requires checked bridge route and socket identity")
        cfg = RobotConfig.model_validate_json(
            (bridge / "resolved-config.json").read_text()
        )
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        base = verify_report(assembly)
        if cfg != RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        ):
            raise ValueError("Configuration mismatch")
        wing = Path(
            f"artifacts/{name}-integral-head-wings-{'v5' if name == 'pip' else 'v4'}"
        )
        verify_report(wing)
        bindings = {}

        def bind(p):
            bindings[str(p)] = sha(p)
            return p

        for p in [
            bridge / "report.json",
            bridge / "resolved-config.json",
            assembly / "report.json",
            assembly / "assembly.step",
            wing / "report.json",
            wing / "head_shell.step",
            wing / "head_rear_cover.step",
        ]:
            bind(p)
        for p in [
            assembly / "assembly.step",
            wing / "head_shell.step",
            wing / "head_rear_cover.step",
        ]:
            if r["source_bindings"][str(p)] != sha(p):
                raise ValueError("Bridge/assembly geometry mismatch")
        report["source_bindings"] = bindings
        report["inherited_assembly_status"] = base["status"]
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        tool = carrier(spec)
        export_step(tool, output / "carrier.step")
        # Put the common flat underside on the print-bed datum, in millimetres.
        printable = Pos(0, 0, -tool.bounding_box().min.Z) * tool
        mesh, cleanup = export_checked_stl(printable, output / "carrier-print.stl")
        report["print_candidate"] = dict(
            file="carrier-print.stl",
            quantity=1,
            volume_mm3=tool.volume,
            bounds_mm=list(printable.bounding_box().size),
            mesh=cleanup,
            material="unselected",
            process="unqualified; flat underside atZ0, no slice or strength result",
        )
        print(name, "import source assembly", flush=True)
        robot = make_robot(cfg)
        inv = np.linalg.inv(robot.forward_kinematics()["head"])
        parts = [
            replace(p, shape=transform_shape(p.shape, inv))
            for p in parts_from_step(
                list(import_step(assembly / "assembly.step").children), base["parts"]
            )
        ]
        for i, p in enumerate(parts):
            if p.name in ["head_shell", "head_rear_cover"]:
                parts[i] = replace(p, shape=import_step(wing / f"{p.name}.step"))
        parts.append(
            Part(
                "pi_component_reservation",
                "head",
                pi_reservation(cfg.head_service),
                "reservation",
                "head",
                (1, 0, 0),
            )
        )
        hardware = {}
        for x in [-5, 5]:
            key = f"head_bridge_bolt_{x}"
            old = next(p for p in parts if p.name == key)
            parts = [p for p in parts if p.name != key]
            hardware[key] = {}
            for suffix in ["screw", "nut", "lower_washer", "upper_washer"]:
                path = bind(bridge / f"{key}_{suffix}.step")
                shape = import_step(path)
                hardware[key][suffix] = shape
                parts.append(replace(old, name=f"{key}_{suffix}", shape=shape))
        q = CameraQueries()
        for x in [5, -5]:
            key = f"head_bridge_bolt_{x}"
            hw = hardware[key]
            placed = Pos(x, 0, 0) * tool
            absent = {"head_rear_cover"} | {
                p.name
                for p in parts
                if p.name.startswith("pi_") or p.name.startswith("head_cover_screw_")
            }
            if x == 5:
                absent |= {
                    p.name for p in parts if p.name.startswith("head_bridge_bolt_-5_")
                }
            stage = [p for p in parts if p.name not in absent]
            hits = q.collide(placed, stage)
            report["fits"].append(
                dict(
                    part=key,
                    kind="seated_carrier",
                    hits=hits,
                    status="fail" if hits else "pass",
                )
            )
            for label, upper, lower in [
                ("nut_on_carrier", hw["nut"], placed),
                ("washer_on_nut", hw["lower_washer"], hw["nut"]),
            ]:
                area = support_area(upper, lower)
                # Moving a supporting aid downward must remove its contact.
                missing = support_area(upper, Pos(0, 0, -2) * lower)
                report["supports"].append(
                    dict(
                        part=key,
                        kind=label,
                        area_mm2=area,
                        displaced_support_mm2=missing,
                        status="pass"
                        if area >= spec["minimum_nut_support_area_mm2"]
                        and missing < 0.02
                        else "fail",
                    )
                )
            loaded = Compound(
                [*placed.solids(), *hw["nut"].solids(), *hw["lower_washer"].solids()]
            )
            for distance in spec["rear_offsets_mm"]:
                for kind, shape, targets in [
                    (
                        "loaded_insertion",
                        loaded,
                        [p for p in stage if not p.name.startswith(key + "_")],
                    ),
                    ("unloaded_withdrawal", placed, stage),
                ]:
                    hits = q.collide(Pos(-distance, 0, 0) * shape, targets)
                    report["samples"].append(
                        dict(
                            part=key,
                            kind=kind,
                            rear_offset_mm=distance,
                            stage="front_first" if x == 5 else "rear_second",
                            absent_parts=sorted(absent),
                            hits=hits,
                            status="fail" if hits else "pass",
                        )
                    )
            screw_driver = import_step(bind(bridge / f"{key}-driver.step"))
            hit = overlap(placed, screw_driver)
            report["fits"].append(
                dict(
                    part=key,
                    kind="driver_with_carrier",
                    hits=[] if hit < 0.02 else [{"part": "driver", "overlap_mm3": hit}],
                    status="pass" if hit < 0.02 else "fail",
                )
            )
            write(output / "progress.json", report)
            print(name, key, "fit", report["fits"][-2]["status"], flush=True)
        for key in ["fits", "supports", "samples"]:
            report[key + "_status"] = (
                "pass"
                if all(row["status"] == "pass" for row in report[key])
                else "fail"
            )
        report["status"] = (
            "partial"
            if all(
                report[k + "_status"] == "pass" for k in ["fits", "supports", "samples"]
            )
            else "fail"
        )
        report["limits"] = [
            "Removable level-held positioning aid, not a permanent robot component or final-torque wrench. Nut and washer load fromabove outside the robot; no inverted or dynamic capture claim.",
            "Contact witnesses prove nominal horizontal supporting area, not strength, creep, friction, print tolerances or a gravity/contact simulation. Open-front guides permit withdrawal but do not latch the hardware.",
            "Head supported, rear cover/Pi removed and harness disconnected. Front stack first while rearabsent; rear second. Loaded insertion assumes own entirestack absent; withdrawal retains engaged screw/nut/washers.",
            "Sampled translations only; no continuous sweep, actual hand/grip, threadengagement or instrumented prototype. Print orientation is a candidate; no slicer or manufacturingrelease.",
        ]
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "report.json"
        }
        write(output / "report.json", report)
        return report
    except Exception:
        report.update(status="fail", error=traceback.format_exc())
        write(output / "report.json", report)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("robot", choices=["pip", "scout"])
    p.add_argument(
        "--specification",
        type=Path,
        default=Path("experiments/head-bridge-carrier.yaml"),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    r = run(args.robot, args.specification, args.output)
    raise SystemExit(r["status"] == "fail")
