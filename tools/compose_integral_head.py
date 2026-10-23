"""Compose checked head shells/socket stacks into a complete candidate assembly.

The source robot's failures remain failed. Nominal added-stock checks and STEP
identity do not qualify continuous clearance, physical service or dynamics.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import Color, Compound, export_step, import_step
from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import transform_shape
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report
from tools.cad_identity import geometry_record, geometry_agrees
from tools.audit_integral_head_wings import volume


def run(name, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        scope=__doc__,
        physical_status="not_run",
        dynamics_status="not_run",
    )
    try:
        paths = [
            *Path("lab").glob("*.py"),
            *Path("tools").glob("*.py"),
            Path("electronics/power-system.yaml"),
        ]
        report["source_hashes"] = copy_source(output, {str(p): p for p in paths})
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        wings = Path(
            f"artifacts/{name}-integral-head-wings-{'v5' if name == 'pip' else 'v4'}"
        )
        bridge = Path(f"artifacts/{name}-head-bridge-service-v3")
        carrier = Path(f"artifacts/{name}-head-bridge-carrier-v1")
        base, wing_report, bridge_report, carrier_report = [
            verify_report(p) for p in [assembly, wings, bridge, carrier]
        ]
        bindings = {}

        def bind(p):
            bindings[str(p)] = sha(p)
            return p

        cfg = RobotConfig.model_validate_json(
            (assembly / "resolved-config.json").read_text()
        )
        for folder in [assembly, wings, bridge, carrier]:
            bind(folder / "report.json")
            bind(folder / "resolved-config.json")
            if (
                RobotConfig.model_validate_json(
                    (folder / "resolved-config.json").read_text()
                )
                != cfg
            ):
                raise ValueError("Configuration mismatch")
        required = [
            assembly / "assembly.step",
            wings / "head_shell.step",
            wings / "head_rear_cover.step",
        ]
        for path in required:
            bind(path)
            if bridge_report["source_bindings"][str(path)] != sha(
                path
            ) or carrier_report["source_bindings"][str(path)] != sha(path):
                raise ValueError("Study geometry binding mismatch")
        if (
            bridge_report["material_retention_status"] != "pass"
            or carrier_report["samples_status"] != "pass"
        ):
            raise ValueError("Expected socket/service evidence absent")
        robot = make_robot(cfg)
        frames = robot.forward_kinematics()
        contract = json.loads(bind(assembly / "interfaces.json").read_text())
        if json.loads(json.dumps(robot.interface())) != contract:
            raise ValueError("Baseline interface changed")
        write(output / "resolved-config.json", cfg.model_dump(mode="json"))
        write(output / "interfaces.json", contract)
        print(name, "importing complete robot", flush=True)
        world = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        records = {r["name"]: r for r in base["parts"]}
        local = [
            replace(
                p,
                shape=transform_shape(p.shape, np.linalg.inv(frames[p.owner])),
                color=tuple(records[p.name]["color"]),
            )
            for p in world
        ]
        replacements = {}
        for part in local:
            if part.name in ["head_shell", "head_rear_cover"]:
                replacements[part.name] = replace(
                    part,
                    shape=Compound(import_step(wings / f"{part.name}.step").solids()),
                )
            elif part.name in ["head_bridge_bolt_-5", "head_bridge_bolt_5"]:
                pieces = []
                for suffix in ["screw", "nut", "upper_washer", "lower_washer"]:
                    path = bind(bridge / f"{part.name}_{suffix}.step")
                    if carrier_report["source_bindings"][str(path)] != sha(path):
                        raise ValueError("Carrier hardware differs")
                    pieces.extend(import_step(path).solids())
                replacements[part.name] = replace(part, shape=Compound(pieces))
        if len(replacements) != 4:
            raise ValueError("Replacement inventory differs")
        combined = [replacements.get(p.name, p) for p in local]
        head_inv = np.linalg.inv(frames["head"])
        targets = [
            replace(p, shape=transform_shape(p.shape, head_inv @ frames[p.owner]))
            for p in combined
            if p.name not in ["head_shell", "head_rear_cover"]
        ]
        q = CameraQueries()
        rows = []
        for original in local:
            if original.name not in replacements:
                continue
            new = replacements[original.name]
            removed = original.shape - new.shape
            added = new.shape - original.shape
            row = dict(
                name=original.name,
                original_volume_mm3=volume(original.shape),
                replacement_volume_mm3=volume(new.shape),
                removed_mm3=volume(removed),
                added_mm3=volume(added),
            )
            if original.name.startswith("head_bridge_bolt_"):
                row["kind"] = "socket_subtraction"
                row["status"] = (
                    "pass"
                    if row["added_mm3"] < 0.02
                    and abs(row["removed_mm3"] - 7.03645640575) < 0.02
                    else "fail"
                )
            else:
                hits = q.collide(added, targets)
                row.update(
                    kind="integral_shell",
                    nominal_added_stock_hits=hits,
                    status="pass" if row["removed_mm3"] < 0.02 and not hits else "fail",
                )
            rows.append(row)
            write(output / "composition-progress.json", rows)
            print(name, original.name, row["status"], flush=True)
        report.update(
            source_assembly_status=base["status"],
            source_assembly=str(assembly),
            replacements=list(replacements),
            composition_checks=rows,
            nominal_composition_status="pass"
            if all(r["status"] == "pass" for r in rows)
            else "fail",
            part_count=len(combined),
            source_bindings=bindings,
        )
        print(name, "exporting complete assembly", flush=True)
        world = [
            replace(
                p, shape=Compound(transform_shape(p.shape, frames[p.owner]).solids())
            )
            for p in combined
        ]
        for p in world:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in world]), output / "assembly.step"
        )
        inventory = [
            dict(
                name=p.name,
                owner=p.owner,
                role=p.role,
                module=p.module,
                color=p.color,
                volume_mm3=p.shape.volume,
            )
            for p in world
        ]
        print(name, "checking STEP roundtrip", flush=True)
        restored = {
            p.name: p.shape
            for p in parts_from_step(
                list(import_step(output / "assembly.step").children), inventory
            )
        }
        identity = []
        for p in world:
            a, b = geometry_record(p.shape), geometry_record(restored[p.name])
            identity.append(
                dict(
                    name=p.name,
                    expected=a,
                    actual=b,
                    status="pass" if geometry_agrees(b, a) else "fail",
                )
            )
            if len(identity) % 25 == 0:
                write(output / "step-roundtrip-progress.json", identity)
                print(name, "STEP identity", len(identity), "/", len(world), flush=True)
        write(output / "step-roundtrip.json", identity)
        passed = all(r["status"] == "pass" for r in identity)
        report.update(
            resolved_config_sha256=sha(output / "resolved-config.json"),
            parts=inventory,
            assembly_step_roundtrip="pass" if passed else "fail",
            geometry_build_status="pass" if passed else "fail",
            status="fail"
            if base["status"] == "fail"
            or not passed
            or report["nominal_composition_status"] == "fail"
            else "partial",
            not_run=[
                "full old-old self-clearance",
                "continuous motion clearance",
                "new task/controller execution",
                "harness/strength/physical qualification",
            ],
            limits=[
                "Same 216/222 CAD group identities and shared kinematic frames; two bridge groups each contain one screw, two washers and one nut.",
                "Removable assembly carrier is not installed or included in robot mass.",
                "Unchanged scene/contact/control evidence retains its original geometry and legacy mass scope; no automatic transfer to this composition.",
                "Full source robot failures remain failed even when this narrower composition and export pass.",
            ],
        )
    except Exception:
        report["error"] = traceback.format_exc()
    report["output_hashes"] = {
        p.name: sha(p)
        for p in output.iterdir()
        if p.is_file() and p.name != "report.json"
    }
    write(output / "report.json", report)
    print(
        name,
        {
            k: report[k]
            for k in [
                "status",
                "geometry_build_status",
                "nominal_composition_status",
                "error",
            ]
            if k in report
        },
        flush=True,
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("name", choices=["pip", "scout"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    run(args.name, args.output)
