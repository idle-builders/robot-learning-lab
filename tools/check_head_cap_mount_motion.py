"""Screen selected unbored cap mount stock through the retained 37 poses."""

import argparse
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
import traceback

import numpy as np
from build123d import Compound, Pos, import_step

from lab.config import RobotConfig
from lab.robot import make_robot
from lab.leg_mechanism import Part, transform_shape
from tools.head_cap_relief import expanded_box
from tools.check_arm_episode import parts_from_step
from tools.check_camera_support import CameraQueries, copy_source, sha, write
from tools.compose_forearm_enclosure import verify_report


def material_point(shape):
    """Find a classified material point, never assume a hollow part's center is solid."""
    for solid in shape.solids():
        bounds = CameraQueries().bounds(solid)
        lo, hi = np.array(list(bounds.min)), np.array(list(bounds.max))
        for fractions in product((0.5, 0.1, 0.9, 0.03, 0.97, 0.01, 0.99), repeat=3):
            point = lo + np.array(fractions) * (hi - lo)
            if solid.is_inside(point.tolist(), tolerance=1e-6):
                return point
    raise ValueError("No classified material point found for detector control")


def run(name, selection, output, envelope_mode="box"):
    root = Path.cwd()
    selection = selection.absolute()
    output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="fail",
        motion_status="not_run",
        scope=__doc__,
        margin_mm=1.5,
        attachment_status="not_run",
        service_status="not_run",
        physical_status="not_run",
        continuous_motion_status="not_run",
    )
    try:
        sources = [
            *Path("lab").glob("*.py"),
            Path("tools/check_head_cap_mount_motion.py"),
            Path("tools/head_cap_relief.py"),
            Path("tools/head_cap_mount_envelope.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/compose_forearm_enclosure.py"),
        ]
        result["source_hashes"] = copy_source(output, {str(p): p for p in sources})
        bindings = {}

        def bind(path):
            path = path.absolute()
            bindings[str(path.relative_to(root))] = sha(path)
            return path

        review = verify_report(selection.parent)
        bind(selection.parent / "report.json")
        if review["status"] != "pass":
            raise ValueError("Selection review failed")
        selected = json.loads(bind(selection).read_text())
        rows = [r for r in selected if r["robot"] == name]
        if len(rows) != 1 or not rows[0]["selected"]:
            raise ValueError("Expected one nonempty robot selection")
        entries = rows[0]["selected"]
        assembly = Path(f"artifacts/{name}-distal-grip-integrated-v1")
        relief = Path(f"artifacts/{name}-head-cap-relief-v3")
        probe = Path(f"artifacts/{name}-head-cap-mount-stock-v3")
        base = verify_report(assembly)
        caps = verify_report(relief)
        stock_report = verify_report(probe)
        for path in [
            assembly / "report.json",
            assembly / "assembly.step",
            assembly / "resolved-config.json",
            relief / "report.json",
            relief / "resolved-config.json",
            probe / "report.json",
            probe / "resolved-config.json",
        ]:
            bind(path)
        assembly_key = str(assembly / "assembly.step")
        for producer in [caps, stock_report]:
            if producer["source_bindings"][assembly_key] != sha(
                assembly / "assembly.step"
            ):
                raise ValueError("Producer and assembly identities differ")
        configs = [
            RobotConfig.model_validate_json(
                (folder / "resolved-config.json").read_text()
            )
            for folder in [assembly, relief, probe]
        ]
        if not configs[0] == configs[1] == configs[2]:
            raise ValueError("Configuration mismatch")
        if (
            caps["fit_status"] != "pass"
            or len(caps["motion"]) != 37
            or any(r["status"] != "pass" for r in caps["motion"])
        ):
            raise ValueError("Requires all 37 cap screens to have passed")
        cfg = configs[0]
        (output / "resolved-config.json").write_text(cfg.model_dump_json(indent=2))
        result.update(
            source_bindings=bindings,
            inherited_assembly_status=base["status"],
            selected_wings=sorted(entries),
            missing_wings=sorted(set(p["name"] for p in caps["parts"]) - set(entries)),
            requested_pose_count=37,
            selected_cases=[],
            poses=[],
        )
        robot = make_robot(cfg)
        stock, envelopes = {}, {}
        result["envelope_mode"] = envelope_mode
        for key, row in entries.items():
            if (
                row not in stock_report["cases"]
                or row["cover"] != key
                or row["status"] != "pass"
            ):
                raise ValueError("Selected row is not the exact passing probe case")
            x, y, _ = row["center_head_mm"]
            folder = probe / key / f"x{x:g}-y{y:g}"
            pieces = [
                import_step(bind(folder / f"{part}.step")) for part in ["boss", "tab"]
            ]
            if any(not p.is_valid or len(p.solids()) != 1 for p in pieces):
                raise ValueError("Stock input invalid or disconnected")
            stock[key] = Compound([solid for p in pieces for solid in p.solids()])
            if envelope_mode == "primitives":
                from tools.head_cap_mount_envelope import envelope

                envelopes[key], record = envelope(
                    robot, stock_report["specification"], row, stock[key]
                )
            else:
                envelopes[key] = expanded_box(stock[key], 1.5)
                record = dict(method="whole stock expanded AABB", margin_mm=1.5)
            result["selected_cases"].append(
                dict(
                    cover=key,
                    clearance_envelope=record,
                    center_head_mm=row["center_head_mm"],
                    input_paths={
                        part: str(folder / f"{part}.step") for part in ["boss", "tab"]
                    },
                )
            )
        cap_shapes = {
            p["name"]: import_step(bind(relief / f"{p['name']}.step"))
            for p in caps["parts"]
        }
        robot = make_robot(cfg)
        neutral = robot.forward_kinematics()
        print(name, "importing exact assembly", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), base["parts"]
        )
        nonhead = [p for p in parts if p.owner != "head"]
        q = CameraQueries()
        combined = []
        for key, shape in stock.items():
            targets = [
                Part(other, "head", s, "unbored_stock", "study", (1, 0, 0))
                for other, s in stock.items()
                if other != key
            ]
            targets += [
                Part(other, "head", s, "unmounted_cap", "study", (1, 0, 0))
                for other, s in cap_shapes.items()
                if other != key
            ]
            combined += [dict(cover=key, **hit) for hit in q.collide(shape, targets)]
        result["combined_stock_conflicts"] = combined
        result["combined_stock_status"] = "fail" if combined else "pass"
        # Translate actual stock into actual yaw housing to verify interference detection.
        target = next(p for p in nonhead if p.name == "neck_yaw_housing")
        target = replace(
            target, shape=transform_shape(target.shape, np.linalg.inv(neutral["head"]))
        )
        original = next(iter(stock.values()))

        first_row = next(iter(entries.values()))
        stock_point = np.array(first_row["center_head_mm"], dtype=float)
        stock_point[2] += stock_report["specification"]["boss_depth_mm"] / 2
        if not any(
            solid.is_inside(stock_point.tolist(), tolerance=1e-6)
            for solid in original.solids()
        ):
            raise ValueError("Unbored post's control point is outside source material")
        target_point = material_point(target.shape)
        displacement = target_point - stock_point
        hits = q.collide(Pos(*displacement) * original, [target])
        result["negative_control"] = dict(
            status="pass" if hits else "fail",
            displacement_head_mm=displacement.tolist(),
            stock_point_head_mm=stock_point.tolist(),
            target_material_point_head_mm=target_point.tolist(),
            collisions=hits,
        )
        for index, row in enumerate(caps["motion"]):
            fk = robot.forward_kinematics(row["pose_rad"], enforce_limits=False)
            located = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape,
                        np.linalg.inv(fk["head"])
                        @ fk[p.owner]
                        @ np.linalg.inv(neutral[p.owner]),
                    ),
                )
                for p in nonhead
            ]
            actual, margin = [], []
            for key, shape in stock.items():
                actual += [dict(cover=key, **hit) for hit in q.collide(shape, located)]
                margin += [
                    dict(cover=key, **hit) for hit in q.collide(envelopes[key], located)
                ]
            result["poses"].append(
                dict(
                    name=row["name"],
                    pose_rad=row["pose_rad"],
                    status="fail" if actual or margin else "pass",
                    exact_collisions=actual,
                    conservative_clearance_collisions=margin,
                )
            )
            write(output / "progress.json", result)
            print(
                name,
                index + 1,
                row["name"],
                "actual",
                len(actual),
                "margin",
                len(margin),
                flush=True,
            )
        result["motion_status"] = (
            "pass" if all(p["status"] == "pass" for p in result["poses"]) else "fail"
        )
        result["status"] = (
            "partial"
            if result["motion_status"] == "pass"
            and not combined
            and hits
            and not result["missing_wings"]
            else "fail"
        )
        result["evaluated_pose_count"] = len(result["poses"])
        result["limits"] = [
            "Original cap geometry is unchanged and source-bound; its prior 37-pose pass is reused only at identical poses.",
            "Both whole-stock boxes and unions of expanded primitives are conservative; consult envelope_mode and per-case construction records.",
            "No screw/head/insert, bore, tool or service geometry is qualified here.",
            "Missing wing locations prevent a complete attachment-set result even when selected stock clears.",
            "Inherited source assembly failures remain; no physical or continuous-motion qualification.",
        ]
        result["output_hashes"] = {
            "resolved-config.json": sha(output / "resolved-config.json")
        }
        write(output / "report.json", result)
        return result
    except Exception:
        result["error"] = traceback.format_exc()
        write(output / "report.json", result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", choices=["pip", "scout"])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envelope-mode", choices=["box", "primitives"], default="box")
    args = parser.parse_args()
    outcome = run(args.robot, args.selection, args.output, args.envelope_mode)
    print(outcome["status"])
    raise SystemExit(1 if outcome["status"] == "fail" else 0)
