"""Compose source-bound detailed body, complete head inventory and XH supply PCB.

Checks only the replacement/new interactions; preserves prior scoped evidence.
Does not establish complete harness, strength, continuous motion or power readiness.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import json
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from build123d import Box, Color, Compound, Pos, Rot, export_step, import_step
from lab.config import RobotConfig, load_config
from lab.export import export_all
from lab.robot import make_robot
from lab.mechanical import part_shape
from lab.leg_mechanism import Part, render_core, transform_shape
from lab.logic_packaging import placement
from lab.power_pack import pack_shapes, torso_containment
from lab.power_system import load_system
from lab.camera_vendor import assembly_overlap as overlap
from lab.logic_supply import compiled_circuit, spice, validate_config
from tools.check_arm_episode import parts_from_step
from lab.artifact_store import retain_dependency
from tools.check_camera_support import CameraQueries, POSES, copy_source, sha, write
from tools.check_head_service import head_part_names
from tools.cad_identity import geometry_record, geometry_agrees
from tools.review_camera_mount import verify_study
from tools.review_vision_power import checked_report


# Reviewed local dependency closure for CAD, head-service geometry, exported
# interfaces, and frame/config identity.  The archived snapshots still retain
# every source file; only this closure is compared across the head study,
# workbench, and current checkout so unrelated simulation changes do not force
# a CAD regeneration.
CAD_INTERFACE_DEPENDENCIES = frozenset(
    {
        "lab/config.py",
        "lab/robot.py",
        "lab/leg_mechanism.py",
        "lab/mechanical.py",
        "lab/export.py",
        "lab/head_service.py",
        "lab/logic_mount.py",
        "lab/imu_mount.py",
        "lab/neck_collar.py",
        "lab/neck_mount.py",
        "lab/camera_support.py",
        "lab/camera_mount.py",
        "lab/camera_vendor.py",
        "lab/visor_mount.py",
        "lab/face_layout.py",
        "lab/vision.py",
        "lab/styling.py",
        "lab/packaging.py",
        "lab/logic_packaging.py",
        "lab/power_pack.py",
        "lab/power_system.py",
        "lab/logic_supply.py",
        "lab/arm_mechanism.py",
        "lab/carrier.py",
        "lab/joint_module.py",
        "lab/neck_mechanism.py",
        "lab/pelvis.py",
        "lab/shoulder_mount.py",
    }
)

# Head-study decisions must use the current geometry and collision checks.
# Ordinary workbench snapshots do not contain these standalone study tools.
HEAD_CHECK_DEPENDENCIES = frozenset(
    {
        "tools/cad_identity.py",
        "tools/check_camera_support.py",
        "tools/check_head_service.py",
        "tools/check_neck_collar.py",
    }
)


def read(path):
    return json.loads(Path(path).read_text())


def verify_snapshot_sources(folder, hashes):
    """Verify every archived source file, including non-CAD dependencies."""
    for key, digest in hashes.items():
        if sha(folder / "source" / key) != digest:
            raise ValueError(f"archived source changed: {key}")


def verify_cad_dependency_closure(*snapshots):
    """Require the reviewed CAD/interface closure in each source snapshot."""
    for name, hashes in snapshots:
        missing = sorted(CAD_INTERFACE_DEPENDENCIES - hashes.keys())
        if missing:
            raise ValueError(
                f"{name} source snapshot missing CAD dependencies: {missing}"
            )
        for key in CAD_INTERFACE_DEPENDENCIES:
            if sha(key) != hashes[key]:
                raise ValueError(f"{name} CAD/interface source drift: {key}")


def verify_head_checks(hashes):
    for key in HEAD_CHECK_DEPENDENCIES:
        if hashes.get(key) != sha(key):
            raise ValueError(
                f"head study requires current geometry/collision check: {key}"
            )


def workspace_path(path):
    """Normalize input/source keys so absolute CLI paths cannot escape snapshots."""
    return Path(path).absolute().relative_to(Path.cwd().resolve())


def compatible_body(current, old):
    a, b = current.model_dump(mode="json"), old.model_dump(mode="json")
    for item in (a, b):
        for key in ("name", "camera", "face", "head_service"):
            item.pop(key)
    if a != b:
        raise ValueError("body composition permits only named head/camera/face changes")
    x, y = make_robot(current), make_robot(old)
    if x.interface()["joint_order"] != y.interface()["joint_order"]:
        raise ValueError("joint order changed")
    for key, value in x.forward_kinematics().items():
        if not np.allclose(value, y.forward_kinematics()[key], atol=1e-12, rtol=0):
            raise ValueError("body frame changed")


def head_role(role):
    if role == "support_camera_vendor_module":
        return "purchased_camera"
    if role == "pi_substrate":
        return "purchased_sbc"
    if "insert_" in role:
        return "insert"
    if "screw_" in role or "nut_" in role:
        return "hardware"
    if role in {
        "head_shell",
        "head_rear_cover",
        "face_lens",
        "cheek",
        "brow",
        "support_camera_rear_carrier",
        "support_camera_front_bezel",
    } or role.startswith("eye_"):
        return "head_polymer"
    raise ValueError(f"unclassified head role: {role}")


def head_binding(geometry, retained_names):
    """Map every shared part once; recover the earlier duplicate-cheek omission."""
    names = head_part_names(geometry)
    roles = [g.role for g in geometry]
    mapping = {}
    used = set()
    for i, (name, g) in enumerate(zip(names, geometry, strict=True)):
        old = f"{g.role}_{i}" if g.role.startswith("eye_") else g.role
        source = (
            name
            if name in retained_names
            else old
            if old in retained_names
            and i == max(j for j, r in enumerate(roles) if r == g.role)
            else None
        )
        if source is not None:
            if source in used:
                raise ValueError("retained head part reused")
            used.add(source)
        mapping[name] = source
    if used != set(retained_names):
        raise ValueError("unmapped retained head parts")
    return mapping


def verify_fit(folder, head_report, board):
    r = read(folder / "report.json")
    unsigned = dict(r)
    expected = unsigned.pop("report_digest_sha256")
    import hashlib

    if (
        hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, indent=2).encode()
        ).hexdigest()
        != expected
    ):
        raise ValueError("board fit report changed")
    if r["status"] != "pass" or r["board_hash"] != sha(board / "logic-supply.step"):
        raise ValueError("board fit identity mismatch")
    for key, digest in r["source_hashes"].items():
        if sha(folder / "source" / key) != digest:
            raise ValueError("board fit retained source changed")
    bodies = [
        digest
        for key, digest in r["source_hashes"].items()
        if key.endswith("/assembly.step")
    ]
    if bodies != [head_report["source_hashes"]["baseline/assembly.step"]]:
        raise ValueError("head and PCB studies use different detailed bodies")
    return r


def run(spec_path, output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="fail",
        builder="head_power_composition",
        scope=__doc__,
        source_hashes={},
        output_hashes={},
    )
    try:
        spec_path = workspace_path(spec_path)
        spec = yaml.safe_load(spec_path.read_text())
        if (
            set(spec)
            != {
                "schema_version",
                "variant",
                "head_study",
                "workbench",
                "board",
                "board_fit",
                "power_seal",
                "power_review",
            }
            or spec["schema_version"] != 1
        ):
            raise ValueError("unsupported assembly specification")
        paths = {k: workspace_path(v) for k, v in spec.items() if k != "schema_version"}
        hs, board, fit, seal, review = (
            paths[k]
            for k in ("head_study", "board", "board_fit", "power_seal", "power_review")
        )
        head_report = verify_study(hs)
        if head_report["status"] != "pass":
            raise ValueError("requires passed head study")
        config = load_config(paths["variant"])
        robot = make_robot(config)
        frames = robot.forward_kinematics()
        if config != RobotConfig.model_validate_json(
            (hs / "resolved-config.json").read_text()
        ):
            raise ValueError("head configuration differs")
        work = read(paths["workbench"] / "report.json")
        verify_snapshot_sources(paths["workbench"], work["source_hashes"])
        if work["checks"]["cad"]["status"] != "pass":
            raise ValueError("workbench CAD failed")
        if work["resolved_config_sha256"] != sha(
            paths["workbench"] / "resolved-config.json"
        ):
            raise ValueError("workbench config changed")
        if config != RobotConfig.model_validate_json(
            (paths["workbench"] / "resolved-config.json").read_text()
        ):
            raise ValueError("workbench config differs")
        verify_cad_dependency_closure(
            ("head study", head_report["source_hashes"]),
            ("workbench", work["source_hashes"]),
        )
        verify_head_checks(head_report["source_hashes"])
        fit_report = verify_fit(fit, head_report, board)
        report["historical_board_fit"] = dict(
            source=str(fit),
            reused="board identity, placement and detailed-body identity only",
            superseded="v7 board/connector-to-pack results used incompatible frames; recomputed here",
        )
        old_config_path = next(
            fit / "source" / key
            for key in fit_report["source_hashes"]
            if key.endswith("/resolved-config.json")
        )
        compatible_body(
            config, RobotConfig.model_validate_json(old_config_path.read_text())
        )
        seal_file = seal / "manifest.json"
        if sha(seal_file) != (seal / "manifest.sha256").read_text().strip():
            raise ValueError("power input seal changed")
        manifest = read(seal_file)
        board_report = checked_report(board, "build-report.json", manifest)
        if board_report["status"] != "pass":
            raise ValueError("PCB build failed")
        contract = read(board / "board-interface.json")
        if (
            contract["outline_mm"] != [38, 34]
            or sorted(contract["mounting_holes_mm"])
            != sorted([[3, 3], [35, 3], [3, 31], [35, 31]])
            or contract["missing_3d_models"]
        ):
            raise ValueError("PCB mount/model contract changed")
        power = read(review / "report.json")
        for key, digest in power["source_bindings"].items():
            if sha(key) != digest:
                raise ValueError(f"power review input changed: {key}")
        circuit = compiled_circuit(board)
        equivalent = []
        for row in power["cases"]:
            folder = Path(row["source"])
            verification = validate_config(
                yaml.safe_load((folder.parent / "verification.yaml").read_text())
            )
            if (folder / "circuit.cir").read_text() != spice(
                circuit, verification, row["original"]["case"]
            ):
                raise ValueError("replacement regulator circuit differs from probe")
            equivalent.append(row["name"])
        report["electrical_evidence"] = dict(
            status="partial",
            review_status=power["status"],
            byte_equivalent_cases=equivalent,
            scope=power["comparison_scope"],
            not_run=power["not_run"],
            legacy_workbench_power_status=work["checks"]["vision_power_allocation"][
                "status"
            ],
        )
        sources = {str(p): p for p in Path("lab").glob("*.py")}
        for p in [
            spec_path,
            paths["variant"],
            Path("robot/base.yaml"),
            Path("electronics/power-system.yaml"),
            Path("electronics/logic-supply/packaging.yaml"),
            Path("tools/compose_robot.py"),
            Path("tools/cad_identity.py"),
            Path("tools/check_arm_episode.py"),
            Path("tools/check_camera_support.py"),
            Path("tools/check_head_service.py"),
            Path("tools/check_neck_collar.py"),
            Path("tools/check_logic_replacement.py"),
            Path("tools/review_camera_mount.py"),
            Path("tools/review_vision_power.py"),
            Path("tools/seal_power_inputs.py"),
        ]:
            sources[str(p)] = p
        # Upstream snapshots are immutable dependencies, never recursively copied.
        report["dependency_manifests"] = {
            prefix: retain_dependency(folder)
            for prefix, folder in [
                ("head", hs), ("board_fit", fit), ("power_seal", seal),
                ("power_review", review), ("board", board),
            ]
        }
        for name in ("report.json", "resolved-config.json", "interfaces.json"):
            sources[f"inputs/workbench/{name}"] = paths["workbench"] / name
        report["source_hashes"] = copy_source(output, sources)
        export_all(robot, output)
        report["resolved_config_sha256"] = sha(output / "resolved-config.json")
        print("Sources bound; importing head and complete detailed body", flush=True)
        raw_head = list(import_step(hs / "head.step").children)
        by_name = {s.label: s for s in raw_head}
        if set(by_name) != set(head_report["geometry"]) or len(raw_head) != len(
            by_name
        ):
            raise ValueError("head STEP inventory differs")
        report["head_step_comparison"] = [
            dict(
                name=name,
                expected=head_report["geometry"][name],
                actual=geometry_record(s),
            )
            for name, s in by_name.items()
        ]
        mismatches = [
            row["name"]
            for row in report["head_step_comparison"]
            if not geometry_agrees(row["actual"], row["expected"])
        ]
        if mismatches:
            raise ValueError(f"head STEP geometry mismatch: {mismatches}")
        link = next(item for item in robot.links if item.name == "head")
        binding = head_binding(link.geometry, by_name)
        local = []
        restored = []
        for i, (name, g) in enumerate(
            zip(head_part_names(link.geometry), link.geometry, strict=True)
        ):
            source = binding[name]
            shape = (
                by_name[source]
                if source
                else Pos(*(v * 1000 for v in g.pos))
                * Rot(*np.degrees(g.rpy))
                * part_shape(robot, link, i)
            )
            if source is None:
                restored.append(name)
            local.append(
                Part(name, "head", shape, head_role(g.role), "head_service", g.color)
            )
        report["head_inventory"] = dict(
            expected=len(link.geometry),
            actual=len(local),
            source_mapping=binding,
            restored_from_shared_recipe=restored,
        )
        body_report = read(hs / "source/baseline/assembly-report.json")
        body = parts_from_step(
            list(import_step(hs / "source/baseline/assembly.step").children),
            body_report["parts"],
        )
        if len(body) != 162 or {
            p.name for p in body if p.name in {"logic_board", "head_shell"}
        } != {"logic_board", "head_shell"}:
            raise ValueError("replacement inventory differs")
        retained = [p for p in body if p.name not in {"logic_board", "head_shell"}]
        for p in retained:
            if p.shape.color is not None:
                p.color = tuple(p.shape.color)
        expected_mount = {
            f"logic_{kind}_{i}" for kind in ("screw", "insert") for i in range(4)
        }
        if {p.name for p in retained if p.module == "logic_mount"} != expected_mount:
            raise ValueError("logic mounting hardware changed")
        raw_board = import_step(board / "logic-supply.step")
        if len(raw_board.children) != 16:
            raise ValueError("PCB model inventory differs")
        pose = dict(
            schema_version=1,
            normal="rearward",
            position_torso_mm=[
                config.dimensions.logic_board_x_m * 1000,
                config.dimensions.logic_board_y_m * 1000,
                config.dimensions.logic_board_z_m * 1000,
            ],
        )
        if pose["position_torso_mm"] != fit_report["board_pose_torso_mm"]:
            raise ValueError("board pose differs from source study")
        loc = placement(contract, pose)
        board_local = loc * Compound(raw_board.solids())
        board_part = Part(
            "logic_board",
            "torso",
            transform_shape(board_local, frames["torso"]),
            "board",
            "logic_mount",
            (0.15, 0.55, 0.25, 1),
        )
        head = [
            replace(p, shape=transform_shape(p.shape, frames["head"])) for p in local
        ]
        combined = retained + [board_part] + head
        if len({p.name for p in combined}) != len(combined):
            raise ValueError("duplicate composed part name")
        q = CameraQueries()
        report["replacement_board_conflicts"] = q.collide(
            board_part.shape, retained + head
        )
        pack_local, _ = pack_shapes(load_system())
        pack_world = transform_shape(pack_local, frames["torso"])
        report["pack_frame"] = dict(
            source="torso-local mm",
            comparison="world mm via torso FK",
            torso_transform_si=frames["torso"].tolist(),
        )
        report["board_pack_overlap_mm3"] = overlap(board_part.shape, pack_world)
        report["board_containment"] = torso_containment(robot, board_local)
        approach = yaml.safe_load(
            Path("electronics/logic-supply/packaging.yaml").read_text()
        )["connector_approach"]
        report["connector_approach"] = []
        for ref in ("J1", "J2", "J3"):
            xy = np.mean(
                [
                    p["xy_mm"]
                    for n, p in contract["pads"].items()
                    if n.startswith(ref + ".")
                ],
                axis=0,
            )
            shape = (
                loc
                * Pos(xy[0], -xy[1], approach["normal_center_mm"])
                * Box(
                    approach["width_mm"],
                    approach["height_mm"],
                    approach["normal_depth_mm"],
                )
            )
            shape = transform_shape(shape, frames["torso"])
            report["connector_approach"].append(
                dict(
                    ref=ref,
                    hits=q.collide(shape, retained + head),
                    pack_overlap_mm3=overlap(shape, pack_world),
                )
            )
        # Recheck both cheek allocations even when a fresh head study already
        # contains them: the historical duplicate-name study omitted one.
        restored_parts = [
            p for p in local if p.name in restored or p.name.startswith("cheek_")
        ]
        report["restored_head_conflicts"] = [
            dict(moving=p.name, **hit)
            for p in restored_parts
            for hit in q.collide(p.shape, [t for t in local if t.name != p.name])
        ]
        report["head_board_motion"] = []
        for yaw, pitch in POSES:
            # Posed shapes are newly allocated each iteration. Previous-pose
            # cache entries cannot be reused and retain large vendor geometry.
            q = CameraQueries()
            posed = robot.forward_kinematics(
                {"neck_yaw_joint": yaw, "head_joint": pitch}
            )
            moved_head = [
                replace(p, shape=transform_shape(p.shape, posed["head"])) for p in local
            ]
            hits = q.collide(board_part.shape, moved_head)
            moved_body = [
                replace(
                    p,
                    shape=transform_shape(
                        p.shape, posed[p.owner] @ np.linalg.inv(frames[p.owner])
                    ),
                )
                for p in retained
            ] + [board_part]
            # Shared head shells retain the existing mount qualification. Check
            # every other populated/visible head part against the detailed body,
            # including decorative pieces absent from older body-only exports.
            body_hits = [
                dict(moving=p.name, **hit)
                for p in local
                if p.name not in {"head_shell", "head_rear_cover"}
                for hit in q.collide(
                    transform_shape(p.shape, posed["head"]), moved_body
                )
            ]
            report["head_board_motion"].append(
                dict(yaw_rad=yaw, pitch_rad=pitch, hits=hits, body_hits=body_hits)
            )
        report["head_service_board"] = []
        q = CameraQueries()
        for distance in (0, 2, 5, 15, 35, 70, 110):
            for p in local:
                if p.name == "head_rear_cover" or p.name.startswith("pi_"):
                    moved = transform_shape(
                        Pos(-distance, 0, 0) * p.shape, frames["head"]
                    )
                    report["head_service_board"].append(
                        dict(
                            part=p.name,
                            rearward_mm=distance,
                            hits=q.collide(moved, [board_part]),
                        )
                    )
        report["parts"] = [
            dict(
                name=p.name,
                owner=p.owner,
                role=p.role,
                module=p.module,
                volume_mm3=p.shape.volume,
                color=list(p.color),
            )
            for p in combined
        ]
        report["part_count"] = len(combined)
        report["not_run"] = [
            "old-old body pairs or new task dynamics",
            "actual Pi populated geometry, harness, connector mating/latches, antenna and thermal keepouts",
            "face mounting; decorative face solids are allocations",
            "complete service/continuous sweeps and physical retention/durability",
            "system power readiness, protection/charging/stop, board thermal and PCB/harness parasitics",
            "mass properties and production price are separate downstream results",
        ]
        print(
            f"Composition checks complete; exporting {len(combined)} groups", flush=True
        )
        for p in combined:
            p.shape.label = p.name
            p.shape.color = Color(*p.color[:3])
        export_step(
            Compound(children=[p.shape for p in combined]), output / "assembly.step"
        )
        print("Validating exported assembly geometry", flush=True)
        imported = parts_from_step(
            list(import_step(output / "assembly.step").children), report["parts"]
        )
        expected_geometry = {p.name: geometry_record(p.shape) for p in combined}
        report["assembly_step_roundtrip"] = [
            dict(
                name=p.name,
                expected=expected_geometry[p.name],
                actual=geometry_record(p.shape),
            )
            for p in imported
        ]
        mismatches = [
            row["name"]
            for row in report["assembly_step_roundtrip"]
            if not geometry_agrees(row["actual"], row["expected"])
        ]
        if mismatches:
            raise ValueError(f"assembly STEP geometry mismatch: {mismatches}")
        render_core(
            combined,
            {p.name: p.shape for p in combined},
            output,
            robot,
            azimuth=155,
            elevation=-20,
        )
        # Battery is an allocation already owned by the mass model, not an extra
        # purchased solid in the assembly table.
        cutaway = [
            p
            for p in combined
            if p.name
            not in {
                "torso_front",
                "torso_rear",
                "head_shell",
                "head_rear_cover",
                "face_lens",
            }
        ]
        cut_folder = output / "cutaway"
        cut_folder.mkdir()
        render_core(
            cutaway,
            {p.name: p.shape for p in cutaway},
            cut_folder,
            robot,
            azimuth=155,
            elevation=-20,
        )
        report["status"] = (
            "pass"
            if (
                not report["replacement_board_conflicts"]
                and report["board_pack_overlap_mm3"] < 0.02
                and report["board_containment"]["status"] == "pass"
                and all(
                    not r["hits"] and r["pack_overlap_mm3"] < 0.02
                    for r in report["connector_approach"]
                )
                and not report["restored_head_conflicts"]
                and all(
                    not r["hits"] and not r["body_hits"]
                    for r in report["head_board_motion"]
                )
                and all(not r["hits"] for r in report["head_service_board"])
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
        and p.name not in {"report.json", "report.sha256", "arm-report.json"}
    }
    write(output / "report.json", report)
    write(output / "arm-report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = run(a.spec, a.output)
    print(
        json.dumps(
            dict(status=r["status"], error=r.get("error"), parts=r.get("part_count"))
        )
    )
    raise SystemExit(0 if r["status"] == "pass" else 1)
