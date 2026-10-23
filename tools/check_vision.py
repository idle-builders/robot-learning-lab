"""Study camera/SBC allocations and selected optics; no installed-hardware claim."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from PIL import Image
from build123d import Box, Compound, Pos, Rot, import_step, export_step
from lab.artifact_store import snapshot_file
from lab.camera import pinhole
from lab.config import load_config, RobotConfig
from lab.leg_mechanism import Part, transform_shape, render_core, overlap
from lab.mechanical import primitive, part_shape
from lab.robot import make_robot
from lab.tasks import PickPlaceTask, build_scene
from lab.vision import load_vision, camera_geometry, pi_geometry
from lab.face_layout import check_face
from tools.check_arm_episode import parts_from_step
from tools.check_neck_collar import ExactQueries


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def allocation(record, rotation, position_mm):
    low, high = np.array(record["min_mm"]), np.array(record["max_mm"])
    shape = Pos(*((low + high) / 2).tolist()) * Box(*(high - low).tolist())
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.array(position_mm) / 1000
    if not np.allclose(rotation.T @ rotation, np.eye(3)) or not np.isclose(
        np.linalg.det(rotation), 1
    ):
        raise ValueError("allocation requires a proper rigid rotation")
    return transform_shape(shape, matrix)


def run(variant, assembly, model_run, episode, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "fail", "scope": __doc__.strip(), "source_hashes": {}}
    try:
        robot = make_robot(load_config(variant))
        manifest = load_vision()
        source = read(assembly / "report.json")
        assert source["status"] == "pass"
        for path, expected in source["source_hashes"].items():
            assert sha(assembly / "source" / path) == expected, path
        for path in ["assembly.step", "interfaces.json"]:
            assert sha(assembly / path) == source["output_hashes"][path], path
        assert (
            sha(assembly / "resolved-config.json") == source["resolved_config_sha256"]
        )
        old = make_robot(
            RobotConfig.model_validate_json(
                (assembly / "resolved-config.json").read_text()
            )
        )
        assert json.loads(json.dumps(old.interface())) == read(
            assembly / "interfaces.json"
        )
        a, b = robot.config.model_dump(), old.config.model_dump()
        for cfg in [a, b]:
            cfg.pop("name")
            cfg.pop("camera")
            cfg.pop("face")
        assert a == b, "only camera selection/placement and face layout may differ"
        assert read(model_run / "interfaces.json") == json.loads(
            json.dumps(robot.interface())
        )
        w = read(model_run / "report.json")
        assert w["checks"]["cad"]["status"] == "pass"
        report["model_run_status"] = w["status"]
        report["power_allocation"] = manifest.power.budget_check
        report["camera_contract"] = robot.interface()["camera_frame"]
        report["face_layout"] = check_face(robot)
        paths = [
            variant,
            Path("robot/base.yaml"),
            Path("electronics/vision.yaml"),
            Path("electronics/power-system.yaml"),
            Path("electronics/logic-supply/verification.yaml"),
            Path(__file__),
            Path("tools/check_neck_collar.py"),
            Path("tools/check_arm_episode.py"),
            *Path("lab").glob("*.py"),
            *Path("docs/design/references/vision").glob("*.pdf"),
        ]
        paths += [
            assembly / n
            for n in [
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "assembly.step",
            ]
        ]
        paths += [
            model_run / n
            for n in [
                "report.json",
                "resolved-config.json",
                "interfaces.json",
                "robot.xml",
            ]
        ]
        paths += list((model_run / "meshes").glob("*.stl"))
        paths += [
            episode / n
            for n in ["report.json", "task.json", "trajectory.jsonl", "interfaces.json"]
        ]
        for path in paths:
            name = str(path.absolute().relative_to(Path.cwd()))
            dest = output / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            snapshot_file(path, dest)
            report["source_hashes"][name] = sha(dest)
        write(output / "resolved-config.json", robot.config.model_dump(mode="json"))
        write(output / "interfaces.json", robot.interface())
        print("Importing retained body", flush=True)
        parts = parts_from_step(
            list(import_step(assembly / "assembly.step").children), source["parts"]
        )
        frames = robot.forward_kinematics()
        head = next(link for link in robot.links if link.name == "head")
        queries = ExactQueries()
        original_parts = parts
        cassette_parts = []
        if robot.config.camera.cassette_enabled:
            from lab.camera_mount import cassette, screws, interface as mount_interface

            parts = [
                replace(
                    p, shape=transform_shape(part_shape(robot, head, 0), frames["head"])
                )
                if p.name == "head_shell"
                else p
                for p in parts
            ]
            cassette_parts = [
                Part(
                    "camera_cassette",
                    "head",
                    transform_shape(cassette(robot), frames["head"]),
                    "printed_candidate",
                    "vision",
                    (0.055, 0.08, 0.085),
                )
            ]
            cassette_parts += [
                Part(
                    f"camera_shell_screw_{i}",
                    "head",
                    transform_shape(s, frames["head"]),
                    "fastener_candidate",
                    "vision",
                    (0.5, 0.5, 0.5),
                )
                for i, s in enumerate(screws(robot))
            ]
            report["camera_mount"] = mount_interface(robot)
            report["camera_mount"]["cassette_body_conflicts"] = queries.collide(
                cassette_parts[0].shape, parts
            )
            report["camera_mount"]["fastener_mechanism_conflicts"] = [
                dict(
                    screw=p.name,
                    hits=queries.collide(
                        p.shape, [q for q in parts if q.name != "head_shell"]
                    ),
                )
                for p in cassette_parts[1:]
            ]
            shell = next(p for p in parts if p.name == "head_shell")
            report["camera_mount"]["fastener_fit"] = [
                dict(
                    screw=p.name,
                    lug_interference_mm3=overlap(p.shape, cassette_parts[0].shape),
                    intended_thread_stock_overlap_mm3=overlap(p.shape, shell.shape),
                    scope="nominal solid screw in undersized pilot; not a thread/pullout model",
                )
                for p in cassette_parts[1:]
            ]
        # Drawing X/Y/Z -> head +Y/+Z/+X. Image axes are a separate contract.
        camera_rotation = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        camera_local = allocation(
            camera_geometry(manifest),
            camera_rotation,
            np.array(robot.camera_position) * 1000,
        )
        camera_world = transform_shape(camera_local, frames["head"])
        report["camera_reservation"] = {
            "position_head_m": list(robot.camera_position),
            "drawing_to_head": camera_rotation.tolist(),
            "body_conflicts": queries.collide(camera_world, parts),
        }
        if cassette_parts:
            report["camera_reservation"]["original_shell_negative_control"] = (
                queries.collide(camera_world, original_parts)
            )
            report["camera_reservation"]["cassette_conflicts"] = queries.collide(
                camera_world, cassette_parts
            )
        decorative = []
        for i, g in enumerate(head.geometry[1:], start=1):
            shape = (
                Pos(*(np.array(g.pos) * 1000).tolist())
                * Rot(*np.degrees(g.rpy).tolist())
                * part_shape(robot, head, i)
            )
            decorative.append(
                Part(
                    "visual_" + g.role + "_" + str(i),
                    "head",
                    transform_shape(shape, frames["head"]),
                    "visual_allocation",
                    "head",
                    g.color,
                )
            )
        report["camera_reservation"]["visual_allocation_conflicts"] = queries.collide(
            camera_world, decorative
        )
        if cassette_parts:
            fixed_face = [p for p in decorative if "camera_cassette" not in p.name]
            report["camera_mount"]["face_conflicts"] = queries.collide(
                cassette_parts[0].shape, fixed_face
            )
            report["camera_mount"]["removal"] = [
                dict(
                    forward_mm=distance,
                    hits=queries.collide(
                        transform_shape(
                            Pos(distance, 0, 0) * cassette(robot), frames["head"]
                        ),
                        parts + fixed_face,
                    ),
                )
                for distance in (0, 5, 15, 30)
            ]
            report["camera_mount"]["removal_prerequisite"] = (
                "two shell screws removed; camera flex disconnected"
            )
        # Cache posed targets once; select only after evaluating each feasible board
        # through neck motion. Head-owned geometry is checked in the neutral frame.
        posed_targets = []
        for yaw in [-1.2, -0.6, 0, 0.6, 1.2]:
            for pitch in [-0.5, 0, 0.5]:
                posed = robot.forward_kinematics(
                    {"neck_yaw_joint": yaw, "head_joint": pitch}
                )
                located = [
                    replace(
                        p,
                        shape=transform_shape(
                            p.shape, posed[p.owner] @ np.linalg.inv(frames[p.owner])
                        ),
                    )
                    for p in parts
                    if p.owner != "head"
                ]
                located += [
                    replace(
                        p,
                        shape=transform_shape(
                            p.shape, posed["head"] @ np.linalg.inv(frames["head"])
                        ),
                    )
                    for p in cassette_parts
                ]
                posed_targets.append((yaw, pitch, posed, located))
        # A face-layout change can move eyes outside their old swept envelope.
        # Check authored eye solids against non-head mechanism/body targets.
        report["face_motion"] = []
        for yaw, pitch, posed, located in posed_targets:
            targets = [p for p in located if p.owner != "head"]
            hits = []
            for eye in decorative:
                if not eye.name.startswith("visual_eye_"):
                    continue
                shape = transform_shape(
                    eye.shape, posed["head"] @ np.linalg.inv(frames["head"])
                )
                hits.extend(
                    dict(eye=eye.name, **hit) for hit in queries.collide(shape, targets)
                )
            report["face_motion"].append(
                dict(yaw_rad=yaw, pitch_rad=pitch, conflicts=hits)
            )
        report["face_motion_status"] = (
            "fail" if any(row["conflicts"] for row in report["face_motion"]) else "pass"
        )
        # Side-mounted board: drawing length rises along head Z; components face inward.
        pi_rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        outer = primitive(head.geometry[0])
        records = []
        candidates = []
        for y in [34.0, 38.0, 40.0, 42.0, 44.0, 48.0]:
            for z in [-6.0, -3.0, 0.0, 3.0]:
                position = [0.0, y, z]
                local = allocation(pi_geometry(manifest), pi_rotation, position)
                shape = transform_shape(local, frames["head"])
                hits = queries.collide(shape, parts)
                hits += queries.collide(shape, cassette_parts)
                camera_overlap = overlap(shape, camera_world)
                outside = local - outer
                volume = outside.volume if outside is not None else 0.0
                row = {
                    "position_head_mm": position,
                    "drawing_to_head": pi_rotation.tolist(),
                    "body_conflicts": hits,
                    "outside_head_silhouette_mm3": max(0, volume),
                    "camera_overlap_mm3": camera_overlap,
                    "envelope_status": "pass"
                    if not hits and volume <= 0.02 and camera_overlap <= 0.02
                    else "fail",
                }
                row["motion_conflicts"] = []
                row["motion_status"] = "not_run"
                if row["envelope_status"] == "pass":
                    for yaw, pitch, posed, located in posed_targets:
                        conflicts = queries.collide(
                            transform_shape(local, posed["head"]), located
                        )
                        if conflicts:
                            row["motion_conflicts"].append(
                                dict(yaw_rad=yaw, pitch_rad=pitch, conflicts=conflicts)
                            )
                    row["motion_status"] = "fail" if row["motion_conflicts"] else "pass"
                records.append(row)
                candidates.append((row, shape, local))
        report["computer_candidates"] = records
        best, pi_world, pi_local = min(
            candidates,
            key=lambda x: (
                x[0]["envelope_status"] != "pass" or x[0]["motion_status"] != "pass",
                len(x[0]["motion_conflicts"]),
                len(x[0]["body_conflicts"]) > 0,
                sum(v["overlap_mm3"] for v in x[0]["body_conflicts"]),
                x[0]["outside_head_silhouette_mm3"],
            ),
        )
        report["selected_computer_reservation"] = best
        print("Allocation candidates checked", flush=True)
        # Motion checks concern only the two new reserved volumes, not old-old pairs.
        motion = []
        for yaw, pitch, posed, located in posed_targets:
            motion.append(
                {
                    "yaw_rad": yaw,
                    "pitch_rad": pitch,
                    "camera_conflicts": queries.collide(
                        transform_shape(camera_local, posed["head"]), located
                    ),
                    "cassette_conflicts": queries.collide(
                        transform_shape(cassette(robot), posed["head"]),
                        [p for p in located if p.owner != "head"],
                    )
                    if cassette_parts
                    else [],
                    "computer_conflicts": queries.collide(
                        transform_shape(pi_local, posed["head"]), located
                    ),
                    "fastener_conflicts": [
                        dict(
                            moving_part=p.name,
                            conflicts=queries.collide(
                                transform_shape(
                                    p.shape,
                                    posed["head"] @ np.linalg.inv(frames["head"]),
                                ),
                                [
                                    target
                                    for target in located
                                    if target.owner != "head"
                                ],
                            ),
                        )
                        for p in cassette_parts[1:]
                    ],
                }
            )
        report["motion_reservations"] = motion
        camera_part = Part(
            "camera_full_envelope",
            "head",
            camera_world,
            "allocation",
            "vision",
            (0.12, 0.2, 0.24),
        )
        pi_part = Part(
            "pi_board_height_allocation",
            "head",
            pi_world,
            "allocation",
            "vision",
            (0.2, 0.65, 0.4),
        )
        report["camera_computer_overlap_mm3"] = overlap(camera_world, pi_world)
        negative = transform_shape(
            allocation(pi_geometry(manifest), pi_rotation, [0.0, 0.0, 0.0]),
            frames["head"],
        )
        report["negative_control"] = queries.collide(negative, parts)
        assert report["negative_control"], (
            "deliberately misplaced SBC must hit mechanism"
        )
        shown = (
            [
                replace(p, color=(0.8, 0.8, 0.75, 0.22))
                if p.name == "head_shell"
                else p
                for p in parts
                if p.name == "head_shell" or p.owner in ["head", "neck_yaw"]
            ]
            + [camera_part, pi_part]
            + cassette_parts
        )
        render_core(shown, {p.name: p.shape for p in shown}, output, robot)
        for p in shown:
            p.shape.label = p.name
        export_step(
            Compound(children=[p.shape for p in shown]),
            output / "head-allocations.step",
        )
        if cassette_parts:
            display = (
                [p for p in shown if p.name != "head_shell"]
                + [p for p in parts if p.name == "head_shell"]
                + fixed_face
            )
            for part in display:
                part.shape.label = part.name
            folder = output / "appearance"
            folder.mkdir()
            render_core(
                display,
                {p.name: p.shape for p in display},
                folder,
                robot,
                azimuth=25,
                elevation=-12,
            )
            export_step(
                Compound(children=[p.shape for p in display]),
                output / "camera-mount.step",
            )
        report["physical_mount"] = "not_run"
        report["bridge_implementation"] = "not_run"
        report["not_run"] = [
            "camera/SBC mounts and hardware",
            "camera recess, exact lens/connector profile and aperture",
            "FFC, USB and power harness/mating/strain relief",
            "RF and thermal",
            "physical streaming/exposure/calibration",
            "new mass and dynamics",
            "complete production pricing",
        ]
        print("Rendering nominal optics at recorded task poses", flush=True)
        contract = read(episode / "interfaces.json")
        assert contract["joints"] == json.loads(json.dumps(robot.interface()["joints"]))
        for current, prior in zip(
            robot.interface()["links"], contract["links"], strict=True
        ):
            assert json.loads(
                json.dumps({k: current[k] for k in ["name", "parent", "pos", "joint"]})
            ) == {k: prior[k] for k in ["name", "parent", "pos", "joint"]}
        task = PickPlaceTask.model_validate(read(episode / "task.json"))
        assert task.fixture == "supported"
        build_scene(robot, task, output / "scene.xml", model_run)
        model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
        data = mujoco.MjData(model)
        assert model.camera("head_camera").fovy[0] == robot.camera_fovy_deg
        phases = {}
        for line in (episode / "trajectory.jsonl").read_text().splitlines():
            row = json.loads(line)
            phases[row["phase"]] = row
        visibility = []
        for fov, label in [
            (41.0, "standard_reference"),
            (67.0, "selected_wide"),
            (70.0, "legacy_reference"),
        ]:
            model.cam_fovy[model.camera("head_camera").id] = fov
            with mujoco.Renderer(model, width=640, height=360) as renderer:
                for phase, row in phases.items():
                    assert (
                        row["joint_state"]["names"] == robot.interface()["joint_order"]
                    )
                    for joint, value in zip(
                        row["joint_state"]["names"],
                        row["joint_state"]["position_rad"],
                        strict=True,
                    ):
                        data.joint(joint).qpos[0] = value
                    metrics = row["task_state"]
                    data.joint("task_block_free").qpos[:] = (
                        metrics["block_position_m"] + metrics["block_quaternion_wxyz"]
                    )
                    mujoco.mj_forward(model, data)
                    renderer.disable_segmentation_rendering()
                    renderer.update_scene(data, camera="head_camera")
                    filename = f"{label}-{phase}.png"
                    Image.fromarray(renderer.render().copy()).save(output / filename)
                    renderer.enable_segmentation_rendering()
                    renderer.update_scene(data, camera="head_camera")
                    seg = renderer.render()
                    count = int(
                        np.count_nonzero(
                            (seg[:, :, 0] == model.geom("task_block_geom").id)
                            & (seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
                        )
                    )
                    distance = float(
                        np.linalg.norm(
                            data.cam_xpos[model.camera("head_camera").id]
                            - data.body("task_block").xpos
                        )
                    )
                    visibility.append(
                        {
                            "optics": label,
                            "phase": phase,
                            "block_visible_pixels": count,
                            "image": filename,
                            "distance_m": distance,
                            "inside_selected_wide_focus_range": distance
                            >= manifest.camera.intrinsics.minimum_focus_m,
                            "intrinsics": pinhole(640, 360, fov),
                        }
                    )
        report["visibility"] = visibility
        report["optics_scope"] = (
            "re-rendered prior supported task phase endpoints; not new dynamics or image-based control; hypothetical standard comparison uses same optical pose"
        )
        report["optics_status"] = (
            "pass"
            if all(
                v["block_visible_pixels"] > 0
                for v in visibility
                if v["optics"] == "selected_wide"
            )
            else "fail"
        )
        reservation_failed = bool(
            report["camera_reservation"]["body_conflicts"]
            or report["camera_reservation"]["visual_allocation_conflicts"]
            or best["envelope_status"] == "fail"
            or best["motion_status"] != "pass"
            or report["camera_computer_overlap_mm3"] > 0.02
            or any(
                row["camera_conflicts"]
                or row["computer_conflicts"]
                or row["cassette_conflicts"]
                or any(item["conflicts"] for item in row["fastener_conflicts"])
                for row in motion
            )
        )
        if cassette_parts:
            mount = report["camera_mount"]
            reservation_failed |= bool(
                mount["cassette_body_conflicts"]
                or mount["face_conflicts"]
                or any(
                    r["hits"]
                    for r in mount["fastener_mechanism_conflicts"] + mount["removal"]
                )
                or report["camera_reservation"]["cassette_conflicts"]
                or any(
                    r["lug_interference_mm3"] > 0.02
                    or r["intended_thread_stock_overlap_mm3"] < 1
                    for r in mount["fastener_fit"]
                )
            )
            mount["geometry_status"] = "fail" if reservation_failed else "pass"
        report["allocation_status"] = "fail" if reservation_failed else "pass"
        report["status"] = (
            "fail"
            if reservation_failed
            or report["power_allocation"]["status"] == "fail"
            or report["optics_status"] == "fail"
            or report["face_layout"]["status"] == "fail"
            or report["face_motion_status"] == "fail"
            else "partial"
        )
        report["output_hashes"] = {
            str(p.relative_to(output)): sha(p)
            for p in output.rglob("*")
            if p.is_file() and "source" not in p.relative_to(output).parts
        }
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = str(exc)
        (output / "error.log").write_text(traceback.format_exc())
    write(output / "report.json", report)
    (output / "report.sha256").write_text(sha(output / "report.json") + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in ["status", "optics_status", "power_allocation", "error"]
                if k in report
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["variant", "assembly", "model-run", "episode", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(args.variant, args.assembly, args.model_run, args.episode, args.output)
    raise SystemExit(1 if result["status"] == "fail" else 0)
