"""Screen a source-sized dual-view and IMU package on Cove-50 Direct Frame.

The current selected CAD source remains untouched. This is a candidate delta:
two oriented Camera Module 3 Wide envelopes and cable-open saddles on the yaw
head, plus an external MinIMU-9 v6 shelf on the torso. It screens volume and
connectedness, not fasteners, cable bend, lens calibration or physical build.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import build123d as bd
import mujoco
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
CONTRACT = ROOT / "experiments/compact_morphology/ape_frame50/frame_contract.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def volume_overlap(a: bd.Shape, b: bd.Shape) -> float:
    overlap = a & b
    return 0.0 if overlap is None else float(overlap.volume)


def load_selected():
    selection = json.loads(DESIGN.read_text())
    if selection["id"] != "cove-50-direct-frame":
        raise ValueError("Current mechanical design is not Cove-50 Direct Frame")
    source = ROOT / selection["generator"]["path"]
    if sha256(source) != selection["generator"]["sha256"]:
        raise ValueError("Selected CAD source no longer matches design pointer")
    spec = importlib.util.spec_from_file_location("selected_cove_direct", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load selected CAD generator")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return selection, module


def camera_envelope(position_mm: tuple[float, float, float], pitch_up_deg: float):
    """The 25x24 mm board plane is vertical; 12.4 mm points along optical +X."""
    angle = math.radians(pitch_up_deg)
    normal = (math.cos(angle), 0.0, math.sin(angle))
    depth = 12.4
    board_center = tuple(position_mm[i] - depth / 2 * normal[i] for i in range(3))
    saddle_center = tuple(position_mm[i] - (depth + 1.0) * normal[i] for i in range(3))
    rotation = bd.Rot(Y=-pitch_up_deg)
    board = bd.Pos(*board_center) * rotation * bd.Box(depth, 24, 25)
    saddle = bd.Pos(*saddle_center) * rotation * (
        bd.Box(2, 34, 29) - bd.Box(4, 16, 14))
    return board, saddle, normal


def build_candidate(source, contract: dict):
    box = source.box
    raw_cage = box((1, 0, 446), (68, 76, 55)) - box((1, 0, 443), (62, 70, 55))
    cameras = contract["sensors"]["cameras"]
    boards = {}
    saddles = {}
    rays = {}
    for name, pitch in (("upper", 30.0), ("lower", -40.0)):
        position = tuple(1000 * v for v in cameras[f"{name}_frame"]["position_m"])
        board, saddle, normal = camera_envelope(position, pitch)
        boards[name], saddles[name] = board, saddle
        rays[name] = {"optical_center_mm": position, "axis_forward": normal}
        # The 3 mm center bridge and side rails join the two face cutouts.
        opening_z = 455 if name == "upper" else 436.5
        opening = box((34, position[1], opening_z), (12, 27, 32 if name == "upper" else 31))
        raw_cage = raw_cage - opening
    head_frame = raw_cage + saddles["upper"] + saddles["lower"]

    # Put the board outside the solid spine rather than inside its old reserve.
    imu_size = tuple(1000 * v for v in contract["sensors"]["body_imu"]["board_envelope_m"])
    imu_position = (25, 0, 296)
    imu_board = box(imu_position, imu_size)
    imu_shelf = box((23, 0, 292.5), (24, 17, 4))
    torso_raw = dict(source.stage_frames())["torso"]
    torso_frame = [v for k, v in torso_raw.items() if k.endswith("_frame")]
    torso_keepout = {k: v for k, v in torso_raw.items() if k == "battery_4s850"}
    return {"head_frame": head_frame, "head_cage_cut": raw_cage,
            "camera_boards": boards,
            "camera_saddles": saddles, "camera_rays": rays,
            "imu_board": imu_board, "imu_shelf": imu_shelf,
            "imu_position_mm": imu_position, "torso_frame": torso_frame,
            "torso_keepout": torso_keepout}


def screen(candidate: dict) -> dict:
    head = candidate["head_frame"]
    boards = candidate["camera_boards"]
    imu = candidate["imu_board"]
    shelf = candidate["imu_shelf"]
    result = {
        "head_connected_solid_count": len(head.solids()),
        "camera_board_frame_overlap_mm3": {name: volume_overlap(head, board)
                                            for name, board in boards.items()},
        "camera_board_board_overlap_mm3": volume_overlap(boards["upper"], boards["lower"]),
        "camera_saddle_cut_cage_overlap_mm3": {
            name: volume_overlap(candidate["head_cage_cut"], saddle)
            for name, saddle in candidate["camera_saddles"].items()},
        "imu_board_torso_frame_overlap_mm3": sum(
            volume_overlap(imu, frame) for frame in candidate["torso_frame"]),
        "imu_board_battery_overlap_mm3": sum(
            volume_overlap(imu, keepout) for keepout in candidate["torso_keepout"].values()),
        "imu_board_shelf_overlap_mm3": volume_overlap(imu, shelf),
        "imu_shelf_spine_overlap_mm3": sum(
            volume_overlap(shelf, frame) for frame in candidate["torso_frame"]),
        "imu_shelf_battery_overlap_mm3": sum(
            volume_overlap(shelf, keepout) for keepout in candidate["torso_keepout"].values()),
    }
    return result


def optical_screen(design: dict, candidate: dict, *, visual: Path | None = None,
                   expected_sha256: str | None = None) -> dict:
    """Render evaluator markers through the revised cage on the pinned visual body."""
    visual = visual or ROOT / design["evidence"]["visual_body"]
    expected_sha256 = expected_sha256 or design["evidence"]["visual_body_sha256"]
    if sha256(visual) != expected_sha256:
        raise ValueError("Visual MJCF no longer matches its evidence hash")
    xml = ET.parse(visual)
    root = xml.getroot()
    asset = root.find("asset")
    world = root.find("worldbody")
    head = next(b for b in root.iter("body") if b.get("name") == "head")
    for geom in list(head.findall("geom")):
        if geom.get("name") in ("cad_head_frame", "cad_head_sensor"):
            head.remove(geom)
    targets = {
        "shown_35cm": ((0.35, 0, 0.457), 0.03),
        "face_1m": ((1, 0, 1.5), 0.095),
        "floor_8cm_40cm": ((0.4, 0, 0.04), 0.04),
        "floor_8cm_25cm": ((0.25, 0, 0.04), 0.04),
        "floor_8cm_12cm": ((0.12, 0, 0.04), 0.04),
    }
    for name, (_, radius) in targets.items():
        body = ET.SubElement(world, "body", name=f"test_{name}", pos="10 0 0")
        ET.SubElement(body, "geom", name=f"test_{name}", type="sphere",
                      size=str(radius), rgba="1 0 0 1", group="0",
                      contype="0", conaffinity="0")
    with tempfile.NamedTemporaryFile(suffix=".stl", dir=visual.parent, delete=False) as mesh_file:
        mesh_path = Path(mesh_file.name)
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=visual.parent,
                                     delete=False) as xml_file:
        xml_path = Path(xml_file.name)
    try:
        # Selected CAD coordinates are millimetres in world space; the head
        # body's neutral origin is (1, 0, 447) mm in the selected visual MJCF.
        local_head = bd.Pos(-1, 0, -447) * candidate["head_frame"]
        bd.export_stl(local_head, mesh_path)
        ET.SubElement(asset, "mesh", name="candidate_head_sensor_frame",
                      file=mesh_path.name, scale="0.001 0.001 0.001")
        ET.SubElement(head, "geom", name="cad_candidate_head_sensor_frame",
                      type="mesh", mesh="candidate_head_sensor_frame",
                      rgba="0.22 0.58 0.53 1", mass="0", group="1",
                      contype="0", conaffinity="0")
        xml_path.write_text(ET.tostring(root, encoding="unicode"))
        model = mujoco.MjModel.from_xml_path(str(xml_path))
    finally:
        mesh_path.unlink(missing_ok=True)
        xml_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    renderer.enable_segmentation_rendering()
    rows = []
    motion_rows = []
    try:
        for name, (position, _) in targets.items():
            for other in targets:
                model.body_pos[model.body(f"test_{other}").id] = (
                    position if other == name else (10, 0, 0))
            mujoco.mj_forward(model, data)
            gid = model.geom(f"test_{name}").id
            for camera in ("upper_rgb", "lower_rgb"):
                renderer.update_scene(data, camera=camera)
                image = renderer.render()
                pixels = int(np.count_nonzero(
                    (image[:, :, 0] == gid) &
                    (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))
                rows.append({"target": name, "camera": camera, "visible_pixels": pixels})

        evidence_report = ROOT / design["evidence"]["report"]
        if sha256(evidence_report) != design["evidence"]["report_sha256"]:
            raise ValueError("Selected CAD evidence report hash changed")
        evidence = json.loads(evidence_report.read_text())
        trace = ROOT / evidence["trace"]
        if sha256(trace) != evidence["trace_sha256"]:
            raise ValueError("Selected CAD trace hash changed")
        archive = np.load(trace, allow_pickle=False)
        poses = archive["qpos"]
        if poses.shape[1] != model.nq:
            raise ValueError("Saved pose vector is incompatible with selected visual MJCF")
        sample_indices = np.unique(np.linspace(0, len(poses) - 1, 20).round().astype(int))
        for frame in sample_indices:
            data.qpos[:] = poses[frame]
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            forward = rotation.reshape(3, 3)[:2, 0]
            forward /= np.linalg.norm(forward)
            for name, (relative, _) in targets.items():
                world_xy = data.qpos[:2] + relative[0] * forward
                world_xyz = (float(world_xy[0]), float(world_xy[1]), relative[2])
                for other in targets:
                    model.body_pos[model.body(f"test_{other}").id] = (
                        world_xyz if other == name else (100, 0, 0))
                mujoco.mj_forward(model, data)
                gid = model.geom(f"test_{name}").id
                for camera in ("upper_rgb", "lower_rgb"):
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render()
                    pixels = int(np.count_nonzero(
                        (image[:, :, 0] == gid) &
                        (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))
                    motion_rows.append({"target": name, "camera": camera,
                                        "saved_pose_index": int(frame),
                                        "visible_pixels": pixels})
    finally:
        renderer.close()
    return {"visual_path": str(visual.relative_to(ROOT)),
            "visual_sha256": expected_sha256,
            "scope": "Neutral-pose ideal marker segmentation through candidate head cage; "
                     "moving screen replays saved qpos and relocates each marker to the same "
                     "ego-relative task distance. No new policy, physical lens, blur, "
                     "board connector, detection or recognition.",
            "rows": rows,
            "saved_motion_trace_sha256": evidence["trace_sha256"],
            "saved_motion_frames_sampled": len(sample_indices),
            "moving_pose_summary": {
                name: {camera: {
                    "visible_frames": sum(r["visible_pixels"] > 0 for r in motion_rows
                                          if r["target"] == name and r["camera"] == camera),
                    "minimum_pixels": min(r["visible_pixels"] for r in motion_rows
                                          if r["target"] == name and r["camera"] == camera),
                    "median_pixels": float(np.median([
                        r["visible_pixels"] for r in motion_rows
                        if r["target"] == name and r["camera"] == camera])),
                } for camera in ("upper_rgb", "lower_rgb")}
                for name in targets},
            "moving_pose_rows": motion_rows}


def shell_solid_screen(candidate: dict, shell_report: dict) -> dict:
    """Check pitched camera board volumes against the parallel Cove shell study."""
    step = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-neutral.step"
    if sha256(step) != shell_report["shell_step_sha256"]:
        raise ValueError("Saved Cove shell STEP changed from its evidence report")
    parts = {part.label: part for part in bd.import_step(str(step)).children}
    head_names = {row["name"] for row in shell_report["shell_parts_by_body"]
                  if row["body"] == "head"}
    board_cutters = {}
    for name, pitch in (("upper", 30), ("lower", -40)):
        center = candidate["camera_rays"][name]["optical_center_mm"]
        axis = candidate["camera_rays"][name]["axis_forward"]
        board_center = tuple(center[i] - 6.2 * axis[i] for i in range(3))
        board_cutters[name] = bd.Pos(*board_center) * bd.Rot(Y=-pitch) * bd.Box(14.4, 26, 27)
    overlaps = []
    cuts = []
    for name in sorted(head_names):
        shape = parts[name]
        for camera, board in candidate["camera_boards"].items():
            overlap = volume_overlap(shape, board)
            if overlap > 1e-3:
                overlaps.append({"shell_part": name, "camera": camera,
                                 "overlap_mm3": overlap})
        if name not in ("face_mask", "muzzle_shell"):
            continue
        before = shape.volume
        for cutter in board_cutters.values():
            shape = shape - cutter
        removed = before - shape.volume
        if removed > 1e-3:
            cuts.append({"shell_part": name, "removed_mm3": removed,
                         "remaining_solid_count": len(shape.solids())})
    return {"saved_shell_step_sha256": shell_report["shell_step_sha256"],
            "current_shell_board_intersections": overlaps,
            "candidate_1mm_board_clearance_cuts": cuts,
            "scope": "Shell CAD volume only; no fastening, stiffness, smooth appearance, "
                     "wire or continuous motion proof."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    design, source = load_selected()
    contract = yaml.safe_load(CONTRACT.read_text())
    candidate = build_candidate(source, contract)
    checks = screen(candidate)
    gross_fit_pass = (
        checks["head_connected_solid_count"] == 1
        and all(v < 1e-6 for v in checks["camera_board_frame_overlap_mm3"].values())
        and checks["camera_board_board_overlap_mm3"] < 1e-6
        and all(v > 0 for v in checks["camera_saddle_cut_cage_overlap_mm3"].values())
        and checks["imu_board_torso_frame_overlap_mm3"] < 1e-6
        and checks["imu_board_battery_overlap_mm3"] < 1e-6
        and checks["imu_board_shelf_overlap_mm3"] < 1e-6
        and checks["imu_shelf_spine_overlap_mm3"] > 0
        and checks["imu_shelf_battery_overlap_mm3"] < 1e-6
    )
    old_head = dict(source.stage_frames())["head"]
    old_torso = dict(source.stage_frames())["torso"]
    old_imu = old_torso["imu_reserve"]
    old_frame = [v for k, v in old_torso.items() if k.endswith("_frame")]
    baseline = {
        "single_camera_module_count": sum("camera_module" in key for key in old_head),
        "existing_camera_frame_overlap_mm3": volume_overlap(
            old_head["head_open_frame"], old_head["camera_module"]),
        "existing_imu_reserve_torso_frame_overlap_mm3": sum(
            volume_overlap(old_imu, frame) for frame in old_frame),
    }
    report = {"design_id": design["id"], "source_sha256": design["generator"]["sha256"],
              "contract_sha256": sha256(CONTRACT),
              "camera_envelope_mm": [12.4, 24, 25],
              "camera_rays": candidate["camera_rays"],
              "imu_board_mm": [20, 13, 3],
              "imu_position_mm": candidate["imu_position_mm"],
              "selected_cad_baseline": baseline, "checks": checks,
              "bare_frame_gross_fit_pass": gross_fit_pass,
              "head_frame_volume_mm3": {
                  "selected": old_head["head_open_frame"].volume,
                  "candidate": candidate["head_frame"].volume},
              "imu_shelf_volume_mm3": candidate["imu_shelf"].volume,
              "optical_screen": optical_screen(design, candidate),
              "scope": "Source-size envelope/connectedness screen. No lens/board PCB datum, "
                       "fasteners, output connector, CSI cable bend, printed tolerance, "
                       "moving-arm sweep or physical part. Optical screen is ideal-marker only."}
    shell_report_path = ROOT / "artifacts/ape-frame50-cove-shell-v1/report.json"
    shell_visual = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-visual.xml"
    if shell_report_path.exists() and shell_visual.exists():
        shell_report = json.loads(shell_report_path.read_text())
        if shell_report.get("source_visual_body_sha256") == design["evidence"]["visual_body_sha256"]:
            report["optional_cove_shell_screen"] = optical_screen(
                design, candidate, visual=shell_visual,
                expected_sha256=shell_report["shell_visual_body_sha256"])
            report["optional_cove_shell_solid_screen"] = shell_solid_screen(
                candidate, shell_report)
            report["optional_cove_shell_board_fit_pass"] = not report[
                "optional_cove_shell_solid_screen"]["current_shell_board_intersections"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected_cad_baseline": baseline, "candidate_checks": checks,
                      "optical_screen": report["optical_screen"]["rows"],
                      "moving_pose_summary": report["optical_screen"]["moving_pose_summary"],
                      "optional_shell_summary": report.get("optional_cove_shell_screen", {}).get(
                          "moving_pose_summary"),
                      "optional_shell_board_intersections": report.get(
                          "optional_cove_shell_solid_screen", {}).get(
                              "current_shell_board_intersections")},
                     indent=2))


if __name__ == "__main__":
    main()
