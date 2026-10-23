"""Compare Cove-50 fixed head-down and torso-down view roles.

Evaluator markers move to a constant base-relative task position in saved
joint poses. This is an optical/occlusion screen, not recognition, camera
packaging, a stationary-world approach or a fresh policy rollout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DESIGN = ROOT / "robot/ape-current-design.json"
SHELL_REPORT = ROOT / "artifacts/ape-frame50-cove-shell-v1/report.json"
SHELL_VISUAL = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-visual.xml"
CAMERAS = {
    "lower_rgb": None,
    "chest_low50": (0.06, 0, 0.265, -50),
    "chest_low50_x50": (0.05, 0, 0.265, -50),
    "chest_low50_x45": (0.045, 0, 0.265, -50),
    "chest_low60": (0.06, 0, 0.265, -60),
    "chest_low70": (0.06, 0, 0.265, -70),
    "chest_mid60": (0.06, 0, 0.315, -60),
}
TARGETS = {
    "floor_center_12cm": ((0.12, 0, 0.04), 0.04),
    "floor_center_25cm": ((0.25, 0, 0.04), 0.04),
    "floor_center_40cm": ((0.4, 0, 0.04), 0.04),
    "floor_right_12cm": ((0.12, -0.15, 0.04), 0.04),
    "floor_right_25cm": ((0.25, -0.15, 0.04), 0.04),
    "small_floor_80cm": ((0.8, 0, 0.015), 0.015),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def xyaxes(elevation_deg: float) -> str:
    angle = math.radians(elevation_deg)
    return f"0 -1 0 {-math.sin(angle):.9f} 0 {math.cos(angle):.9f}"


def load_model(visual: Path, *, targets: dict | None = None) -> mujoco.MjModel:
    targets = TARGETS if targets is None else targets
    xml = ET.parse(visual)
    root = xml.getroot()
    torso = next(b for b in root.iter("body") if b.get("name") == "torso")
    for name, candidate in CAMERAS.items():
        if candidate is None:
            continue
        x, y, z, pitch = candidate
        ET.SubElement(torso, "camera", name=name,
                      pos=f"{x} {y} {z - 0.231}",
                      xyaxes=xyaxes(pitch), fovy="67")
    world = root.find("worldbody")
    for name, (_, radius) in targets.items():
        body = ET.SubElement(world, "body", name=f"probe_{name}", pos="10 0 0")
        ET.SubElement(body, "geom", name=f"probe_{name}", type="sphere",
                      size=str(radius), rgba="1 0 0 1", group="0",
                      contype="0", conaffinity="0")
    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=visual.parent,
                                     delete=False) as temporary:
        temporary.write(ET.tostring(root, encoding="unicode"))
        scene = Path(temporary.name)
    try:
        return mujoco.MjModel.from_xml_path(str(scene))
    finally:
        scene.unlink()


def chest_board_screen(shell_report: dict) -> list[dict]:
    """Source-sized board fit for the useful low-chest 50-degree view."""
    import build123d as bd
    from cove_sensor_package import camera_envelope, load_selected, volume_overlap

    _, selected = load_selected()
    step = ROOT / "artifacts/ape-frame50-cove-shell-v1/cove50-shell-neutral.step"
    if sha256(step) != shell_report["shell_step_sha256"]:
        raise ValueError("Saved Cove shell STEP changed from its evidence report")
    parts = {part.label: part for part in bd.import_step(str(step)).children}
    front = parts["torso_front"]
    torso = dict(selected.stage_frames())["torso"]
    results = []
    for x in (45, 50, 60):
        board, _, axis = camera_envelope((x, 0, 265), -50)
        board_center = (x - 6.2 * axis[0], 0, 265 - 6.2 * axis[2])
        cutter = bd.Pos(*board_center) * bd.Rot(Y=50) * bd.Box(14.4, 26, 27)
        cut = front - cutter
        results.append({"optical_center_mm": [x, 0, 265],
                        "board_shell_overlap_mm3": volume_overlap(board, front),
                        "board_structural_frame_overlap_mm3": sum(
                            volume_overlap(board, shape) for name, shape in torso.items()
                            if name.endswith("_frame")),
                        "board_battery_overlap_mm3": volume_overlap(
                            board, torso["battery_4s850"]),
                        "shell_1mm_clearance_cut_removed_mm3": front.volume - cut.volume,
                        "shell_solid_count_after_cut": len(cut.solids())})
    return results


def run(output: Path) -> dict:
    design = json.loads(DESIGN.read_text())
    if design["id"] != "cove-50-direct-frame":
        raise ValueError("Current selected design is not Cove-50")
    shell = json.loads(SHELL_REPORT.read_text())
    if shell["source_visual_body_sha256"] != design["evidence"]["visual_body_sha256"]:
        raise ValueError("Shell is not based on the selected Cove-50 visual body")
    if sha256(SHELL_VISUAL) != shell["shell_visual_body_sha256"]:
        raise ValueError("Cove shell visual body changed from its evidence report")
    frame_report = ROOT / design["evidence"]["report"]
    if sha256(frame_report) != design["evidence"]["report_sha256"]:
        raise ValueError("Selected direct-frame report changed")
    trace_info = json.loads(frame_report.read_text())
    trace = ROOT / trace_info["trace"]
    if sha256(trace) != trace_info["trace_sha256"]:
        raise ValueError("Selected saved motion trace changed")
    model = load_model(SHELL_VISUAL)
    data = mujoco.MjData(model)
    poses = np.load(trace, allow_pickle=False)["qpos"]
    if poses.shape[1] != model.nq:
        raise ValueError("Saved pose trace has incompatible joint ordering/width")
    indices = np.unique(np.linspace(0, len(poses) - 1, 20).round().astype(int))
    renderer = mujoco.Renderer(model, width=640, height=480)
    renderer.enable_segmentation_rendering()
    rows = []
    try:
        for frame in (-1, *indices):
            data.qpos[:] = model.qpos0 if frame == -1 else poses[frame]
            origin_xy = data.qpos[:2].copy()
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            basis = rotation.reshape(3, 3)[:2, :2]
            for name, (relative, _) in TARGETS.items():
                xy = origin_xy + basis @ np.asarray(relative[:2])
                xyz = (float(xy[0]), float(xy[1]), relative[2])
                for other in TARGETS:
                    model.body_pos[model.body(f"probe_{other}").id] = (
                        xyz if other == name else (100, 0, 0))
                mujoco.mj_forward(model, data)
                gid = model.geom(f"probe_{name}").id
                for camera in CAMERAS:
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render()
                    pixels = int(np.count_nonzero(
                        (image[:, :, 0] == gid) &
                        (image[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))))
                    rows.append({"frame": int(frame), "target": name,
                                 "camera": camera, "visible_pixels": pixels})
    finally:
        renderer.close()
    summary = {}
    for target in TARGETS:
        summary[target] = {}
        for camera in CAMERAS:
            neutral = next(r["visible_pixels"] for r in rows if r["frame"] == -1
                           and r["target"] == target and r["camera"] == camera)
            moving = [r["visible_pixels"] for r in rows if r["frame"] >= 0
                      and r["target"] == target and r["camera"] == camera]
            summary[target][camera] = {
                "neutral_pixels": neutral,
                "moving_visible_frames": sum(v > 0 for v in moving),
                "moving_minimum_pixels": min(moving),
                "moving_median_pixels": float(np.median(moving)),
            }
    return {"design_id": design["id"], "shell_visual_sha256": shell["shell_visual_body_sha256"],
            "trace_sha256": trace_info["trace_sha256"],
            "camera_candidates_m_degrees": CAMERAS,
            "low_chest_board_screen": chest_board_screen(shell),
            "target_definitions": TARGETS,
            "samples": len(indices), "summary": summary, "rows": rows,
            "scope": "Virtual torso views on the current Cove shell study; evaluator sphere pixels "
                     "at constant ego-relative positions across saved old-policy qpos. "
                     "No source-size chest package, stationary approach, sharpness, "
                     "detector, current-body policy or physical view."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
