"""Compare mounted and counterfactual cameras on native 50 cm reach poses.

The policy/physics already ran. This reads its full qpos and renders the same
2 cm toy through fixed cameras, with an evaluator-only segmentation mask.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-cove50-squat-reach-handoff-20261010"
BODY = ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"
OUTPUT = ROOT / "artifacts/ape-sensors-50cm/reach-views-v2"
SEEDS = (9711, 9712)
CAMERAS = ("upper_rgb", "lower_rgb", "belly_down", "forearm_side")
FRAME_SIZE = (320, 240)
SAMPLE_EVERY = 5  # 10 Hz from native 50 Hz qpos.


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_for_replay() -> mujoco.MjModel:
    xml = ET.fromstring(BODY.read_text())
    # Local MuJoCo predates the saved native runtime's contact-sensor schema.
    # Removal affects display only; no physics is stepped here.
    xml.remove(xml.find("sensor"))
    world = xml.find("worldbody")
    toy = ET.SubElement(world, "body", name="audit_toy_body")
    ET.SubElement(toy, "freejoint", name="audit_toy_root")
    ET.SubElement(toy, "geom", name="audit_toy", type="box",
                  size=".01 .01 .01", mass=".00625", rgba="1 .22 .08 1",
                  contype="0", conaffinity="0")
    torso = xml.find(".//body[@name='torso']")
    ET.SubElement(torso, "camera", name="belly_down", pos=".05 .02 .05",
                  xyaxes="0 -1 0 .965926 0 .258819", fovy="67")
    forearm = xml.find(".//body[@name='right_forearm']")
    # Optical center is outside the forearm stock-mass box and frame capsule.
    # This is still a placement hypothesis, not a packaged camera module.
    ET.SubElement(forearm, "camera", name="forearm_side", pos=".03 -.04 -.05",
                  xyaxes="1 0 0 0 1 0", fovy="67")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    if model.nq != 35:
        raise ValueError(f"Native robot+toy trace needs 35 qpos, got {model.nq}")
    model.geom_rgba[model.geom_rgba[:, 3] == 0, 3] = 1
    model.geom_group[:] = 1
    model.geom_group[model.geom("floor").id] = 0
    model.geom_group[model.geom("audit_toy").id] = 0
    return model


def toy_mask(segmentation: np.ndarray, toy_geom_id: int) -> np.ndarray:
    return ((segmentation[:, :, 0] == toy_geom_id) &
            (segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM))


def capture(renderer: mujoco.Renderer, data: mujoco.MjData, camera: str,
            normal: mujoco.MjvOption, ideal: mujoco.MjvOption,
            toy_geom_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    renderer.update_scene(data, camera=camera, scene_option=normal)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    visible = toy_mask(renderer.render(), toy_geom_id).copy()
    renderer.disable_segmentation_rendering()
    renderer.enable_depth_rendering()
    depth = renderer.render().copy()
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=camera, scene_option=ideal)
    renderer.enable_segmentation_rendering()
    without_robot = toy_mask(renderer.render(), toy_geom_id).copy()
    renderer.disable_segmentation_rendering()
    return rgb, visible, without_robot, depth


def image_panel(rgb: np.ndarray, visible: np.ndarray, *, label: str,
                t: float, phase: str, distance_mm: float) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    yy, xx = np.where(visible)
    if len(xx):
        cv2.rectangle(bgr, (int(xx.min()), int(yy.min())),
                      (int(xx.max()), int(yy.max())), (86, 228, 126), 2)
    cv2.rectangle(bgr, (0, 0), (319, 47), (25, 25, 25), -1)
    cv2.putText(bgr, f"{label}: {len(xx)} toy px", (9, 20),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 245, 240), 1, cv2.LINE_AA)
    cv2.putText(bgr, f"t {t:.1f}s {phase} | hand {distance_mm:.0f}mm", (9, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .42, (200, 210, 215), 1, cv2.LINE_AA)
    return bgr


def run_seed(model: mujoco.MjModel, seed: int) -> tuple[list[dict], list[dict], dict]:
    name = f"selected-continuous-{seed}-roll025"
    trace_path, report_path = SOURCE / f"{name}.npz", SOURCE / f"{name}.json"
    trace = np.load(trace_path)
    report = json.loads(report_path.read_text())
    if report["body_sha256"] != sha(BODY) or trace["qpos"].shape != (775, 35):
        raise ValueError(f"Native source changed: {name}")
    if len(report["timeseries"]) != 775 or not np.isclose(float(trace["step_dt_s"]), .02):
        raise ValueError("Native phase/timing mismatch")
    data = mujoco.MjData(model)
    normal, ideal = mujoco.MjvOption(), mujoco.MjvOption()
    normal.geomgroup[:] = 0
    normal.geomgroup[0] = normal.geomgroup[1] = 1
    ideal.geomgroup[:] = 0
    ideal.geomgroup[0] = 1
    toy_id = model.geom("audit_toy").id
    video_path = OUTPUT / f"{name}-four-view.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (640, 480))
    if not writer.isOpened():
        raise RuntimeError("Could not write reach-view MP4")
    rows = []
    try:
        with mujoco.Renderer(model, width=FRAME_SIZE[0], height=FRAME_SIZE[1]) as renderer:
            for i in range(0, len(trace["qpos"]), SAMPLE_EVERY):
                data.qpos[:] = trace["qpos"][i]
                mujoco.mj_forward(model, data)
                point = report["timeseries"][i]
                t = float(point["time_s"])
                phase = str(trace["phase"][i])
                distance = float(point["cube_to_hand_distance_m"])
                panels = []
                for camera in CAMERAS:
                    rgb, visible, ideal_mask, depth = capture(
                        renderer, data, camera, normal, ideal, toy_id)
                    ys, xs = np.where(visible)
                    count, ideal_count = len(xs), int(ideal_mask.sum())
                    rows.append({"seed": seed, "sample_index": i, "time_s": round(t, 3),
                                 "phase": phase, "camera": camera,
                                 "cube_to_hand_distance_m": round(distance, 5),
                                 "toy_visible_pixels": count,
                                 "toy_pixels_without_robot": ideal_count,
                                 "self_occluded_pixels": max(0, ideal_count - count),
                                 "toy_bbox_xyxy": None if count == 0 else
                                 [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                                 "toy_median_optical_depth_m": None if count == 0 else
                                 round(float(np.median(depth[visible])), 4)})
                    panels.append(image_panel(rgb, visible, label=camera,
                                              t=t, phase=phase,
                                              distance_mm=distance * 1000))
                writer.write(np.vstack((np.hstack(panels[:2]), np.hstack(panels[2:]))))
    finally:
        writer.release()
    # A separate counterfactual distance screen at three recorded robot poses.
    distances = []
    with mujoco.Renderer(model, width=FRAME_SIZE[0], height=FRAME_SIZE[1]) as renderer:
        for i in (0, 350, 650):
            for target_x in (.12, .25, .40):
                pose = trace["qpos"][i].copy()
                pose[-7:-4] = [target_x, .104, .01]
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                for camera in CAMERAS:
                    _, visible, ideal_mask, depth = capture(
                        renderer, data, camera, normal, ideal, toy_id)
                    distances.append({"seed": seed, "native_pose_index": i,
                                      "phase": str(trace["phase"][i]),
                                      "counterfactual_toy_x_m": target_x,
                                      "camera": camera,
                                      "toy_visible_pixels": int(visible.sum()),
                                      "toy_pixels_without_robot": int(ideal_mask.sum())})
    source = {"seed": seed, "trace_sha256": sha(trace_path),
              "report_sha256": sha(report_path), "video": str(video_path.relative_to(ROOT)),
              "video_sha256": sha(video_path), "frames": 155}
    return rows, distances, source


def main() -> None:
    pointer = json.loads((ROOT / "robot/ape-current-design.json").read_text())
    if pointer.get("status") != "rejected_no_selected_mechanical_design":
        raise ValueError("Historical body label needs re-evaluation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = model_for_replay()
    all_rows, all_distances, sources = [], [], []
    for seed in SEEDS:
        rows, distances, source = run_seed(model, seed)
        all_rows.extend(rows)
        all_distances.extend(distances)
        sources.append(source)
    with (OUTPUT / "native-reach-view-rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)
    summary = {}
    for camera in CAMERAS:
        summary[camera] = {}
        for phase in ("posture", "preposition", "reach", "hand_within_50mm"):
            subset = [row for row in all_rows if row["camera"] == camera and
                      (row["phase"] == phase if phase != "hand_within_50mm" else
                       row["cube_to_hand_distance_m"] <= .05)]
            pixels = [row["toy_visible_pixels"] for row in subset]
            summary[camera][phase] = {"frames": len(subset),
                                      "toy_visible_frames": sum(p > 0 for p in pixels),
                                      "median_visible_pixels": None if not any(pixels) else
                                      int(np.median([p for p in pixels if p])),
                                      "max_visible_pixels": max(pixels, default=0),
                                      "outside_fov_frames": sum(row["toy_pixels_without_robot"] == 0
                                                                for row in subset),
                                      "fully_self_occluded_frames": sum(
                                          row["toy_visible_pixels"] == 0 and
                                          row["toy_pixels_without_robot"] > 0 for row in subset)}
    result = {"schema": "robot-learning-lab.cove50-native-reach-camera-observability.v1",
              "classification": "rendered native articulated pose and moving toy replay; no new physics",
              "current_pointer_status": pointer["status"],
              "source_body_sha256": sha(BODY),
              "analysis_sha256": sha(Path(__file__)),
              "source_report_sha256": sha(SOURCE / "report.json"),
              "camera_models": {"upper_rgb": "historical body head upper, 67 degree vertical FOV",
                                "lower_rgb": "historical body head lower, 67 degree vertical FOV",
                                "belly_down": "counterfactual rigid torso mount (0.05,0.02,0.05)m, down 75 degrees",
                                "forearm_side": "counterfactual optical center outside right forearm at (0.03,-0.04,-0.05)m, local down; module not packaged"},
              "image_size_px": list(FRAME_SIZE), "sample_period_s": .1,
              "sources": sources, "summary": summary,
              "counterfactual_distance_rows": all_distances,
              "limits": ["The native trace's robot and toy qpos are replayed; no camera affects the saved policy or contacts.",
                         "The Direct Frame is rejected. The forearm and belly cameras have no physical mounting, focus, cable, mass, exposure or power qualification. Forearm optical center clears its simplified stock box; source-sized module fit is unproved.",
                         "Pixel support and visibility are evaluator segmentation, not image recognition or grasp verification; the native reach never achieves possession.",
                         "The counterfactual distance screen moves the toy only for rendering and is not a reachable physics episode."]}
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({camera: summary[camera]["hand_within_50mm"] for camera in CAMERAS}, indent=2))


if __name__ == "__main__":
    main()
