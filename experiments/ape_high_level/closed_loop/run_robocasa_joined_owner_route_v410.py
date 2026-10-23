"""Join a range-localized island route to fresh owner-assisted object finding.

This is a kinematic, scripted-owner integration probe. The actor sees range,
RGB and optional depth, not simulator pose or segmentation. Slow recognition
only proposes a crop; a fresh fast RGB track after that latency is required
before an owner-confirmed visual claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robosuite.models.tasks import ManipulationTask

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
from active_scan import Vision  # noqa: E402
from probe_robocasa_eye_height import HOPE, add_can  # noqa: E402
from run_fast_robocasa_owner_v403 import FastActor  # noqa: E402
from run_robocasa_layout2_route_v407 import (  # noqa: E402
    GOAL, START, SensorGrid, evaluator_clear, frame_with_map, planar_scan, sha256,
)
from run_robocasa_layout2_localized_route_v408 import RangeLocalizer  # noqa: E402


OUT = ROOT / "artifacts/ape-robocasa-joined-owner-route-v410"
TEACH_CROP = ROOT / "artifacts/ape-robocasa-kitchen-pilot/taught-rgb-crop.png"
THRESHOLD = .35  # frozen from the earlier ape recognition stack
MAX_FRESH_AGE_S = .20


def make_scene() -> mujoco.MjModel:
    random.seed(3201)
    np.random.seed(3201)
    arena = KitchenArena(layout_id=2, style_id=1,
                         rng=np.random.default_rng(3), clutter_mode=0)
    task = ManipulationTask(mujoco_arena=arena, mujoco_robots=[],
                            mujoco_objects=list(arena.fixtures.values()))
    ET.SubElement(task.asset, "mesh", name="probe_tomato_sauce",
                  file=str(HOPE / "TomatoSauce.obj"), scale="0.01 0.01 0.01")
    ET.SubElement(task.asset, "texture", name="probe_tomato_sauce_tex",
                  type="2d", file=str(HOPE / "TomatoSauce.png"))
    ET.SubElement(task.asset, "material", name="probe_tomato_sauce_mat",
                  texture="probe_tomato_sauce_tex", specular="0.1")
    add_can(task.worldbody, "shown_can", (-10., -10., .43))
    add_can(task.worldbody, "replica_can", (3., -2.3, .98))
    ET.SubElement(task.worldbody, "camera", name="head", pos="1.05 -4.2 .46",
                  xyaxes="1 0 0 0 0 1", fovy="70")
    ET.SubElement(task.worldbody, "camera", name="owner_view", pos="1.05 -4.2 .46",
                  xyaxes="0 1 0 0 0 1", fovy="70")
    return task.get_model()


def capture(renderer: mujoco.Renderer, data: mujoco.MjData,
            camera: str, with_depth: bool) -> tuple[np.ndarray, np.ndarray | None, float]:
    begin = time.perf_counter()
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    depth = None
    if with_depth:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=camera)
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()
    return rgb, depth, time.perf_counter() - begin


def evaluator_box(renderer: mujoco.Renderer, data: mujoco.MjData,
                  model: mujoco.MjModel, name: str) -> tuple[list[int] | None, int]:
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera="owner_view")
    segmentation = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    ys, xs = np.nonzero(segmentation[:, :, 0] == model.geom(name + "_geom").id)
    return ([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
            if len(xs) else None), int(len(xs))


def iou(a: list[float] | None, b: list[int] | None) -> float:
    if a is None or b is None:
        return 0.
    left, top, right, bottom = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., right - left) * max(0., bottom - top)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union else 0.


def side_depth_point(box: list[float], depth: np.ndarray,
                     camera_xy_est: np.ndarray) -> list[float] | None:
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    cx0, cx1 = max(0, int(u - (x1 - x0) * .18)), min(640, int(u + (x1 - x0) * .18) + 1)
    cy0, cy1 = max(0, int(v - (y1 - y0) * .18)), min(480, int(v + (y1 - y0) * .18) + 1)
    patch = depth[cy0:cy1, cx0:cx1]
    valid = patch[np.isfinite(patch) & (patch >= .15) & (patch <= 2.8)]
    if len(valid) < 9:
        return None
    distance = float(np.quantile(valid, .25))
    focal = 480 / (2 * math.tan(math.radians(70 / 2)))
    # owner_view looks -x; image right is +y and up is +z.
    return [float(camera_xy_est[0] - distance),
            float(camera_xy_est[1] + (u - 320) * distance / focal),
            float(.46 + (240 - v) * distance / focal)]


def owner_frame(rgb: np.ndarray, label: str, box: list[float] | None,
                gt_box: list[int] | None) -> np.ndarray:
    image = Image.new("RGB", (1040, 528), "#101619")
    image.paste(Image.fromarray(rgb), (0, 48))
    draw = ImageDraw.Draw(image)
    draw.text((10, 12), label, fill="white")
    draw.text((650, 100), "owner-facing view after localized route", fill="#d4eef2")
    draw.text((650, 124), "green: current actor box", fill="#50ec93")
    draw.text((650, 148), "blue: evaluator-only item box", fill="#70baff")
    for candidate, color in ((gt_box, "#70baff"), (box, "#50ec93")):
        if candidate is not None:
            draw.rectangle((candidate[0], candidate[1] + 48,
                            candidate[2], candidate[3] + 48), outline=color, width=3)
    return np.asarray(image)


def episode(response: str, use_depth: bool, vision: Vision,
            exemplar: np.ndarray) -> dict:
    model = make_scene()
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    head_id, owner_id = model.camera("head").id, model.camera("owner_view").id
    target_id = model.body("shown_can").id
    rng = np.random.default_rng(410)
    localizer = RangeLocalizer(5.0, grow=True)
    grid = SensorGrid()
    actual = START.copy()
    estimate = np.r_[START, 0.]
    trace: list[dict] = []
    frames: list[np.ndarray] = []
    t_s = 0.
    state = "navigate_to_owner_viewpoint"
    for step in range(100):
        model.cam_pos[head_id, :2] = actual
        model.cam_pos[owner_id, :2] = actual
        mujoco.mj_forward(model, data)
        scan = planar_scan(model, data, actual, 5., rng)
        estimate, pose_status = localizer.correct(scan, estimate)
        grid.update(scan, estimate[:2], 5.)
        rgb, _, capture_wall = capture(renderer, data, "head", False)
        requested, plan_status, path = grid.plan(estimate[:2], GOAL)
        row = {"stage": "navigate", "step": step, "time_s": round(t_s, 3),
               "actual_xy_m": actual.tolist(), "estimated_xy_m": estimate[:2].tolist(),
               "pose_error_m": float(np.linalg.norm(actual - estimate[:2])),
               "pose_status": pose_status, "plan_status": plan_status,
               "request_xy_m": requested.tolist() if requested is not None else None,
               "range_returns": sum(value is not None for value in scan),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "capture_wall_s": capture_wall}
        trace.append(row)
        frames.append(frame_with_map(rgb, grid, estimate[:2], path,
                                     "route / " + plan_status, step, 5.))
        t_s += max(.1, capture_wall)
        if np.linalg.norm(estimate[:2] - GOAL) <= .15:
            state = ("arrived_at_owner_viewpoint" if np.linalg.norm(actual - GOAL) <= .25
                     else "uncertain_arrival_stop")
            break
        if pose_status["status"] not in ("mapped", "registered"):
            state = "localization_uncertain_stop"
            break
        if requested is None:
            state = plan_status
            break
        delta = requested - estimate[:2]
        true_delta = delta * (1. + .30 / 5.89)
        checks = [evaluator_clear(model, data, actual + f * true_delta)
                  for f in (.25, .5, .75, 1.)]
        row["evaluator_segment_clear"] = all(ok for ok, _ in checks)
        if not row["evaluator_segment_clear"]:
            state = "geometry_overlap_request_stop"
            break
        grid.visits[grid.cell(estimate[:2])] += 1
        estimate[:2] += delta
        actual += true_delta
    if state != "arrived_at_owner_viewpoint":
        outcome = state
    else:
        model.cam_pos[owner_id, :2] = actual
        mujoco.mj_forward(model, data)
        before, _, capture_wall = capture(renderer, data, "owner_view", use_depth)
        t_s += max(.1, capture_wall)
        frames.append(owner_frame(before, "look toward owner: item not yet shown", None, None))
        trace.append({"stage": "ask_owner_to_show", "time_s": round(t_s, 3),
                      "action": "please_show_taught_can_and_hold_still"})
        if response == "delayed":
            for _ in range(2):
                rgb, _, age = capture(renderer, data, "owner_view", use_depth)
                t_s += max(.1, age)
                frames.append(owner_frame(rgb, "wait for owner response", None, None))
                trace.append({"stage": "owner_response_wait", "time_s": round(t_s, 3)})
            trace.append({"stage": "ask_owner_again", "time_s": round(t_s, 3),
                          "action": "please_hold_it_still_in_view"})
        if response == "ignored":
            for _ in range(3):
                rgb, _, age = capture(renderer, data, "owner_view", use_depth)
                t_s += max(.1, age)
                frames.append(owner_frame(rgb, "owner did not show item", None, None))
                trace.append({"stage": "owner_response_wait", "time_s": round(t_s, 3)})
            outcome = "unknown_ask_for_help"
        else:
            model.body_pos[target_id] = [actual[0] - .55, actual[1], .43]
            mujoco.mj_forward(model, data)
            shown_rgb, _, age = capture(renderer, data, "owner_view", use_depth)
            shown_image = Image.fromarray(shown_rgb)
            begin = time.perf_counter()
            proposals = vision.detect(shown_image)
            scores = (vision.feature([shown_image.crop(p["box"]) for p in proposals]) @ exemplar
                      if proposals else np.empty(0))
            ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
            recognition_wall = time.perf_counter() - begin
            t_s += max(.1, age + recognition_wall)
            chosen = ranked[0] if ranked and float(ranked[0][1]) >= THRESHOLD else None
            fresh_track = None
            point = None
            point_error = None
            final_iou = 0.
            gt_box, gt_pixels = evaluator_box(renderer, data, model, "shown_can")
            frames.append(owner_frame(shown_rgb, "slow recognizer proposes; wait for fresh image",
                                      chosen[0]["box"] if chosen else None, gt_box))
            trace.append({"stage": "recognize_after_owner_show", "time_s": round(t_s, 3),
                          "proposal_count": len(proposals),
                          "top_similarity": float(ranked[0][1]) if ranked else None,
                          "selected_box": chosen[0]["box"] if chosen else None,
                          "recognition_wall_s": recognition_wall,
                          "shown_visible_pixels_evaluator": gt_pixels})
            if response == "withdrawn":
                # The long recognizer's positive frame is now obsolete. The
                # owner withdraws the item before the required fresh capture.
                model.body_pos[target_id] = [-10., -10., .43]
                mujoco.mj_forward(model, data)
                trace.append({"stage": "owner_withdraws_during_recognition",
                              "time_s": round(t_s, 3)})
            if chosen is None:
                outcome = "visible_item_not_reacquired"
            else:
                crop = shown_image.crop(chosen[0]["box"])
                template = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
                tracker = FastActor(template, use_depth=False)
                for fresh_index in range(3):
                    rgb, depth, capture_wall = capture(renderer, data, "owner_view", use_depth)
                    begin = time.perf_counter()
                    fresh_track = tracker.observe(Image.fromarray(rgb), None,
                                                  float(estimate[0]))
                    age = capture_wall + (time.perf_counter() - begin)
                    if age > MAX_FRESH_AGE_S:
                        fresh_track["status"] = "stale_packet"
                        tracker.reset_continuity()
                    if fresh_track["box"] is not None and depth is not None:
                        point = side_depth_point(fresh_track["box"], depth, estimate[:2])
                    gt_box, gt_pixels = evaluator_box(renderer, data, model, "shown_can")
                    final_iou = iou(fresh_track["box"], gt_box)
                    frames.append(owner_frame(rgb, f"fresh post-recognition frame {fresh_index}: "
                                              + fresh_track["status"], fresh_track["box"], gt_box))
                    t_s += max(.1, age)
                    trace.append({"stage": "fresh_owner_view", "time_s": round(t_s, 3),
                                  "frame_age_s": age, "actor": fresh_track,
                                  "depth_point_xyz_m": point,
                                  "evaluator_iou": final_iou,
                                  "shown_visible_pixels_evaluator": gt_pixels})
                    if fresh_track["status"] == "visually_reacquired_unlocalized":
                        break
                if point is not None:
                    point_error = float(np.linalg.norm(np.array(point) - model.body_pos[target_id]))
                trace.append({"stage": "owner_confirms_taught_item", "time_s": round(t_s, 3),
                              "owner_reply": "yes_that_is_the_taught_can"})
                outcome = ("owner_confirmed_metric_find" if use_depth else "owner_confirmed_visual_find")
                if fresh_track is None or fresh_track["status"] != "visually_reacquired_unlocalized":
                    outcome = "unconfirmed_after_slow_recognition"
    renderer.close()
    OUT.mkdir(parents=True, exist_ok=True)
    video = OUT / f"layout2-{response}-{'rgbd' if use_depth else 'rgb'}.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    return {"response": response, "depth": use_depth, "outcome": outcome,
            "scene_fingerprint": scene_fingerprint, "route_state": state,
            "final_actual_xy_m": actual.tolist(), "final_estimated_xy_m": estimate[:2].tolist(),
            "final_pose_error_m": float(np.linalg.norm(actual - estimate[:2])),
            "recognition_wall_s": (recognition_wall if state == "arrived_at_owner_viewpoint"
                                   and response != "ignored" else None),
            "depth_point_error_m": (point_error if state == "arrived_at_owner_viewpoint"
                                    and response != "ignored" else None),
            "final_evaluator_iou": (final_iou if state == "arrived_at_owner_viewpoint"
                                    and response != "ignored" else None),
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "trace": trace}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(TEACH_CROP).convert("RGB")])[0]
    episodes = []
    for response, depth in (("compliant", False), ("compliant", True),
                            ("delayed", True), ("ignored", True),
                            ("withdrawn", True)):
        result = episode(response, depth, vision, exemplar)
        episodes.append(result)
        print(response, "RGB-D" if depth else "RGB", result["outcome"],
              round(result["final_pose_error_m"], 3),
              result["depth_point_error_m"], flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "route_source_sha256": sha256(Path(__file__).with_name("run_robocasa_layout2_route_v407.py")),
              "localizer_source_sha256": sha256(Path(__file__).with_name("run_robocasa_layout2_localized_route_v408.py")),
              "taught_crop_sha256": sha256(TEACH_CROP),
              "scope": "joined kinematic range route and owner visual find in RoboCasa layout2",
              "slow_recognition_threshold": THRESHOLD,
              "fast_packet_age_limit_s": MAX_FRESH_AGE_S,
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
