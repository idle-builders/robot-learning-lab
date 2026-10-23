"""Timing-aware fast RGB tracking in live RoboCasa owner re-show episodes.

The owner teaches a textured item once in style 1 using the existing generic
detector and a click. Later styles reuse that memory. A cheap whole-frame
template search runs on each new frame (threshold fixed on style 1), followed
by two-frame image/depth stability. The owner and eye rig are still scripted;
this is not a physical ape or a general recognizer.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
sys.path.insert(0, str(Path(__file__).parent))
from active_scan import Vision  # noqa: E402
from probe_robocasa_eye_height import WIDTH, HEIGHT, capture, scene
from run_robocasa_owner_reveal_v400 import SCAN_X, depth_location, decorate, iou


OUT = ROOT / "artifacts/ape-robocasa-fast-owner-v403"
THRESHOLD = .60  # chosen on style 1 before the style 2/3 run
SCALES = (.8, .9, 1.0, 1.1, 1.2)
MAX_PACKET_AGE_S = .20


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FastActor:
    def __init__(self, template: np.ndarray, use_depth: bool) -> None:
        self.template = template
        self.use_depth = use_depth
        self.previous: dict | None = None
        self.streak = 0

    def reset_continuity(self) -> None:
        self.previous = None
        self.streak = 0

    def observe(self, image: Image.Image, depth: np.ndarray | None,
                camera_x_est: float) -> dict:
        start = time.perf_counter()
        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        options = []
        for scale in SCALES:
            scaled = cv2.resize(self.template, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
            response = cv2.matchTemplate(bgr, scaled, cv2.TM_CCOEFF_NORMED)
            _, score, _, xy = cv2.minMaxLoc(response)
            options.append((score, [xy[0], xy[1], xy[0] + scaled.shape[1],
                                    xy[1] + scaled.shape[0]], scale))
        score, best_box, scale = max(options, key=lambda option: option[0])
        box = best_box if score >= THRESHOLD else None
        point = depth_location(box, depth, camera_x_est) if box is not None and depth is not None else None
        same = False
        if box is not None and self.previous is not None:
            if self.use_depth:
                previous_point = self.previous["point"]
                same = (point is not None and previous_point is not None and
                        np.linalg.norm(np.array(point) - np.array(previous_point)) <= .08)
            else:
                same = (camera_x_est == self.previous["camera_x_est"] and
                        iou(box, self.previous["box"]) >= .45)
        self.streak = self.streak + 1 if same else (1 if box is not None else 0)
        self.previous = {"box": box, "point": point, "camera_x_est": camera_x_est} if box is not None else None
        status = ("metric_localized" if self.streak >= 2 and self.use_depth and point is not None
                  else "visually_reacquired_unlocalized" if self.streak >= 2 and not self.use_depth
                  else "candidate_unconfirmed" if box is not None else "no_candidate")
        return {"status": status, "box": box, "point_xyz_m": point,
                "similarity": float(score) if box is not None else None,
                "top_similarity": float(score), "scale": scale,
                "streak": self.streak,
                "inference_wall_s": round(time.perf_counter() - start, 4)}


def teach_once() -> tuple[np.ndarray, dict]:
    random.seed(3101)
    np.random.seed(3101)
    model = scene(style_id=1)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    model.cam_pos[model.camera("head").id, 0] = SCAN_X[0]
    mujoco.mj_forward(model, data)
    ids = {name: model.geom(name + "_geom").id for name in ("shown_can", "counter_can")}
    rgb, _, pixels, boxes = capture(renderer, data, "head", ids)
    renderer.close()
    image = Image.fromarray(rgb)
    gt_box = boxes["shown_can"]
    if gt_box is None or pixels["shown_can"] < 100:
        raise RuntimeError("Teaching can not visible")
    click = [int((gt_box[0] + gt_box[2]) / 2), int((gt_box[1] + gt_box[3]) / 2)]
    vision = Vision(teach_selector="generic-proposal")
    start = time.perf_counter()
    proposals = vision.detect(image)
    actor_box, detail = vision.select_click(image, click, proposals)
    latency = time.perf_counter() - start
    if actor_box is None:
        raise RuntimeError(f"Owner click not covered by generic proposal: {detail}")
    crop = image.crop(actor_box)
    OUT.mkdir(parents=True, exist_ok=True)
    crop.save(OUT / "taught-crop.png")
    template = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
    return template, {"owner_click": click, "actor_box": actor_box,
                      "evaluator_box": gt_box, "proposals": len(proposals),
                      "detector_wall_s": latency, "selection": detail,
                      "crop_sha256": sha256(OUT / "taught-crop.png")}


def episode(style: int, response: str, use_depth: bool, template: np.ndarray) -> dict:
    scene_seed = 3100 + style
    random.seed(scene_seed)
    np.random.seed(scene_seed)
    model = scene(style_id=style)
    scene_fingerprint = hashlib.sha256(
        b"".join(array.tobytes() for array in
                 (model.body_pos, model.geom_pos, model.geom_quat,
                  model.geom_size, model.geom_rgba, model.light_diffuse))).hexdigest()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    can_ids = {name: model.geom(name + "_geom").id for name in ("shown_can", "counter_can")}
    camera_id = model.camera("head").id
    target_id = model.body("shown_can").id
    actor = FastActor(template, use_depth)
    camera_x = SCAN_X[0]
    t_s = 0.0
    events: list[dict] = []
    frames: list[np.ndarray] = []

    def observe(stage: str) -> dict:
        nonlocal t_s
        model.cam_pos[camera_id, 0] = camera_x
        mujoco.mj_forward(model, data)
        start = time.perf_counter()
        renderer.disable_segmentation_rendering()
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera="head")
        rgb = renderer.render().copy()
        depth = None
        if use_depth:
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera="head")
            depth = renderer.render().copy()
            renderer.disable_depth_rendering()
        image = Image.fromarray(rgb)
        decision = actor.observe(image, depth, camera_x)
        actor_path_wall = time.perf_counter() - start
        if actor_path_wall > MAX_PACKET_AGE_S:
            decision["status"] = "stale_packet"
            actor.reset_continuity()
        # The presented packet is consumed before the next authored action.
        # If a frame takes longer than the nominal 100 ms tick, simulated time
        # advances by the measured capture + actor delay. Evaluator work is
        # intentionally outside this timing path.
        t_s += max(.1, actor_path_wall)
        eval_start = time.perf_counter()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="head")
        seg = renderer.render().copy()
        renderer.disable_segmentation_rendering()
        pixels = {}
        boxes = {}
        for name, geom_id in can_ids.items():
            ys, xs = np.nonzero(seg[:, :, 0] == geom_id)
            pixels[name] = int(len(xs))
            boxes[name] = ([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
                           if len(xs) else None)
        evaluator_wall = time.perf_counter() - eval_start
        row = {"time_s": round(t_s, 3), "stage": stage,
               "camera_x_est_m": camera_x,
               "sensors": {"rgb": True, "depth": use_depth,
                           "frame_age_s": round(actor_path_wall, 4),
                           "camera_pose_age_s": round(actor_path_wall, 4)},
               "actor": decision,
               "evaluator": {"shown_visible_pixels": pixels["shown_can"],
                             "shown_box": boxes["shown_can"],
                             "counter_visible_pixels": pixels["counter_can"],
                             "target_xyz_m": model.body_pos[target_id].tolist()},
               "capture_and_inference_wall_s": round(actor_path_wall, 4),
               "tick_overrun_s": round(max(0.0, actor_path_wall - .1), 4),
               "evaluator_wall_s": round(evaluator_wall, 4),
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()}
        events.append(row)
        frames.append(decorate(image, stage=stage, actor=decision,
                               eval_box=boxes["shown_can"],
                               line=f"x={camera_x:.2f}m  {decision['status']}  score={decision['top_similarity']:.3f}"))
        return row

    # The owner-taught item is placed behind the adult-height counter before
    # the search; the actor is told neither the event nor its hidden pose.
    model.body_pos[target_id] = [3.20, -0.25, .975]
    outcome = None
    for x in SCAN_X:
        camera_x = x
        row = observe("actor_search_view")
        if row["actor"]["status"] in ("metric_localized", "visually_reacquired_unlocalized"):
            outcome = "premature_claim_during_search"
            break
        t_s += .3
    if outcome is None:
        events.append({"time_s": round(t_s, 3), "stage": "actor_asks_owner_to_show",
                       "action": "please_show_item_at_my_eye_height"})
        actor.reset_continuity()
        if response == "delayed":
            for _ in range(2):
                observe("owner_response_wait")
            events.append({"time_s": round(t_s, 3), "stage": "actor_retries_show_request",
                           "action": "please_hold_item_still_in_view"})
            actor.reset_continuity()
        if response == "ignored":
            for _ in range(3):
                observe("owner_ignores_request")
        else:
            start = model.body_pos[target_id].copy()
            end = np.array([camera_x - .35, -1.15, .43])
            move_start_s = t_s
            while t_s - move_start_s < .4:
                fraction = min(1.0, (t_s - move_start_s + .1) / .4)
                model.body_pos[target_id] = start + fraction * (end - start)
                observe("owner_moves_item")
            model.body_pos[target_id] = end
            for _ in range(3):
                row = observe("owner_holds_item_still")
                if row["actor"]["status"] in ("metric_localized", "visually_reacquired_unlocalized"):
                    outcome = row["actor"]["status"]
                    break
        if outcome is None:
            outcome = "unknown_ask_for_help"
    video = OUT / f"style{style}-{response}-{'rgbd' if use_depth else 'rgb'}.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    renderer.close()
    accepts = [e for e in events if (e.get("actor") or {}).get("status") in
               ("metric_localized", "visually_reacquired_unlocalized")]
    first = accepts[0] if accepts else None
    location_error = None
    if first and first["actor"]["point_xyz_m"] is not None:
        location_error = float(np.linalg.norm(np.array(first["actor"]["point_xyz_m"]) -
                                              np.array(first["evaluator"]["target_xyz_m"])))
    return {"style": style, "response": response, "depth": use_depth,
            "scene_seed": scene_seed, "scene_fingerprint": scene_fingerprint,
            "outcome": outcome, "first_location_error_m": location_error,
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "events": events}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    template, teaching = teach_once()
    episodes = []
    for style in (1, 2, 3):
        for response in ("compliant", "delayed", "ignored"):
            for use_depth in (False, True):
                result = episode(style, response, use_depth, template)
                episodes.append(result)
                print(style, response, "RGB-D" if use_depth else "RGB",
                      result["outcome"], result["first_location_error_m"], flush=True)
                (OUT / "report.partial.json").write_text(json.dumps(episodes, indent=2) + "\n")
    report = {"scope": "fast RGB template search and owner re-show in 3 styles; no body or native action",
              "source_sha256": sha256(Path(__file__)),
              "base_scene_source_sha256": sha256(Path(__file__).with_name("probe_robocasa_eye_height.py")),
              "teaching": teaching, "threshold": THRESHOLD, "scales": SCALES,
              "max_packet_age_s": MAX_PACKET_AGE_S,
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
