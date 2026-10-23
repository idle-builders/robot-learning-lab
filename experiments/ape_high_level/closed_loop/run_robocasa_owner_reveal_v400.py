"""Live RGB-to-social-decision transfer in a textured RoboCasa kitchen.

This joins the existing RGB proposal/retrieval model to an observation-driven
search/ask/reobserve controller. The owner and aisle eye-rig movement are
scripted surrogates. There is no physical ape, navigation policy, contact,
realistic human, or claim that the aisle is safe for the robot's full body.
Simulator segmentation and object positions are evaluator-only.

Run with PYTORCH_ENABLE_MPS_FALLBACK=1 and the local RoboCasa environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/teach_find"))
sys.path.insert(0, str(Path(__file__).parent))
from active_scan import Vision  # noqa: E402
from probe_robocasa_eye_height import WIDTH, HEIGHT, capture, scene  # noqa: E402


OUT = ROOT / "artifacts/ape-robocasa-owner-reveal-v400"
SCAN_X = (1.05, 1.65, 2.25, 2.85, 3.45, 4.05)
CAMERA_Y = -2.30
CAMERA_Z = 0.46
FOVY_DEG = 70
SIMILARITY_THRESHOLD = 0.35  # prior Cove-50 recognizer threshold
STABLE_IOU = 0.45
STABLE_DISTANCE_M = 0.08
DEPTH_MIN_M, DEPTH_MAX_M = 0.15, 2.80


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x1 - x0) * max(0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def depth_location(box: list[float], depth: np.ndarray, camera_x_est: float) -> list[float] | None:
    x0, y0, x1, y1 = box
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    cx0, cx1 = max(0, int(u - (x1 - x0) * .18)), min(WIDTH, int(u + (x1 - x0) * .18) + 1)
    cy0, cy1 = max(0, int(v - (y1 - y0) * .18)), min(HEIGHT, int(v + (y1 - y0) * .18) + 1)
    patch = depth[cy0:cy1, cx0:cx1]
    valid = patch[np.isfinite(patch) & (patch >= DEPTH_MIN_M) & (patch <= DEPTH_MAX_M)]
    if len(valid) < 9:
        return None
    distance = float(np.quantile(valid, .25))
    focal = HEIGHT / (2 * math.tan(math.radians(FOVY_DEG / 2)))
    return [float(camera_x_est + (u - WIDTH / 2) * distance / focal),
            float(CAMERA_Y + distance),
            float(CAMERA_Z + (HEIGHT / 2 - v) * distance / focal)]


class Actor:
    def __init__(self, vision: Vision, exemplar: np.ndarray, use_depth: bool) -> None:
        self.vision = vision
        self.exemplar = exemplar
        self.use_depth = use_depth
        self.previous: dict | None = None
        self.streak = 0

    def observe(self, image: Image.Image, depth: np.ndarray | None,
                camera_x_est: float) -> dict:
        start = time.perf_counter()
        proposals = self.vision.detect(image)
        scores = (self.vision.feature([image.crop(p["box"]) for p in proposals]) @ self.exemplar
                  if proposals else np.empty(0))
        ranked = sorted(zip(proposals, scores), key=lambda pair: float(pair[1]), reverse=True)
        top = ranked[0] if ranked else None
        selected = top if top and float(top[1]) >= SIMILARITY_THRESHOLD else None
        box = selected[0]["box"] if selected else None
        point = depth_location(box, depth, camera_x_est) if box is not None and depth is not None else None
        same = False
        if box is not None and self.previous is not None:
            prev_box = self.previous["box"]
            if self.use_depth:
                prev_point = self.previous["point"]
                same = (point is not None and prev_point is not None and
                        np.linalg.norm(np.array(point) - np.array(prev_point)) <= STABLE_DISTANCE_M)
            else:
                same = iou(box, prev_box) >= STABLE_IOU and camera_x_est == self.previous["camera_x_est"]
        self.streak = self.streak + 1 if same else (1 if box is not None else 0)
        self.previous = {"box": box, "point": point, "camera_x_est": camera_x_est} if box is not None else None
        status = ("metric_localized" if self.streak >= 2 and self.use_depth and point is not None
                  else "visually_reacquired_unlocalized" if self.streak >= 2 and not self.use_depth
                  else "candidate_unconfirmed" if box is not None
                  else "no_candidate")
        return {"status": status, "box": box, "point_xyz_m": point,
                "similarity": float(selected[1]) if selected else None,
                "top_similarity": float(top[1]) if top else None,
                "top_generic_label": top[0]["generic_label"] if top else None,
                "proposals": len(proposals), "streak": self.streak,
                "inference_wall_s": round(time.perf_counter() - start, 4)}


def decorate(image: Image.Image, *, stage: str, actor: dict | None,
             eval_box: list[int] | None, line: str) -> np.ndarray:
    canvas = Image.new("RGB", (WIDTH, HEIGHT + 48), "#10151a")
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 5), stage, fill="white")
    draw.text((12, 25), line[:96], fill="#d1e6e8")
    if eval_box is not None:
        draw.rectangle((eval_box[0], eval_box[1] + 48, eval_box[2], eval_box[3] + 48),
                       outline="#41bdff", width=2)
    if actor and actor["box"] is not None:
        box = actor["box"]
        draw.rectangle((box[0], box[1] + 48, box[2], box[3] + 48),
                       outline="#41e282" if actor["status"] in ("metric_localized", "visually_reacquired_unlocalized")
                       else "#f1a52d", width=3)
    return np.asarray(canvas)


def run_episode(vision: Vision, *, style: int, response: str, use_depth: bool,
                out: Path) -> dict:
    model = scene(style_id=style)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    target_body = model.body("shown_can").id
    target_geom = model.geom("shown_can_geom").id
    counter_geom = model.geom("counter_can_geom").id
    camera_id = model.camera("head").id
    can_ids = {"shown_can": target_geom, "counter_can": counter_geom}
    frames: list[np.ndarray] = []
    events: list[dict] = []
    t_s = 0.0
    camera_x = SCAN_X[0]
    last_image: Image.Image | None = None

    def observe(stage: str, actor: Actor | None = None) -> dict:
        nonlocal t_s, last_image
        model.cam_pos[camera_id, 0] = camera_x
        mujoco.mj_forward(model, data)
        rgb, depth, visible, boxes = capture(renderer, data, "head", can_ids)
        image = Image.fromarray(rgb)
        last_image = image
        observation = actor.observe(image, depth if use_depth else None, camera_x) if actor else None
        row = {"time_s": round(t_s, 3), "stage": stage, "camera_x_est_m": camera_x,
               "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
               "sensors": {"rgb": True, "depth": use_depth, "camera_pose_age_s": 0.0},
               "actor": observation,
               "evaluator": {"shown_visible_pixels": visible["shown_can"],
                             "shown_box": boxes["shown_can"],
                             "counter_visible_pixels": visible["counter_can"],
                             "target_xyz_m": model.body_pos[target_body].tolist()}}
        events.append(row)
        detail = (f"x={camera_x:.2f}m  {observation['status']}  "
                  f"sim={observation['similarity']}" if observation else f"x={camera_x:.2f}m  owner teaches by click")
        frames.append(decorate(image, stage=stage, actor=observation,
                               eval_box=boxes["shown_can"], line=detail))
        t_s += 0.1
        return row

    # The owner supplies a click, not the evaluator label or object identity.
    teach = observe("owner_show_for_teaching")
    teach_box = teach["evaluator"]["shown_box"]
    if teach_box is None or teach["evaluator"]["shown_visible_pixels"] < 100:
        renderer.close()
        return {"style": style, "response": response, "depth": use_depth,
                "outcome": "teaching_unobservable", "events": events}
    click = [int((teach_box[0] + teach_box[2]) / 2), int((teach_box[1] + teach_box[3]) / 2)]
    assert last_image is not None
    teach_image = last_image
    teach_proposals = vision.detect(teach_image)
    selected_box, click_detail = vision.select_click(teach_image, click, teach_proposals)
    if selected_box is None:
        renderer.close()
        return {"style": style, "response": response, "depth": use_depth,
                "outcome": "teaching_proposal_missing", "click": click,
                "events": events}
    exemplar = vision.feature([teach_image.crop(selected_box)])[0]
    actor = Actor(vision, exemplar, use_depth)
    events[0]["teaching"] = {"owner_click": click, "actor_box": selected_box,
                             "selection": click_detail}

    # An owner places the taught instance on the far counter. The owner script
    # changes simulator state; the actor does not receive this pose or event.
    model.body_pos[target_body] = [3.20, -0.25, 0.975]
    outcome = None
    for x in SCAN_X:
        camera_x = x
        row = observe("actor_search_view", actor)
        decision = row["actor"]["status"]
        if decision in ("metric_localized", "visually_reacquired_unlocalized"):
            outcome = "premature_claim_during_search"
            break
        # A coverage-based high-level request; eye-rig translation is an
        # unqualified kinematic skill surrogate, with no physical stop model.
        t_s += 0.3
    if outcome is None:
        events.append({"time_s": round(t_s, 3), "stage": "actor_asks_owner_to_show",
                       "action": "please_show_item_at_my_eye_height"})
        actor.previous = None
        actor.streak = 0
        if response == "delayed":
            for _ in range(2):
                observe("owner_response_wait", actor)
            events.append({"time_s": round(t_s, 3), "stage": "actor_retries_show_request",
                           "action": "please_hold_item_still_in_view"})
            actor.previous = None
            actor.streak = 0
        if response != "ignored":
            # Continuous 0.4 s simulated owner-object motion; no articulated
            # hand or audio. Actor consumes each exposed frame.
            start = model.body_pos[target_body].copy()
            end = np.array([camera_x - .35, -1.15, .43])
            for k in range(1, 5):
                model.body_pos[target_body] = start + k / 4 * (end - start)
                observe("owner_moves_item", actor)
            for _ in range(3):
                row = observe("owner_holds_item_still", actor)
                if row["actor"]["status"] in ("metric_localized", "visually_reacquired_unlocalized"):
                    outcome = row["actor"]["status"]
                    break
        else:
            for _ in range(3):
                observe("owner_ignores_request", actor)
        if outcome is None:
            outcome = "unknown_ask_for_help"

    video = out / f"style{style}-{response}-{'rgbd' if use_depth else 'rgb'}.mp4"
    imageio.mimsave(video, frames, fps=5, codec="libx264", quality=7)
    renderer.close()
    accepted = [e for e in events if (e.get("actor") or {}).get("status") in
                ("metric_localized", "visually_reacquired_unlocalized")]
    first = accepted[0] if accepted else None
    error = None
    if first and first["actor"]["point_xyz_m"] is not None:
        error = float(np.linalg.norm(np.array(first["actor"]["point_xyz_m"]) -
                                     np.array(first["evaluator"]["target_xyz_m"])))
    false_claims = sum(e["evaluator"]["shown_visible_pixels"] < 100 for e in accepted)
    return {"style": style, "response": response, "depth": use_depth,
            "outcome": outcome, "false_claims": false_claims,
            "first_location_error_m": error, "teaching_box": selected_box,
            "video": {"path": str(video), "sha256": sha256(video), "frames": len(frames)},
            "events": events}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vision = Vision(teach_selector="generic-proposal")
    episodes: list[dict] = []
    for style in (1, 2):
        for response in ("compliant", "delayed", "ignored"):
            for use_depth in (False, True):
                episode = run_episode(vision, style=style, response=response,
                                      use_depth=use_depth, out=OUT)
                episodes.append(episode)
                print(style, response, "RGB-D" if use_depth else "RGB",
                      episode["outcome"], episode.get("first_location_error_m"), flush=True)
                (OUT / "report.partial.json").write_text(json.dumps(episodes, indent=2) + "\n")
    report = {"scope": "live RGB perception to search/ask/reobserve; scripted owner and eye-rig movement, no physical body",
              "source_sha256": sha256(Path(__file__)),
              "scene_source_sha256": sha256(Path(__file__).with_name("probe_robocasa_eye_height.py")),
              "thresholds": {"similarity": SIMILARITY_THRESHOLD, "stable_iou": STABLE_IOU,
                             "stable_distance_m": STABLE_DISTANCE_M},
              "episodes": episodes}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUT / "report.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
