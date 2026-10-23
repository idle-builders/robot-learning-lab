"""Replay V35's owner inference window through two plausible fixed camera views.

The paired camera observations use the saved V35 timing and poses. The actor
receives RGB-D and a generic person detector; simulator labels are used only
to score what each view actually contained. This is a sensor replay, not a
fresh full-route or articulated-body run.
"""

from __future__ import annotations

import hashlib
import json
import math
import copy
from pathlib import Path
import sys

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/native_route_guard"))

import robocasa_first_return_person_reacquire_turn_v576 as turn  # noqa: E402
import owner_inference_response as response  # noqa: E402
import robocasa_person_interaction as interaction  # noqa: E402
import run_owner_concurrent_reacquire_v571 as detector  # noqa: E402
import run_robocasa_joined_owner_route_v410 as owner  # noqa: E402
import run_robocasa_two_moving_people_v586 as moving  # noqa: E402
import screen_robocasa_owner_person_v489 as screen  # noqa: E402
import screen_robocasa_owner_safe_stop_v494 as camera  # noqa: E402
import screen_robocasa_two_people_v582 as two  # noqa: E402


SOURCE = ROOT / "artifacts/ape-native-route-guard/v35-inference-response/report.json"
OUT = ROOT / "artifacts/ape-sensors-50cm/v35-owner-side-camera-replay"
CASES = ("wide_buffered_turning", "wide_buffered_static", "wide_no_entry")
DELAYS_S = (0., .1, .2, .3, .4, .5, .6, .7, .8)
DETECT_DELAYS_S = (0., .2, .5, .8)
OWNER_CAMERA = "owner_up25"
FLANK_CAMERA = "owner_wide25"
RGB_CHANGE = 30
DEPTH_CHANGE_M = .05
MOTION_PIXELS = 350
PIXEL_GATES = (350, 3000, 10000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def subject_xy(summary, t):
    condition = summary["condition"]
    if condition == "wide_no_entry":
        return np.asarray(summary["bystander_stage_start_evaluator_xy_m"])
    if condition == "wide_buffered_static":
        return np.asarray([.5562619673, -4.2441418102])
    start = np.asarray([.5562619673, -4.2441418102])
    target = np.asarray(summary["bystander_target_evaluator_xy_m"])
    fraction = min(1., .62 * max(0., t-summary["worker_start_s"])
                   / np.linalg.norm(target-start))
    return start+(target-start)*fraction


def image_motion(baseline_rgb, baseline_depth, rgb, depth):
    rgb_pixels = int(np.count_nonzero(np.max(cv2.absdiff(rgb, baseline_rgb), axis=2)
                                      > RGB_CHANGE))
    valid = (np.isfinite(depth) & np.isfinite(baseline_depth) &
             (depth >= .15) & (baseline_depth >= .15) &
             (depth <= 2.8) & (baseline_depth <= 2.8))
    depth_pixels = int(np.count_nonzero(valid &
                   (np.abs(depth-baseline_depth) > DEPTH_CHANGE_M)))
    return {"rgb_changed_pixels": rgb_pixels,
            "depth_changed_pixels": depth_pixels,
            "actor_motion_cue": rgb_pixels >= MOTION_PIXELS and
                                depth_pixels >= MOTION_PIXELS}


def annotate(rgb, condition, view, delay, cue, person_pixels, detections):
    frame = rgb.copy()
    for detection in detections:
        box = list(map(int, detection["box_xyxy"]))
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (255, 230, 0), 2)
        cv2.putText(frame, f"person {detection['score']:.2f}",
                    (box[0], max(22, box[1]-6)), cv2.FONT_HERSHEY_SIMPLEX,
                    .53, (255, 230, 0), 2)
    lines = [f"{condition} | {view} | +{delay:.1f}s",
             f"RGB/depth changed {cue['rgb_changed_pixels']}/{cue['depth_changed_pixels']}",
             f"cue {cue['actor_motion_cue']} | person pixels evaluator {person_pixels}"]
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (10, 28+index*25), cv2.FONT_HERSHEY_SIMPLEX,
                    .59, (255, 255, 255), 3)
        cv2.putText(frame, line, (10, 28+index*25), cv2.FONT_HERSHEY_SIMPLEX,
                    .59, (30, 30, 30), 1)
    return frame


def replay_case(summary, model, data, renderer):
    condition = summary["condition"]
    ape = np.asarray(summary["worker_ape_start_xy_m"])
    model.body_pos[model.body("audit_ape").id, :2] = ape
    # V589 freezes the presented owner at the post-turn, 40 cm south pose.
    old_show = interaction.SHOW_HAND.copy()
    interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
    try:
        turn._show_owner(model, 1., .4)
    finally:
        interaction.SHOW_HAND = old_show
    camera.aim_camera(model, OWNER_CAMERA, ape, np.asarray([1., -2.8]), 15)
    camera.aim_camera(model, FLANK_CAMERA, ape, ape+[-1., 0.], 15)
    baseline = {}
    rows, panels = [], []
    for delay in DELAYS_S:
        t = summary["worker_start_s"]+delay
        two.set_distractor(model, subject_xy(summary, t))
        mujoco.mj_forward(model, data)
        image_pair = []
        for view, cam in (("owner", OWNER_CAMERA), ("left_flank", FLANK_CAMERA)):
            rgb, depth, capture_wall = owner.capture(renderer, data, cam, True)
            if view not in baseline:
                baseline[view] = (rgb.copy(), depth.copy())
            cue = image_motion(*baseline[view], rgb, depth)
            box, person_pixels = screen.evaluator_box(
                renderer, data, model, cam, "distractor_torso_geom")
            owner_box, owner_pixels = screen.evaluator_box(
                renderer, data, model, cam, "owner_torso_geom")
            item_box, item_pixels = screen.evaluator_box(
                renderer, data, model, cam, "shown_can_geom")
            found = (detector.detect_person(rgb) if delay in DETECT_DELAYS_S
                     else {"detections": [], "worker_wall_s": None})
            detections = []
            for item in found["detections"]:
                point = camera.depth_point(item["box_xyxy"], depth,
                                           data, model, cam)
                detections.append({**item,
                                   "depth_point_xyz_m": point,
                                   "torso_iou_evaluator": {
                                       "side_person": two.box_iou(
                                           item["box_xyxy"], box),
                                       "owner": two.box_iou(
                                           item["box_xyxy"], owner_box)}})
            row = {"condition": condition, "delay_s": delay,
                   "time_s": t, "view": view,
                   "capture_wall_s": capture_wall,
                   "person_detector_wall_s": found["worker_wall_s"],
                   **cue, "person_torso_pixels_evaluator": person_pixels,
                   "person_torso_box_evaluator": box,
                   "owner_torso_pixels_evaluator": owner_pixels,
                   "owner_torso_box_evaluator": owner_box,
                   "shown_item_pixels_evaluator": item_pixels,
                   "shown_item_box_evaluator": item_box,
                   "detections": detections}
            rows.append(row)
            frame = annotate(rgb, condition, view, delay, cue,
                             person_pixels, detections)
            image_pair.append(frame)
            if delay in (0., .2, .5, .8):
                path = OUT / f"{condition}-{view}-{delay:.1f}s.png"
                Image.fromarray(frame).save(path)
                row["image"] = {"path": str(path), "sha256": sha(path)}
        panels.append(np.concatenate(image_pair, axis=1))
    return rows, panels


def two_packet_alarm(rows, pixels):
    streak = 0
    for sample in rows:
        active = (sample["rgb_changed_pixels"] >= pixels and
                  sample["depth_changed_pixels"] >= pixels)
        streak = streak+1 if active else 0
        if streak >= 2:
            return sample["delay_s"]
    return None


def main():
    source = json.loads(SOURCE.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(moving.SCENE_XML))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=640, height=480)
    all_rows, panels, outcomes = [], [], []
    for condition in CASES:
        summary = next(row for row in source["summaries"]
                       if row["condition"] == condition)
        rows, frames = replay_case(summary, model, data, renderer)
        all_rows.extend(rows)
        panels.extend(frames)
        for view in ("owner", "left_flank"):
            subset = [x for x in rows if x["view"] == view]
            sensitivity = {str(pixels): two_packet_alarm(subset, pixels)
                           for pixels in PIXEL_GATES}
            confirmed = [sample["delay_s"] for sample in subset
                         if any(d["torso_iou_evaluator"]["side_person"] >= .3
                                for d in sample["detections"])]
            outcomes.append({"condition": condition, "view": view,
                             "two_packet_motion_alarm_delay_s": sensitivity["350"],
                             "pixel_gate_sensitivity": sensitivity,
                             "first_side_person_detection_delay_s":
                             min(confirmed) if confirmed else None})
            print(condition, view,
                  [(x["delay_s"], x["actor_motion_cue"],
                    x["person_torso_pixels_evaluator"],
                    [(round(d["score"], 2),
                      {k: round(v, 2) for k, v in
                       d["torso_iou_evaluator"].items()})
                     for d in x["detections"]])
                   for x in subset], flush=True)
    renderer.close()
    fast = next(row for row in source["summaries"]
                if row["condition"] == "wide_respond_fast")
    modeled_response = []
    for delay in (.2, .3, .4, .5, .6, None):
        trial = copy.deepcopy(fast)
        if delay is None:
            trial["response_parameters"] = None
            trial["actor_alarm_s"] = None
        else:
            trial["actor_alarm_s"] = trial["worker_start_s"]+delay
        score = response.score_inference_gap(trial)
        modeled_response.append({"alarm_delay_s": delay,
                                 "minimum_proxy_gap_m": score["minimum_gap_m"],
                                 "first_below_5cm_s": score["first_below_5cm_s"]})
    video = OUT / "owner-vs-left-flank-v35.mp4"
    with imageio.get_writer(video, fps=3, codec="libx264",
                            macro_block_size=None) as writer:
        for frame in panels:
            writer.append_data(frame)
    report = {"schema": "robot-learning-lab.ape-v35-side-camera-replay.v1",
              "task": "T-5ytcbv",
              "scope": "Camera RGB-D and generic-person detector replay at saved V35 owner inference poses/times; image motion cue is an actor computation from two frames. Simulator person pixels score observability only. Fixed 46 cm camera positions and ideal rendering; no selected body, shell occlusion, optical defects, gait blur or live control branch.",
              "source_sha256": sha(Path(__file__)),
              "v35_report_sha256": sha(SOURCE),
              "scene_xml_sha256": sha(moving.SCENE_XML),
              "current_design_pointer_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "person_detector_weights_sha256": sha(
                  detector.reacquire.DETECTOR / "model.safetensors"),
              "parameters": {"rgb_change": RGB_CHANGE,
                             "depth_change_m": DEPTH_CHANGE_M,
                             "motion_pixels": MOTION_PIXELS,
                             "camera_fovy_degrees": {
                                 cam: float(model.cam_fovy[model.camera(cam).id])
                                 for cam in (OWNER_CAMERA, FLANK_CAMERA)}},
              "rows": all_rows, "outcomes": outcomes,
              "same_kinematic_response_law_counterfactual": modeled_response,
              "video": {"path": str(video), "sha256": sha(video),
                        "frames": len(panels)}}
    (OUT / "report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
