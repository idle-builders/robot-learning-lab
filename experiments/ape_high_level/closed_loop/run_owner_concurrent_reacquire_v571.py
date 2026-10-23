"""Run person inference on a worker while the main thread monitors range.

This is one real wall-clock software concurrency screen at the held pre-show
pose. Scene motion and the retreat remain scripted; no native joint skill is
exercised. An image captured before the range alarm is discarded afterward.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
import torch

import robocasa_owner_person as person
import robocasa_person_interaction as interaction
import robocasa_owner_reacquire_rgbd_v560 as reacquire
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_person_joined_v497 as joined
from analyze_robocasa_owner_monitor_v501 import changed_beams
from run_robocasa_dynamic_suite_v439 import motion
from run_robocasa_scan_height_v433 import height_scan
from screen_robocasa_owner_safe_stop_v494 import aim_camera


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "artifacts/ape-robocasa-first-return-person-reacquire-v566/report.json"
OUT = ROOT / "artifacts/ape-robocasa-concurrent-owner-reacquire-v571"
SCAN = height_scan((.30,))
START_TIME_S = 4.8
EXTRA_OWNER_APPROACH_M = .8
APPROACH_DURATION_S = 2.
SCAN_PERIOD_S = .1
RESPONSE_DELAY_S = .2
RETREAT_SPEED_M_S = .25
RETREAT_DISTANCE_M = .5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detect_person(rgb):
    start = time.perf_counter()
    processor, detector, device = reacquire._model()
    tensors = processor(images=Image.fromarray(rgb), text=["person ."],
                        return_tensors="pt").to(device)
    with torch.no_grad():
        output = detector(**tensors)
    found = processor.post_process_grounded_object_detection(
        output, tensors.input_ids, threshold=.25, text_threshold=.25,
        target_sizes=[rgb.shape[:2]])[0]
    detections = sorted(({"label": str(label), "score": float(score),
                          "box_xyxy": [float(value) for value in box]}
                         for label, score, box in zip(found["text_labels"],
                                                      found["scores"], found["boxes"])),
                        key=lambda row: row["score"], reverse=True)
    return {"detections": detections,
            "worker_wall_s": time.perf_counter() - start}


def main():
    source = json.loads(SOURCE.read_text())
    ep = next(item for item in source["episodes"]
              if item["response"] == "early_owner_mid_sector_blackout")
    held = next(row for row in reversed(ep["trace"])
                if row["stage"] == "pre_show_backtrack")
    held_xy = np.asarray(held["ape_xy_m"])
    restage = next(row for row in ep["trace"]
                   if row["stage"] == "choose_pre_show_restage")["chosen"]
    assert restage["distance_m"] >= RETREAT_DISTANCE_M
    assert restage["known_free_path"]
    model = joined.scene_with_owner_and_prop(ep["scenario"]["from_xy_m"])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)
    prop = motion(ep["scenario"])
    ape_geom = model.geom("audit_ape_geom").id
    person_geoms = [model.geom(name).id for name in person.PERSON_GEOMS]
    actual = held_xy.copy()
    model.body_pos[model.body("audit_ape").id, :2] = actual
    model.body_pos[model.body("route_obstacle").id, :2] = prop(START_TIME_S)
    person.set_owner_pose(model, interaction.TORSO_XY + [0., -.4],
                          interaction.REST_HAND + [0., -.4, 0.],
                          show_can=False)
    aim_camera(model, interaction.CAMERA, actual, actual + [0., 1.], 20)
    mujoco.mj_forward(model, data)
    before_rgb, _, capture_wall = owner.capture(
        renderer, data, interaction.CAMERA, False)
    OUT.mkdir(parents=True, exist_ok=True)
    before_path = OUT / "before-worker.png"
    Image.fromarray(before_rgb).save(before_path)

    baseline = None
    streak = 0
    alarm_elapsed = None
    first_below_5cm = None
    minimum = {"gap_m": 1., "time_s": None}
    rows, scan_wall_times = [], []
    with ThreadPoolExecutor(max_workers=1) as pool:
        begin = time.perf_counter()
        future = pool.submit(detect_person, before_rgb)
        next_tick = begin
        next_scan_elapsed = 0.
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
                now = time.perf_counter()
            elapsed = now - begin
            t = START_TIME_S + elapsed
            extra = EXTRA_OWNER_APPROACH_M * min(1., elapsed / APPROACH_DURATION_S)
            amount = (min(RETREAT_DISTANCE_M,
                          RETREAT_SPEED_M_S * max(0., elapsed - alarm_elapsed -
                                                 RESPONSE_DELAY_S))
                      if alarm_elapsed is not None else 0.)
            actual[:] = held_xy + [0., -amount]
            model.body_pos[model.body("audit_ape").id, :2] = actual
            model.body_pos[model.body("route_obstacle").id, :2] = prop(t)
            person.set_owner_pose(model,
                                  interaction.TORSO_XY + [0., -.4 - extra],
                                  interaction.REST_HAND + [0., -.4 - extra, 0.],
                                  show_can=False)
            mujoco.mj_forward(model, data)
            gap = min(float(mujoco.mj_geomDistance(
                model, data, ape_geom, geom, 1., None)) for geom in person_geoms)
            if gap < minimum["gap_m"]:
                minimum = {"gap_m": gap, "time_s": round(t, 4)}
            if gap < .05 and first_below_5cm is None:
                first_below_5cm = round(t, 4)
            row = {"time_s": round(t, 4), "elapsed_wall_s": elapsed,
                   "owner_extra_south_m": extra, "ape_xy_m": actual.tolist(),
                   "ape_person_gap_evaluator_m": gap}
            if elapsed >= next_scan_elapsed:
                packet = SCAN(model, data, actual, 5.,
                              np.random.default_rng(14000 + len(scan_wall_times)))
                if baseline is None:
                    baseline = packet
                changed = changed_beams(baseline, packet)
                streak = streak + 1 if changed >= 3 else 0
                if streak >= 2 and alarm_elapsed is None:
                    alarm_elapsed = elapsed
                    row["actor_action"] = "cancel_stale_person_image_and_retract"
                row["range_changed_beams"] = changed
                row["range_motion_streak"] = streak
                scan_wall_times.append(elapsed)
                next_scan_elapsed += SCAN_PERIOD_S
                while next_scan_elapsed <= elapsed:
                    next_scan_elapsed += SCAN_PERIOD_S
            rows.append(row)
            if future.done() and elapsed >= 3.:
                break
            if elapsed > 30.:
                break
            next_tick += .02
            if next_tick < now:
                next_tick = now + .02
        finished_wall = time.perf_counter() - begin
        result = future.result()
    aim_camera(model, interaction.CAMERA, actual, actual + [0., 1.], 20)
    mujoco.mj_forward(model, data)
    after_rgb, _, _ = owner.capture(renderer, data, interaction.CAMERA, False)
    after_path = OUT / "after-monitor.png"
    Image.fromarray(after_rgb).save(after_path)
    renderer.close()
    intervals = np.diff(scan_wall_times)
    report = {"source_sha256": sha(Path(__file__)),
              "source_episode_sha256": sha(SOURCE),
              "person_detector_weights_sha256": sha(reacquire.DETECTOR / "model.safetensors"),
              "scope": "actual wall-clock ThreadPoolExecutor Grounding DINO person inference on one captured RGB frame while main thread simulates owner motion, 10Hz ideal mid-range packet monitor, .2s delayed .25m/s surrogate retreat, and 20ms intended geometry loop; not a joined full episode or native skill",
              "source_actor_known_free_retreat_path": restage,
              "initial_capture_wall_s": capture_wall,
              "detector_worker": result,
              "detector_finished_wall_s": finished_wall,
              "detection_discarded_after_motion": alarm_elapsed is not None,
              "range_alarm_time_s": (START_TIME_S + alarm_elapsed
                                     if alarm_elapsed is not None else None),
              "range_scans": len(scan_wall_times),
              "range_scan_wall_interval_max_s": float(np.max(intervals)),
              "range_scan_wall_interval_p95_s": float(np.quantile(intervals, .95)),
              "retreat_distance_m": float(np.linalg.norm(actual - held_xy)),
              "minimum_ape_person_gap": minimum,
              "first_person_gap_below_5cm_s": first_below_5cm,
              "before_image": {"path": str(before_path), "sha256": sha(before_path)},
              "after_image": {"path": str(after_path), "sha256": sha(after_path)},
              "trace": rows}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print({key: report[key] for key in
           ("detector_finished_wall_s", "range_alarm_time_s", "range_scans",
            "range_scan_wall_interval_max_s", "retreat_distance_m",
            "minimum_ape_person_gap", "first_person_gap_below_5cm_s",
            "detection_discarded_after_motion")}, flush=True)


if __name__ == "__main__":
    main()
