"""Run a real image worker while independent range/depth watch a rear entrant.

This is a timed sensor-role challenge from frozen real-capture owner stops. The
image worker searches a static forward view. A separate, same-mesh scene moves
a textured floor prop behind the stationary proxy while the worker runs.
Candidate validity and modeled contact are scored separately: an alarm can
invalidate an image without supplying any physical escape capability.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_real_capture_owner_loop import SCENES
import run_real_capture_conditional_search as conditional
from run_real_capture_teach_find import colour_histogram


ROOT = Path(__file__).resolve().parents[3]
ART = ROOT / "artifacts/ape-real-room-next-captures-20261014"
PRECOMMIT = ART / "conditional_search_rear_hazard_precommit.json"
OUT = ART / "conditional_search_rear_hazard"
ANGLES = np.arange(-180., 180., 5., dtype=np.float32)
TICK_S = .1
BODY_RADIUS_M = .17
START_DISTANCE_M = 1.0
END_DISTANCE_M = .05
START_MOVE_S = .2
MOVE_DURATION_S = 1.6


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def trajectory(t: float) -> float:
    progress = np.clip((t - START_MOVE_S) / MOVE_DURATION_S, 0., 1.)
    return START_DISTANCE_M + (END_DISTANCE_M - START_DISTANCE_M) * float(progress)


def setup(case: dict):
    key = case["scene_key"]
    summary = json.loads((ROOT / case["teaching_summary"]).read_text())
    source = json.loads((ROOT / summary["source_capture"]).read_text())
    screen = json.loads((ROOT / summary["scenario_placement_screen"]).read_text())
    placement = next(x for x in screen["scenes"] if x["scene_key"] == key)
    stop = json.loads((ROOT / summary["teaching"]["owner_approach_trace"]).read_text())["trace"][-1]
    kwargs = {"floor_y": source["floor_y_1pct_m"],
              "scale": source.get("scale_to_metres", 1.),
              "source_z_up": source.get("source_z_up", False)}
    scene_path = ROOT / source["source_mesh"]
    worker_scene = TexturedCaptureScene(scene_path, **kwargs)
    monitor_scene = TexturedCaptureScene(scene_path, **kwargs)
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    target = case["target"]
    distractor = case["distractor"]
    target_prop = TexturedFloorObject(props / f"{target}.obj", scale=.012)
    distractor_prop = TexturedFloorObject(props / f"{distractor}.obj", scale=.012)
    hazard_prop = TexturedFloorObject(props / f"{case['hazard_object']}.obj", scale=.012)
    base_props = {target: (target_prop, tuple(placement["find_target_xz"]), 0.),
                  distractor: (distractor_prop,
                               tuple(placement["find_distractor_xz"]), 0.)}
    monitor_scene.set_props(base_props)
    teaching = summary["teaching"]
    rgb = Image.open(ROOT / teaching["source_rgb"]).convert("RGB")
    vision = conditional.ConditionalVision("generic-proposal")
    vision._teaching_frame_seen = True
    vision.search_mode = case["search_mode"]
    crop = teaching["selected_box_xyxy"]
    exemplar = vision.feature([rgb.crop(tuple(crop))])[0]
    colour = colour_histogram(np.asarray(rgb), crop)
    actual_xy = np.asarray(stop["evaluator_actual_xz_m"], float)
    actual_yaw = float(stop["evaluator_actual_heading_deg"])
    forward = np.array([math.sin(math.radians(actual_yaw)),
                        math.cos(math.radians(actual_yaw))])
    for distance in (1., .8, .6, .4, .2):
        if not monitor_scene.floor_supported(*(actual_xy - distance * forward)):
            raise ValueError(f"Rear hazard lacks floor support: {key} {distance}")
    return {"summary": summary, "source": source, "placement": placement,
            "stop": stop, "worker_scene": worker_scene,
            "monitor_scene": monitor_scene, "base_props": base_props,
            "hazard_prop": hazard_prop, "hazard_radius_m": hazard_prop.footprint_radius_m,
            "vision": vision, "exemplar": exemplar, "colour": colour,
            "actual_xy": actual_xy, "actual_yaw": actual_yaw,
            "forward": forward, "cfg": SCENES[key]}


def run_case(case: dict, branch: str) -> dict:
    if branch not in ("parked", "rear_entrant"):
        raise ValueError(branch)
    state = setup(case)
    folder = OUT / case["id"] / branch
    folder.mkdir(parents=True, exist_ok=True)
    monitor = state["monitor_scene"]
    xy = state["actual_xy"]
    yaw = state["actual_yaw"]
    eye = np.array([xy[0], monitor.floor_y + .42, xy[1]])
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    baseline_range = monitor.range_fan(eye, yaw, owner_xz=None,
                                       angles_deg=ANGLES)
    baseline_rgb, baseline_depth = monitor.observe(eye, forward,
        owner_xz=None, width=160, height=90)
    Image.fromarray(baseline_rgb).save(folder / "baseline_rgb.png")
    Image.fromarray(np.where(np.isfinite(baseline_depth),
        np.clip(baseline_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            folder / "baseline_depth.png")
    capture_wall = []
    original_camera = conditional.baseline.camera
    wall_origin = None

    def timed_camera(*args, **kwargs):
        result = original_camera(*args, **kwargs)
        capture_wall.append(time.perf_counter() - wall_origin)
        return result

    def image_worker():
        conditional.baseline.camera = timed_camera
        try:
            stop = state["stop"]
            return conditional.original_run_search(
                state["worker_scene"], state["vision"], xy, yaw,
                np.asarray(stop["actor_pose_xy_m"], float),
                float(stop["actor_heading_deg"]),
                state["exemplar"],
                float(state["summary"]["teaching"]["acceptance_similarity"]),
                True, folder / "search",
                start_time_s=float(stop["decision_time_s"] + .4),
                search_xz=tuple(state["placement"]["find_target_xz"]),
                search_distractor_xz=tuple(
                    state["placement"]["find_distractor_xz"]),
                owner_start_xz=tuple(state["cfg"]["start_xz"]),
                target_name=case["target"],
                distractor_name=case["distractor"],
                identity_mode="multi_cue", colour_exemplar=state["colour"],
                colour_acceptance=float(state["summary"]["teaching"][
                    "colour_acceptance_similarity"]))
        finally:
            conditional.baseline.camera = original_camera

    packets = []
    video_frames = []
    monitor_video = folder / "monitor_forward_rgbd.mp4"
    first_alarm = {"all_around_range": None, "front_range": None,
                   "head_depth": None}
    first_gap_breach = None
    max_lateness = 0.
    cancel_requested_s = None
    worker_cancel_succeeded = None
    with ThreadPoolExecutor(max_workers=1) as pool, imageio.get_writer(
            str(monitor_video), fps=5, codec="libx264", quality=7,
            macro_block_size=16) as writer:
        wall_origin = time.perf_counter()
        future = pool.submit(image_worker)
        tick = 0
        while not future.done():
            scheduled = wall_origin + tick * TICK_S
            remaining = scheduled - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            elapsed = time.perf_counter() - wall_origin
            max_lateness = max(max_lateness, elapsed - tick * TICK_S)
            if elapsed > 25.:
                raise TimeoutError("Image worker exceeded bounded monitor episode")
            distance = trajectory(elapsed) if branch == "rear_entrant" else None
            if distance is None:
                monitor.set_props(state["base_props"])
            else:
                hazard_xy = xy - distance * state["forward"]
                monitor.set_props({**state["base_props"],
                    case["hazard_object"]: (state["hazard_prop"],
                                            tuple(hazard_xy), 0.)})
            ranges = monitor.range_fan(eye, yaw, owner_xz=None,
                                       angles_deg=ANGLES)
            delta = baseline_range[0] - ranges[0]
            changed = (delta > .12) & (ranges[0] < .60)
            front = (np.abs(ANGLES) <= 70)
            range_all_alarm = bool(np.any(changed))
            range_front_alarm = bool(np.any(changed & front))
            if range_all_alarm and cancel_requested_s is None:
                cancel_requested_s = round(elapsed, 4)
                worker_cancel_succeeded = future.cancel()
            head_depth_alarm = False
            depth_change_px = None
            if tick % 2 == 0:
                rgb, depth = monitor.observe(eye, forward, owner_xz=None,
                                             width=160, height=90)
                valid = np.isfinite(depth) & np.isfinite(baseline_depth)
                new_near = valid & ((baseline_depth - depth) > .12) & (depth < .60)
                depth_change_px = int(new_near.sum())
                head_depth_alarm = depth_change_px >= 20
                display = cv2.resize(rgb, (640, 360),
                                     interpolation=cv2.INTER_NEAREST)
                label = (f"t={elapsed:.2f}s | rear range: "
                         f"{'ALARM' if range_all_alarm else 'watch'}")
                cv2.putText(display, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (255, 255, 255), 2)
                writer.append_data(display)
                rgb_file = folder / f"monitor_rgb_{len(video_frames):03d}.png"
                depth_file = folder / f"monitor_depth_{len(video_frames):03d}.png"
                Image.fromarray(rgb).save(rgb_file)
                Image.fromarray(np.where(np.isfinite(depth),
                    np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(depth_file)
                video_frames.append({"packet_index": tick,
                    "rgb": relative(rgb_file), "depth": relative(depth_file)})
            for name, alarm in (("all_around_range", range_all_alarm),
                                ("front_range", range_front_alarm),
                                ("head_depth", head_depth_alarm)):
                if alarm and first_alarm[name] is None:
                    first_alarm[name] = round(elapsed, 4)
            gap = (None if distance is None else
                   distance - BODY_RADIUS_M - state["hazard_radius_m"])
            if gap is not None and gap < .05 and first_gap_breach is None:
                first_gap_breach = round(elapsed, 4)
            packets.append({"tick": tick, "wall_elapsed_s": round(elapsed, 5),
                "hazard_center_distance_m": distance,
                "evaluator_proxy_gap_m": gap,
                "low_range_m": [float(x) if np.isfinite(x) else None
                                for x in ranges[0]],
                "mid_range_m": [float(x) if np.isfinite(x) else None
                                for x in ranges[1]],
                "all_around_range_alarm": range_all_alarm,
                "front_range_alarm": range_front_alarm,
                "head_depth_changed_near_pixels": depth_change_px,
                "head_depth_alarm": head_depth_alarm})
            tick += 1
        result = future.result()
        worker_wall_s = time.perf_counter() - wall_origin
    image_candidate = result["action"] == "candidate_to_owner_for_confirmation"
    saved_search = json.loads((folder / "search/trace.json").read_text())
    model_wall_s = sum(x["inference_wall_s"] for x in
                       saved_search["observations"]) + float(
        (saved_search["verification"] or {}).get("inference_wall_s", 0.))
    sensor_decisions = {name: {
        "alarm_before_worker_release": alarm is not None and alarm < worker_wall_s,
        "candidate_valid_after_monitor": bool(image_candidate and
            (alarm is None or alarm >= worker_wall_s)),
        "alarm_wall_elapsed_s": alarm}
        for name, alarm in first_alarm.items()}
    sensor_decisions["capture_only"] = {
        "alarm_before_worker_release": False,
        "candidate_valid_after_monitor": image_candidate,
        "alarm_wall_elapsed_s": None}
    search_trace = folder / "search/trace.json"
    trace = {"schema": "ape.real-capture-rear-hazard-search.v1",
             "scene_key": case["scene_key"], "branch": branch,
             "search_mode": case["search_mode"],
             "teaching_summary": case["teaching_summary"],
             "source_mesh": state["source"]["source_mesh"],
             "search_trace": relative(search_trace),
             "search_video": result["video"],
             "monitor_video": relative(monitor_video),
             "worker_wall_s": worker_wall_s,
             "worker_model_inference_wall_s": model_wall_s,
             "cancel_requested_wall_elapsed_s": cancel_requested_s,
             "worker_cancel_succeeded": worker_cancel_succeeded,
             "range_angles_deg": ANGLES.tolist(),
             "baseline_low_range_m": [float(x) if np.isfinite(x) else None
                                      for x in baseline_range[0]],
             "baseline_depth": relative(folder / "baseline_depth.png"),
             "camera_capture_wall_elapsed_s": capture_wall,
             "monitor_packets": packets,
             "monitor_video_frames": video_frames,
             "monitor_packet_count": len(packets),
             "monitor_max_wall_lateness_s": max_lateness,
             "sensor_decisions": sensor_decisions,
             "first_proxy_gap_below_5cm_s": first_gap_breach,
             "candidate_before_monitor": image_candidate,
             "evaluator_selected_target_iou": (result["verification"] or {}).get(
                 "evaluator_selected_target_iou", 0.),
             "limits": "A separately scripted textured floor prop approaches from behind. Range/depth packets are sampled concurrently with a real image worker on a stationary 17cm proxy. The monitor requests cancellation but no physical stop/escape is executed; 5cm gap is evaluator-only proxy geometry."}
    trace_path = folder / "episode_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "branch": branch,
            "worker_wall_s": worker_wall_s,
            "candidate_before_monitor": image_candidate,
            "sensor_decisions": sensor_decisions,
            "first_proxy_gap_below_5cm_s": first_gap_breach,
            "monitor_packet_count": len(packets),
            "monitor_max_wall_lateness_s": max_lateness,
            "search_video": result["video"],
            "monitor_video": relative(monitor_video),
            "trace": relative(trace_path)}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for entry in frozen["source_manifest"]:
        for key, digest in entry["sha256"].items():
            if sha(ROOT / entry[key]) != digest:
                raise ValueError(f"Frozen source changed: {entry['id']} {key}")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in frozen["cases"]:
        for branch in ("parked", "rear_entrant"):
            print(case["id"], branch, flush=True)
            rows.append(run_case(case, branch))
            (ART / "conditional_search_rear_hazard_report.json").write_text(
                json.dumps({"schema": "ape.rear-hazard-timed-search-report.v1",
                    "precommit": relative(PRECOMMIT), "rows": rows,
                    "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
