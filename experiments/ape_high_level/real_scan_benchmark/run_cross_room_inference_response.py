"""Run timed near-body sensing and a bounded proxy response during image search.

Four real-capture rooms use different hazard bearings. The image worker is the
real generic proposal search; the separate monitor renders range and RGB-D as
the textured prop moves. A sensor alarm issues a kinematic escape command
from the measured ray bearing. This is a provisional action interface, not a
qualified motor policy or physical safety result.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import setup, ANGLES, BODY_RADIUS_M
import run_real_capture_conditional_search as conditional


PRECOMMIT = ART / "cross_room_inference_response_precommit.json"
OUT = ART / "cross_room_inference_response"
REPORT = ART / "cross_room_inference_response_report.json"
TICK_S = .1
SPEED_M_S = .25
RESPONSE_DELAY_S = .1
MAX_DISPLACEMENT_M = .6
MONITOR_HZ = 5


def incoming_xy(start: np.ndarray, incoming: np.ndarray, t: float) -> np.ndarray:
    progress = np.clip((t - .2) / 1.6, 0., 1.)
    center_distance = 1.0 - .95 * float(progress)
    return start + incoming * center_distance


def actor_xy(start: np.ndarray, escape: np.ndarray | None,
             command_time: float | None, t: float) -> np.ndarray:
    if escape is None or command_time is None:
        return start.copy()
    distance = min(MAX_DISPLACEMENT_M,
                   max(0., t - command_time - RESPONSE_DELAY_S) * SPEED_M_S)
    return start + distance * escape


def proxy_gap(actor: np.ndarray, obstacle: np.ndarray, radius: float) -> float:
    return float(np.linalg.norm(actor - obstacle) - BODY_RADIUS_M - radius)


def static_gap(scene, xy: np.ndarray) -> float:
    rays = []
    for angle in ANGLES:
        yaw = math.radians(float(angle))
        rays.append([xy[0], scene.floor_y + .1, xy[1],
                     math.sin(yaw), 0., math.cos(yaw)])
    from open3d import core as o3c
    result = scene.static.cast_rays(o3c.Tensor(np.asarray(rays, np.float32)))[
        "t_hit"].numpy()
    return float(np.min(result) - BODY_RADIUS_M)


def decorate(rgb: np.ndarray, *, t: float, label: str, actor: np.ndarray,
             start: np.ndarray, hazard: np.ndarray | None, gap: float | None,
             alarm: bool) -> np.ndarray:
    display = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_NEAREST)
    cv2.putText(display, f"{label} | t={t:.2f}s | " +
                ("RESPONSE" if alarm else "watch"), (8, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 2)
    if gap is not None:
        cv2.putText(display, f"evaluator gap={gap:.2f}m", (8, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 2)
    cx, cy, scale = 540, 105, 85
    cv2.rectangle(display, (453, 18), (630, 194), (25, 25, 25), -1)
    cv2.putText(display, "EVALUATOR MAP", (459, 36),
                cv2.FONT_HERSHEY_SIMPLEX, .32, (255, 255, 255), 1)
    cv2.circle(display, (cx, cy), 3, (130, 130, 130), -1)
    ax = int(cx + (actor[0] - start[0]) * scale)
    ay = int(cy - (actor[1] - start[1]) * scale)
    cv2.circle(display, (ax, ay), int(BODY_RADIUS_M * scale),
               (90, 220, 90), 2)
    if hazard is not None:
        hx = int(cx + (hazard[0] - start[0]) * scale)
        hy = int(cy - (hazard[1] - start[1]) * scale)
        cv2.circle(display, (hx, hy), 6, (40, 45, 245), -1)
    return display


def run_one(case: dict, arm: str, parked: bool) -> dict:
    state = setup(case)
    folder = OUT / case["id"] / ("parked" if parked else arm)
    folder.mkdir(parents=True, exist_ok=True)
    monitor = state["monitor_scene"]
    start = state["actual_xy"]
    yaw = state["actual_yaw"]
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    eye = np.array([start[0], monitor.floor_y + .42, start[1]])
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    baseline_ranges = monitor.range_fan(eye, yaw, owner_xz=None,
                                         angles_deg=ANGLES)
    baseline_rgb, baseline_depth = monitor.observe(eye, forward,
        owner_xz=None, width=160, height=90)
    Image.fromarray(baseline_rgb).save(folder / "baseline_rgb.png")
    Image.fromarray(np.where(np.isfinite(baseline_depth),
        np.clip(baseline_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            folder / "baseline_depth.png")
    worker_scene = state["worker_scene"]
    stop = state["stop"]
    def worker():
        return conditional.original_run_search(
            worker_scene, state["vision"], start, yaw,
            np.asarray(stop["actor_pose_xy_m"], float),
            float(stop["actor_heading_deg"]), state["exemplar"],
            float(state["summary"]["teaching"]["acceptance_similarity"]),
            True, folder / "search",
            start_time_s=float(stop["decision_time_s"] + .4),
            search_xz=tuple(state["placement"]["find_target_xz"]),
            search_distractor_xz=tuple(
                state["placement"]["find_distractor_xz"]),
            owner_start_xz=tuple(state["cfg"]["start_xz"]),
            target_name=case["target"], distractor_name=case["distractor"],
            identity_mode="multi_cue", colour_exemplar=state["colour"],
            colour_acceptance=float(state["summary"]["teaching"][
                "colour_acceptance_similarity"]))
    packets = []
    video_frames = []
    first_alarm_capture = None
    first_alarm_delivery = None
    first_alarm_angle = None
    escape = None
    max_lateness = 0.
    monitor_video = folder / "monitor.mp4"
    with ThreadPoolExecutor(max_workers=1) as pool, imageio.get_writer(
            str(monitor_video), fps=MONITOR_HZ, codec="libx264", quality=7,
            macro_block_size=16) as writer:
        origin = time.perf_counter()
        future = pool.submit(worker)
        tick = 0
        while not future.done():
            target = origin + tick * TICK_S
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            capture_s = time.perf_counter() - origin
            if capture_s > 25.:
                raise TimeoutError("Image worker exceeded bounded monitor run")
            max_lateness = max(max_lateness, capture_s - tick * TICK_S)
            actor = actor_xy(start, escape, first_alarm_delivery, capture_s)
            hazard = None if parked else incoming_xy(start, incoming, capture_s)
            if hazard is None:
                monitor.set_props(state["base_props"])
            else:
                monitor.set_props({**state["base_props"],
                    case["hazard_object"]: (state["hazard_prop"],
                                            tuple(hazard), 0.)})
            at_eye = np.array([actor[0], monitor.floor_y + .42, actor[1]])
            rays = monitor.range_fan(at_eye, yaw, owner_xz=None,
                                     angles_deg=ANGLES)
            delivery_s = time.perf_counter() - origin
            with np.errstate(invalid="ignore"):
                early = (baseline_ranges[0] - rays[0] > .08) & (rays[0] < .9)
                late = (baseline_ranges[0] - rays[0] > .12) & (rays[0] < .6)
            active = (early if arm != "range360_late" else late).copy()
            if arm == "front_range_early":
                active &= np.abs(ANGLES) <= 70
            if arm in ("head_depth", "capture_only"):
                active[:] = False
            if first_alarm_delivery is None and np.any(active):
                indices = np.flatnonzero(active)
                selected = indices[np.argmin(rays[0, indices])]
                first_alarm_capture = capture_s
                first_alarm_delivery = delivery_s
                first_alarm_angle = float(ANGLES[selected])
                opposite = math.radians(yaw + first_alarm_angle + 180.)
                escape = np.array([math.sin(opposite), math.cos(opposite)])
            depth_alarm = False
            depth_bearing = None
            if tick % 2 == 0:
                rgb, depth = monitor.observe(at_eye, forward,
                    owner_xz=None, width=160, height=90)
                if escape is None:
                    valid = np.isfinite(depth) & np.isfinite(baseline_depth)
                    changed = valid & (baseline_depth - depth > .08) & (depth < .9)
                    depth_alarm = bool(changed.sum() >= 20)
                    if depth_alarm:
                        xs = np.where(changed)[1]
                        focal = 90 / (2 * math.tan(math.radians(75.) / 2))
                        depth_bearing = math.degrees(math.atan(
                            (float(xs.mean()) - 80.) / focal))
                gap = None if hazard is None else proxy_gap(
                    actor, hazard, state["hazard_radius_m"])
                display = decorate(rgb, t=capture_s, label=arm,
                    actor=actor, start=start, hazard=hazard, gap=gap,
                    alarm=first_alarm_delivery is not None)
                writer.append_data(display)
                index = len(video_frames)
                rgb_path = folder / f"monitor_rgb_{index:03d}.png"
                depth_path = folder / f"monitor_depth_{index:03d}.png"
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(np.where(np.isfinite(depth),
                    np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                        depth_path)
                video_frames.append({"packet_index": tick,
                    "capture_wall_s": capture_s, "rgb": relative(rgb_path),
                    "depth": relative(depth_path)})
            if first_alarm_delivery is None and arm == "head_depth" and depth_alarm:
                first_alarm_capture = capture_s
                first_alarm_delivery = time.perf_counter() - origin
                first_alarm_angle = depth_bearing
                opposite = math.radians(yaw + depth_bearing + 180.)
                escape = np.array([math.sin(opposite), math.cos(opposite)])
            gap = None if hazard is None else proxy_gap(
                actor, hazard, state["hazard_radius_m"])
            packets.append({"tick": tick,
                "capture_wall_s": round(capture_s, 5),
                "range_delivery_wall_s": round(delivery_s, 5),
                "actor_xz_m": actor.tolist(),
                "hazard_xz_m": None if hazard is None else hazard.tolist(),
                "evaluator_proxy_gap_m": gap,
                "low_range_m": [float(x) if np.isfinite(x) else None
                                for x in rays[0]],
                "mid_range_m": [float(x) if np.isfinite(x) else None
                                for x in rays[1]],
                "range360_early_alarm": bool(np.any(early)),
                "range360_late_alarm": bool(np.any(late)),
                "front_range_early_alarm": bool(np.any(
                    early & (np.abs(ANGLES) <= 70))),
                "head_depth_alarm": depth_alarm})
            tick += 1
        image_result = future.result()
        worker_wall_s = time.perf_counter() - origin
    # Score the entire response path at 20 ms, including between image frames.
    scores = []
    for t in np.arange(0., worker_wall_s + .0001, .02):
        actor = actor_xy(start, escape, first_alarm_delivery, float(t))
        hazard = None if parked else incoming_xy(start, incoming, float(t))
        hazard_gap = None if hazard is None else proxy_gap(
            actor, hazard, state["hazard_radius_m"])
        prop_gap = min(proxy_gap(actor, np.asarray(
            state["placement"][f"find_{name}_xz"]),
            state["base_props"][key][0].footprint_radius_m)
            for name, key in (("target", case["target"]),
                              ("distractor", case["distractor"])))
        scores.append({"t_s": round(float(t), 3),
            "actor_xz_m": actor.tolist(),
            "hazard_gap_m": hazard_gap,
            "prop_gap_m": prop_gap,
            "static_gap_m": static_gap(monitor, actor),
            "floor_supported": bool(monitor.floor_supported(*actor))})
    min_gap = min(x["hazard_gap_m"] for x in scores
                  if x["hazard_gap_m"] is not None) if not parked else None
    min_prop_gap = min(x["prop_gap_m"] for x in scores)
    min_static_gap = min(x["static_gap_m"] for x in scores)
    floor_all = all(x["floor_supported"] for x in scores)
    monitor_video_path = relative(monitor_video)
    search_trace = folder / "search/trace.json"
    trace = {"schema": "ape.cross-room-inference-response.v1",
        "case": case["id"], "arm": arm, "parked": parked,
        "scene_key": case["scene_key"], "teaching_summary": case["teaching_summary"],
        "search_trace": relative(search_trace),
        "search_video": image_result["video"],
        "monitor_video": monitor_video_path,
        "worker_wall_s": worker_wall_s,
        "alarm_capture_s": first_alarm_capture,
        "alarm_delivery_s": first_alarm_delivery,
        "alarm_relative_angle_deg": first_alarm_angle,
        "response_delay_s": RESPONSE_DELAY_S,
        "response_speed_m_s": SPEED_M_S,
        "response_max_displacement_m": MAX_DISPLACEMENT_M,
        "response_direction_xz": None if escape is None else escape.tolist(),
        "baseline_low_range_m": [float(x) if np.isfinite(x) else None
                                 for x in baseline_ranges[0]],
        "range_angles_deg": ANGLES.tolist(),
        "monitor_packets": packets,
        "monitor_video_frames": video_frames,
        "monitor_max_wall_lateness_s": max_lateness,
        "evaluator_20ms": scores,
        "min_hazard_gap_m": min_gap,
        "min_prop_gap_m": min_prop_gap,
        "min_static_gap_m": min_static_gap,
        "floor_supported_throughout": floor_all,
        "image_candidate_before_monitor": image_result["action"] ==
            "candidate_to_owner_for_confirmation",
        "image_valid_after_monitor": bool(
            image_result["action"] == "candidate_to_owner_for_confirmation" and
            (first_alarm_delivery is None or
             first_alarm_delivery >= worker_wall_s)),
        "owner_action_after_monitor": ("ask_for_new_safe_view" if
            first_alarm_delivery is not None else
            image_result["action"]),
        "limits": "Image worker sees the original static scene and pose; monitor uses the same mesh with one scripted moving prop. Alarm drives a 0.25m/s kinematic proxy command after 0.1s, not selected-body motor control. Evaluator geometry scores gap, floor and static contact; it does not select the response."}
    trace_path = folder / "episode_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "arm": arm, "parked": parked,
        "trace": relative(trace_path), "search_video": image_result["video"],
        "monitor_video": monitor_video_path,
        "worker_wall_s": worker_wall_s,
        "alarm_delivery_s": first_alarm_delivery,
        "min_hazard_gap_m": min_gap,
        "min_prop_gap_m": min_prop_gap,
        "min_static_gap_m": min_static_gap,
        "floor_supported_throughout": floor_all,
        "image_valid_after_monitor": trace["image_valid_after_monitor"]}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    for case in frozen["cases"]:
        if sha(ROOT / case["teaching_summary"]) != case["summary_sha256"]:
            raise ValueError("Teaching summary changed")
    rows = []
    for case in frozen["cases"]:
        for arm in frozen["arms"]:
            print(case["id"], arm, flush=True)
            rows.append(run_one(case, arm, False))
            REPORT.write_text(json.dumps({"schema":
                "ape.cross-room-inference-response-report.v1",
                "precommit": relative(PRECOMMIT), "rows": rows,
                "limits": frozen["limits"]}, indent=2) + "\n")
        print(case["id"], "parked", flush=True)
        rows.append(run_one(case, "range360_early", True))
        REPORT.write_text(json.dumps({"schema":
            "ape.cross-room-inference-response-report.v1",
            "precommit": relative(PRECOMMIT), "rows": rows,
            "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps({"runs": len(rows),
        "alarms": sum(r["alarm_delivery_s"] is not None for r in rows),
        "gap_under_5cm": sum(r["min_hazard_gap_m"] is not None and
                             r["min_hazard_gap_m"] < .05 for r in rows)},
        indent=2))


if __name__ == "__main__":
    main()
