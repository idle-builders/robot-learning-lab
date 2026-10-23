"""Keep sensing live while rendering diagnostic monitor video after the run.

Each trial starts at an audited first-response endpoint. A second scripted prop
enters during the actual image worker. Healthy versus missing low-range packets
change only the actor's delivered sensor evidence. After the prop leaves, the
same fixed generic-first/wide-expanded item policy resumes with continuous
range monitoring during both later searches.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import imageio.v2 as imageio
import numpy as np
import cv2
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES, BODY_RADIUS_M
import run_cross_room_inference_response as motion
import run_item_inference_response_19 as cohort
import run_real_capture_conditional_search as conditional
import run_shared_world_return as first
import run_shared_world_wide_recovery as recovery


OUT = ROOT / "assets/real_indoor_scenes/monitored_return_deferred_20261015"
PRECOMMIT = ART / "monitored_return_deferred_precommit.json"
REPORT = ART / "monitored_return_deferred_report.json"
LANES = ART / "monitored_return_lane_screen.json"
TICK_S = .1
OUTAGE = (.25, 1.25)
ARMS = ("healthy_deferred_video", "range_outage_depth_deferred_video")
BACKUP_AIM_DELAY_S = .2
BACKUP_DEPTH_INTERVAL_S = .2
BACKUP_CHANGED_PIXELS = 20


class TimedStaticScene:
    def __init__(self, scene, origin: float):
        self.scene = scene
        self.origin = origin
        self.captures = []

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def set_props(self, placements):
        self.scene.set_props(placements)

    def observe(self, *args, **kwargs):
        self.captures.append({"channel": "rgbd",
            "capture_wall_s": time.perf_counter() - self.origin})
        return self.scene.observe(*args, **kwargs)

    def range_fan(self, *args, **kwargs):
        self.captures.append({"channel": "range",
            "capture_wall_s": time.perf_counter() - self.origin})
        return self.scene.range_fan(*args, **kwargs)


def owner_answer(result: dict) -> str | None:
    if result["action"] != "candidate_to_owner_for_confirmation":
        return None
    return "no" if result["wrong_confident_candidate"] else "yes"


def monitored_search(state: dict, case: dict, folder, start: np.ndarray,
                     actor: np.ndarray, *, entrant: bool, arm: str,
                     wide: bool = False) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    deferred_video = arm in ARMS or arm == "healthy"
    monitor = state["monitor_scene"]
    monitor.set_props(state["base_props"])
    yaw = state["actual_yaw"]
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    eye = np.array([start[0], monitor.floor_y + .42, start[1]])
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    baseline = monitor.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)
    packets, frames = [], []
    alarm_capture = alarm_delivery = alarm_angle = health_lost = None
    alarm_source = None
    head_track_bearing = None
    head_forward = None
    head_previous_depth = None
    head_samples = []
    head_next_capture_s = None
    escape = None
    origin = time.perf_counter()
    worker_scene = (first.TimedWorkerScene(state["worker_scene"], origin,
        case, state, start, incoming) if entrant else
        TimedStaticScene(state["worker_scene"], origin))
    original_offsets = conditional.baseline.SEARCH_YAW_OFFSETS_DEG
    original_mode = state["vision"].search_mode
    if wide:
        conditional.baseline.SEARCH_YAW_OFFSETS_DEG = recovery.OFFSETS
        state["vision"].search_mode = "expanded"
    else:
        state["vision"].search_mode = "generic"
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, imageio.get_writer(
                str(folder / "monitor.mp4"), fps=5, codec="libx264",
                quality=6, macro_block_size=16) as writer:
            future = pool.submit(first.search, state, case, worker_scene,
                start, actor, folder / "search",
                float(state["stop"]["decision_time_s"] + .4))
            tick = 0
            while not future.done():
                target = origin + tick * TICK_S
                remaining = target - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
                capture_s = time.perf_counter() - origin
                if capture_s > 30.:
                    raise TimeoutError("Return image worker exceeded bounded trial")
                actor_xy = motion.actor_xy(start, escape, alarm_delivery,
                                            capture_s)
                hazard = (motion.incoming_xy(start, incoming, capture_s)
                          if entrant else None)
                if hazard is None:
                    monitor.set_props(state["base_props"])
                else:
                    monitor.set_props({**state["base_props"],
                        case["hazard_object"]: (state["hazard_prop"],
                                                tuple(hazard), 0.)})
                at_eye = np.array([actor_xy[0], monitor.floor_y + .42,
                                   actor_xy[1]])
                missing = bool(entrant and arm in
                               ("range_outage", "range_outage_depth",
                                "range_outage_depth_deferred_video") and
                               OUTAGE[0] <= capture_s < OUTAGE[1])
                rays = (None if missing else monitor.range_fan(at_eye, yaw,
                    owner_xz=None, angles_deg=ANGLES))
                delivery_s = time.perf_counter() - origin
                if missing and health_lost is None:
                    health_lost = delivery_s
                    head_next_capture_s = capture_s + BACKUP_AIM_DELAY_S
                if rays is None:
                    changed = np.zeros(len(ANGLES), bool)
                else:
                    with np.errstate(invalid="ignore"):
                        changed = (baseline[0] - rays[0] > .08) & (rays[0] < .9)
                    if (arm in ("range_outage_depth",
                                "range_outage_depth_deferred_video") and
                            head_track_bearing is None):
                        with np.errstate(invalid="ignore"):
                            track = (baseline[0] - rays[0] > .02) & (rays[0] < 1.3)
                        if np.any(track):
                            ids = np.flatnonzero(track)
                            chosen = ids[np.argmin(rays[0, ids])]
                            head_track_bearing = float(ANGLES[chosen])
                            if abs(head_track_bearing) <= 90.:
                                look = math.radians(yaw + head_track_bearing)
                                head_forward = np.array([math.sin(look), -.58,
                                                         math.cos(look)])
                if alarm_delivery is None and np.any(changed):
                    ids = np.flatnonzero(changed)
                    chosen = ids[np.argmin(rays[0, ids])]
                    alarm_capture, alarm_delivery = capture_s, delivery_s
                    alarm_angle = float(ANGLES[chosen])
                    alarm_source = "low_range"
                    opposite = math.radians(yaw + alarm_angle + 180.)
                    escape = np.array([math.sin(opposite), math.cos(opposite)])
                head_alarm = False
                head_sample_index = None
                if (arm in ("range_outage_depth",
                            "range_outage_depth_deferred_video") and missing and
                    head_forward is not None and escape is None and
                    head_next_capture_s is not None and
                    capture_s >= head_next_capture_s):
                    head_capture_s = time.perf_counter() - origin
                    head_rgb, head_depth = monitor.observe(at_eye,
                        head_forward, owner_xz=None, width=160, height=90)
                    head_delivery_s = time.perf_counter() - origin
                    changed_pixels = None
                    if head_previous_depth is not None:
                        valid = (np.isfinite(head_previous_depth) &
                                 np.isfinite(head_depth))
                        changed_pixels = int((valid &
                            (head_previous_depth - head_depth > .08) &
                            (head_depth < .9)).sum())
                        head_alarm = changed_pixels >= BACKUP_CHANGED_PIXELS
                    head_sample_index = len(head_samples)
                    rgb_path = folder / f"head_rgb_{head_sample_index:03d}.png"
                    depth_path = folder / f"head_depth_{head_sample_index:03d}.png"
                    Image.fromarray(head_rgb).save(rgb_path)
                    Image.fromarray(np.where(np.isfinite(head_depth),
                        np.clip(head_depth * 1000, 0, 65535), 0).astype(
                            np.uint16)).save(depth_path)
                    head_samples.append({"capture_wall_s": head_capture_s,
                        "delivery_wall_s": head_delivery_s,
                        "changed_pixels": changed_pixels,
                        "alarm": head_alarm,
                        "rgb": relative(rgb_path),
                        "depth": relative(depth_path)})
                    head_previous_depth = head_depth
                    head_next_capture_s = head_capture_s + BACKUP_DEPTH_INTERVAL_S
                    if head_alarm and alarm_delivery is None:
                        alarm_capture = head_capture_s
                        alarm_delivery = head_delivery_s
                        alarm_angle = head_track_bearing
                        alarm_source = "aimed_head_depth"
                        opposite = math.radians(yaw + alarm_angle + 180.)
                        escape = np.array([math.sin(opposite),
                                           math.cos(opposite)])
                gap = (motion.proxy_gap(actor_xy, hazard,
                        state["hazard_radius_m"]) if entrant else None)
                packets.append({"tick": tick,
                    "capture_wall_s": round(capture_s, 6),
                    "delivery_wall_s": round(delivery_s, 6),
                    "source_health": "missing" if missing else "healthy",
                    "actor_xz_m": actor_xy.tolist(),
                    "hazard_xz_m": hazard.tolist() if hazard is not None else None,
                    "evaluator_proxy_gap_m": gap,
                    "alarm": bool(np.any(changed)),
                    "head_depth_alarm": head_alarm,
                    "head_depth_sample_index": head_sample_index,
                    "low_range_m": ([float(x) if np.isfinite(x) else None
                                     for x in rays[0]] if rays is not None else None)})
                if tick % 2 == 0:
                    if not deferred_video:
                        rgb, _ = monitor.observe(at_eye, forward,
                            owner_xz=None, width=160, height=90)
                        display = motion.decorate(rgb, t=capture_s,
                            label=("SECOND ENTRY" if entrant else "RETURN SEARCH") +
                                  (" / RANGE LOST" if missing else ""),
                            actor=actor_xy, start=start, hazard=hazard, gap=gap,
                            alarm=alarm_delivery is not None)
                        writer.append_data(display)
                    frames.append({"tick": tick, "capture_wall_s": capture_s})
                tick += 1
            result = future.result()
            worker_wall_s = time.perf_counter() - origin
            if deferred_video:
                # Re-render exact saved monitor states for diagnostics only.
                # RGB video work no longer blocks range/depth acquisition.
                for frame in frames:
                    packet = packets[frame["tick"]]
                    at = np.asarray(packet["actor_xz_m"], float)
                    obstacle = packet["hazard_xz_m"]
                    if obstacle is None:
                        monitor.set_props(state["base_props"])
                    else:
                        monitor.set_props({**state["base_props"],
                            case["hazard_object"]: (state["hazard_prop"],
                                                    tuple(obstacle), 0.)})
                    rendered_eye = np.array([at[0], monitor.floor_y + .42,
                                             at[1]])
                    rgb, _ = monitor.observe(rendered_eye, forward,
                        owner_xz=None, width=160, height=90)
                    display = motion.decorate(rgb,
                        t=packet["capture_wall_s"],
                        label=("REPLAY SECOND ENTRY" if entrant else
                               "REPLAY RETURN") +
                              (" / RANGE LOST" if packet["source_health"] ==
                               "missing" else ""),
                        actor=at, start=start,
                        hazard=np.asarray(obstacle, float) if obstacle is not
                            None else None,
                        gap=packet["evaluator_proxy_gap_m"],
                        alarm=alarm_delivery is not None and
                              packet["capture_wall_s"] >= alarm_delivery)
                    writer.append_data(display)
    finally:
        conditional.baseline.SEARCH_YAW_OFFSETS_DEG = original_offsets
        state["vision"].search_mode = original_mode
    endpoint = motion.actor_xy(start, escape, alarm_delivery, worker_wall_s)
    gaps = []
    if entrant:
        for t in np.arange(0., worker_wall_s + .0001, .02):
            actor_xy = motion.actor_xy(start, escape, alarm_delivery, float(t))
            hazard = motion.incoming_xy(start, incoming, float(t))
            gaps.append(motion.proxy_gap(actor_xy, hazard,
                                         state["hazard_radius_m"]))
    invalidated = bool(alarm_delivery is not None or health_lost is not None)
    head_video = None
    if head_samples:
        head_video_path = folder / "aimed_head_depth.mp4"
        with imageio.get_writer(str(head_video_path), fps=5,
                codec="libx264", quality=7,
                macro_block_size=16) as writer:
            for sample in head_samples:
                rgb = np.asarray(Image.open(ROOT / sample["rgb"]).convert("RGB"))
                display = cv2.resize(rgb, (640, 360),
                                     interpolation=cv2.INTER_NEAREST)
                label = ("AIMED HEAD " +
                    ("ALARM" if sample["alarm"] else "observe") +
                    f" | t={sample['capture_wall_s']:.2f}s")
                cv2.putText(display, label, (8, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (255, 255, 255), 2)
                writer.append_data(display)
        head_video = relative(head_video_path)
    trace = {"schema": "ape.monitored-return-phase.v1",
        "case": case["id"], "arm": arm, "entrant": entrant, "wide": wide,
        "origin_actor_xz_m": start.tolist(),
        "entry_relative_deg": case["entry_relative_deg"],
        "source_health_outage_window_s": OUTAGE if arm in (
            "range_outage", "range_outage_depth",
            "range_outage_depth_deferred_video") and
            entrant else None,
        "backup_head_track_bearing_deg": head_track_bearing,
        "backup_head_yaw_supported": head_forward is not None,
        "backup_aim_delay_s": (BACKUP_AIM_DELAY_S if arm in
            ("range_outage_depth", "range_outage_depth_deferred_video") and
            entrant else None),
        "backup_depth_interval_s": (BACKUP_DEPTH_INTERVAL_S if arm in
            ("range_outage_depth", "range_outage_depth_deferred_video") and
            entrant else None),
        "backup_head_samples": head_samples,
        "backup_head_video": head_video,
        "monitor_packets": packets, "worker_world_captures": worker_scene.captures,
        "monitor_video_frames": frames,
        "monitor_video_render_mode": ("post_run_diagnostic_replay" if
            deferred_video else "live_in_loop"),
        "monitor_video": relative(folder / "monitor.mp4"),
        "search_trace": relative(folder / "search/trace.json"),
        "search_video": result["video"],
        "search_action": result["action"],
        "search_wrong_candidate": result["wrong_confident_candidate"],
        "alarm_capture_s": alarm_capture,
        "alarm_delivery_s": alarm_delivery,
        "alarm_relative_angle_deg": alarm_angle,
        "alarm_source": alarm_source,
        "source_health_lost_s": health_lost,
        "image_invalidated": invalidated,
        "worker_wall_s": worker_wall_s,
        "response_endpoint_xz_m": endpoint.tolist(),
        "min_evaluator_proxy_gap_m": min(gaps) if gaps else None,
        "owner_answer": None if invalidated else owner_answer(result)}
    path = folder / "phase_trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def fresh_clearance(state: dict, position: np.ndarray) -> list[dict]:
    scene = state["monitor_scene"]
    scene.set_props(state["base_props"])
    packets = []
    for index in range(3):
        if index:
            time.sleep(.1)
        captured = time.perf_counter()
        eye = np.array([position[0], scene.floor_y + .42, position[1]])
        rays = scene.range_fan(eye, state["actual_yaw"], owner_xz=None,
                               angles_deg=ANGLES)
        finite = rays[0][np.isfinite(rays[0])]
        packets.append({"capture_wall_s": captured,
            "nearest_low_range_m": float(finite.min()) if len(finite) else None,
            "healthy": bool(len(finite))})
    return packets


def run_one(case: dict, lane: dict, arm: str) -> dict:
    earlier_path = OUT / case["id"] / arm / "episode_trace.json"
    earlier = json.loads((ROOT / case["first_response_trace"]).read_text())
    state = cohort.generalized_setup(case)
    start = np.asarray(earlier["response_endpoint_xz_m"], float)
    actor = (np.asarray(state["stop"]["actor_pose_xy_m"], float) + start -
             state["actual_xy"])
    folder = earlier_path.parent
    phase1 = monitored_search(state, case, folder / "second_entry", start,
                              actor, entrant=True, arm=arm)
    result = {"schema": "ape.monitored-return-stress.v1",
        "case": case["id"], "arm": arm,
        "first_response_trace": case["first_response_trace"],
        "lane_screen": relative(LANES),
        "second_entry_bearing_deg": case["entry_relative_deg"],
        "second_entry_trace": relative(folder / "second_entry/phase_trace.json"),
        "first_hazard_gap_m": earlier["min_evaluator_proxy_gap_m"],
        "second_hazard_gap_m": phase1["min_evaluator_proxy_gap_m"],
        "second_search_invalidated": phase1["image_invalidated"],
        "phase_traces": [relative(folder / "second_entry/phase_trace.json")],
        "post_exit_clearance": None,
        "owner_answer": None,
        "outcome": None,
        "limits": "Second entrant is a scripted small prop; ideal 17cm disk response and scripted departure. First owner route/teach/response is an audited saved episode. The in-flight image keeps its initial pose and is invalidated on alarm or range loss. Source health is supplied as an injected fault; no automatic device fault diagnosis, measured body stop, IMU/joint/contact feedback or physical owner."}
    if phase1["min_evaluator_proxy_gap_m"] < .05:
        result["outcome"] = "modeled_hazard_breach_hold_unknown"
    else:
        endpoint = np.asarray(phase1["response_endpoint_xz_m"], float)
        packets = fresh_clearance(state, endpoint)
        result["post_exit_clearance"] = packets
        clear = all(p["healthy"] and p["nearest_low_range_m"] >
                    BODY_RADIUS_M + .05 for p in packets)
        if not clear:
            result["outcome"] = "clearance_unavailable_hold_unknown"
        else:
            actor += endpoint - start
            phase2 = monitored_search(state, case, folder / "fresh_return",
                endpoint, actor, entrant=False, arm="healthy")
            result["phase_traces"].append(relative(folder /
                                                "fresh_return/phase_trace.json"))
            answer = phase2["owner_answer"]
            if answer != "yes":
                phase3 = monitored_search(state, case, folder / "wide_return",
                    endpoint, actor, entrant=False, arm="healthy", wide=True)
                result["phase_traces"].append(relative(folder /
                                                  "wide_return/phase_trace.json"))
                answer = phase3["owner_answer"]
            result["owner_answer"] = answer
            result["outcome"] = ("fresh_owner_confirmed" if answer == "yes"
                                 else "hold_unknown")
    earlier_path.write_text(json.dumps(result, indent=2) + "\n")
    return {"case": case["id"], "arm": arm,
        "trace": relative(earlier_path), "outcome": result["outcome"],
        "second_hazard_gap_m": result["second_hazard_gap_m"],
        "owner_answer": result["owner_answer"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    screen = json.loads(LANES.read_text())
    source = json.loads(first.REPORT.read_text())
    by_case = {row["case"]: row for row in source["rows"]}
    cases = []
    for lane in screen["rows"]:
        if lane["selected_relative_bearing_deg"] is None:
            continue
        case = next(c for c in cohort.case_rows() if c["id"] == lane["case"])
        case = {**case,
            "entry_relative_deg": lane["selected_relative_bearing_deg"],
            "first_response_trace": by_case[lane["case"]]["trace"]}
        cases.append(case)
    frozen = {"schema": "ape.monitored-return-deferred-precommit.v1",
        "lane_screen": relative(LANES),
        "lane_screen_sha256": sha(LANES),
        "first_response_report": relative(first.REPORT),
        "first_response_report_sha256": sha(first.REPORT),
        "cases": cases, "arms": ARMS,
        "range_outage_s": OUTAGE,
        "policy": "Compare healthy all-around low range with primary range outage plus a prior-range-directed +/-90deg head-depth backup. Head depth captures begin 0.2s after observed source loss and repeat every 0.2s; >=20 pixels approaching by >0.08m across consecutive frames below 0.9m issue the same ideal opposite-bearing response. Independent range/depth acquisition continues during image work; monitor RGB video is reconstructed only after the worker finishes, from saved timed state. After scripted entrant departure, require three 0.1s-separated clearance packets and generic-first/fixed wide item return under continued monitoring. A modeled <5cm hazard gap blocks item success.",
        "limits": "14 of 19 saved post-response endpoints support screened second-entrant and escape geometry; five unsupported endpoints remain in lane screen. No human or articulated body."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Monitored return precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["id"] / arm / "episode_trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"case": case["id"], "arm": arm,
                    "trace": relative(path), "outcome": trace["outcome"],
                    "second_hazard_gap_m": trace["second_hazard_gap_m"],
                    "owner_answer": trace["owner_answer"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_one(case, {}, arm)
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.monitored-return-deferred-report.v1",
                "precommit": relative(PRECOMMIT),
                "processed": len(rows), "rows": rows}, indent=2) + "\n")
            print(case["id"], arm, row["outcome"],
                  row["second_hazard_gap_m"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
