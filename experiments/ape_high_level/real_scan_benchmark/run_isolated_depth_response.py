"""Respond to a second entrant in a sensor/action process during image work.

The real generic image worker sees the same timed prop on a separate room
renderer. A spawned process owns near-body sensing and an ideal bounded proxy
response, so slow recognition cannot delay its sensor or action scheduler.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
import run_cross_room_inference_response as motion
import run_item_inference_response_19 as cohort
import run_shared_world_return as first


LANES = ART / "monitored_return_lane_screen.json"
PRECOMMIT = ART / "isolated_depth_response_precommit.json"
REPORT = ART / "isolated_depth_response_report.json"
OUT = ROOT / "assets/real_indoor_scenes/isolated_depth_response_20261015"
TICK_S = .1
OUTAGE = (.25, 1.25)
ARMS = ("healthy", "range_outage", "range_outage_depth")
HEAD_AIM_DELAY_S = .2
HEAD_SAMPLE_INTERVAL_S = .2
HEAD_CHANGED_PIXELS = 20


def sensor_process(conn, case: dict, start_xy: list[float], yaw: float,
                   arm: str, folder_path: str) -> None:
    try:
        summary = json.loads((ROOT / case["teaching_summary"]).read_text())
        source = json.loads((ROOT / summary["source_capture"]).read_text())
        screen = json.loads((ROOT / summary["scenario_placement_screen"]).read_text())
        placement = next(x for x in screen["scenes"] if
                         x["scene_key"] == case["scene_key"])
        scene = TexturedCaptureScene(ROOT / source["source_mesh"],
            floor_y=source["floor_y_1pct_m"],
            scale=source.get("scale_to_metres", 1.),
            source_z_up=source.get("source_z_up", False))
        props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
        base = {case["target"]: (TexturedFloorObject(
            props / f"{case['target']}.obj", scale=.012),
            tuple(placement["find_target_xz"]), 0.),
            case["distractor"]: (TexturedFloorObject(
            props / f"{case['distractor']}.obj", scale=.012),
            tuple(placement["find_distractor_xz"]), 0.)}
        hazard_prop = TexturedFloorObject(props /
            f"{case['hazard_object']}.obj", scale=.012)
        scene.set_props(base)
        start = np.asarray(start_xy, float)
        bearing = math.radians(yaw + case["entry_relative_deg"])
        incoming = np.array([math.sin(bearing), math.cos(bearing)])
        eye = np.array([start[0], scene.floor_y + .42, start[1]])
        baseline = scene.range_fan(eye, yaw, owner_xz=None,
                                   angles_deg=ANGLES)
        conn.send({"kind": "ready"})
        message = conn.recv()
        if message["kind"] != "start":
            raise ValueError("Sensor process did not receive start")
        origin = float(message["origin"])
        folder = ROOT / folder_path
        folder.mkdir(parents=True, exist_ok=True)
        packets = []
        head_samples = []
        alarm_capture = alarm_delivery = alarm_angle = health_lost = None
        alarm_source = None
        escape = None
        head_track = None
        head_forward = None
        head_previous = None
        head_next = None
        requested_stop_s = None
        tick = 0
        while True:
            target = origin + tick * TICK_S
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            capture_s = time.perf_counter() - origin
            if capture_s > 25.:
                raise TimeoutError("Isolated monitor exceeded bounded image search")
            actor = motion.actor_xy(start, escape, alarm_delivery, capture_s)
            hazard = motion.incoming_xy(start, incoming, capture_s)
            scene.set_props({**base, case["hazard_object"]: (
                hazard_prop, tuple(hazard), 0.)})
            at_eye = np.array([actor[0], scene.floor_y + .42, actor[1]])
            missing = bool(arm != "healthy" and
                           OUTAGE[0] <= capture_s < OUTAGE[1])
            rays = (None if missing else scene.range_fan(at_eye, yaw,
                    owner_xz=None, angles_deg=ANGLES))
            delivery_s = time.perf_counter() - origin
            if missing and health_lost is None:
                health_lost = delivery_s
                head_next = capture_s + HEAD_AIM_DELAY_S
            if rays is None:
                changed = np.zeros(len(ANGLES), bool)
            else:
                with np.errstate(invalid="ignore"):
                    changed = (baseline[0] - rays[0] > .08) & (rays[0] < .9)
                    track = (baseline[0] - rays[0] > .02) & (rays[0] < 1.3)
                if arm == "range_outage_depth" and head_track is None and np.any(track):
                    ids = np.flatnonzero(track)
                    chosen = ids[np.argmin(rays[0, ids])]
                    head_track = float(ANGLES[chosen])
                    if abs(head_track) <= 90.:
                        look = math.radians(yaw + head_track)
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
            if (arm == "range_outage_depth" and missing and
                head_forward is not None and escape is None and
                head_next is not None and capture_s >= head_next):
                head_capture_s = time.perf_counter() - origin
                rgb, depth = scene.observe(at_eye, head_forward,
                    owner_xz=None, width=160, height=90)
                head_delivery_s = time.perf_counter() - origin
                changed_pixels = None
                if head_previous is not None:
                    valid = np.isfinite(head_previous) & np.isfinite(depth)
                    changed_pixels = int((valid &
                        (head_previous - depth > .08) & (depth < .9)).sum())
                    head_alarm = changed_pixels >= HEAD_CHANGED_PIXELS
                head_sample_index = len(head_samples)
                rgb_path = folder / f"head_rgb_{head_sample_index:03d}.png"
                depth_path = folder / f"head_depth_{head_sample_index:03d}.png"
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(np.where(np.isfinite(depth),
                    np.clip(depth * 1000, 0, 65535), 0).astype(
                        np.uint16)).save(depth_path)
                head_samples.append({"capture_wall_s": head_capture_s,
                    "delivery_wall_s": head_delivery_s,
                    "changed_pixels": changed_pixels,
                    "alarm": head_alarm,
                    "rgb": relative(rgb_path),
                    "depth": relative(depth_path)})
                head_previous = depth
                head_next = head_capture_s + HEAD_SAMPLE_INTERVAL_S
                if head_alarm and alarm_delivery is None:
                    alarm_capture = head_capture_s
                    alarm_delivery = head_delivery_s
                    alarm_angle = head_track
                    alarm_source = "aimed_head_depth"
                    opposite = math.radians(yaw + alarm_angle + 180.)
                    escape = np.array([math.sin(opposite), math.cos(opposite)])
            packets.append({"tick": tick,
                "capture_wall_s": round(capture_s, 6),
                "delivery_wall_s": round(delivery_s, 6),
                "source_health": "missing" if missing else "healthy",
                "actor_xz_m": actor.tolist(),
                "hazard_xz_m": hazard.tolist(),
                "range_change_alarm": bool(np.any(changed)),
                "head_depth_alarm": head_alarm,
                "head_depth_sample_index": head_sample_index,
                "low_range_m": ([float(x) if np.isfinite(x) else None
                                 for x in rays[0]] if rays is not None else None)})
            tick += 1
            if conn.poll():
                message = conn.recv()
                if message["kind"] == "stop":
                    requested_stop_s = float(message["worker_end_s"])
                    break
        endpoint = motion.actor_xy(start, escape, alarm_delivery,
                                    requested_stop_s)
        conn.send({"kind": "result", "packets": packets,
            "head_samples": head_samples,
            "head_track_bearing_deg": head_track,
            "alarm_capture_s": alarm_capture,
            "alarm_delivery_s": alarm_delivery,
            "alarm_relative_angle_deg": alarm_angle,
            "alarm_source": alarm_source,
            "source_health_lost_s": health_lost,
            "response_endpoint_xz_m": endpoint.tolist(),
            "baseline_low_range_m": [float(x) if np.isfinite(x) else None
                                      for x in baseline[0]]})
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def run_one(case: dict, lane: dict, arm: str) -> dict:
    state = cohort.generalized_setup(case)
    folder = OUT / case["id"] / arm
    folder.mkdir(parents=True, exist_ok=True)
    start = np.asarray(lane["response_endpoint_xz_m"], float)
    actor = (np.asarray(state["stop"]["actor_pose_xy_m"], float) + start -
             state["actual_xy"])
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=sensor_process,
        args=(child, case, start.tolist(), state["actual_yaw"], arm,
              relative(folder)))
    process.start()
    child.close()
    if not parent.poll(45.):
        process.terminate()
        process.join()
        raise TimeoutError("Isolated sensor scene did not become ready")
    ready = parent.recv()
    if ready["kind"] != "ready":
        process.join()
        raise RuntimeError(f"Sensor process failed: {ready}")
    origin = time.perf_counter() + .2
    parent.send({"kind": "start", "origin": origin})
    bearing = math.radians(state["actual_yaw"] + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    worker_scene = first.TimedWorkerScene(state["worker_scene"], origin,
                                         case, state, start, incoming)
    remaining = origin - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(first.search, state, case, worker_scene, start,
            actor, folder / "search",
            float(state["stop"]["decision_time_s"] + .4))
        image_result = future.result(timeout=25.)
    worker_end_s = time.perf_counter() - origin
    parent.send({"kind": "stop", "worker_end_s": worker_end_s})
    if not parent.poll(30.):
        process.terminate()
        process.join()
        raise TimeoutError("Isolated monitor did not return saved packets")
    response = parent.recv()
    process.join(timeout=10.)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError("Isolated monitor did not exit")
    parent.close()
    if response["kind"] != "result":
        raise RuntimeError(f"Sensor process failed: {response}")
    packets = response["packets"]
    alarm_delivery = response["alarm_delivery_s"]
    alarm_angle = response["alarm_relative_angle_deg"]
    escape = None
    if alarm_angle is not None:
        direction = math.radians(state["actual_yaw"] + alarm_angle + 180.)
        escape = np.array([math.sin(direction), math.cos(direction)])
    scores = []
    for t in np.arange(0., worker_end_s + .0001, .02):
        actor_xy = motion.actor_xy(start, escape, alarm_delivery, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        scores.append(motion.proxy_gap(actor_xy, hazard,
                                       state["hazard_radius_m"]))
    min_gap = min(scores)
    monitor = state["monitor_scene"]
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    frames = []
    with imageio.get_writer(str(folder / "monitor.mp4"), fps=5,
            codec="libx264", quality=6, macro_block_size=16) as writer:
        for packet in packets[::2]:
            hazard = np.asarray(packet["hazard_xz_m"], float)
            actor_xy = np.asarray(packet["actor_xz_m"], float)
            monitor.set_props({**state["base_props"],
                case["hazard_object"]: (state["hazard_prop"],
                                        tuple(hazard), 0.)})
            eye = np.array([actor_xy[0], monitor.floor_y + .42, actor_xy[1]])
            rgb, _ = monitor.observe(eye, forward, owner_xz=None,
                                     width=160, height=90)
            display = motion.decorate(rgb,
                t=packet["capture_wall_s"], label="ISOLATED RESPONSE REPLAY",
                actor=actor_xy, start=start, hazard=hazard,
                gap=motion.proxy_gap(actor_xy, hazard,
                                     state["hazard_radius_m"]),
                alarm=alarm_delivery is not None and
                      packet["capture_wall_s"] >= alarm_delivery)
            writer.append_data(display)
            frames.append(packet["tick"])
    head_video = None
    if response["head_samples"]:
        head_path = folder / "aimed_head_depth.mp4"
        with imageio.get_writer(str(head_path), fps=5,
                codec="libx264", quality=7,
                macro_block_size=16) as writer:
            for sample in response["head_samples"]:
                rgb = np.asarray(Image.open(ROOT / sample["rgb"]).convert("RGB"))
                writer.append_data(np.repeat(np.repeat(rgb, 4, axis=0),
                                             4, axis=1))
        head_video = relative(head_path)
    trace = {"schema": "ape.isolated-depth-response.v1",
        "case": case["id"], "entry_relative_deg": case["entry_relative_deg"],
        "arm": arm,
        "actor_xz_m": start.tolist(),
        "worker_wall_s": worker_end_s,
        "worker_world_captures": worker_scene.captures,
        "search_trace": relative(folder / "search/trace.json"),
        "search_video": image_result["video"],
        "search_action": image_result["action"],
        "monitor_packets": packets,
        "head_samples": response["head_samples"],
        "head_video": head_video,
        "head_track_bearing_deg": response["head_track_bearing_deg"],
        "source_health_lost_s": response["source_health_lost_s"],
        "alarm_capture_s": response["alarm_capture_s"],
        "alarm_delivery_s": alarm_delivery,
        "alarm_relative_angle_deg": alarm_angle,
        "alarm_source": response["alarm_source"],
        "response_endpoint_xz_m": response["response_endpoint_xz_m"],
        "min_evaluator_proxy_gap_m": min_gap,
        "image_invalidated": bool(alarm_delivery is not None or
                                  response["source_health_lost_s"] is not None),
        "baseline_low_range_m": response["baseline_low_range_m"],
        "monitor_video": relative(folder / "monitor.mp4"),
        "monitor_video_packet_ticks": frames,
        "monitor_video_render_mode": "post_run_diagnostic_replay",
        "max_monitor_capture_interval_s": max((b["capture_wall_s"] -
            a["capture_wall_s"] for a,b in zip(packets, packets[1:])),
            default=0.),
        "limits": "A separate spawned sensor/action process drives an ideal 17cm disk response while the generic image worker sees a same-clock moving-prop copy. The image result is invalidated on alarm or source loss. No fresh item return is run in this phase; diagnostic videos are reconstructed after worker completion. Head yaw, clean depth and disk response remain unqualified counterfactuals."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "trace": relative(path),
        "max_monitor_capture_interval_s": trace["max_monitor_capture_interval_s"],
        "worker_wall_s": worker_end_s,
        "arm": arm, "min_evaluator_proxy_gap_m": min_gap,
        "alarm_source": response["alarm_source"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    lanes = json.loads(LANES.read_text())
    original = {x["id"]: x for x in cohort.case_rows()}
    selected = [x for x in lanes["rows"] if
                x["selected_relative_bearing_deg"] is not None]
    cases = [{**original[x["case"]],
        "entry_relative_deg": x["selected_relative_bearing_deg"]}
        for x in selected]
    frozen = {"schema": "ape.isolated-depth-response-precommit.v1",
        "lane_screen": relative(LANES), "lane_screen_sha256": sha(LANES),
        "cases": cases, "arms": ARMS,
        "requested_range_period_s": TICK_S,
        "range_outage_s": OUTAGE,
        "head_aim_delay_s": HEAD_AIM_DELAY_S,
        "head_sample_period_s": HEAD_SAMPLE_INTERVAL_S,
        "policy": "Independent spawned sensor/action process compares healthy 360deg low/mid range, primary range missing 0.25-1.25s, and that missing source with an ideal +/-90deg head-depth backup. The first healthy range packet can choose a changed-object bearing within 1.3m. On outage, head aims after 0.2s; >=20 closer pixels below 0.9m between consecutive 160x90 depth captures issue the same ideal opposite-bearing 0.1s-delayed 0.25m/s disk response. The real generic image worker sees the same moving prop concurrently, but any alarm/source loss invalidates its image. Diagnostic video is rendered after timing is complete.",
        "limits": "14 screened held positions; ideal prop/disk and head yaw/depth; no physical source, measured head/stop/contact or owner-task return in this phase."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Isolated response precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case, lane in zip(cases, selected):
        for arm in ARMS:
            path = OUT / case["id"] / arm / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"case": case["id"], "trace": relative(path),
                    "max_monitor_capture_interval_s": trace[
                        "max_monitor_capture_interval_s"],
                    "worker_wall_s": trace["worker_wall_s"],
                    "arm": arm,
                    "min_evaluator_proxy_gap_m": trace[
                        "min_evaluator_proxy_gap_m"],
                    "alarm_source": trace["alarm_source"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_one(case, lane, arm)
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.isolated-depth-response-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(case["id"], arm, row["min_evaluator_proxy_gap_m"],
                  row["alarm_source"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
