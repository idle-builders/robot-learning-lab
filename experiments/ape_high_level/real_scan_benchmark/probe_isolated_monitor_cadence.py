"""Measure a separate low-range monitor while the real image worker runs.

This isolates the time-critical range sampler in a spawned process. It does
not yet choose or execute a response; the question is whether the measured
source cadence survives concurrent generic image inference and rendering.
The image worker still sees the same scripted moving prop on the same clock.
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

from real_capture_dynamic_scene import TexturedCaptureScene, TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
import run_cross_room_inference_response as motion
import run_item_inference_response_19 as cohort
import run_shared_world_return as first


LANES = ART / "monitored_return_lane_screen.json"
PRECOMMIT = ART / "isolated_monitor_cadence_precommit.json"
REPORT = ART / "isolated_monitor_cadence_report.json"
OUT = ROOT / "assets/real_indoor_scenes/isolated_monitor_cadence_20261015"
TICK_S = .1


def sensor_process(conn, case: dict, start_xy: list[float], yaw: float,
                   origin: float | None = None) -> None:
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
        packets = []
        tick = 0
        while True:
            target = origin + tick * TICK_S
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            capture_s = time.perf_counter() - origin
            if capture_s > 25.:
                raise TimeoutError("Isolated monitor exceeded bounded image search")
            hazard = motion.incoming_xy(start, incoming, capture_s)
            scene.set_props({**base, case["hazard_object"]: (
                hazard_prop, tuple(hazard), 0.)})
            rays = scene.range_fan(eye, yaw, owner_xz=None,
                                   angles_deg=ANGLES)
            delivery_s = time.perf_counter() - origin
            with np.errstate(invalid="ignore"):
                changed = (baseline[0] - rays[0] > .08) & (rays[0] < .9)
            packets.append({"tick": tick,
                "capture_wall_s": round(capture_s, 6),
                "delivery_wall_s": round(delivery_s, 6),
                "hazard_xz_m": hazard.tolist(),
                "range_change_alarm": bool(np.any(changed)),
                "low_range_m": [float(x) if np.isfinite(x) else None
                                for x in rays[0]]})
            tick += 1
            if conn.poll():
                message = conn.recv()
                if message["kind"] == "stop":
                    break
        conn.send({"kind": "result", "packets": packets,
            "baseline_low_range_m": [float(x) if np.isfinite(x) else None
                                      for x in baseline[0]]})
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def run_one(case: dict, lane: dict) -> dict:
    state = cohort.generalized_setup(case)
    folder = OUT / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    start = np.asarray(lane["response_endpoint_xz_m"], float)
    actor = (np.asarray(state["stop"]["actor_pose_xy_m"], float) + start -
             state["actual_xy"])
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=sensor_process,
        args=(child, case, start.tolist(), state["actual_yaw"]))
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
    parent.send({"kind": "stop"})
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
    monitor = state["monitor_scene"]
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    frames = []
    with imageio.get_writer(str(folder / "monitor.mp4"), fps=5,
            codec="libx264", quality=6, macro_block_size=16) as writer:
        for packet in packets[::2]:
            hazard = np.asarray(packet["hazard_xz_m"], float)
            monitor.set_props({**state["base_props"],
                case["hazard_object"]: (state["hazard_prop"],
                                        tuple(hazard), 0.)})
            eye = np.array([start[0], monitor.floor_y + .42, start[1]])
            rgb, _ = monitor.observe(eye, forward, owner_xz=None,
                                     width=160, height=90)
            display = motion.decorate(rgb,
                t=packet["capture_wall_s"], label="ISOLATED RANGE REPLAY",
                actor=start, start=start, hazard=hazard,
                gap=motion.proxy_gap(start, hazard,
                                     state["hazard_radius_m"]),
                alarm=packet["range_change_alarm"])
            writer.append_data(display)
            frames.append(packet["tick"])
    trace = {"schema": "ape.isolated-monitor-cadence.v1",
        "case": case["id"], "entry_relative_deg": case["entry_relative_deg"],
        "actor_xz_m": start.tolist(),
        "worker_wall_s": worker_end_s,
        "worker_world_captures": worker_scene.captures,
        "search_trace": relative(folder / "search/trace.json"),
        "search_video": image_result["video"],
        "search_action": image_result["action"],
        "monitor_packets": packets,
        "baseline_low_range_m": response["baseline_low_range_m"],
        "monitor_video": relative(folder / "monitor.mp4"),
        "monitor_video_packet_ticks": frames,
        "monitor_video_render_mode": "post_run_diagnostic_replay",
        "max_monitor_capture_interval_s": max((b["capture_wall_s"] -
            a["capture_wall_s"] for a,b in zip(packets, packets[1:])),
            default=0.),
        "limits": "Sensor-only cadence probe. A separate spawned process renders 360deg low/mid range while the real generic image worker uses a same-clock moving-prop copy. No action response or owner return is run here; diagnostic monitor RGB video is reconstructed after worker completion."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "trace": relative(path),
        "max_monitor_capture_interval_s": trace["max_monitor_capture_interval_s"],
        "worker_wall_s": worker_end_s}


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
    frozen = {"schema": "ape.isolated-monitor-cadence-precommit.v1",
        "lane_screen": relative(LANES), "lane_screen_sha256": sha(LANES),
        "cases": cases, "requested_range_period_s": TICK_S,
        "policy": "Compare actual low-range capture intervals with a spawned independent sensor process while the real generic image worker runs against the same timed scripted entrant. Keep diagnostic monitor RGB rendering out of the timed path; no action command or item return in this scheduler probe.",
        "limits": "14 geometry-screened held positions only; spawned process and ideal simulated range are not physical sensor timings."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Isolated monitor precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case, lane in zip(cases, selected):
        path = OUT / case["id"] / "trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"case": case["id"], "trace": relative(path),
                "max_monitor_capture_interval_s": trace[
                    "max_monitor_capture_interval_s"],
                "worker_wall_s": trace["worker_wall_s"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case, lane)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema":
            "ape.isolated-monitor-cadence-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["id"], row["max_monitor_capture_interval_s"], flush=True)


if __name__ == "__main__":
    main()
