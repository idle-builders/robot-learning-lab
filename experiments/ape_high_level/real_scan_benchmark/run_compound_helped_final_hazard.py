"""Interrupt helped final confirmation with an independently sensed entrant."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
import time

import imageio.v2 as imageio
import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
from run_compound_owner_point_recovery import case_rows, make_scene, OUT as HELP_OUT
from run_compound_owner_point_approach import OUT as APPROACH_OUT, final_recheck
import run_cross_room_inference_response as motion
import run_shared_world_return as shared


LANES = ART / "compound_helped_hazard_lane_screen.json"
PRECOMMIT = ART / "compound_helped_final_hazard_precommit.json"
REPORT = ART / "compound_helped_final_hazard_report.json"
OUT = ROOT / "assets/real_indoor_scenes/compound_helped_final_hazard_20261015"
ARMS = ("healthy", "range_outage")
OUTAGE = (.25, 1.25)
PERIOD_S = .1
HAZARD = "Popcorn"
# Frozen trial score. The independent audit also scores the rendered 0.0793 m
# footprint; retain this value so the original precommit and traces replay.
HAZARD_RADIUS_M = .06


def placements(case: dict) -> dict:
    screen = json.loads((ROOT / case["screen"]).read_text())
    row = next(x for x in screen["scenes"] if x.get("scene_key") ==
               case["scene_key"])
    props = ROOT / ".cache/ape-high-level/hope_meshes_eval"
    return {case["target"]: (TexturedFloorObject(
        props / f"{case['target']}.obj", scale=.012),
        tuple(row["find_target_xz"]), 0.),
        case["distractor"]: (TexturedFloorObject(
            props / f"{case['distractor']}.obj", scale=.012),
            tuple(row["find_distractor_xz"]), 0.)}


def sensor_process(conn, case: dict, start_xy: list[float], yaw: float,
                   arm: str) -> None:
    try:
        scene = make_scene(case, True)
        base = placements(case)
        hazard_prop = TexturedFloorObject(ROOT /
            f".cache/ape-high-level/hope_meshes_eval/{HAZARD}.obj",
            scale=.012)
        start = np.asarray(start_xy, float)
        angle = math.radians(yaw + case["entry_relative_deg"])
        incoming = np.array([math.sin(angle), math.cos(angle)])
        eye = np.array([start[0], scene.floor_y + .42, start[1]])
        scene.set_props(base)
        baseline = scene.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)
        conn.send({"kind": "ready"})
        message = conn.recv()
        if message["kind"] != "start":
            raise ValueError("Missing common start clock")
        origin = float(message["origin"])
        packets = []
        alarm_capture = alarm_delivery = alarm_angle = health_lost = None
        escape = None
        tick = 0
        stop_s = None
        while True:
            target = origin + tick * PERIOD_S
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            captured = time.perf_counter() - origin
            if captured > 25.:
                raise TimeoutError("Image worker exceeded bounded hazard trial")
            actor = motion.actor_xy(start, escape, alarm_delivery, captured)
            hazard = motion.incoming_xy(start, incoming, captured)
            scene.set_props({**base, HAZARD: (hazard_prop, tuple(hazard), 0.)})
            eye = np.array([actor[0], scene.floor_y + .42, actor[1]])
            missing = arm == "range_outage" and OUTAGE[0] <= captured < OUTAGE[1]
            rays = None if missing else scene.range_fan(eye, yaw,
                owner_xz=None, angles_deg=ANGLES)
            delivered = time.perf_counter() - origin
            if missing and health_lost is None:
                health_lost = delivered
            changed = np.zeros(len(ANGLES), bool)
            if rays is not None:
                with np.errstate(invalid="ignore"):
                    changed = (baseline[0] - rays[0] > .08) & (rays[0] < .9)
            if alarm_delivery is None and np.any(changed):
                ids = np.flatnonzero(changed)
                selected = ids[np.argmin(rays[0, ids])]
                alarm_capture, alarm_delivery = captured, delivered
                alarm_angle = float(ANGLES[selected])
                opposite = math.radians(yaw + alarm_angle + 180.)
                escape = np.array([math.sin(opposite), math.cos(opposite)])
            packets.append({"tick": tick,
                "capture_wall_s": round(captured, 6),
                "delivery_wall_s": round(delivered, 6),
                "source_health": "missing" if missing else "healthy",
                "actor_xz_m": actor.tolist(),
                "hazard_xz_m": hazard.tolist(),
                "range_change_alarm": bool(np.any(changed)),
                "low_range_m": ([float(x) if np.isfinite(x) else None
                    for x in rays[0]] if rays is not None else None)})
            tick += 1
            if conn.poll():
                message = conn.recv()
                if message["kind"] == "stop":
                    stop_s = float(message["worker_end_s"])
                    break
        endpoint = motion.actor_xy(start, escape, alarm_delivery, stop_s)
        conn.send({"kind": "result", "packets": packets,
            "baseline_low_range_m": [float(x) if np.isfinite(x) else None
                                     for x in baseline[0]],
            "alarm_capture_s": alarm_capture,
            "alarm_delivery_s": alarm_delivery,
            "alarm_relative_angle_deg": alarm_angle,
            "source_health_lost_s": health_lost,
            "response_endpoint_xz_m": endpoint.tolist()})
    except Exception as exc:
        conn.send({"kind": "error", "message": repr(exc)})
    finally:
        conn.close()


def run_one(case: dict, lane: dict, arm: str) -> dict:
    folder = OUT / case["scene_key"] / arm
    folder.mkdir(parents=True, exist_ok=True)
    prior = json.loads((APPROACH_OUT / case["scene_key"] / "trace.json").read_text())
    walk = prior["walk"]
    start = np.asarray(walk["actual_final_xz_m"], float)
    yaw = float(walk["actions"][-1]["evaluator_actual_yaw_deg"])
    state = {"actual": start.copy(), "actual_yaw": yaw,
        "actor": np.asarray(walk["actor_final_xz_m"], float),
        "actor_yaw": float(walk["actions"][-1]["actor_yaw_deg"])}
    point = json.loads((HELP_OUT / case["scene_key"] /
                        "accurate_present/trace.json").read_text())
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=sensor_process,
        args=(child, case, start.tolist(), yaw, arm))
    process.start()
    child.close()
    if not parent.poll(45.):
        process.terminate(); process.join()
        raise TimeoutError("Near-body process did not become ready")
    ready = parent.recv()
    if ready["kind"] != "ready":
        process.join()
        raise RuntimeError(f"Near-body process failed: {ready}")
    origin = time.perf_counter() + .2
    parent.send({"kind": "start", "origin": origin})
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    worker_scene = make_scene(case, True)
    hazard_prop = TexturedFloorObject(ROOT /
        f".cache/ape-high-level/hope_meshes_eval/{HAZARD}.obj", scale=.012)
    wrapped = shared.TimedWorkerScene(worker_scene, origin, case,
        {"hazard_prop": hazard_prop}, start, incoming)
    remaining = origin - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(final_recheck, wrapped, case, point, state,
            folder / "interrupted_recheck")
        worker_result = future.result(timeout=25.)
    worker_end = time.perf_counter() - origin
    parent.send({"kind": "stop", "worker_end_s": worker_end})
    if not parent.poll(30.):
        process.terminate(); process.join()
        raise TimeoutError("Near-body process did not return")
    response = parent.recv()
    process.join(timeout=10.)
    if process.is_alive():
        process.terminate(); process.join()
        raise TimeoutError("Near-body process did not exit")
    parent.close()
    if response["kind"] != "result":
        raise RuntimeError(f"Near-body process failed: {response}")
    packets = response["packets"]
    alarm = response["alarm_delivery_s"]
    escape = None
    if response["alarm_relative_angle_deg"] is not None:
        opposite = math.radians(yaw + response["alarm_relative_angle_deg"] + 180.)
        escape = np.array([math.sin(opposite), math.cos(opposite)])
    gaps = []
    for t in np.arange(0., worker_end + .0001, .02):
        actor = motion.actor_xy(start, escape, alarm, float(t))
        hazard = motion.incoming_xy(start, incoming, float(t))
        gaps.append(motion.proxy_gap(actor, hazard, HAZARD_RADIUS_M))
    min_gap = min(gaps)
    invalidated = alarm is not None or response["source_health_lost_s"] is not None
    scene = make_scene(case, True)
    base = placements(case)
    forward = np.array([math.sin(math.radians(yaw)), -.58,
                        math.cos(math.radians(yaw))])
    diagnostic_ticks = []
    with imageio.get_writer(str(folder / "monitor.mp4"), fps=5,
            codec="libx264", quality=6, macro_block_size=16) as writer:
        for packet in packets[::2]:
            actor = np.asarray(packet["actor_xz_m"], float)
            hazard = np.asarray(packet["hazard_xz_m"], float)
            scene.set_props({**base, HAZARD: (hazard_prop, tuple(hazard), 0.)})
            eye = np.array([actor[0], scene.floor_y + .42, actor[1]])
            rgb, _ = scene.observe(eye, forward, owner_xz=None,
                                   width=160, height=90)
            writer.append_data(motion.decorate(rgb,
                t=packet["capture_wall_s"], label="HELPED FINAL CHECK",
                actor=actor, start=start, hazard=hazard,
                gap=motion.proxy_gap(actor, hazard, HAZARD_RADIUS_M),
                alarm=alarm is not None and packet["capture_wall_s"] >= alarm))
            diagnostic_ticks.append(packet["tick"])
    trace = {"schema": "ape.compound-helped-final-hazard.v1",
        "case": case["scene_key"], "arm": arm,
        "source_action_trace": relative(APPROACH_OUT / case["scene_key"] / "trace.json"),
        "lane_screen": relative(LANES),
        "entry_relative_deg": case["entry_relative_deg"],
        "start_xz_m": start.tolist(), "heading_deg": yaw,
        "worker_wall_s": worker_end,
        "worker_world_captures": wrapped.captures,
        "worker_recheck_trace": relative(folder /
            "interrupted_recheck/episode_trace.json"),
        "worker_recheck_video": worker_result["video"],
        "worker_action_before_health_gate": worker_result["action"],
        "usable_action": ("invalidated_hold_unknown" if invalidated else
                          worker_result["action"]),
        "image_invalidated": invalidated,
        "monitor_packets": packets,
        "monitor_video": relative(folder / "monitor.mp4"),
        "monitor_video_packet_ticks": diagnostic_ticks,
        "baseline_low_range_m": response["baseline_low_range_m"],
        "source_health_lost_s": response["source_health_lost_s"],
        "alarm_capture_s": response["alarm_capture_s"],
        "alarm_delivery_s": alarm,
        "alarm_relative_angle_deg": response["alarm_relative_angle_deg"],
        "response_endpoint_xz_m": response["response_endpoint_xz_m"],
        "min_evaluator_proxy_gap_m": min_gap,
        "max_monitor_capture_interval_s": max((b["capture_wall_s"] -
            a["capture_wall_s"] for a,b in zip(packets, packets[1:])), default=0.),
        "limits": "Three of six helped standoffs with evaluator-screened entry/escape geometry; the other three remain unsupported. Independent range process and concurrent actual image worker share a monotonic clock and textured scene. Diagnostic monitor video is postrun. The image worker stays at its original pose and its result is discarded after alarm/source loss. No post-interruption owner return in this phase; ideal disk response and rigid prop."}
    path = folder / "trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["scene_key"], "arm": arm,
        "trace": relative(path), "min_gap_m": min_gap,
        "usable_action": trace["usable_action"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    lanes = json.loads(LANES.read_text())
    case_map = {c["scene_key"]: c for c in case_rows()}
    selected = [r for r in lanes["rows"] if
                r["selected_relative_bearing_deg"] is not None]
    if len(lanes["rows"]) != 6 or len(selected) != 3:
        raise ValueError("Expected six screened, three viable final poses")
    cases = [{**case_map[r["case"]], "entry_relative_deg": r[
        "selected_relative_bearing_deg"], "hazard_object": HAZARD}
        for r in selected]
    frozen = {"schema": "ape.compound-helped-final-hazard-precommit.v1",
        "lane_screen": relative(LANES), "lane_screen_sha256": sha(LANES),
        "cases": cases, "arms": ARMS, "range_outage_s": OUTAGE,
        "period_s": PERIOD_S, "hazard_object": HAZARD,
        "hazard_radius_m": HAZARD_RADIUS_M,
        "policy": "During a real fresh final owner-image worker after owner help and guarded item approach, an independent spawned near-body process samples two-height all-around range each 0.1s. A changed return issues the same ideal delayed opposite-bearing disk response; source loss invalidates in-flight identity. Compare healthy and 0.25-1.25s missing-range arms without retuning the visual policy. Worker sees the same timed prop on an independent textured renderer; supervisor discards worker result if alarm or source loss.",
        "limits": "Three geometry-supported final poses, three unsupported retained; static scripted owner/task before challenge, ideal kinematic response and rigid low prop. This phase ends at interruption and does not prove post-alert return or physical safety."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("Final-hazard precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    for case in cases:
        for arm in ARMS:
            path = OUT / case["scene_key"] / arm / "trace.json"
            if path.exists():
                trace = json.loads(path.read_text())
                row = {"case": case["scene_key"], "arm": arm,
                    "trace": relative(path),
                    "min_gap_m": trace["min_evaluator_proxy_gap_m"],
                    "usable_action": trace["usable_action"]}
            else:
                if args.max_new and launched >= args.max_new:
                    break
                row = run_one(case, {}, arm)
                launched += 1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":
                "ape.compound-helped-final-hazard-report.v1",
                "precommit": relative(PRECOMMIT), "processed": len(rows),
                "rows": rows}, indent=2) + "\n")
            print(row["case"], arm, row["min_gap_m"],
                  row["usable_action"], flush=True)
        if args.max_new and launched >= args.max_new:
            break


if __name__ == "__main__":
    main()
