"""Resume teach/find from fresh sensor evidence after an inference interruption.

The previous candidate is invalid. Three fresh low-range packets must support
a held view before a new generic-first conditional search. A separate range
monitor continues while the new image worker runs. Pose after escape is only
command-integrated and cannot authorize a metric reach or fetch.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import setup, ANGLES
from run_cross_room_inference_response import actor_xy, incoming_xy
import run_real_capture_conditional_search as conditional


PRECOMMIT = ART / "cross_room_recovery_precommit.json"
REPORT = ART / "cross_room_recovery_report.json"
OUT = ART / "cross_room_recovery"
TICK_S = .1


class PersistentPropScene:
    """Keep the stopped entrant when the unchanged search sets task props."""

    def __init__(self, scene, name, prop, position):
        self.scene = scene
        self.hazard = {name: (prop, tuple(position), 0.)}

    def set_props(self, placements):
        self.scene.set_props({**placements, **self.hazard})

    def __getattr__(self, name):
        return getattr(self.scene, name)


def fresh_clearance(trace: dict) -> tuple[bool, dict]:
    packets = trace["monitor_packets"][-3:]
    if len(packets) != 3:
        return False, {"reason": "fewer_than_three_packets"}
    times = [float(x["capture_wall_s"]) for x in packets]
    minima = [min((v for v in x["low_range_m"] if v is not None),
                  default=math.inf) for x in packets]
    age = float(trace["worker_wall_s"]) - times[-1]
    accepted = bool(all(x > .30 and math.isfinite(x) for x in minima) and
                    all(0 < b - a <= .15 for a, b in zip(times, times[1:])) and
                    0 <= age <= .2 and
                    max(minima) - min(minima) <= .03)
    return accepted, {"capture_wall_s": times,
        "nearest_low_range_m": minima,
        "age_at_worker_release_s": age,
        "rule": "three 360deg low packets; nearest finite >0.30m; <=0.15s spacing; <=0.20s latest age; <=0.03m spread",
        "accepted": accepted}


def run_case(case: dict, condition: str) -> dict:
    state = setup(case)
    previous = json.loads((ROOT / case["interrupted_trace"]).read_text())
    gate_ok, gate = fresh_clearance(previous)
    folder = OUT / case["id"] / condition
    folder.mkdir(parents=True, exist_ok=True)
    if not gate_ok:
        result = {"case": case["id"], "condition": condition,
                  "action": "ask_owner_to_pause_for_clearance",
                  "fresh_clearance": gate,
                  "limits": "No new item claim without three fresh observed clearance packets."}
        (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    stop = state["stop"]
    start = state["actual_xy"]
    yaw = state["actual_yaw"]
    alarm = float(previous["alarm_delivery_s"])
    direction = np.asarray(previous["response_direction_xz"], float)
    new_xy = actor_xy(start, direction, alarm, float(previous["worker_wall_s"]))
    commanded_delta = new_xy - start
    actor_estimate = np.asarray(stop["actor_pose_xy_m"], float) + commanded_delta
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    hazard_xy = incoming_xy(start, incoming, float(previous["worker_wall_s"]))
    target_present = condition == "find_present"
    props = state["base_props"].copy()
    if not target_present:
        props.pop(case["target"])
    props[case["hazard_object"]] = (state["hazard_prop"], tuple(hazard_xy), 0.)
    monitor = state["monitor_scene"]
    monitor.set_props(props)
    worker_scene = PersistentPropScene(state["worker_scene"],
        case["hazard_object"], state["hazard_prop"], hazard_xy)
    eye = np.array([new_xy[0], monitor.floor_y + .42, new_xy[1]])
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    baseline_ranges = monitor.range_fan(eye, yaw, owner_xz=None,
                                         angles_deg=ANGLES)
    baseline_rgb, baseline_depth = monitor.observe(eye, forward,
        owner_xz=None, width=160, height=90)
    Image.fromarray(baseline_rgb).save(folder / "baseline_rgb.png")
    Image.fromarray(np.where(np.isfinite(baseline_depth),
        np.clip(baseline_depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
            folder / "baseline_depth.png")
    vision = state["vision"]
    start_time_s = float(stop["decision_time_s"] + .4 +
                         previous["worker_wall_s"] + .3)

    def worker():
        return conditional.conditional_run_search(
            worker_scene, vision, new_xy, yaw, actor_estimate,
            float(stop["actor_heading_deg"]), state["exemplar"],
            float(state["summary"]["teaching"]["acceptance_similarity"]),
            target_present, folder / "search", start_time_s=start_time_s,
            search_xz=tuple(state["placement"]["find_target_xz"]),
            search_distractor_xz=tuple(
                state["placement"]["find_distractor_xz"]),
            owner_start_xz=tuple(state["cfg"]["start_xz"]),
            target_name=case["target"], distractor_name=case["distractor"],
            identity_mode="multi_cue", colour_exemplar=state["colour"],
            colour_acceptance=float(state["summary"]["teaching"][
                "colour_acceptance_similarity"]))

    packets = []
    frames = []
    monitor_video = folder / "monitor.mp4"
    first_unsafe = None
    max_lateness = 0.
    with ThreadPoolExecutor(max_workers=1) as pool, imageio.get_writer(
            str(monitor_video), fps=5, codec="libx264", quality=7,
            macro_block_size=16) as writer:
        origin = time.perf_counter()
        future = pool.submit(worker)
        tick = 0
        while not future.done():
            due = origin + tick * TICK_S
            remaining = due - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            capture_s = time.perf_counter() - origin
            max_lateness = max(max_lateness, capture_s - tick * TICK_S)
            if capture_s > 25.:
                raise TimeoutError("Recovery image worker exceeded 25s")
            rays = monitor.range_fan(eye, yaw, owner_xz=None,
                                     angles_deg=ANGLES)
            delivered_s = time.perf_counter() - origin
            finite = rays[0][np.isfinite(rays[0])]
            nearest = float(finite.min()) if len(finite) else math.inf
            with np.errstate(invalid="ignore"):
                changed = (baseline_ranges[0] - rays[0] > .08) & (rays[0] < .9)
            unsafe = not math.isfinite(nearest) or nearest <= .30 or bool(changed.any())
            if unsafe and first_unsafe is None:
                first_unsafe = delivered_s
            if tick % 2 == 0:
                rgb, depth = monitor.observe(eye, forward,
                    owner_xz=None, width=160, height=90)
                display = cv2.resize(rgb, (640, 360),
                                     interpolation=cv2.INTER_NEAREST)
                cv2.putText(display,
                    f"RECOVERY {condition} | t={capture_s:.2f}s | " +
                    ("UNSAFE" if unsafe else "monitoring"),
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .52,
                    (255, 255, 255), 2)
                writer.append_data(display)
                index = len(frames)
                rgb_path = folder / f"monitor_rgb_{index:03d}.png"
                depth_path = folder / f"monitor_depth_{index:03d}.png"
                Image.fromarray(rgb).save(rgb_path)
                Image.fromarray(np.where(np.isfinite(depth),
                    np.clip(depth * 1000, 0, 65535), 0).astype(np.uint16)).save(
                        depth_path)
                frames.append({"tick": tick, "capture_wall_s": capture_s,
                    "rgb": relative(rgb_path), "depth": relative(depth_path)})
            packets.append({"tick": tick, "capture_wall_s": capture_s,
                "delivery_wall_s": delivered_s,
                "nearest_low_range_m": nearest if math.isfinite(nearest) else None,
                "low_range_m": [float(x) if np.isfinite(x) else None
                                for x in rays[0]],
                "unsafe": unsafe})
            tick += 1
        search_result = future.result()
        worker_wall_s = time.perf_counter() - origin
    old_rgb = json.loads((ROOT / previous["search_trace"]).read_text())[
        "observations"][0]["rgb_file"]
    old_rgb_path = (ROOT / previous["search_trace"]).parent / old_rgb
    new_trace = json.loads((folder / "search/generic/trace.json").read_text())
    new_rgb_path = folder / "search/generic" / new_trace["observations"][0]["rgb_file"]
    different_image = sha(old_rgb_path) != sha(new_rgb_path)
    if not different_image:
        raise ValueError("Recovery repeated the interrupted RGB image")
    final_action = ("ask_owner_to_pause_for_clearance" if first_unsafe is not None
                    else search_result["action"])
    trace = {"schema": "ape.cross-room-post-interrupt-recovery.v1",
        "case": case["id"], "condition": condition,
        "interrupted_trace": case["interrupted_trace"],
        "fresh_clearance": gate,
        "pose_source": "unverified_command_integrated_displacement",
        "motion_target": None,
        "actor_new_actual_xz_m": new_xy.tolist(),
        "actor_command_estimate_xy_m": actor_estimate.tolist(),
        "hazard_stopped_xz_m": hazard_xy.tolist(),
        "search_start_time_s": start_time_s,
        "search_trace": relative(folder / "search/episode_trace.json"),
        "search_video": search_result["video"],
        "monitor_video": relative(monitor_video),
        "worker_wall_s": worker_wall_s,
        "monitor_packets": packets,
        "monitor_frames": frames,
        "monitor_max_wall_lateness_s": max_lateness,
        "first_unsafe_delivery_s": first_unsafe,
        "old_rgb_sha256": sha(old_rgb_path),
        "fresh_rgb_sha256": sha(new_rgb_path),
        "image_action_before_health_gate": search_result["action"],
        "action": final_action,
        "limits": "The entrant holds still after the first response. The sensor monitor remains live during a real second image worker but the same scripted owner handles candidate review inside that worker. New pose is command-integrated, not independently localized; no metric motion target, selected body, grasp or possession."}
    path = folder / "result.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "condition": condition,
        "clearance_accepted": gate_ok, "action": final_action,
        "fresh_image": different_image,
        "monitor_alarm": first_unsafe is not None,
        "search_video": search_result["video"],
        "monitor_video": relative(monitor_video),
        "result": relative(path)}


def main() -> None:
    frozen = json.loads(PRECOMMIT.read_text())
    for name, item in frozen["inputs"].items():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    rows = []
    for case in frozen["cases"]:
        if sha(ROOT / case["interrupted_trace"]) != case["interrupted_sha256"]:
            raise ValueError("Interrupted trace changed")
        for condition in frozen["conditions"]:
            print(case["id"], condition, flush=True)
            rows.append(run_case(case, condition))
            REPORT.write_text(json.dumps({"schema":
                "ape.cross-room-recovery-report.v1",
                "precommit": relative(PRECOMMIT), "rows": rows,
                "limits": frozen["limits"]}, indent=2) + "\n")
    print(json.dumps({"runs": len(rows),
        "clearance_accepted": sum(x["clearance_accepted"] for x in rows),
        "present_confirmed": sum(x["condition"] == "find_present" and
            x["action"] == "owner_confirmed_visual_identity" for x in rows),
        "absent_unknown": sum(x["condition"] == "find_absent" and
            x["action"] == "unidentified_ask_new_view_or_help" for x in rows)},
        indent=2))


if __name__ == "__main__":
    main()
