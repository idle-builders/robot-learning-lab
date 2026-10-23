"""Timed shared-hazard teach/find interruption and fresh-item return.

The worker and independent near-body monitor render the same precommitted
hazard trajectory against separate instances of the same capture mesh. The
worker's in-flight image result is discarded after an alarm. A new search is
then captured at the response endpoint after the scripted hazard exits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import time

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES, BODY_RADIUS_M
import run_item_inference_response_19 as cohort
import run_cross_room_inference_response as prior
import run_real_capture_conditional_search as conditional


OUT = ROOT / "assets/real_indoor_scenes/shared_world_return_20261015"
PRECOMMIT = ART / "shared_world_return_precommit.json"
REPORT = ART / "shared_world_return_report.json"
TICK_S = .1
MONITOR_HZ = 5
MAX_WORKER_S = 30.


class TimedWorkerScene:
    """Render the worker's observations at the monitor's shared clock time."""

    def __init__(self, scene, origin: float, case: dict, state: dict,
                 start: np.ndarray, incoming: np.ndarray) -> None:
        self.scene = scene
        self.origin = origin
        self.case = case
        self.state = state
        self.start = start
        self.incoming = incoming
        self.task_props = {}
        self.captures = []

    def __getattr__(self, name):
        return getattr(self.scene, name)

    def set_props(self, placements):
        self.task_props = dict(placements)

    def _refresh(self, channel: str):
        elapsed = time.perf_counter() - self.origin
        hazard = prior.incoming_xy(self.start, self.incoming, elapsed)
        self.scene.set_props({**self.task_props,
            self.case["hazard_object"]: (self.state["hazard_prop"],
                                          tuple(hazard), 0.)})
        self.captures.append({"channel": channel,
            "capture_wall_s": round(elapsed, 6),
            "hazard_xz_m": hazard.tolist()})

    def observe(self, *args, **kwargs):
        self._refresh("rgbd")
        return self.scene.observe(*args, **kwargs)

    def range_fan(self, *args, **kwargs):
        self._refresh("range")
        return self.scene.range_fan(*args, **kwargs)


def search(state: dict, case: dict, scene, actual: np.ndarray,
           actor: np.ndarray, out, start_time_s: float) -> dict:
    stop = state["stop"]
    return conditional.original_run_search(
        scene, state["vision"], actual, state["actual_yaw"], actor,
        float(stop["actor_heading_deg"]), state["exemplar"],
        float(state["summary"]["teaching"]["acceptance_similarity"]),
        True, out, start_time_s=start_time_s,
        search_xz=tuple(state["placement"]["find_target_xz"]),
        search_distractor_xz=tuple(state["placement"]["find_distractor_xz"]),
        owner_start_xz=tuple(state["cfg"]["start_xz"]),
        target_name=case["target"], distractor_name=case["distractor"],
        identity_mode="multi_cue", colour_exemplar=state["colour"],
        colour_acceptance=float(state["summary"]["teaching"][
            "colour_acceptance_similarity"]))


def run_one(case: dict) -> dict:
    state = cohort.generalized_setup(case)
    folder = OUT / case["id"]
    folder.mkdir(parents=True, exist_ok=True)
    monitor = state["monitor_scene"]
    start = state["actual_xy"].copy()
    yaw = state["actual_yaw"]
    bearing = math.radians(yaw + case["entry_relative_deg"])
    incoming = np.array([math.sin(bearing), math.cos(bearing)])
    eye = np.array([start[0], monitor.floor_y + .42, start[1]])
    forward = np.array([state["forward"][0], -.58, state["forward"][1]])
    baseline = monitor.range_fan(eye, yaw, owner_xz=None, angles_deg=ANGLES)
    packets = []
    frames = []
    alarm_capture = alarm_delivery = alarm_angle = None
    escape = None
    origin = time.perf_counter()
    worker_scene = TimedWorkerScene(state["worker_scene"], origin, case,
                                    state, start, incoming)
    with ThreadPoolExecutor(max_workers=1) as pool, imageio.get_writer(
            str(folder / "monitor.mp4"), fps=MONITOR_HZ, codec="libx264",
            quality=6, macro_block_size=16) as writer:
        future = pool.submit(search, state, case, worker_scene, start,
            np.asarray(state["stop"]["actor_pose_xy_m"], float),
            folder / "interrupted_search",
            float(state["stop"]["decision_time_s"] + .4))
        tick = 0
        while not future.done():
            target = origin + tick * TICK_S
            remaining = target - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            capture_s = time.perf_counter() - origin
            if capture_s > MAX_WORKER_S:
                raise TimeoutError("Image worker exceeded timed trial")
            actor = prior.actor_xy(start, escape, alarm_delivery, capture_s)
            hazard = prior.incoming_xy(start, incoming, capture_s)
            monitor.set_props({**state["base_props"],
                case["hazard_object"]: (state["hazard_prop"], tuple(hazard), 0.)})
            at_eye = np.array([actor[0], monitor.floor_y + .42, actor[1]])
            rays = monitor.range_fan(at_eye, yaw, owner_xz=None,
                                     angles_deg=ANGLES)
            delivery_s = time.perf_counter() - origin
            with np.errstate(invalid="ignore"):
                changed = (baseline[0] - rays[0] > .08) & (rays[0] < .9)
            if alarm_delivery is None and np.any(changed):
                ids = np.flatnonzero(changed)
                selected = ids[np.argmin(rays[0, ids])]
                alarm_capture, alarm_delivery = capture_s, delivery_s
                alarm_angle = float(ANGLES[selected])
                opposite = math.radians(yaw + alarm_angle + 180.)
                escape = np.array([math.sin(opposite), math.cos(opposite)])
            gap = prior.proxy_gap(actor, hazard, state["hazard_radius_m"])
            packets.append({"tick": tick,
                "capture_wall_s": round(capture_s, 6),
                "delivery_wall_s": round(delivery_s, 6),
                "actor_xz_m": actor.tolist(), "hazard_xz_m": hazard.tolist(),
                "evaluator_proxy_gap_m": gap,
                "alarm": bool(np.any(changed)),
                "low_range_m": [float(x) if np.isfinite(x) else None
                                for x in rays[0]]})
            if tick % 2 == 0:
                rgb, depth = monitor.observe(at_eye, forward, owner_xz=None,
                                             width=160, height=90)
                display = prior.decorate(rgb, t=capture_s, label="shared world",
                    actor=actor, start=start, hazard=hazard, gap=gap,
                    alarm=alarm_delivery is not None)
                writer.append_data(display)
                frames.append({"tick": tick, "capture_wall_s": capture_s})
            tick += 1
        interrupted = future.result()
        worker_wall_s = time.perf_counter() - origin
    endpoint = prior.actor_xy(start, escape, alarm_delivery, worker_wall_s)
    # The external hazard leaves before the ape gathers new evidence. The
    # actor must use three new near-body packets at its reached pose.
    monitor.set_props(state["base_props"])
    clearance = []
    for _ in range(3):
        at_eye = np.array([endpoint[0], monitor.floor_y + .42, endpoint[1]])
        rays = monitor.range_fan(at_eye, yaw, owner_xz=None,
                                 angles_deg=ANGLES)
        finite = rays[0][np.isfinite(rays[0])]
        clearance.append(float(finite.min()) if len(finite) else None)
    clear = bool(all(x is not None and x > BODY_RADIUS_M + .05
                     for x in clearance))
    fresh = None
    if clear:
        state["worker_scene"].set_props(state["base_props"])
        actor_offset = endpoint - start
        actor = (np.asarray(state["stop"]["actor_pose_xy_m"], float)
                 + actor_offset)
        fresh = search(state, case, state["worker_scene"], endpoint,
            actor, folder / "fresh_search",
            float(state["stop"]["decision_time_s"] + .4 + worker_wall_s))
    scores = []
    for t in np.arange(0., worker_wall_s + .0001, .02):
        actor = prior.actor_xy(start, escape, alarm_delivery, float(t))
        hazard = prior.incoming_xy(start, incoming, float(t))
        scores.append(prior.proxy_gap(actor, hazard,
                                      state["hazard_radius_m"]))
    min_gap = min(scores)
    owner_answer = None
    if fresh is not None and fresh["action"] == "candidate_to_owner_for_confirmation":
        owner_answer = "yes" if not fresh["wrong_confident_candidate"] else "no"
    outcome = ("fresh_owner_confirmed" if owner_answer == "yes" else
               "owner_rejected_candidate" if owner_answer == "no" else
               "clearance_unavailable" if not clear else
               "fresh_item_unidentified")
    trace = {"schema": "ape.shared-world-return.v1",
        "case": case["id"], "entry_relative_deg": case["entry_relative_deg"],
        "scene_key": case["scene_key"], "source_summary": case["teaching_summary"],
        "interrupted_search_trace": relative(folder / "interrupted_search/trace.json"),
        "interrupted_search_action": interrupted["action"],
        "interrupted_search_invalidated": alarm_delivery is not None,
        "worker_world_captures": worker_scene.captures,
        "monitor_packets": packets, "monitor_video_frames": frames,
        "monitor_video": relative(folder / "monitor.mp4"),
        "alarm_capture_s": alarm_capture,
        "alarm_delivery_s": alarm_delivery,
        "alarm_relative_angle_deg": alarm_angle,
        "response_endpoint_xz_m": endpoint.tolist(),
        "worker_wall_s": worker_wall_s,
        "min_evaluator_proxy_gap_m": min_gap,
        "post_exit_low_range_min_m": clearance,
        "fresh_search_trace": (relative(folder / "fresh_search/trace.json")
                               if fresh is not None else None),
        "fresh_search_action": fresh["action"] if fresh is not None else None,
        "fresh_search_wrong_confident_candidate": (
            fresh["wrong_confident_candidate"] if fresh is not None else None),
        "scripted_owner_answer": owner_answer,
        "outcome": outcome,
        "limits": "Separate same-mesh renderers share one clock and prop trajectory. The in-flight search stays at its initial camera pose and is always invalidated on alarm. Hazard departure is scripted; response is an ideal disk at 0.25m/s after 0.1s. Clearance uses three immediate packets, not elapsed persistence. Owner answer uses evaluator overlap; no articulated motion, measured stop, person, physical pose, IMU or contact."}
    path = folder / "episode_trace.json"
    path.write_text(json.dumps(trace, indent=2) + "\n")
    return {"case": case["id"], "trace": relative(path),
        "outcome": outcome, "alarm": alarm_delivery is not None,
        "min_evaluator_proxy_gap_m": min_gap,
        "fresh_search_action": trace["fresh_search_action"],
        "scripted_owner_answer": owner_answer}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new", type=int, default=0)
    args = parser.parse_args()
    cases = cohort.case_rows()
    frozen = {"schema": "ape.shared-world-return-precommit.v1",
        "source_cohort": relative(cohort.PRECOMMIT),
        "source_cohort_sha256": sha(cohort.PRECOMMIT),
        "cases": cases,
        "policy": "Frozen generic-proposal multi-cue taught-item search; 360-degree two-height change monitor; invalidate image at delivered alarm; ideal opposite-bearing escape; scripted hazard departure; three fresh clearance packets; new item search and owner confirmation.",
        "attempt": "One complete interruption and return per screened real indoor room",
        "success": "Alarm before <5cm proxy gap and fresh correct owner-confirmed item candidate",
        "fallback": "Hold and ask when clearance or item evidence is unavailable",
        "limits": "19 screened of 20 real-capture rooms; no human hazard or physical gait/body qualification."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text()) != frozen:
            raise ValueError("Shared-world trial precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen, indent=2) + "\n")
    rows = []
    launched = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for case in cases:
        path = OUT / case["id"] / "episode_trace.json"
        if path.exists():
            trace = json.loads(path.read_text())
            row = {"case": case["id"], "trace": relative(path),
                "outcome": trace["outcome"],
                "alarm": trace["alarm_delivery_s"] is not None,
                "min_evaluator_proxy_gap_m": trace["min_evaluator_proxy_gap_m"],
                "fresh_search_action": trace["fresh_search_action"],
                "scripted_owner_answer": trace["scripted_owner_answer"]}
        else:
            if args.max_new and launched >= args.max_new:
                break
            row = run_one(case)
            launched += 1
        rows.append(row)
        REPORT.write_text(json.dumps({"schema": "ape.shared-world-return-report.v1",
            "precommit": relative(PRECOMMIT), "processed": len(rows),
            "rows": rows}, indent=2) + "\n")
        print(case["id"], row["outcome"], row["min_evaluator_proxy_gap_m"],
              flush=True)


if __name__ == "__main__":
    main()
