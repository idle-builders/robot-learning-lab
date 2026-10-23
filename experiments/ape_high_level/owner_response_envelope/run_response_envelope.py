"""Measure the virtual owner's required retreat response, not a body skill.

This reuses the complete V597 episode unchanged. Only the response speed while
the person detector is in flight and its configured response delay differ.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import sys
import threading

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/ape_high_level/closed_loop"))

import audit_robocasa_first_return_person_reacquire_v578 as audit
import robocasa_first_return_person_reacquire_turn_v597 as turn
import robocasa_person_interaction as interaction
import run_robocasa_first_return_person_reacquire_v577 as baseline
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_layout2_multitrack_v485 as multi
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


OUT = ROOT / "artifacts/ape-owner-response-envelope/v2"
CASES = (
    ("reference", .25, .2),
    ("slow_015", .15, .2),
    ("slow_010", .10, .2),
    ("slow_005", .05, .2),
    ("slow_0025", .025, .2),
    ("late_080", .25, .8),
    ("late_160", .25, 1.6),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(ep, geometry, speed, delay, submissions, fault_packets):
    trace = ep["trace"]
    alarms = [row for row in trace if row["stage"] ==
              "owner_reacquire_image_cancelled"]
    first_alarm = alarms[0] if alarms else None
    ticks = [row for row in trace if row["stage"] ==
             "owner_reacquire_guard_tick"]
    first_motion = None
    if ticks:
        first_xy = np.asarray(ticks[0]["ape_xy_m"])
        first_motion = next((row for row in ticks if
                             np.linalg.norm(np.asarray(row["ape_xy_m"]) -
                                            first_xy) >= .001), None)
    gap = geometry["person_exact_nearest"]["gap_m"]
    return {
        "response_speed_m_s": speed,
        "response_delay_s": delay,
        "detector_submissions": submissions,
        "frozen_mid_range_packets": fault_packets,
        "scene_fingerprint": ep["scene_fingerprint"],
        "pre_show_choice": next(row for row in trace if row["stage"] ==
                                "choose_pre_show_standoff"),
        "pre_show_end": next(row for row in trace if row["stage"] ==
                             "pre_show_motion_assessment"),
        "alarm_reason": first_alarm["alarm_reason"] if first_alarm else None,
        "alarm_time_s": first_alarm["alarm_time_s"] if first_alarm else None,
        "first_retreat_motion_time_s": (first_motion["time_s"] if
                                         first_motion else None),
        "observed_alarm_to_first_motion_s": (
            round(first_motion["time_s"] - first_alarm["alarm_time_s"], 4)
            if first_motion is not None and first_alarm is not None else None),
        "retreat_distance_at_cancel_m": (first_alarm["retreat_distance_m"]
                                          if first_alarm else None),
        "person_min_exact_gap_m": gap,
        "person_min_exact_gap_time_s": geometry["person_exact_nearest"]["time_s"],
        "person_5cm_proxy_margin_pass": gap >= .05,
        "episode_outcome": ep["outcome"],
        "video": ep["video"],
        "audit": geometry,
    }


def main():
    policy = json.loads(joined.CONFIG.read_text())
    scene = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.OUT = OUT
    joined.owner_turn = turn.owner_turn
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    original = {"hand": interaction.SHOW_HAND.copy(), "scan": turn.SCAN,
                "pool": turn.ThreadPoolExecutor,
                "speed": turn.BACKTRACK_SPEED_M_S,
                "delay": turn.REACQUIRE_RESPONSE_DELAY_S,
                "margin": route.MARGIN_M}
    results = []
    episodes = []
    try:
        for name, speed, delay in CASES:
            in_worker = threading.Event()
            submissions = [0]
            fault_packets = [0]
            sector_center = [None]
            frozen_packet = [None]
            real_scan = baseline.owner_scan(True)

            class GuardPool(ThreadPoolExecutor):
                def submit(self, *args, **kwargs):
                    submissions[0] += 1
                    # The pre-show backtrack uses the unchanged reference speed.
                    turn.BACKTRACK_SPEED_M_S = speed
                    in_worker.set()
                    return super().submit(*args, **kwargs)

                def __exit__(self, *args):
                    try:
                        return super().__exit__(*args)
                    finally:
                        turn.BACKTRACK_SPEED_M_S = original["speed"]
                        in_worker.clear()

            def scan(model, data, xy, max_range_m, rng):
                packet = real_scan(model, data, xy, max_range_m, rng)
                if not in_worker.is_set():
                    return packet
                if sector_center[0] is None:
                    tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
                    sector_center[0] = np.asarray(max(
                        tracks, key=lambda row: row["hits"])["center_xy_m"])
                fault_packets[0] += 1
                direction = sector_center[0] - np.asarray(xy)
                bearing = math.atan2(direction[1], direction[0])
                angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
                delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
                mask = np.abs(delta) <= math.radians(40.)
                if frozen_packet[0] is None:
                    frozen_packet[0] = list(packet)
                return [frozen_packet[0][i] if mask[i] else value
                        for i, value in enumerate(packet)]

            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = scan
            turn.ThreadPoolExecutor = GuardPool
            turn.BACKTRACK_SPEED_M_S = original["speed"]
            turn.REACQUIRE_RESPONSE_DELAY_S = delay
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .4
            turn.FIRST_RETURN_OWNER_S = 1.
            turn.LATE_OWNER_APPROACH_M = .8
            turn.SECOND_RETURN_OWNER_M = 0.
            turn.DEPTH_NOISE_SIGMA_M = 0.
            turn.DEPTH_DROPOUT_FRACTION = 0.
            turn.PROP_SCRIPT = motion(scene)
            turn.PACKET_OUT = OUT / f"{name}-depth-packets.npz"
            response = f"owner_response_envelope_{name}"
            ep = joined.episode(scene, response, True, prior, policy,
                                vision, exemplar)
            ep["guard_range_fault_packets_injected"] = fault_packets[0]
            ep["response_speed_m_s"] = speed
            ep["response_delay_s"] = delay
            geometry = audit.audit_episode(ep)
            geometry.pop("geometry_samples", None)
            result = summarize(ep, geometry, speed, delay,
                               submissions[0], fault_packets[0])
            results.append({"name": name, **result})
            episodes.append(ep)
            print(name, ep["outcome"],
                  "gap_cm", round(100 * result["person_min_exact_gap_m"], 2),
                  "alarm_to_motion_s", result["observed_alarm_to_first_motion_s"],
                  flush=True)
    finally:
        interaction.SHOW_HAND = original["hand"]
        turn.SCAN = original["scan"]
        turn.ThreadPoolExecutor = original["pool"]
        turn.BACKTRACK_SPEED_M_S = original["speed"]
        turn.REACQUIRE_RESPONSE_DELAY_S = original["delay"]
        route.MARGIN_M = .10
        route.LIDAR_Z_M = .30
    matrix_path = ROOT / "artifacts/ape-cove50-transfer-20261008/backward-varied300-matrix.json"
    native_matrix = json.loads(matrix_path.read_text())
    report = {
        "source_sha256": sha256(__file__),
        "owner_turn_source_sha256": sha256(turn.__file__),
        "joined_episode_source_sha256": sha256(joined.__file__),
        "geometry_audit_source_sha256": sha256(audit.__file__),
        "person_detector_weights_sha256": sha256(turn.person_reacquire.DETECTOR / "model.safetensors"),
        "config_sha256": sha256(joined.CONFIG),
        "prior_map_sha256": sha256(joined.PRIOR),
        "native_backward_matrix_path": str(matrix_path),
        "native_backward_matrix_sha256": sha256(matrix_path),
        "native_backward_checkpoint_sha256": native_matrix["checkpoint_sha256"],
        "scope": "One complete textured RoboCasa late-owner reference scene with scripted owner and kinematic ape. Frozen 30 cm owner sector; live ideal independent 45 cm upper range. Only in-worker retreat speed and response delay vary. 5 cm is a proxy simulation margin, not human or body safety qualification.",
        "cases": results,
        "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
