"""Full furnished-owner turns at different map-supported pre-show distances.

The owner actor, recognition, guards, and virtual sensors are imported intact.
Only its pre-show reserve requirement and interrupt response delay vary.
The ape and owner translations remain scripted kinematic surrogates.
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

OUT = ROOT / "artifacts/ape-owner-standoff-response/v2"
CASES = (
    ("baseline_delayed", "moving", .35, 1.6),
    ("half_meter_delayed", "moving", .45, 1.6),
    ("six_tenths_delayed", "moving", .55, 1.6),
    ("six_tenths_prompt", "moving", .55, .2),
    ("baseline_stationary", "stationary", .35, .2),
    ("half_meter_stationary", "stationary", .45, .2),
    ("six_tenths_stationary", "stationary", .55, .2),
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
                "pool": turn.ThreadPoolExecutor, "reserve": turn.PRE_SHOW_RESERVE_M,
                "delay": turn.REACQUIRE_RESPONSE_DELAY_S}
    source_paths = (Path(__file__), Path(turn.__file__), Path(joined.__file__),
                    Path(audit.__file__), joined.CONFIG, joined.PRIOR)
    initial_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    rows, episodes = [], []
    try:
        for name, response_kind, reserve, delay in CASES:
            in_worker = threading.Event()
            frozen = [None]
            sector_center = [None]
            submissions = [0]
            fault_packets = [0]
            real_scan = baseline.owner_scan(True)

            class GuardPool(ThreadPoolExecutor):
                def submit(self, *args, **kwargs):
                    submissions[0] += 1
                    in_worker.set()
                    return super().submit(*args, **kwargs)

                def __exit__(self, *args):
                    try:
                        return super().__exit__(*args)
                    finally:
                        in_worker.clear()

            def scan(model, data, xy, max_range_m, rng):
                packet = real_scan(model, data, xy, max_range_m, rng)
                if not in_worker.is_set():
                    return packet
                if sector_center[0] is None:
                    track = max(multi.CAPTURED[-1].multi_history[-1]["tracks"],
                                key=lambda row: row["hits"])
                    sector_center[0] = np.asarray(track["center_xy_m"])
                fault_packets[0] += 1
                direction = sector_center[0] - np.asarray(xy)
                bearing = math.atan2(direction[1], direction[0])
                angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
                delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
                mask = np.abs(delta) <= math.radians(40.)
                if frozen[0] is None:
                    frozen[0] = list(packet)
                return [frozen[0][i] if mask[i] else value
                        for i, value in enumerate(packet)]

            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = scan
            turn.ThreadPoolExecutor = GuardPool
            turn.PRE_SHOW_RESERVE_M = reserve
            turn.REACQUIRE_RESPONSE_DELAY_S = delay
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .4
            turn.FIRST_RETURN_OWNER_S = 1.
            turn.LATE_OWNER_APPROACH_M = .8 if response_kind == "moving" else 0.
            turn.SECOND_RETURN_OWNER_M = 0.
            turn.DEPTH_NOISE_SIGMA_M = 0.
            turn.DEPTH_DROPOUT_FRACTION = 0.
            turn.PROP_SCRIPT = motion(scene)
            turn.PACKET_OUT = OUT / f"{name}-depth-packets.npz"
            ep = joined.episode(scene, f"{response_kind}_standoff_{name}",
                                True, prior, policy, vision, exemplar)
            geometry = audit.audit_episode(ep)
            geometry.pop("geometry_samples", None)
            trace = ep["trace"]
            chosen = next((x["chosen"] for x in trace if x["stage"] ==
                           "choose_pre_show_standoff"), None)
            assessment = next((x for x in trace if x["stage"] ==
                               "pre_show_motion_assessment"), None)
            alarm = next((x for x in trace if x["stage"] ==
                          "owner_reacquire_image_cancelled"), None)
            row = {
                "name": name, "response_kind": response_kind,
                "required_reserve_m": reserve, "response_delay_s": delay,
                "chosen_distance_m": chosen["distance_m"] if chosen else None,
                "chosen_reserve_m": chosen["reserve_m"] if chosen else None,
                "chosen_path_known_free": chosen["known_free_path"] if chosen else None,
                "pre_show_owner_track_healthy": (assessment.get("owner_track_healthy")
                                                 if assessment else None),
                "alarm_reason": alarm["alarm_reason"] if alarm else None,
                "alarm_time_s": alarm["alarm_time_s"] if alarm else None,
                "frozen_mid_sector_packets": fault_packets[0],
                "detector_submissions": submissions[0],
                "minimum_exact_person_gap_m": geometry["person_exact_nearest"]["gap_m"],
                "minimum_exact_person_gap_time_s": geometry["person_exact_nearest"]["time_s"],
                "five_cm_proxy_margin": geometry["person_exact_nearest"]["gap_m"] >= .05,
                "outcome": ep["outcome"], "video": ep["video"],
                "scene_fingerprint": ep["scene_fingerprint"], "geometry": geometry,
            }
            rows.append(row)
            episodes.append(ep)
            print(name, row["chosen_distance_m"], row["outcome"],
                  "gap_cm", round(100 * row["minimum_exact_person_gap_m"], 2),
                  "alarm", row["alarm_reason"], flush=True)
    finally:
        interaction.SHOW_HAND = original["hand"]
        turn.SCAN = original["scan"]
        turn.ThreadPoolExecutor = original["pool"]
        turn.PRE_SHOW_RESERVE_M = original["reserve"]
        turn.REACQUIRE_RESPONSE_DELAY_S = original["delay"]
        route.MARGIN_M = .10
        route.LIDAR_Z_M = .30
    final_hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    if initial_hashes != final_hashes:
        raise RuntimeError("Shared source changed during paired experiment")
    report = {
        "task": "T-j66rxy", "source_hashes": initial_hashes,
        "scope": "Complete textured RoboCasa owner episodes. 50 cm reference geometry, scripted owner and kinematic ape; no selected body or qualified retreat. The 5 cm gap is an evaluation proxy, not a human-safety threshold.",
        "cases": rows, "episodes": episodes,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
