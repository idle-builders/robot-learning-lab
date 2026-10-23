"""Healthy late-moving and stationary owner controls for the sector alarm.

The other 30 cm beams still return during asynchronous person inference.
This is an injected coverage question in the kinematic development rig.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np
from PIL import Image

import audit_robocasa_first_return_person_reacquire_v578 as audit
import robocasa_first_return_person_reacquire_turn_v594 as turn
import robocasa_person_interaction as interaction
import run_robocasa_first_return_person_reacquire_v577 as baseline
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
import run_robocasa_layout2_multitrack_v485 as multi
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-reacquire-sector-healthy-v595"
RESPONSE = "early_owner_mid_sector_blackout_late_inference"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    policy = json.loads(joined.CONFIG.read_text())
    case = next(row for row in policy["cases"] if row["id"] == "late_crossing")
    prior = np.load(joined.PRIOR)
    OUT.mkdir(parents=True, exist_ok=True)
    joined.OUT = OUT
    joined.owner_turn = turn.owner_turn
    route.MARGIN_M = .15
    vision = owner.Vision(teach_selector="generic-proposal")
    exemplar = vision.feature([Image.open(owner.TEACH_CROP).convert("RGB")])[0]
    old_hand = interaction.SHOW_HAND.copy()
    old_scan = turn.SCAN
    old_pool = turn.ThreadPoolExecutor
    episodes = []
    audits = []
    try:
        for condition, response, late_approach in (
                ("healthy_late", f"{RESPONSE}_healthy", .8),
                ("healthy_stationary", "stationary_owner_mid_sector_blackout_healthy", 0.)):
            in_worker = threading.Event()
            missing = [0]
            sector_center = [None]
            masked_beams = [0]
            real_scan = baseline.owner_scan(True)

            class GuardPool(ThreadPoolExecutor):
                def submit(self, *args, **kwargs):
                    in_worker.set()
                    return super().submit(*args, **kwargs)

                def __exit__(self, *args):
                    try:
                        return super().__exit__(*args)
                    finally:
                        in_worker.clear()

            def scan(model, data, xy, max_range_m, rng):
                packet = real_scan(model, data, xy, max_range_m, rng)
                if condition == "sector_outage" and in_worker.is_set():
                    if sector_center[0] is None:
                        tracks = multi.CAPTURED[-1].multi_history[-1]["tracks"]
                        sector_center[0] = np.asarray(max(
                            tracks, key=lambda row: row["hits"])["center_xy_m"])
                    missing[0] += 1
                    direction = sector_center[0] - np.asarray(xy)
                    bearing = math.atan2(direction[1], direction[0])
                    angles = np.linspace(-math.pi, math.pi, len(packet), endpoint=False)
                    delta = (angles - bearing + math.pi) % (2 * math.pi) - math.pi
                    mask = np.abs(delta) <= math.radians(40.)
                    masked_beams[0] += int(np.count_nonzero(mask))
                    return [None if mask[index] else value
                            for index, value in enumerate(packet)]
                return packet

            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = scan
            turn.ThreadPoolExecutor = GuardPool
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .4
            turn.FIRST_RETURN_OWNER_S = 1.
            turn.LATE_OWNER_APPROACH_M = late_approach
            turn.SECOND_RETURN_OWNER_M = 0.
            turn.DEPTH_NOISE_SIGMA_M = 0.
            turn.DEPTH_DROPOUT_FRACTION = 0.
            turn.PROP_SCRIPT = motion(case)
            turn.PACKET_OUT = OUT / f"{condition}-depth-packets.npz"
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            ep["guard_range_outage_packets_injected"] = missing[0]
            ep["guard_range_owner_sector_masked_beams"] = masked_beams[0]
            ep["guard_range_owner_sector_half_width_degrees"] = 40.
            episodes.append(ep)
            geometry = audit.audit_episode(ep)
            geometry.pop("geometry_samples", None)
            audits.append(geometry)
            stages = {row["stage"] for row in ep["trace"]}
            print(condition, ep["outcome"], "injected", missing[0],
                  "cancelled", "owner_reacquire_image_cancelled" in stages,
                  "gap", geometry["person_exact_nearest"], flush=True)
    finally:
        interaction.SHOW_HAND = old_hand
        turn.SCAN = old_scan
        turn.ThreadPoolExecutor = old_pool
        route.MARGIN_M = .10
        route.LIDAR_Z_M = .30
    report = {
        "source_sha256": sha256(__file__),
        "joined_route_source_sha256": sha256(joined.__file__),
        "turn_source_sha256": sha256(turn.__file__),
        "person_detector_weights_sha256": sha256(turn.person_reacquire.DETECTOR / "model.safetensors"),
        "config_sha256": sha256(joined.CONFIG),
        "prior_map_sha256": sha256(joined.PRIOR),
        "scope": "Healthy mid-height range controls with the V594 sector-coverage alarm: late moving-owner interruption and stationary-owner full show/find after initial five-scan owner-sector blackout. No range loss during asynchronous detection. Kinematic ape, scripted owner; not a qualified physical stop or final sensor suite.",
        "episodes": episodes,
        "independent_geometry_audit_source_sha256": sha256(audit.__file__),
        "independent_geometry_audits": audits,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
