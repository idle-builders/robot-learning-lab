"""Matched late-owner approach with healthy versus absent guard range packets.

Only the 30 cm scan during asynchronous person inference is changed. The
outage begins before the first guard tick and ends when inference closes.
This is an injected sensor-health question in the kinematic development rig.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image

import audit_robocasa_first_return_person_reacquire_v578 as audit
import robocasa_first_return_person_reacquire_turn_v576 as turn
import robocasa_person_interaction as interaction
import run_robocasa_first_return_person_reacquire_v577 as baseline
import run_robocasa_joined_owner_route_v410 as owner
import run_robocasa_layout2_route_v407 as route
import run_robocasa_person_joined_v497 as joined
from run_robocasa_dynamic_suite_v439 import motion


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-reacquire-range-loss-v591"
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
        for condition in ("healthy", "outage"):
            in_worker = threading.Event()
            missing = [0]
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
                if condition == "outage" and in_worker.is_set():
                    missing[0] += 1
                    return [None] * len(packet)
                return packet

            interaction.SHOW_HAND = np.asarray([1.30, -2.98, .55])
            turn.SCAN = scan
            turn.ThreadPoolExecutor = GuardPool
            turn.OWNER_APPROACH_M = 0.
            turn.FIRST_RETURN_OWNER_M = .4
            turn.FIRST_RETURN_OWNER_S = 1.
            turn.LATE_OWNER_APPROACH_M = .8
            turn.SECOND_RETURN_OWNER_M = 0.
            turn.DEPTH_NOISE_SIGMA_M = 0.
            turn.DEPTH_DROPOUT_FRACTION = 0.
            turn.PROP_SCRIPT = motion(case)
            turn.PACKET_OUT = OUT / f"{condition}-depth-packets.npz"
            response = f"{RESPONSE}_{condition}"
            ep = joined.episode(case, response, True, prior, policy,
                                vision, exemplar)
            ep["guard_range_outage_packets_injected"] = missing[0]
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
        "scope": "Matched full late-owner RoboCasa episode; the sole sensor change is complete 30 cm range loss during asynchronous person detection. Kinematic ape, scripted owner; injected health fault, not an automatic diagnosis or qualified physical stop.",
        "episodes": episodes,
        "independent_geometry_audit_source_sha256": sha256(audit.__file__),
        "independent_geometry_audits": audits,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
