"""V347: a moving item challenges person-presence and naive motion cues.

The actor uses only captured RGB, frozen DINO proposals, and capture times.
The scripted item trajectory and person/negative labels are evaluator inputs.
This is a diagnostic replay, not a motion authorization controller.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

import challenge_person_belief_v346 as challenge
from active_scan import DETECTOR
from articulated_person_scene import (NEAR_HAND_XYZ_M, START_HAND_XYZ_M,
                                      add_person)
from person_belief import PersonBelief
from run import ROOT, sha
from run_late_corridor_intrusion_v323 import LateCorridorRoom


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v347"
PREVIOUS = ROOT / "artifacts/ape-closed-loop-cove50-v346/person-belief-challenge.json"
MOTION_PIXEL_DELTA = 40
MOTION_PIXEL_COUNT = 1000


class MovingItemRoom(LateCorridorRoom):
    def _update_moving_obstacle(self) -> None:
        super()._update_moving_obstacle()
        fraction = np.clip((self.time_s - 16.2) / .4, 0., 1.)
        location = (START_HAND_XYZ_M + fraction *
                    (NEAR_HAND_XYZ_M - START_HAND_XYZ_M))
        self.model.body_pos[self.model.body("loop_TomatoSauce").id] = (
            location + np.asarray((0., -.08, 0.)))


def actor_motion(row: dict) -> dict:
    belief = PersonBelief()
    previous_rgb = None
    motion_counts = []
    any_motion = False
    any_limb = False
    last_state = None
    for sample in row["actor_samples"]:
        packet = np.load(ROOT / sample["actor_packet_path"])
        lower = packet["lower_rgb"].astype(np.int16)
        if previous_rgb is None:
            moving = 0
        else:
            delta = np.max(np.abs(lower - previous_rgb), axis=2)
            moving = int(np.count_nonzero(
                delta[:, 288:] > MOTION_PIXEL_DELTA))
        previous_rgb = lower
        motion_counts.append(moving)
        any_motion |= moving >= MOTION_PIXEL_COUNT
        last_state = belief.ingest(
            sample["time_s"],
            sample["actor_upper_person_detections"],
            sample["actor_lower_limb_detections"])
        any_limb |= last_state["lower_limb_state"] == "visible_now"
    confirmed = last_state["person_presence"] == "confirmed"
    return {
        "case": row["case"], "positive_person": row["positive_person"],
        "person_state": last_state["person_presence"],
        "lower_right_motion_pixels_per_capture": motion_counts,
        "visual_motion_cue": any_motion,
        "lower_limb_seen_once": any_limb,
        "person_and_motion": confirmed and any_motion,
        "person_motion_and_limb": confirmed and any_motion and any_limb,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(DETECTOR)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        DETECTOR).to(device).eval()
    original_room, original_out = challenge.LateCorridorRoom, challenge.OUT
    original_coatstand = challenge.add_coatstand
    try:
        challenge.LateCorridorRoom = MovingItemRoom
        challenge.OUT = OUT
        moving_rows = []
        for kind, stand_factory in (
            ("coatstand", original_coatstand),
            ("static_mannequin", add_person),
        ):
            challenge.add_coatstand = stand_factory
            for seed in challenge.SEEDS:
                moving_rows.append(challenge.run_case(
                    seed, {"name": f"{kind}_moving_item",
                           "light_scale": 1., "head_yaw_rad": 1.2,
                           "coatstand": True},
                    False, processor, model, device))
    finally:
        challenge.LateCorridorRoom = original_room
        challenge.OUT = original_out
        challenge.add_coatstand = original_coatstand
    previous = json.loads(PREVIOUS.read_text())
    rows = [actor_motion(row) for row in previous["rows"] + moving_rows]
    def tally(name: str) -> dict:
        return {
            "positive_triggered": sum(r["positive_person"] and r[name]
                                      for r in rows),
            "negative_triggered": sum(not r["positive_person"] and r[name]
                                      for r in rows),
        }
    report = {
        "scope": "Frozen V345 visual person belief challenged by four additional no-person scenes: a static torso/head jacket stand or a static full-body arm mannequin, each with an unrelated moving can following the near-hand path in two seeds. Diagnostic motion counts come only from successive lower RGB frames. This cannot establish human agency, ownership, contact safety, or generalized specificity.",
        "source_sha256": sha(SOURCE),
        "v346_source_sha256": sha(Path(challenge.__file__)),
        "v346_report_sha256": sha(PREVIOUS),
        "mannequin_scene_source_sha256": sha(Path(add_person.__code__.co_filename)),
        "detector_weights_sha256": sha(DETECTOR / "model.safetensors"),
        "motion_rule": {"lower_image_x_min_px": 288,
                        "channel_delta_min": MOTION_PIXEL_DELTA,
                        "changed_pixels_min": MOTION_PIXEL_COUNT,
                        "scope": "exploratory right-half lower-camera motion cue"},
        "rows": rows,
        "score": {"person_and_motion": tally("person_and_motion"),
                  "person_motion_and_limb": tally("person_motion_and_limb")},
        "moving_negative_rows": moving_rows,
    }
    (OUT / "interaction-motion-challenge.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(report["score"], flush=True)


if __name__ == "__main__":
    main()
