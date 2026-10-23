"""Inject a one-beam heading fault into the frozen range-route actor.

The physical room and camera path do not rotate. After frame 20, reported
beam bearings are shifted by 5 degrees either consistently or alternating.
This approximates a bad capture-time IMU/head transform, not an IMU model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_robocasa_layout2_localized_route_v408 as actor
from run_robocasa_layout2_route_v407 import frame_with_map as layout2_frame
from run_robocasa_layout2_route_v407 import model_with_layout2, sha256
from run_robocasa_layout4_transfer_v412 import frame_layout4, layout4_scene


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-heading-fault-v417"
FAULT_STEP = 20
BEAM_DEG = 360 / 72
original_scan = actor.planar_scan


def biased_scan(mode: str):
    calls = 0

    def read(model, data, xy, max_range_m, rng):
        nonlocal calls
        scan = original_scan(model, data, xy, max_range_m, rng)
        shift = 0
        if calls >= FAULT_STEP:
            shift = 1 if mode == "fixed_plus5" else (1 if calls % 2 == 0 else -1)
            scan = np.roll(np.asarray(scan, dtype=object), shift).tolist()
        calls += 1
        return scan

    return read


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for layout in (2, 4):
        for mode in ("fixed_plus5", "alternating_plusminus5"):
            actor.model_with_layout2 = model_with_layout2 if layout == 2 else layout4_scene
            actor.frame_with_map = layout2_frame if layout == 2 else frame_layout4
            actor.planar_scan = biased_scan(mode)
            actor.OUT = OUT / f"layout{layout}-{mode}"
            episode = actor.episode(5., .30, "range_growing")
            cases.append({"layout": layout, "mode": mode, "fault_step": FAULT_STEP,
                          "beam_shift_deg": BEAM_DEG,
                          "episode": episode})
            print(layout, mode, episode["outcome"], episode["steps"],
                  round(episode["end_pose_error_m"], 3),
                  episode["unsafe_requested_steps"], flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_actor_source_sha256": sha256(Path(actor.__file__)),
              "fault": "reported scan bearing shifted after frame20; true camera/body heading fixed",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
