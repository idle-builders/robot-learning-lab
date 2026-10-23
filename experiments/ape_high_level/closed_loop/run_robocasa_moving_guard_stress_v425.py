"""Stress the 20 ms moving-prop range stop with noise and missing reads."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import run_robocasa_moving_prop_v423 as trial
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-moving-guard-stress-v425"
original_range = trial.raw_nearest_range


def guard_sensor(mode: str):
    calls = 0
    rng = np.random.default_rng(425)

    def measure(model, data, pose):
        nonlocal calls
        value = original_range(model, data, pose)
        # The first intrusion substep is call 700 in the matched baseline.
        if mode == "missing_60ms" and 700 <= calls < 703:
            value = math.inf
        elif mode == "missing_100ms" and 700 <= calls < 705:
            value = math.inf
        elif mode == "noise_5mm" and math.isfinite(value):
            value = max(.001, value + rng.normal(0., .005))
        elif mode == "noise_15mm" and math.isfinite(value):
            value = max(.001, value + rng.normal(0., .015))
        calls += 1
        return value

    return measure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for mode in ("nominal", "noise_5mm", "noise_15mm",
                 "missing_60ms", "missing_100ms"):
        trial.raw_nearest_range = original_range if mode == "nominal" else guard_sensor(mode)
        trial.OUT = OUT / mode
        episode = trial.episode(True)
        cases.append({"mode": mode, "episode": episode})
        print(mode, episode["outcome"], episode["steps"],
              round(episode["min_true_prop_gap_m"], 4),
              round(episode["end_time_s"], 3), flush=True)
    report = {"source_sha256": sha256(Path(__file__)),
              "frozen_trial_source_sha256": sha256(Path(trial.__file__)),
              "guard_missing_window": "three or five consecutive 20ms reads beginning at intrusion",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
