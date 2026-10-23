"""Replace the ideal 20 ms guard with discrete 100 ms complete scans.

This is a conservative complete-scan timing abstraction, not a C1 device
model: a spinning lidar can stream individual rays between whole scans.
The prop still follows the V423 trajectory, including after a stop.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import run_robocasa_moving_prop_v423 as trial
from audit_robocasa_moving_prop_v424 import audit_episode
from audit_robocasa_post_stop_v427 import replay
from run_robocasa_layout2_route_v407 import sha256


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-scan-cadence-v428"
NOMINAL_RANGE = trial.raw_nearest_range


def complete_scan_sensor(phase: int):
    call = 0
    last = math.inf

    def measure(model, data, pose):
        nonlocal call, last
        if call % 5 == phase:
            last = NOMINAL_RANGE(model, data, pose)
        call += 1
        return last

    return measure


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for phase in range(5):
        label = f"complete_scan_100ms_phase{phase}"
        trial.raw_nearest_range = complete_scan_sensor(phase)
        trial.OUT = OUT / label
        episode = trial.episode(True)
        immediate_audit = audit_episode(episode)
        held = replay(label, episode)
        case = {"mode": label, "episode": episode,
                "immediate_audit": {key: value for key, value in immediate_audit.items()
                                    if key != "closest"},
                "held_pose": {key: value for key, value in held.items() if key != "trace"}}
        cases.append(case)
        print(label, episode["outcome"],
              "stop_gap", round(episode["min_true_prop_gap_m"], 4),
              "post_stop_gap", round(held["minimum_post_stop_gap_m"], 4), flush=True)
    trial.raw_nearest_range = NOMINAL_RANGE
    report = {"source_sha256": sha256(Path(__file__)),
              "trial_source_sha256": sha256(Path(trial.__file__)),
              "nominal_guard_period_s": .1,
              "action_substep_s": trial.SUBSTEP_S,
              "scope": "ideal discrete complete scans at 100ms with five phase offsets; no streamed rays, sensor packaging or physical braking",
              "cases": cases}
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
