"""Replay saved Isaac low scans through a sensor-only near-body alarm."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import random
from statistics import median

from near_body_watchdog import NearBodyWatchdog


EPISODES = {143: "moving_hazard", 185: "moving_hazard",
            155: "static_control", 177: "safe_moving_control",
            194: "healthy_gated_control"}


def episode_path(root: Path, version: int, filename: str) -> Path:
    matches = glob.glob(str(root / f"*-v{version}" / "evidence" / filename))
    if len(matches) != 1:
        raise FileNotFoundError((version, filename, matches))
    return Path(matches[0])


def run(root: Path, version: int, *, seed: int | None = None,
        noise_fraction: float = 0., dropout_fraction: float = 0.,
        samples_per_frame: int = 1) -> dict:
    ranges = json.loads(episode_path(root, version, "range-fan-trace.json").read_text())["frames"]
    actions = json.loads(episode_path(root, version, "online-action-trace.json").read_text())["actions"]
    assert len(ranges) == len(actions)
    rng = random.Random(seed)
    monitor = NearBodyWatchdog()
    first = None
    observations = []
    for index, (scan, action) in enumerate(zip(ranges, actions)):
        low = [beam["distance_m"] for beam in scan["planes"][0]["beams"]]
        if seed is not None:
            noisy = [[None if value is None or rng.random() < dropout_fraction
                      else max(.001, value + rng.gauss(0., noise_fraction * value))
                      for value in low] for _ in range(samples_per_frame)]
            combined = []
            for beam_index in range(len(low)):
                present = [sample[beam_index] for sample in noisy
                           if sample[beam_index] is not None]
                combined.append(median(present) if present else None)
            low = combined
        preceding = (actions[index - 1]["command_for_next_frame"]["forward_step_m"]
                     if index else 0.)
        result = monitor.observe(low, scan["timeline_time_s"], preceding)
        if result["alarm"] and first is None:
            first = index
        observations.append({"frame": index, **result})
    return {"version": version, "role": EPISODES[version],
            "first_alarm_frame": first, "observations": observations}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/ape-realistic-people"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = [run(args.root, version) for version in EPISODES]
    fault_screens = []
    for noise, dropout in ((.05, .05), (.10, .10)):
        for samples_per_frame in (1, 3, 5):
            for version in EPISODES:
                trials = [run(args.root, version, seed=seed,
                              noise_fraction=noise, dropout_fraction=dropout,
                              samples_per_frame=samples_per_frame)
                          for seed in range(20)]
                fault_screens.append({"version": version, "role": EPISODES[version],
                                      "noise_fraction": noise,
                                      "dropout_fraction": dropout,
                                      "samples_per_frame": samples_per_frame,
                                      "alarm_frames": [item["first_alarm_frame"]
                                                       for item in trials]})
    report = {"scope": "saved-packet counterfactual; no closed-loop response",
              "sources": {str(version): {name: {
                  "path": str(path),
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for name in ("range-fan-trace.json", "online-action-trace.json")
                  for path in [episode_path(args.root, version, name)]}
                  for version in EPISODES},
              "algorithm": {"low_height_m": .15,
                            "spatial_patch": "median of >=4 valid returns in 5 adjacent 5-degree beams",
                            "critical_patch_m": .27,
                            "closing_patch_m": .31,
                            "trend": "3 stationary observations, both declines >=0.005m, total >=0.035m within 0.8s",
                            "fault_model": "independent Gaussian fractional range noise and random return loss; fixed saved scene state for 1/3/5 scan repeats"},
              "baseline": baseline, "fault_screens": fault_screens}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"baseline": {row["version"]: row["first_alarm_frame"]
                                   for row in baseline},
                      "fault_screens": [{key: value for key, value in row.items()
                                         if key != "alarm_frames"} | {
                                         "alarms": sum(frame is not None
                                                       for frame in row["alarm_frames"]),
                                         "early_by_11": sum(frame is not None and frame <= 11
                                                            for frame in row["alarm_frames"])}
                                        for row in fault_screens]}, indent=2))


if __name__ == "__main__":
    main()
