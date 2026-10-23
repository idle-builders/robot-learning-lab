"""Check the old alarm against actual scans made during visual inference."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import random

from near_body_watchdog import NearBodyWatchdog


EPISODES = (143, 185, 155, 177, 194)


def packet(root: Path, version: int, filename: str) -> dict:
    matches = glob.glob(str(root / f"*-v{version}" / "evidence" / filename))
    if len(matches) != 1:
        raise FileNotFoundError((version, filename, matches))
    return json.loads(Path(matches[0]).read_text())


def run(root: Path, version: int, seed: int | None = None,
        noise: float = 0., dropout: float = 0.) -> dict:
    capture = packet(root, version, "range-fan-trace.json")["frames"]
    monitor = packet(root, version, "continuous-vision-monitor.json")["frames"]
    actions = packet(root, version, "online-action-trace.json")["actions"]
    assert len(capture) == len(monitor) == len(actions)
    rng = random.Random(seed)
    alarm = NearBodyWatchdog()
    first = None
    observations = []

    def fault(values: list[float | None]) -> list[float | None]:
        if seed is None:
            return values
        return [None if value is None or rng.random() < dropout
                else max(.001, value + rng.gauss(0., noise * value))
                for value in values]

    for frame, (scan, checks) in enumerate(zip(capture, monitor)):
        prior_command = actions[frame - 1]["command_for_next_frame"] if frame else {}
        # A body turn changes which object/face is nearest. Without inertial
        # ego-motion, the scalar trend must be reset rather than invented.
        if abs(prior_command.get("body_yaw_delta_deg", 0.)) > 2.:
            alarm.reset_motion_history()
        readings = [(scan["timeline_time_s"],
                     [beam["distance_m"] for beam in scan["planes"][0]["beams"]],
                     prior_command.get("forward_step_m", 0.))]
        readings.extend((sample["time_s"], sample["range_m_by_plane"]["0"], 0.)
                        for sample in checks["samples"])
        for time_s, low, translation in readings:
            result = alarm.observe(fault(low), time_s, translation)
            observations.append({"frame": frame, "time_s": time_s, **result})
            if first is None and result["alarm"]:
                first = {"frame": frame, "time_s": round(time_s, 3),
                         "reason": result["reason"]}
    return {"version": version, "first_alarm": first,
            "sample_count": len(observations), "observations": observations}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/ape-realistic-people"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    baseline = [run(args.root, v) for v in EPISODES]
    faults = []
    for noise, dropout in ((.05, .05), (.10, .10)):
        for version in EPISODES:
            trials = [run(args.root, version, seed=seed,
                          noise=noise, dropout=dropout)["first_alarm"]
                      for seed in range(20)]
            faults.append({"version": version, "noise": noise,
                           "dropout": dropout, "alarms": trials})
    sources = {}
    for version in EPISODES:
        sources[str(version)] = {}
        for filename in ("range-fan-trace.json", "continuous-vision-monitor.json",
                         "online-action-trace.json"):
            paths = glob.glob(str(args.root / f"*-v{version}" / "evidence" / filename))
            assert len(paths) == 1
            path = Path(paths[0])
            sources[str(version)][filename] = {"path": str(path),
                                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {"scope": "actual opportunistic inference-time scans from old trajectories; no altered actions",
              "method": "V198 spatial rule; trend reset after commanded body yaw >2deg without measured inertial transform",
              "sources": sources,
              "baseline": baseline, "faults": faults}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"baseline": {x["version"]: x["first_alarm"] for x in baseline},
                      "faults": [{"version": x["version"], "noise": x["noise"],
                                  "alarm_count": sum(y is not None for y in x["alarms"]),
                                  "by_frame_11": sum(y is not None and y["frame"] <= 11
                                                     for y in x["alarms"])}
                                 for x in faults]}, indent=2))


if __name__ == "__main__":
    main()
