"""Replay existing path guard with exact kinematic eye pose and real saved scans.

The exact commanded eye pose is an idealized ablation, not a sensor input
available to an articulated ape. Actions remain the original saved actions.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from fan_view_route import PathTimedOccupancyGuard


EPISODES = (143, 185, 155, 177, 194)


def source(root: Path, version: int, filename: str) -> Path:
    matches = glob.glob(str(root / f"*-v{version}" / "evidence" / filename))
    if len(matches) != 1:
        raise FileNotFoundError((version, filename, matches))
    return Path(matches[0])


def replay(root: Path, version: int) -> dict:
    filenames = ("range-fan-trace.json", "continuous-vision-monitor.json",
                 "online-action-trace.json")
    paths = {name: source(root, version, name) for name in filenames}
    packets = {name: json.loads(path.read_text()) for name, path in paths.items()}
    ranges = packets[filenames[0]]["frames"]
    monitor = packets[filenames[1]]["frames"]
    actions = packets[filenames[2]]["actions"]
    guard = PathTimedOccupancyGuard(clearance_m=.20, horizon_s=.8)
    angles = np.arange(-90, 95, 5, dtype=float)
    rows = []
    for frame, (scan, checks, action) in enumerate(zip(ranges, monitor, actions)):
        pose = np.asarray(scan["planes"][0]["origin_xyz_m"][:2], dtype=float)
        heading = scan["body_heading_deg_commanded"]
        low = np.asarray([beam["distance_m"] if beam["distance_m"] is not None
                          else np.nan for beam in scan["planes"][0]["beams"]])
        guard.observe(low, angles, pose, heading, scan["timeline_time_s"])
        for sample in checks["samples"]:
            low = np.asarray([value if value is not None else np.nan
                              for value in sample["range_m_by_plane"]["0"]])
            guard.observe(low, angles, pose, heading, sample["time_s"])
        latest = (checks["samples"][-1]["time_s"] if checks["samples"]
                  else scan["timeline_time_s"])
        assessment = guard.assess(
            pose, heading, action["command_for_next_frame"]["forward_step_m"],
            max(0., action["decision_time_s"] - latest))
        rows.append({"frame": frame, "latest_low_scan_time_s": latest,
                     "decision_time_s": action["decision_time_s"],
                     **assessment})
    return {"version": version, "rows": rows,
            "first_alarm_frame": next((row["frame"] for row in rows
                                       if row["action"] != "allow"), None),
            "sources": {name: {"path": str(path),
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in paths.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/ape-realistic-people"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = [replay(args.root, version) for version in EPISODES]
    result = {"scope": "exact kinematic eye pose + actual saved scans, unchanged actions; diagnostic only",
              "guard": "PathTimedOccupancyGuard, 0.20m clearance, 0.8s horizon",
              "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({item["version"]: item["first_alarm_frame"]
                      for item in episodes}, indent=2))


if __name__ == "__main__":
    main()
