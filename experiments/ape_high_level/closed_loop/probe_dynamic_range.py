"""Probe range-change tracking on saved v14 actor pose/timing traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dynamic_range import DynamicRangeTracker
from run import ROOT, Room, sensed_range


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[35, 61, 62, 63, 64])
    args = parser.parse_args()
    source = ROOT / "artifacts/ape-closed-loop-cove50-v14"
    rows = []
    room = Room()
    try:
        for seed in args.seeds:
            for scenario in ("baseline", "continuous_obstacle"):
                path = (source / f"{scenario}-seed{seed}-0-chest_rgbd_range_recovery-none"
                        / "trace.json")
                if not path.exists():
                    continue
                trace = json.loads(path.read_text())
                room.reset()
                room.layout(seed, absent=False)
                if scenario == "continuous_obstacle":
                    room.configure_continuous_obstacle(seed)
                tracker = DynamicRangeTracker()
                seen_times = set()
                for event in trace["events"]:
                    t = event["t_s"]
                    if t in seen_times:
                        continue
                    seen_times.add(t)
                    pose = np.asarray(event["pose_estimate_xyyaw"])
                    room.actual[:] = pose
                    room.estimated[:] = pose
                    room.time_s = t
                    room._update_moving_obstacle()
                    room._forward()
                    observation = tracker.update(
                        sensed_range(room, seed, int(round(t * 10))), pose, t)
                    rows.append({"seed": seed, "scenario": scenario, "t_s": t,
                                 "range_tracker": observation,
                                 "prop_xy_evaluator_only": room.obstacle_xy[3].tolist()})
    finally:
        room.close()
    output = source / "dynamic-range-probe.json"
    output.write_text(json.dumps({"rows": rows,
                                  "note": "Offline actor-pose replay; prop position is scorer-only."},
                                 indent=2) + "\n")
    for row in rows:
        if row["seed"] == 64 and row["scenario"] == "continuous_obstacle":
            state = row["range_tracker"]
            print(row["t_s"], state["status"], state["center_xy_m"],
                  state["velocity_xy_m_s"],
                  state["projected_surface_distance_0p4s_m"],
                  "prop", row["prop_xy_evaluator_only"])
    print(output)


if __name__ == "__main__":
    main()
