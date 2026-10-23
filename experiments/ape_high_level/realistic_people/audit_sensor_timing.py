"""Summarize wall-time costs in a captured Isaac sensor-to-decision loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(run_dir: Path) -> dict:
    actions = json.loads((run_dir / "evidence/online-action-trace.json").read_text())["actions"]
    if not actions or any("sensor_pipeline_wall_s" not in row for row in actions):
        raise ValueError("Run has no instrumented sensor timing")
    keys = ("frame_ready_to_actor_s", "lidar_processing_s", "range_fan_s",
            "frame_ready_to_control_read_s", "control_read_start_to_actor_s",
            "social_decision_s", "owner_decision_s", "frame_ready_to_command_s",
            "control_read_start_to_command_s")
    rows = []
    for action in actions:
        pipeline = action["sensor_pipeline_wall_s"]
        row = {"frame": action["frame"]}
        row.update({key: pipeline[key] for key in keys if key in pipeline})
        for camera, phases in pipeline["camera_phases_s"].items():
            for phase, duration in phases.items():
                row[f"{camera}_{phase}"] = duration
        social = action.get("social_actor") or {}
        row["social_fast_frame_age_s"] = social.get("frame_age_s")
        rows.append(row)
    metrics = {}
    for key in rows[0]:
        if key == "frame":
            continue
        values = np.array([row[key] for row in rows if row.get(key) is not None],
                          dtype=float)
        if values.size:
            metrics[key] = {"count": int(values.size), "median_s": float(np.median(values)),
                            "p95_s": float(np.percentile(values, 95)),
                            "max_s": float(np.max(values))}
    return {"run": str(run_dir), "frames": len(rows), "metrics": metrics, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir)
    (args.run_dir / "sensor-timing-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
