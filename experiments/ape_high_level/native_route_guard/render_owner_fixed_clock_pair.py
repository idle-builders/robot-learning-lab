"""Matched-clock evaluator plan videos for the owner policy comparison."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

import owner_fixed_clock_pair as trial
import render_owner_surface_track_escape as renderer


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.runner.original()["summary"]
    videos = {}
    for route in trial.v50.PATHS:
        items = [next(item for item in report["results"]
                      if item["policy"] == mode and item["path"] == route)
                 for mode in trial.POLICIES]
        labels = ("side-only watch", "early nearest escape",
                  "late surface-track escape")
        times = np.arange(min(item["run"]["resume_time_s"] for item in items),
                          max(item["run"]["last_time_s"] for item in items)+.0001,
                          .1)
        video = trial.OUT / "comparisons" / f"{route}-matched-clock.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(item, float(t), label, summary)
                    for item, label in zip(items, labels)], axis=1))
        videos[route] = {"path": str(video), "sha256": renderer.sha(video),
                         "frames": len(times), "shape": [620, 1620, 3],
                         "kind": "evaluator_plan_view_from_20ms_samples"}
    result = {"schema": "robot-learning-lab.ape-fixed-clock-videos.v1",
              "report_sha256": renderer.sha(report_path),
              "source_sha256": renderer.sha(__file__), "videos": videos}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
