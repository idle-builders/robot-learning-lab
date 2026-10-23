"""Evaluator plan comparison of the short and longer owner meeting poses."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_longer_meeting_pose as trial
import render_owner_surface_track_escape as renderer


def main():
    short_path = trial.v54.OUT / "report.json"
    long_path = trial.OUT / "report.json"
    short = json.loads(short_path.read_text())
    long = json.loads(long_path.read_text())
    summary = trial.runner.original()["summary"]
    short_item = next(item for item in short["results"]
                      if item["policy"] == "side_only" and
                      item["path"] == "inbound_holdout")
    long_item = {**next(item for item in long["results"]
                        if item["policy"] == "side_only" and
                        item["path"] == "inbound_holdout"),
                 "case": trial.v54.CASE}
    items = [short_item, long_item]
    times = np.arange(min(item["run"]["resume_time_s"] for item in items),
                      max(item["run"]["last_time_s"] for item in items)+.0001,
                      .1)
    video = trial.OUT / "comparisons" / "inward-short-vs-long-meeting.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            writer.append_data(np.concatenate([
                renderer.panel(item, float(t), label, summary)
                for item, label in zip(items,
                    ("35 cm meeting pose", "60 cm meeting pose"))], axis=1))
    result = {"schema": "robot-learning-lab.ape-meeting-distance-video.v1",
              "report_sha256": [renderer.sha(short_path),
                                renderer.sha(long_path)],
              "source_sha256": renderer.sha(__file__),
              "video": {"path": str(video), "sha256": renderer.sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
