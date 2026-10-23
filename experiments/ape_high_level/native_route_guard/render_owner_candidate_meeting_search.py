"""Evaluator-only comparison of three policies on the changed prop turn."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_candidate_meeting_search as trial
import render_owner_surface_track_escape as renderer


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.runner.original()["summary"]
    modes = ("side_only", "nearest_always", "surface_track")
    labels = ("side-only watch", "early nearest escape",
              "surface-track escape")
    items = [{**next(row for row in report["results"]
                     if row["policy"] == mode and
                     row["path"] == "new_west_south_turn"),
              "case": trial.v54.CASE} for mode in modes]
    times = np.arange(min(item["run"]["resume_time_s"] for item in items),
                      max(item["run"]["last_time_s"] for item in items)+.0001,
                      .1)
    video = trial.OUT / "comparisons" / "changed-turn-three-policies.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(renderer.v50.PATHS)
    try:
        renderer.v50.PATHS.update(trial.PATHS)
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(item, float(t), label, summary)
                    for item, label in zip(items, labels)], axis=1))
    finally:
        renderer.v50.PATHS.clear()
        renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-candidate-search-video.v1",
              "report_sha256": renderer.sha(report_path),
              "source_sha256": renderer.sha(__file__),
              "video": {"path": str(video), "sha256": renderer.sha(video),
                        "frames": len(times), "shape": [620, 1620, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
