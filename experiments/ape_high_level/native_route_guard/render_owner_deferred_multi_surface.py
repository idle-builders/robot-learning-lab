"""Evaluator plan view of held deferral versus live multi-surface response."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_deferred_multi_surface as trial
import owner_far_west_turn_holdout as v61
import render_owner_deferred_show as renderer
import render_owner_surface_track_escape as base_renderer


def main():
    old = json.loads((v61.OUT / "report.json").read_text())
    new = json.loads((trial.OUT / "report.json").read_text())
    left = {"path": "far_west_turn", "run": old["run"]}
    right = next(row for row in new["results"]
                 if row["path"] == "far_west_turn")
    summary = trial.runner.original()["summary"]
    times = np.arange(min(left["run"]["resume_time_s"],
                          right["run"]["resume_time_s"]),
                      max(left["run"]["last_time_s"],
                          right["run"]["last_time_s"])+.0001, .1)
    video = trial.OUT / "comparisons" / "far-west-held-vs-responsive-defer.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(base_renderer.v50.PATHS)
    try:
        base_renderer.v50.PATHS["far_west_turn"] = trial.PATHS["far_west_turn"]
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(left, float(t), "V61 defer + hold",
                                   summary, True),
                    renderer.panel(right, float(t), "V64 defer + respond",
                                   summary, True)], axis=1))
    finally:
        base_renderer.v50.PATHS.clear()
        base_renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-deferred-multi-surface-video.v1",
              "v61_report_sha256": base_renderer.sha(v61.OUT / "report.json"),
              "v64_report_sha256": base_renderer.sha(trial.OUT / "report.json"),
              "source_sha256": base_renderer.sha(__file__),
              "video": {"path": str(video), "sha256": base_renderer.sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    manifest = trial.OUT / "comparison.json"
    manifest.write_text(json.dumps(result, indent=2)+"\n")
    print(manifest)


if __name__ == "__main__":
    main()
