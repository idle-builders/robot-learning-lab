"""Evaluator plan view: low-only versus mid-only deferred hazard watch."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_deferred_range_height_ablation as trial
import render_owner_deferred_show as renderer
import render_owner_surface_track_escape as base_renderer


def main():
    report = json.loads((trial.OUT / "report.json").read_text())
    low = next(row for row in report["results"]
               if row["profile"] == "low_05cm" and
               row["base_path"] == "far_west_turn" and
               row["finish_after_checkpoint_s"] == 5.20)
    mid = next(row for row in report["results"]
               if row["profile"] == "mid_30cm" and
               row["base_path"] == "far_west_turn" and
               row["finish_after_checkpoint_s"] == 5.20)
    left = {"path": "far_west_turn", "run": low["run"]}
    right = {"path": "far_west_turn", "run": mid["run"]}
    summary = trial.runner.original()["summary"]
    times = np.arange(min(left["run"]["resume_time_s"],
                          right["run"]["resume_time_s"]),
                      max(left["run"]["last_time_s"],
                          right["run"]["last_time_s"])+.0001, .1)
    video = trial.OUT / "comparisons" / "far-west-low-vs-mid-range.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(base_renderer.v50.PATHS)
    try:
        base_renderer.v50.PATHS["far_west_turn"] = trial.v64.PATHS["far_west_turn"]
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(left, float(t), "5 cm monitor",
                                   summary, True),
                    renderer.panel(right, float(t), "30 cm monitor",
                                   summary, True)], axis=1))
    finally:
        base_renderer.v50.PATHS.clear()
        base_renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-deferred-range-height-video.v1",
              "report_sha256": base_renderer.sha(trial.OUT / "report.json"),
              "source_sha256": base_renderer.sha(__file__),
              "video": {"path": str(video), "sha256": base_renderer.sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
