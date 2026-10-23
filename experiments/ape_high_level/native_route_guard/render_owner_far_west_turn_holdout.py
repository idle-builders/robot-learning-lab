"""Evaluator plan-view contrast for the V60/V61 frozen-policy turn change."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_far_west_turn_holdout as trial
import render_owner_deferred_show as deferred_renderer
import render_owner_surface_track_escape as renderer


def main():
    old_report = json.loads((trial.v60.OUT / "report.json").read_text())
    old = next(row for row in old_report["results"]
               if row["path"] == "new_west_south_turn")
    new_report = json.loads((trial.OUT / "report.json").read_text())
    new = {"path": "far_west_turn", "run": new_report["run"]}
    summary = trial.v60.runner.original()["summary"]
    times = np.arange(min(old["run"]["resume_time_s"],
                          new["run"]["resume_time_s"]),
                      max(old["run"]["last_time_s"],
                          new["run"]["last_time_s"])+.0001, .1)
    path = trial.OUT / "comparisons" / "bounded-vs-far-west-turn.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(renderer.v50.PATHS)
    try:
        renderer.v50.PATHS.update(trial.v60.v59.v56.PATHS)
        renderer.v50.PATHS["far_west_turn"] = trial.END_XY_M
        with imageio.get_writer(path, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    deferred_renderer.panel(old, float(t),
                                            "V60 west + defer", summary, True),
                    deferred_renderer.panel(new, float(t),
                                            "V61 farther-west turn", summary, True)],
                    axis=1))
    finally:
        renderer.v50.PATHS.clear()
        renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-far-west-holdout-video.v1",
              "v60_report_sha256": renderer.sha(trial.v60.OUT / "report.json"),
              "v61_report_sha256": renderer.sha(trial.OUT / "report.json"),
              "source_sha256": renderer.sha(__file__),
              "video": {"path": str(path), "sha256": renderer.sha(path),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    manifest = trial.OUT / "comparison.json"
    manifest.write_text(json.dumps(result, indent=2)+"\n")
    print(manifest)


if __name__ == "__main__":
    main()
