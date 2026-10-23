"""Evaluator plan-view comparison of normal and fast far-west turns."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_deferred_turn_speed_stress as trial
import render_owner_deferred_show as renderer
import render_owner_surface_track_escape as base_renderer


def main():
    base = json.loads((trial.v64.OUT / "report.json").read_text())
    stress = json.loads((trial.OUT / "report.json").read_text())
    left = next(row for row in base["results"]
                if row["path"] == "far_west_turn")
    right = next(row for row in stress["results"]
                 if row["variant"] == "far_west_fast")
    right = {"path": "far_west_turn", "run": right["run"]}
    summary = trial.runner.original()["summary"]
    times = np.arange(min(left["run"]["resume_time_s"],
                          right["run"]["resume_time_s"]),
                      max(left["run"]["last_time_s"],
                          right["run"]["last_time_s"])+.0001, .1)
    video = trial.OUT / "comparisons" / "far-west-normal-vs-fast.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(base_renderer.v50.PATHS)
    original_prop_at = base_renderer.v50.prop_at

    def fast_prop_at(time_s, checkpoint_s, endpoint):
        elapsed = time_s-checkpoint_s
        if elapsed <= trial.TURN_START_S:
            return original_prop_at(time_s, checkpoint_s, endpoint)
        u = float(np.clip((elapsed-trial.TURN_START_S)/
                          (4.85-trial.TURN_START_S), 0., 1.))
        eased = u*u*(3.-2.*u)
        return trial.TURN_START_XY_M+eased*(endpoint-trial.TURN_START_XY_M)

    try:
        base_renderer.v50.PATHS["far_west_turn"] = trial.v64.PATHS["far_west_turn"]
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                first = renderer.panel(left, float(t),
                                       "V64 normal final turn", summary, True)
                base_renderer.v50.prop_at = fast_prop_at
                second = renderer.panel(right, float(t),
                                        "V65 faster final turn", summary, True)
                base_renderer.v50.prop_at = original_prop_at
                writer.append_data(np.concatenate([first, second], axis=1))
    finally:
        base_renderer.v50.prop_at = original_prop_at
        base_renderer.v50.PATHS.clear()
        base_renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-deferred-turn-speed-video.v1",
              "v64_report_sha256": base_renderer.sha(trial.v64.OUT / "report.json"),
              "v65_report_sha256": base_renderer.sha(trial.OUT / "report.json"),
              "source_sha256": base_renderer.sha(__file__),
              "video": {"path": str(video), "sha256": base_renderer.sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
