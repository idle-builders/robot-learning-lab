"""Evaluator plan-view comparison of fresh versus unflagged stale range."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_deferred_stale_range as v68
import render_owner_deferred_show as renderer
import render_owner_surface_track_escape as base_renderer


def main():
    healthy = json.loads((v68.v64.OUT / "report.json").read_text())
    stale = json.loads((v68.OUT / "report.json").read_text())
    left = next(x for x in healthy["results"] if x["path"] ==
                "far_west_turn")
    right = next(x for x in stale["results"] if x["path"] ==
                 "far_west_turn")
    summary = v68.runner.original()["summary"]
    times = np.arange(min(left["run"]["resume_time_s"],
                          right["run"]["resume_time_s"]),
                      max(left["run"]["last_time_s"],
                          right["run"]["last_time_s"])+.0001, .1)
    video = v68.OUT / "comparisons/far-west-fresh-vs-stale.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(base_renderer.v50.PATHS)
    try:
        base_renderer.v50.PATHS["far_west_turn"] = \
            v68.v64.PATHS["far_west_turn"]
        with imageio.get_writer(video, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(left, float(t), "fresh 10 Hz range",
                                   summary, True),
                    renderer.panel(right, float(t), "0.6 s unflagged stale",
                                   summary, True)], axis=1))
    finally:
        base_renderer.v50.PATHS.clear()
        base_renderer.v50.PATHS.update(old_paths)
    manifest = {"schema": "robot-learning-lab.ape-deferred-stale-video.v1",
                "v64_report_sha256": base_renderer.sha(v68.v64.OUT /
                                                          "report.json"),
                "v68_report_sha256": base_renderer.sha(v68.OUT / "report.json"),
                "source_sha256": base_renderer.sha(__file__),
                "video": {"path": str(video),
                          "sha256": base_renderer.sha(video),
                          "frames": len(times), "shape": [620, 1080, 3],
                          "kind": "evaluator_plan_view_from_20ms_samples"}}
    (v68.OUT / "comparison.json").write_text(json.dumps(manifest,
                                                            indent=2)+"\n")
    print(v68.OUT / "comparison.json")


if __name__ == "__main__":
    main()
