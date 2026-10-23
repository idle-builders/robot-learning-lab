"""Evaluator plan view of the same far-west turn: nearest versus all clusters."""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np

import owner_multi_surface_escape as trial
import owner_far_west_reactive_escape as v62
import render_owner_surface_track_escape as renderer


def main():
    old = json.loads((v62.OUT / "report.json").read_text())
    new = json.loads((trial.OUT / "report.json").read_text())
    left = {"case": trial.v54.CASE, "path": "far_west_turn",
            "run": old["run"]}
    right = {**next(row for row in new["results"]
                    if row["path"] == "far_west_turn"),
             "case": trial.v54.CASE}
    summary = trial.runner.original()["summary"]
    times = np.arange(min(left["run"]["resume_time_s"],
                          right["run"]["resume_time_s"]),
                      max(left["run"]["last_time_s"],
                          right["run"]["last_time_s"])+.0001, .1)
    path = trial.OUT / "comparisons" / "far-west-nearest-vs-multi-surface.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    old_paths = dict(renderer.v50.PATHS)
    try:
        renderer.v50.PATHS["far_west_turn"] = trial.v61.END_XY_M
        with imageio.get_writer(path, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                writer.append_data(np.concatenate([
                    renderer.panel(left, float(t),
                                   "V62 nearest surface", summary),
                    renderer.panel(right, float(t),
                                   "V63 multiple surfaces", summary)], axis=1))
    finally:
        renderer.v50.PATHS.clear()
        renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-multi-surface-video.v1",
              "v62_report_sha256": renderer.sha(v62.OUT / "report.json"),
              "v63_report_sha256": renderer.sha(trial.OUT / "report.json"),
              "source_sha256": renderer.sha(__file__),
              "video": {"path": str(path), "sha256": renderer.sha(path),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    manifest = trial.OUT / "comparison.json"
    manifest.write_text(json.dumps(result, indent=2)+"\n")
    print(manifest)


if __name__ == "__main__":
    main()
