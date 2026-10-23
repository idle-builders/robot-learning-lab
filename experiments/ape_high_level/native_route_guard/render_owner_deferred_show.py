"""Evaluator plan-view videos: V56 completed show versus V60 deferral."""

from __future__ import annotations

import json

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_deferred_show as trial
import render_owner_surface_track_escape as renderer


def panel(item, time_s, title, summary, deferred):
    if deferred:
        run = {**item["run"], "second_inference_start_s": item["run"][
            "scripted_unprompted_return_clock_s"]}
        item = {**item, "run": run, "case": trial.v54.CASE}
    else:
        item = {**item, "case": trial.v54.CASE}
    image = renderer.panel(item, time_s, title, summary)
    if deferred:
        cv2.rectangle(image, (0, 36), (539, 65), (16, 22, 27), -1)
        phase = ("move west" if time_s < item["run"][
            "scripted_unprompted_return_clock_s"]-.6 else
            "deferred show + range watch")
        cv2.putText(image, f"t={time_s:.2f}s  {phase}", (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, .40, (190, 207, 214), 1)
    return image


def main():
    old_report_path = trial.v59.v56.OUT / "report.json"
    new_report_path = trial.OUT / "report.json"
    old = json.loads(old_report_path.read_text())
    new = json.loads(new_report_path.read_text())
    summary = trial.runner.original()["summary"]
    outputs = {}
    old_paths = dict(renderer.v50.PATHS)
    try:
        renderer.v50.PATHS.update(trial.v59.v56.PATHS)
        for name in ("new_west_south_turn", "outbound_control"):
            old_item = next(row for row in old["results"]
                            if row["policy"] == "surface_track" and
                            row["path"] == name)
            new_item = next(row for row in new["results"]
                            if row["path"] == name)
            times = np.arange(min(old_item["run"]["resume_time_s"],
                                  new_item["run"]["resume_time_s"]),
                              max(old_item["run"]["last_time_s"],
                                  new_item["run"]["last_time_s"])+.0001, .1)
            video = trial.OUT / "comparisons" / f"{name}-show-vs-defer.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_writer(video, fps=10, codec="libx264",
                                    macro_block_size=None) as writer:
                for t in times:
                    writer.append_data(np.concatenate([
                        panel(old_item, float(t), "V56 southwest show",
                              summary, False),
                        panel(new_item, float(t), "V60 west + defer",
                              summary, True)], axis=1))
            outputs[name] = {"path": str(video),
                             "sha256": renderer.sha(video),
                             "frames": len(times),
                             "shape": [620, 1080, 3],
                             "kind": "evaluator_plan_view_from_20ms_samples"}
    finally:
        renderer.v50.PATHS.clear()
        renderer.v50.PATHS.update(old_paths)
    result = {"schema": "robot-learning-lab.ape-deferred-show-comparison.v1",
              "v56_report_sha256": renderer.sha(old_report_path),
              "v60_report_sha256": renderer.sha(new_report_path),
              "source_sha256": renderer.sha(__file__),
              "comparisons": outputs}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
