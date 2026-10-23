"""Show baseline and V44 meeting-pose geometry on a common clock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_scanned_meeting_pose as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(item, scenario, summary, target_time, title):
    run = item["run"]
    samples = run["evaluator_gap_samples"]
    t = min(target_time, run["last_time_s"])
    row = min(samples, key=lambda sample: abs(sample["time_s"]-t))
    t = row["time_s"]
    ape = np.asarray(row["ape_xy_m"])
    person = trial.v38.human_xy(item["case"], summary, t,
                                run["request_time_s"],
                                run["second_inference_start_s"])
    prop = trial.v41.v40.prop_at(t, run["checkpoint_time_s"], scenario)
    image = np.full((680, 680, 3), (16, 22, 27), dtype=np.uint8)
    bounds = (.35, 2.75, -5.55, -3.95)

    def xy(pos):
        return (int(55+570*(pos[0]-bounds[0])/(bounds[1]-bounds[0])),
                int(575-450*(pos[1]-bounds[2])/(bounds[3]-bounds[2])))

    cv2.putText(image, title, (20, 32), cv2.FONT_HERSHEY_SIMPLEX,
                .72, (240, 244, 246), 2)
    cv2.putText(image, f"t={t:.2f}s; evaluator geometry", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 205, 212), 1)
    for x in np.arange(.5, 2.8, .25):
        cv2.line(image, xy((x, bounds[2])), xy((x, bounds[3])),
                 (38, 46, 52), 1)
    for y in np.arange(-5.5, -3.9, .25):
        cv2.line(image, xy((bounds[0], y)), xy((bounds[1], y)),
                 (38, 46, 52), 1)
    path = np.asarray([xy(sample["ape_xy_m"]) for sample in samples
                       if sample["time_s"] <= t])
    if len(path) > 1:
        cv2.polylines(image, [path], False, (70, 170, 100), 3, cv2.LINE_AA)
    for name, point, radius, color in (
            ("prop", prop, 32, (65, 180, 230)),
            ("person", person, 26, (85, 110, 235)),
            ("ape", ape, 33, (90, 205, 105))):
        center = xy(point)
        cv2.circle(image, center, radius, color, -1)
        cv2.putText(image, name, (center[0]+12, center[1]-radius-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .53, color, 2)
    for x, name, gap in ((20, "person", row["gap_m"]),
                         (335, "prop", row["prop_gap_m"])):
        color = (90, 205, 105) if gap >= .05 else (80, 90, 245)
        cv2.putText(image, f"{name}: {100*gap:.1f} cm", (x, 625),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return image


def main():
    new_path = trial.OUT / "report.json"
    old_path = trial.v43.OUT / "report.json"
    new = json.loads(new_path.read_text())
    old = json.loads(old_path.read_text())
    summary = trial.v38.original()["summary"]
    videos = {}
    for prop_case in ("development", "early_entry", "holdout_closer_finish"):
        case = "blind_return_flank_fast_response"
        baseline = next(item for item in old["results"] if item["prop_case"] ==
                        prop_case and item["case"] == case and
                        item["mode"] == "baseline")
        restage = next(item for item in new["results"] if item["prop_case"] ==
                       prop_case and item["case"] == case)
        end = min(baseline["run"]["last_time_s"],
                  restage["run"]["last_time_s"])
        times = np.arange(baseline["run"]["resume_time_s"], end+.0001, .1)
        path = trial.OUT / "comparisons" / \
            f"{prop_case}-baseline-vs-scanned-pose.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=10, codec="libx264",
                                macro_block_size=None) as writer:
            for t in times:
                left = panel(baseline, trial.v41.PROP_CASES[prop_case],
                             summary, float(t), "Baseline: react after show")
                right = panel(restage, trial.v41.PROP_CASES[prop_case],
                              summary, float(t), "V44: scan-checked meeting pose")
                writer.append_data(np.concatenate((left, right), axis=1))
        videos[prop_case] = {"path": str(path), "sha256": sha(path),
                             "frames": len(times), "shape": [680, 1360, 3],
                             "kind": "evaluator_plan_view_from_20ms_samples"}
    result = {"schema": "robot-learning-lab.ape-owner-scanned-pose-comparisons.v1",
              "v43_report_sha256": sha(old_path),
              "v44_report_sha256": sha(new_path),
              "source_sha256": sha(__file__), "videos": videos}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
