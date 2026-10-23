"""Render V46's three scan planes on one evaluator-only timeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_low_prop_scan_planes as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(item, time_s, summary, config):
    run = item["run"]
    ended = time_s > run["last_time_s"]
    t = min(time_s, run["last_time_s"])
    sample = min(run["evaluator_gap_samples"],
                 key=lambda row: abs(row["time_s"]-t))
    t = sample["time_s"]
    ape = np.asarray(sample["ape_xy_m"])
    person = trial.v38.human_xy(item["case"], summary, t,
                                run["request_time_s"],
                                run["second_inference_start_s"])
    prop = trial.v41.v40.prop_at(t, run["checkpoint_time_s"], config)
    image = np.full((620, 540, 3), (16, 22, 27), dtype=np.uint8)
    bounds = (.45, 2.25, -5.48, -4.05)

    def xy(point):
        return (int(35+475*(point[0]-bounds[0])/(bounds[1]-bounds[0])),
                int(545-430*(point[1]-bounds[2])/(bounds[3]-bounds[2])))

    cv2.putText(image, item["profile"], (15, 29),
                cv2.FONT_HERSHEY_SIMPLEX, .66, (240, 245, 247), 2)
    phase = (run["outcome"] if ended else
             "moving to meeting pose" if run["second_inference_start_s"] is None
             or t < run["second_inference_start_s"] else
             "owner inference + live watch")
    cv2.putText(image, f"t={t:.2f}s  {phase[:33]}", (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX, .41, (190, 207, 214), 1)
    cv2.putText(image, "Evaluator plan view; not actor input", (15, 77),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (150, 173, 186), 1)
    for x in np.arange(.5, 2.3, .25):
        cv2.line(image, xy((x, bounds[2])), xy((x, bounds[3])),
                 (38, 46, 52), 1)
    for y in np.arange(-5.4, -4.0, .25):
        cv2.line(image, xy((bounds[0], y)), xy((bounds[1], y)),
                 (38, 46, 52), 1)
    route = np.asarray([xy(row["ape_xy_m"]) for row in
                        run["evaluator_gap_samples"] if row["time_s"] <= t])
    if len(route) > 1:
        cv2.polylines(image, [route], False, (65, 165, 100), 2, cv2.LINE_AA)
    for name, point, radius, color in (
            ("low prop", prop, 27, (55, 180, 235)),
            ("person", person, 22, (85, 115, 235)),
            ("ape", ape, 28, (90, 205, 105))):
        center = xy(point)
        cv2.circle(image, center, radius, color, -1)
        cv2.putText(image, name, (center[0]+6, center[1]-radius-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
    color = ((90, 205, 105) if sample["prop_gap_m"] >= .05 else
             (80, 90, 245))
    cv2.putText(image, f"prop gap: {sample['prop_gap_m']*100:.1f} cm",
                (15, 586), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2)
    if ended:
        cv2.putText(image, "EPISODE ENDED", (320, 586),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 215, 220), 1)
    return image


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.v38.original()["summary"]
    case = "parked_after_clear"
    items = [next(item for item in report["results"] if item["profile"] ==
                  profile and item["case"] == case)
             for profile in trial.PROFILES]
    start_s = min(item["run"]["resume_time_s"] for item in items)
    end_s = max(item["run"]["last_time_s"] for item in items)
    times = np.arange(start_s, end_s+.0001, .1)
    video = trial.OUT / "comparisons" / "low-prop-mid-vs-low-vs-dual.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            frames = [panel(item, float(t), summary,
                            report["prop_motion"]) for item in items]
            writer.append_data(np.concatenate(frames, axis=1))
    result = {"schema": "robot-learning-lab.ape-owner-low-prop-comparison.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(__file__),
              "video": {"path": str(video), "sha256": sha(video),
                        "frames": len(times), "shape": [620, 1620, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
