"""Evaluator-only plan comparison for the fixture-clear inward prop turn."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_fixture_clear_prop as v50
import owner_full_angle_escape as v51


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(item, time_s, title, summary):
    run = item["run"]
    ended = time_s > run["last_time_s"]
    sample = min(run["evaluator_gap_samples"],
                 key=lambda row: abs(row["time_s"]-min(time_s,
                                                        run["last_time_s"])))
    ape = np.asarray(sample["ape_xy_m"])
    person = v50.v38.human_xy(
        item["case"], summary, time_s, run["request_time_s"],
        run["second_inference_start_s"])
    prop = v50.prop_at(time_s, run["checkpoint_time_s"],
                       v50.PATHS[item["path"]])
    frame = np.full((620, 540, 3), (16, 22, 27), dtype=np.uint8)
    bounds = (.45, 2.35, -5.55, -4.0)

    def xy(point):
        return (int(35+475*(point[0]-bounds[0])/(bounds[1]-bounds[0])),
                int(545-430*(point[1]-bounds[2])/(bounds[3]-bounds[2])))

    cv2.putText(frame, title, (15, 29), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (240, 245, 247), 2)
    phase = (run["outcome"] if ended else
             "meeting pose" if time_s < run["second_inference_start_s"] else
             "owner inference + live watch")
    cv2.putText(frame, f"t={time_s:.2f}s  {phase[:35]}", (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (190, 207, 214), 1)
    cv2.putText(frame, "Evaluator plan view; not actor input", (15, 77),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (150, 173, 186), 1)
    for x in np.arange(.5, 2.35, .25):
        cv2.line(frame, xy((x, bounds[2])), xy((x, bounds[3])),
                 (38, 46, 52), 1)
    for y in np.arange(-5.5, -4., .25):
        cv2.line(frame, xy((bounds[0], y)), xy((bounds[1], y)),
                 (38, 46, 52), 1)
    route = np.asarray([xy(row["ape_xy_m"]) for row in
                        run["evaluator_gap_samples"]
                        if row["time_s"] <= time_s])
    if len(route) > 1:
        cv2.polylines(frame, [route], False, (65, 165, 100), 2, cv2.LINE_AA)
    for name, point, radius, color in (
            ("low prop", prop, 27, (55, 180, 235)),
            ("person", person, 22, (85, 115, 235)),
            ("ape", ape, 28, (90, 205, 105))):
        center = xy(point)
        cv2.circle(frame, center, radius, color, -1)
        cv2.putText(frame, name, (center[0]+6, center[1]-radius-5),
                    cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
    color = ((90, 205, 105) if sample["prop_gap_m"] >= .05 else
             (80, 90, 245))
    cv2.putText(frame, f"prop gap: {sample['prop_gap_m']*100:.1f} cm",
                (15, 586), cv2.FONT_HERSHEY_SIMPLEX, .62, color, 2)
    return frame


def main():
    baseline_path = v50.OUT / "report.json"
    escape_path = v51.OUT / "report.json"
    baseline = json.loads(baseline_path.read_text())
    escape = json.loads(escape_path.read_text())
    summary = v50.v38.original()["summary"]
    key = ("inbound_holdout", "parked_after_clear")

    def select(report):
        return next(row for row in report["results"]
                    if (row["path"], row["case"]) == key)

    items = [select(baseline), select(escape)]
    times = np.arange(min(row["run"]["resume_time_s"] for row in items),
                      max(row["run"]["last_time_s"] for row in items)+.0001,
                      .1)
    video = v51.OUT / "comparisons" / "inward-turn-before-vs-escape.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            writer.append_data(np.concatenate([
                panel(item, float(t), label, summary)
                for item, label in zip(items, ("side-only watch", "all-angle escape"))
            ], axis=1))
    result = {"schema": "robot-learning-lab.ape-full-angle-video.v1",
              "baseline_report_sha256": sha(baseline_path),
              "escape_report_sha256": sha(escape_path),
              "source_sha256": sha(__file__),
              "video": {"path": str(video), "sha256": sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = v51.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
