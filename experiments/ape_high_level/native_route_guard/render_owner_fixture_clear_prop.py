"""Two matched V50 paths on a shared evaluator clock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_fixture_clear_prop as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def panel(item, time_s, summary):
    run = item["run"]
    ended = time_s > run["last_time_s"]
    sample = min(run["evaluator_gap_samples"],
                 key=lambda row: abs(row["time_s"]-min(time_s,
                                                        run["last_time_s"])))
    ape = np.asarray(sample["ape_xy_m"])
    person = trial.v38.human_xy(
        item["case"], summary, time_s, run["request_time_s"],
        run["second_inference_start_s"])
    prop = trial.prop_at(time_s, run["checkpoint_time_s"],
                         trial.PATHS[item["path"]])
    image = np.full((620, 540, 3), (16, 22, 27), dtype=np.uint8)
    bounds = (.45, 2.35, -5.55, -4.0)

    def xy(point):
        return (int(35+475*(point[0]-bounds[0])/(bounds[1]-bounds[0])),
                int(545-430*(point[1]-bounds[2])/(bounds[3]-bounds[2])))

    title = "outward control" if item["path"] == "outbound_control" else \
        "inward turn"
    cv2.putText(image, title, (15, 29), cv2.FONT_HERSHEY_SIMPLEX,
                .67, (240, 245, 247), 2)
    phase = (run["outcome"] if ended else
             "meeting pose" if run["second_inference_start_s"] is None or
             time_s < run["second_inference_start_s"] else
             "owner inference + live watch")
    cv2.putText(image, f"t={time_s:.2f}s  {phase[:34]}", (15, 54),
                cv2.FONT_HERSHEY_SIMPLEX, .40, (190, 207, 214), 1)
    cv2.putText(image, "Evaluator plan view; not actor input", (15, 77),
                cv2.FONT_HERSHEY_SIMPLEX, .39, (150, 173, 186), 1)
    for x in np.arange(.5, 2.35, .25):
        cv2.line(image, xy((x, bounds[2])), xy((x, bounds[3])),
                 (38, 46, 52), 1)
    for y in np.arange(-5.5, -4., .25):
        cv2.line(image, xy((bounds[0], y)), xy((bounds[1], y)),
                 (38, 46, 52), 1)
    route = np.asarray([xy(row["ape_xy_m"]) for row in
                        run["evaluator_gap_samples"]
                        if row["time_s"] <= time_s])
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
    return image


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.v38.original()["summary"]
    case = "parked_after_clear"
    items = [next(row for row in report["results"] if row["path"] == path
                  and row["case"] == case) for path in trial.PATHS]
    times = np.arange(min(row["run"]["resume_time_s"] for row in items),
                      max(row["run"]["last_time_s"] for row in items)+.0001,
                      .1)
    video = trial.OUT / "comparisons" / "outward-vs-inward-owner.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            writer.append_data(np.concatenate([
                panel(item, float(t), summary) for item in items], axis=1))
    result = {"schema": "robot-learning-lab.ape-fixture-clear-video.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(__file__),
              "video": {"path": str(video), "sha256": sha(video),
                        "frames": len(times), "shape": [620, 1080, 3],
                        "kind": "evaluator_plan_view_from_20ms_samples"}}
    path = trial.OUT / "comparison.json"
    path.write_text(json.dumps(result, indent=2)+"\n")
    print(path)


if __name__ == "__main__":
    main()
