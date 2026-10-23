"""Render an evaluator-only plan view beside V41's complete camera videos."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_joint_hazard_response as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def draw(case, item, report, summary, *, output_root=None):
    run = item["run"]
    samples = run["evaluator_gap_samples"]
    watch = [row for row in json.loads(Path(run["trace"]["path"]).read_text())
             if row["stage"] == "second_inference_range_watch"]
    switch = next((row["time_s"] for row in watch
                   if row["response_step_info"] is not None and
                   row["response_step_info"]["action"] ==
                   "swerve_southwest"), None)
    out = (output_root or trial.OUT) / "comparisons" / \
        f"{case}-overhead-gap-replay.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    bounds = (.75, 2.65, -5.45, -4.0)

    def position(xy):
        x0, x1, y0, y1 = bounds
        return (int(110+570*(xy[0]-x0)/(x1-x0)),
                int(595-510*(xy[1]-y0)/(y1-y0)))

    with imageio.get_writer(out, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for target_time in np.arange(22., 24.81, .05):
            row = min(samples, key=lambda sample: abs(
                sample["time_s"]-target_time))
            t = row["time_s"]
            ape = np.asarray(row["ape_xy_m"])
            person = trial.v40.v38.human_xy(
                "blind_return_flank_fast_response", summary, t,
                run["request_time_s"], run["second_inference_start_s"])
            prop = trial.v40.prop_at(t, report["checkpoint_s"],
                                     trial.PROP_CASES[case])
            image = np.full((680, 800, 3), (16, 22, 26), dtype=np.uint8)
            cv2.putText(image, f"{case} / t={t:.2f}s", (20, 31),
                        cv2.FONT_HERSHEY_SIMPLEX, .67, (235, 240, 245), 2)
            cv2.putText(image,
                        "EVALUATOR PLAN VIEW: schematic icons; MuJoCo gaps",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, .53,
                        (190, 205, 212), 1)
            for x in np.arange(1., 2.6, .25):
                cv2.line(image, position((x, bounds[2])),
                         position((x, bounds[3])), (37, 45, 50), 1)
            for y in np.arange(-5.25, -4.0, .25):
                cv2.line(image, position((bounds[0], y)),
                         position((bounds[1], y)), (37, 45, 50), 1)
            trajectory = [position(sample["ape_xy_m"]) for sample in samples
                          if sample["time_s"] <= t]
            if len(trajectory) > 1:
                cv2.polylines(image, [np.asarray(trajectory)], False,
                              (65, 180, 100), 3, cv2.LINE_AA)
            cv2.circle(image, position(prop), 48, (55, 180, 220), -1)
            cv2.circle(image, position(person), 23, (75, 110, 230), -1)
            cv2.circle(image, position(ape), 35, (90, 205, 105), -1)
            for name, xy, color in (("prop", prop, (70, 210, 240)),
                                    ("person", person, (120, 140, 245)),
                                    ("ape", ape, (115, 230, 135))):
                at = position(xy)
                cv2.putText(image, name, (at[0]+14, at[1]-38),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            if switch is not None and t >= switch:
                cv2.putText(image, "range closure: southwest response",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, .58,
                            (110, 215, 230), 1)
            for column, label, gap in ((20, "person", row["gap_m"]),
                                       (390, "prop", row["prop_gap_m"])):
                color = ((90, 205, 105) if gap >= .05 else (95, 95, 240))
                cv2.putText(image, f"{label} gap {100*gap:.1f} cm",
                            (column, 646), cv2.FONT_HERSHEY_SIMPLEX,
                            .7, color, 2)
            writer.append_data(image)
    return {"path": str(out), "sha256": sha(out),
            "frames": len(np.arange(22., 24.81, .05)),
            "shape": [680, 800, 3],
            "kind": "evaluator_replay_from_20ms_gap_samples",
            "source_trace_sha256": run["trace"]["sha256"]}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    summary = trial.v40.v38.original()["summary"]
    videos = {item["prop_case"]: draw(item["prop_case"], item, report,
                                       summary) for item in report["results"]}
    result = {"schema": "robot-learning-lab.ape-owner-joint-hazard-comparisons.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(Path(__file__)),
              "overhead_videos": videos}
    (trial.OUT / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
