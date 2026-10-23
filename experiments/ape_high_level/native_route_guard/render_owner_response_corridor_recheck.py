"""Pair V40's open-loop and rechecked owner-response continuations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_response_corridor_recheck as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuation(run):
    reader = imageio.get_reader(run["video"]["path"])
    try:
        frames = [frame for index, frame in enumerate(reader)
                  if index >= run["source_video_frames"]]
    finally:
        reader.close()
    assert frames
    return frames


def draw_pair(prop_case, left, right):
    banks = (continuation(left["run"]), continuation(right["run"]))
    out = trial.OUT / "comparisons" / f"{prop_case}-open-vs-recheck.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    count = max(len(bank) for bank in banks)
    with imageio.get_writer(out, fps=5, codec="libx264",
                            macro_block_size=None) as writer:
        for index in range(count):
            canvas = np.full((390, 1280, 3), (16, 22, 25), dtype=np.uint8)
            for column, (item, bank) in enumerate(zip((left, right), banks)):
                x0 = 10+column*640
                image = bank[min(index, len(bank)-1)]
                canvas[62:377, x0:x0+620] = cv2.resize(image, (620, 315))
                mode = "open loop" if item["mode"] == "open_loop" else "range recheck"
                cv2.putText(canvas, f"{prop_case}: {mode}", (x0, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, .61, (235, 240, 245), 2)
                run = item["run"]
                line = (f"person {100*run['minimum_proxy_gap_m']:.1f}cm / "
                        f"prop {100*run['minimum_prop_gap_m']:.1f}cm min")
                cv2.putText(canvas, line, (x0, 49),
                            cv2.FONT_HERSHEY_SIMPLEX, .52, (160, 225, 220), 1)
            writer.append_data(canvas)
    return {"path": str(out), "sha256": sha(out), "frames": count,
            "shape": [390, 1280, 3],
            "left_source_video_sha256": left["run"]["video"]["sha256"],
            "right_source_video_sha256": right["run"]["video"]["sha256"],
            "alignment": "paired continuation frame index; packet timestamps remain in source traces"}


def draw_overhead(prop_case, left, right, checkpoint_s, summary):
    """Evaluator-only plan view; source camera videos remain separate."""
    bounds = (.75, 2.65, -5.45, -4.0)
    run_pair = (left["run"], right["run"])
    samples = [run["evaluator_gap_samples"] for run in run_pair]
    out = trial.OUT / "comparisons" / f"{prop_case}-overhead-gap-replay.mp4"
    times = np.arange(22.0, 24.81, .05)

    def xy_pixel(xy, column):
        x0, x1, y0, y1 = bounds
        return (int(50+column*640+520*(xy[0]-x0)/(x1-x0)),
                int(610-500*(xy[1]-y0)/(y1-y0)))

    def ink(canvas, line, at, *, color=(235, 240, 245), scale=.56):
        cv2.putText(canvas, line, at, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1)

    with imageio.get_writer(out, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in times:
            canvas = np.full((680, 1280, 3), (15, 22, 27), dtype=np.uint8)
            ink(canvas, "EVALUATOR PLAN VIEW: icons schematic; gap numbers use MuJoCo geometry",
                (18, 30), scale=.62)
            for column, (item, bank) in enumerate(zip((left, right), samples)):
                run = item["run"]
                row = min(bank, key=lambda sample: abs(sample["time_s"]-t))
                row_t = row["time_s"]
                xbase = column*640
                label = "open loop" if column == 0 else "range recheck"
                ink(canvas, f"{prop_case} / {label} / t={row_t:.2f}s",
                    (xbase+18, 62), scale=.65)
                for x in np.arange(1.0, 2.6, .25):
                    a = xy_pixel((x, bounds[2]), column)
                    b = xy_pixel((x, bounds[3]), column)
                    cv2.line(canvas, a, b, (36, 45, 48), 1)
                for y in np.arange(-5.25, -4.0, .25):
                    a = xy_pixel((bounds[0], y), column)
                    b = xy_pixel((bounds[1], y), column)
                    cv2.line(canvas, a, b, (36, 45, 48), 1)
                ape = np.asarray(row["ape_xy_m"])
                person = trial.v38.human_xy(
                    "blind_return_flank_fast_response", summary, row_t,
                    run["request_time_s"], run["second_inference_start_s"])
                prop = trial.prop_at(row_t, checkpoint_s,
                                     trial.PROP_CASES[prop_case])
                start = np.asarray(summary["ape_after_episode_xy_m"])
                cv2.line(canvas, xy_pixel(start, column),
                         xy_pixel(start+[0., -.25], column),
                         (85, 125, 140), 2, cv2.LINE_AA)
                cv2.circle(canvas, xy_pixel(prop, column), 48,
                           (50, 184, 220), -1)
                cv2.circle(canvas, xy_pixel(person, column), 23,
                           (80, 105, 230), -1)
                cv2.circle(canvas, xy_pixel(ape, column), 35,
                           (90, 205, 105), -1)
                ink(canvas, "prop", tuple(np.add(xy_pixel(prop, column), (8, -55))),
                    color=(70, 210, 240))
                ink(canvas, "person", tuple(np.add(xy_pixel(person, column), (8, -28))),
                    color=(120, 140, 245))
                ink(canvas, "ape", tuple(np.add(xy_pixel(ape, column), (8, 47))),
                    color=(110, 230, 135))
                person_gap, prop_gap = row["gap_m"], row["prop_gap_m"]
                pc = (90, 205, 105) if person_gap >= .05 else (95, 95, 240)
                rc = (90, 205, 105) if prop_gap >= .05 else (95, 95, 240)
                ink(canvas, f"person gap {100*person_gap:.1f} cm", (xbase+30, 642),
                    color=pc, scale=.64)
                ink(canvas, f"prop gap {100*prop_gap:.1f} cm", (xbase+340, 642),
                    color=rc, scale=.64)
                if item["response_halt_s"] is not None and row_t >= item["response_halt_s"]:
                    ink(canvas, "range halt active", (xbase+30, 88),
                        color=(90, 190, 235))
            writer.append_data(canvas)
    return {"path": str(out), "sha256": sha(out), "frames": len(times),
            "shape": [680, 1280, 3],
            "kind": "evaluator_replay_from_20ms_gap_samples",
            "alignment": "same scripted world time; person and prop positions are evaluator truth"}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    lookup = {(item["prop_case"], item["mode"]): item
              for item in report["results"]}
    videos = {case: draw_pair(case, lookup[(case, "open_loop")],
                             lookup[(case, "recheck_065")])
              for case in trial.PROP_CASES}
    summary = trial.v38.original()["summary"]
    overhead = {case: draw_overhead(
        case, lookup[(case, "open_loop")], lookup[(case, "recheck_065")],
        report["checkpoint_s"], summary) for case in trial.PROP_CASES}
    result = {"schema": "robot-learning-lab.ape-owner-corridor-comparisons.v1",
              "report_sha256": sha(report_path),
              "source_sha256": sha(Path(__file__)),
              "videos": videos,
              "overhead_videos": overhead}
    (trial.OUT / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
