"""Captioned V31 comparison on the matched V27 late-right episode video."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import body_aware_held_sensor as study
import held_body_intrusion_trial as late
from kitchen_trial import sha


START_S = 2.4
END_S = 5.0


def main() -> None:
    case = "late_right_pause"
    world = 0
    report = json.loads((study.OUT / "report.json").read_text())
    rows = {(r["mode"], r["world"], r["case"]): r for r in report["results"]}
    chosen = {mode: rows[mode, world, case] for mode in
              ("front_body_15hz", "omni_body_10hz",
               "omni_body_masked_10hz", "split_body_10hz")}
    source = late.OUT / "bounded_guard/videos" / \
        f"{case}-mapped_low_guard-world{world}.mp4"
    trace = late.OUT / "bounded_guard/episodes" / \
        f"{case}-mapped_low_guard-world{world}.npz"
    with np.load(trace, allow_pickle=False) as saved:
        times = saved["time_s"].copy()
        gaps = saved["gap_evaluator_m"].copy()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Missing V27 video: {source}")
    movie = study.OUT / "late-right-body-aware-sensor-world0.mp4"
    writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (960, 710))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    frames = 0
    try:
        for index in range(100):
            ok, original = cap.read()
            if not ok:
                raise RuntimeError(f"Missing source frame {index}")
            sample = index*4
            t = float(times[sample])
            if not START_S <= t <= END_S:
                continue
            gap = float(gaps[sample])
            scene = original[:480].copy()
            cv2.rectangle(scene, (0, 0), (959, 43), (29, 32, 37), -1)
            cv2.putText(scene, "HELD APE: BODY-AWARE RANGE VIEWS",
                        (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .68,
                        (230, 236, 238), 2, cv2.LINE_AA)
            foot = np.full((230, 960, 3), (29, 32, 37), dtype=np.uint8)
            lines = [
                (f"t={t:.2f}s  modeled full-body person gap={gap:+.3f}m",
                 (82, 101, 230) if gap < .05 else (227, 233, 238)),
                ("Forward 140 deg: " +
                 ("ALERT" if chosen["front_body_15hz"]["first_two_packet_actor_alert_s"]
                  is not None and t >= chosen["front_body_15hz"]["first_two_packet_actor_alert_s"]
                  else "no alert"), (169, 184, 195)),
                ("Raw front-mounted 360 deg: " +
                 ("ALERT (also in static control)" if t >= 2.92 else "no alert"),
                 (82, 101, 230) if t >= 2.92 else (169, 184, 195)),
                ("Same 360 deg, ideal self mask: " +
                 ("ALERT" if t >= chosen["omni_body_masked_10hz"]
                  ["first_two_packet_actor_alert_s"] else "no alert"),
                 (68, 201, 230) if t >= 3.42 else (169, 184, 195)),
                ("Front plus left/right flanks: " +
                 ("ALERT" if t >= chosen["split_body_10hz"]
                  ["first_two_packet_actor_alert_s"] else "no alert"),
                 (68, 201, 230) if t >= 3.42 else (169, 184, 195)),
            ]
            for n, (line, color) in enumerate(lines):
                cv2.putText(foot, line, (12, 27+n*30),
                            cv2.FONT_HERSHEY_SIMPLEX, .56, color, 1, cv2.LINE_AA)
            cv2.putText(foot,
                        "Historical body + scripted capsule; exact-pose rays; alerts execute no escape",
                        (12, 180), cv2.FONT_HERSHEY_SIMPLEX, .50,
                        (166, 183, 196), 1, cv2.LINE_AA)
            x0, x1 = 28, 932
            cv2.line(foot, (x0, 210), (x1, 210), (100, 113, 125), 4)
            markers = ((2.92, "self alert", (82, 101, 230)),
                       (3.10, "entry", (163, 184, 194)),
                       (3.42, "real alert", (68, 201, 230)),
                       (chosen["omni_body_masked_10hz"]["first_evaluator_below_5cm_s"],
                        "5 cm", (82, 101, 230)))
            for when, label, color in markers:
                if when is None:
                    continue
                x = round(x0+(when-START_S)/(END_S-START_S)*(x1-x0))
                cv2.line(foot, (x, 200), (x, 220), color, 2)
                cv2.putText(foot, label, (x-18, 197),
                            cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1, cv2.LINE_AA)
            cursor = round(x0+(t-START_S)/(END_S-START_S)*(x1-x0))
            cv2.circle(foot, (cursor, 210), 5, (245, 245, 245), -1)
            writer.write(np.concatenate([scene, foot], axis=0))
            frames += 1
    finally:
        writer.release()
        cap.release()
    manifest = {"schema": "robot-learning-lab.ape-body-aware-held-video.v1",
                "case": case, "world": world, "frames": frames,
                "path": str(movie), "sha256": sha(movie),
                "source_video_sha256": sha(source),
                "source_trace_sha256": sha(trace),
                "report_sha256": sha(study.OUT / "report.json"),
                "source_sha256": sha(Path(__file__))}
    (study.OUT / "video.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(movie, frames)


if __name__ == "__main__":
    main()
