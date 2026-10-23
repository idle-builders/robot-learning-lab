"""Short annotated late-entry clip comparing held-state warning coverage."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import held_body_intrusion_trial as late
import held_monitor_probe as probe
from kitchen_trial import sha


START_S = 2.4
END_S = 5.0


def main() -> None:
    case = "late_right_pause"
    world = 0
    report = json.loads((probe.OUT / "report.json").read_text())
    rows = {(r["mode"], r["world"], r["case"]): r for r in report["results"]}
    front = rows["front_140deg_15hz", world, case]
    omni = rows["ideal_omni_10hz", world, case]
    source = late.OUT / "bounded_guard/videos" / \
        f"{case}-mapped_low_guard-world{world}.mp4"
    trace = late.OUT / "bounded_guard/episodes" / \
        f"{case}-mapped_low_guard-world{world}.npz"
    with np.load(trace, allow_pickle=False) as saved:
        times = saved["time_s"].copy()
        gaps = saved["gap_evaluator_m"].copy()
        actions = saved["action"].copy()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Missing V27 source video")
    out = probe.OUT / "late-right-front-vs-ideal-omni-world0.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             12.5, (960, 650))
    if not writer.isOpened():
        raise RuntimeError("MP4 writer failed")
    frames = 0
    try:
        for index in range(100):
            okay, original = cap.read()
            if not okay:
                raise RuntimeError(f"Missing source frame {index}")
            sample = index*4
            t = float(times[sample])
            if not START_S <= t <= END_S:
                continue
            gap = float(gaps[sample])
            scene = original[:480].copy()
            cv2.rectangle(scene, (0, 0), (959, 43), (29, 32, 37), -1)
            cv2.putText(scene, "HELD APE: FRONT FAN VS IDEAL ALL-AROUND LOW VIEW",
                        (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .64,
                        (230, 236, 238), 2, cv2.LINE_AA)
            foot = np.full((170, 960, 3), (29, 32, 37), dtype=np.uint8)
            cv2.putText(foot, f"t={t:.2f} s  {actions[sample]}  "
                              f"body-person gap={gap:+.3f} m",
                        (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .66,
                        (82, 101, 230) if gap < .05 else (227, 233, 238),
                        1, cv2.LINE_AA)
            front_alert = front["first_two_packet_actor_alert_s"]
            omni_alert = omni["first_two_packet_actor_alert_s"]
            cv2.putText(foot,
                        f"Front 140 deg / 15 Hz: "
                        f"{'ALERT' if front_alert is not None and t >= front_alert else 'no alert'}",
                        (12, 57), cv2.FONT_HERSHEY_SIMPLEX, .62,
                        (77, 96, 226) if front_alert is not None and
                        t >= front_alert else (169, 184, 195), 1, cv2.LINE_AA)
            cv2.putText(foot,
                        f"Ideal 360 deg / 10 Hz: "
                        f"{'ALERT' if omni_alert is not None and t >= omni_alert else 'no alert'}"
                        "  (no escape action)",
                        (12, 86), cv2.FONT_HERSHEY_SIMPLEX, .62,
                        (68, 201, 230) if omni_alert is not None and
                        t >= omni_alert else (169, 184, 195), 1, cv2.LINE_AA)
            cv2.putText(foot,
                        "Ideal pose/rays; scripted capsule; historical kinematic stop tail",
                        (12, 113), cv2.FONT_HERSHEY_SIMPLEX, .51,
                        (166, 183, 196), 1, cv2.LINE_AA)
            x0, x1 = 28, 932
            cv2.line(foot, (x0, 151), (x1, 151), (100, 113, 125), 4)
            markers = ((3.0, "end", (217, 217, 217)),
                       (3.10, "entry", (163, 184, 194)),
                       (omni_alert, "alert", (68, 201, 230)),
                       (omni["first_evaluator_below_5cm_s"], "5cm",
                        (82, 101, 230)))
            for when, label, color in markers:
                if when is None:
                    continue
                x = round(x0+(when-START_S)/(END_S-START_S)*(x1-x0))
                cv2.line(foot, (x, 140), (x, 160), color, 2)
                cv2.putText(foot, label, (x-17, 137),
                            cv2.FONT_HERSHEY_SIMPLEX, .36, color, 1, cv2.LINE_AA)
            cursor = round(x0+(t-START_S)/(END_S-START_S)*(x1-x0))
            cv2.circle(foot, (cursor, 151), 5, (245, 245, 245), -1)
            writer.write(np.concatenate([scene, foot], axis=0))
            frames += 1
    finally:
        writer.release()
        cap.release()
    manifest = {"schema": "robot-learning-lab.ape-v28-held-monitor-video.v1",
                "case": case, "world": world, "frames": frames,
                "path": str(out), "sha256": sha(out),
                "source_video_sha256": sha(source),
                "trace_sha256": sha(trace),
                "v28_report_sha256": sha(probe.OUT / "report.json"),
                "source_sha256": sha(Path(__file__))}
    (probe.OUT / "comparison-video.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(out, frames)


if __name__ == "__main__":
    main()
