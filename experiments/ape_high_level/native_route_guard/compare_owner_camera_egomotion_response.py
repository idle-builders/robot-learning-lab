"""Time-align V37 raw versus yaw-stabilized head-camera watches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_camera_egomotion_response as trial


MODES = ("naive", "yaw_compensated")
CONDITIONS = ("wide_buffered_static", "wide_respond_fast")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(outcome):
    summary = outcome["summary"]
    reader = imageio.get_reader(summary["sensor_video"]["path"])
    frames = [frame for frame in reader]
    reader.close()
    times = [0.] + [sample["time_s"]-summary["worker_start_s"]
                    for sample in summary["inference_probe_samples"]]
    assert len(times) == len(frames)
    return frames, np.asarray(times), summary


def status(summary, t):
    if summary["actor_alarm_s"] is None:
        return "NO ALARM / ITEM TASK CONTINUES"
    alarm = summary["actor_alarm_s"]-summary["worker_start_s"]
    if t < alarm:
        return "MONITORING"
    if summary["first_action_s"] is None:
        return "ALARM / ITEM IMAGE DISCARDED"
    first = summary["first_action_s"]-summary["worker_start_s"]
    return ("ALARM / ITEM IMAGE DISCARDED" if t < first
            else "MODELED SOUTH RESPONSE")


def make_video(report, condition):
    selected = {row["mode"]: row for row in report["outcomes"]
                if row["condition"] == condition and row["mode"] in MODES}
    assert set(selected) == set(MODES)
    tracks = [track(selected[mode]) for mode in MODES]
    end = min(float(times[-1]) for _, times, _ in tracks)
    ticks = np.arange(0., end+.00001, .1)
    path = trial.OUT / f"{condition}-raw-versus-yaw-stabilized.mp4"
    with imageio.get_writer(path, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for t in ticks:
            panels = []
            for mode, (frames, times, summary) in zip(MODES, tracks):
                index = min(len(frames)-1, max(0, int(
                    np.searchsorted(times, t, side="right")-1)))
                frame = frames[index].copy()
                sample = (summary["inference_probe_samples"][index-1]
                          if index > 0 else None)
                cv2.rectangle(frame, (0, 0), (640, 92), (18, 18, 18), -1)
                title = ("RAW CAMERA CHANGE" if mode == "naive"
                         else "YAW-STABILIZED RGB-D")
                cv2.putText(frame, title, (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, .67,
                            (255, 255, 255), 2)
                if sample is not None:
                    detail = (f"changed RGB/depth: {sample['rgb_changed_pixels']}/"
                              f"{sample['depth_changed_pixels']}  |  cue: "
                              f"{'YES' if sample['actor_motion_cue'] else 'NO'}")
                    cv2.putText(frame, detail, (10, 68),
                                cv2.FONT_HERSHEY_SIMPLEX, .54,
                                (255, 255, 255), 2)
                cv2.rectangle(frame, (0, 436), (640, 480), (15, 15, 15), -1)
                caption = f"+{t:.1f}s | {status(summary, t)}"
                cv2.putText(frame, caption, (10, 465),
                            cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255, 255, 255), 2)
                panels.append(frame)
            writer.append_data(np.concatenate(panels, axis=1))
    return {"path": str(path), "sha256": sha(path), "frames": len(ticks)}


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    videos = {condition: make_video(report, condition)
              for condition in CONDITIONS}
    output = {"schema": "robot-learning-lab.ape-v37-camera-comparison.v1",
              "source_sha256": sha(Path(__file__)),
              "report_sha256": sha(report_path),
              "alignment": "camera frame at or before each 100 ms tick relative to item-inference worker start",
              "videos": videos}
    (trial.OUT / "comparison.json").write_text(json.dumps(output, indent=2)+"\n")


if __name__ == "__main__":
    main()
