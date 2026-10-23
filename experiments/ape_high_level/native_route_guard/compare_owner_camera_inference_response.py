"""Time-align V36 live owner/flank sensor streams and action state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

import owner_camera_inference_response as trial


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def track(outcome):
    summary = outcome["summary"]
    video = summary["sensor_video"]["path"]
    reader = imageio.get_reader(video)
    frames = [frame for frame in reader]
    reader.close()
    times = [0.] + [sample["time_s"]-summary["worker_start_s"]
                    for sample in summary["inference_probe_samples"]]
    assert len(times) == len(frames)
    return frames, np.asarray(times), summary


def status(summary, t):
    if summary["actor_alarm_s"] is None:
        return "WATCHING / no alarm"
    alarm = summary["actor_alarm_s"]-summary["worker_start_s"]
    first = summary["first_action_s"]-summary["worker_start_s"]
    if t < alarm:
        return "WATCHING"
    if t < first:
        return "ALARM / stale image discarded"
    return "MODELED SOUTH RESPONSE"


def main():
    report_path = trial.OUT / "report.json"
    report = json.loads(report_path.read_text())
    turning = {outcome["sensor"]: outcome for outcome in report["outcomes"]
               if outcome["condition"] == "wide_respond_fast"}
    assert set(turning) == set(trial.SENSORS)
    tracks = [track(turning[sensor]) for sensor in trial.SENSORS]
    end = min(float(track_times[-1]) for _, track_times, _ in tracks)
    ticks = np.arange(0., end+.00001, .1)
    frames_out = []
    for t in ticks:
        images = []
        for frames, times, summary in tracks:
            index = min(len(frames)-1, max(0, int(np.searchsorted(times, t, side="right")-1)))
            frame = frames[index].copy()
            caption = f"inference +{t:.1f}s | {status(summary, t)}"
            cv2.rectangle(frame, (0, 436), (640, 480), (15, 15, 15), -1)
            cv2.putText(frame, caption, (10, 465), cv2.FONT_HERSHEY_SIMPLEX,
                        .58, (255, 255, 255), 2)
            images.append(frame)
        paired = np.concatenate(images, axis=1)
        frames_out.append(paired)
    video = trial.OUT / "owner-vs-flank-live-inference.mp4"
    with imageio.get_writer(video, fps=10, codec="libx264",
                            macro_block_size=None) as writer:
        for frame in frames_out:
            writer.append_data(frame)
    comparison = {"schema": "robot-learning-lab.ape-v36-camera-comparison.v1",
                  "source_sha256": sha(Path(__file__)),
                  "report_sha256": sha(report_path),
                  "alignment": "sensor frame at or before each 100 ms tick relative to inference worker start; holds last frame after a branch ends",
                  "video": {"path": str(video), "sha256": sha(video),
                            "frames": len(frames_out)},
                  "owner_alarm_after_worker_s": round(
                      turning["owner_rgbd_motion"]["summary"]["actor_alarm_s"]-
                      turning["owner_rgbd_motion"]["summary"]["worker_start_s"], 3),
                  "flank_alarm_after_worker_s": round(
                      turning["left_flank_rgbd_motion"]["summary"]["actor_alarm_s"]-
                      turning["left_flank_rgbd_motion"]["summary"]["worker_start_s"], 3)}
    (trial.OUT / "comparison.json").write_text(json.dumps(comparison, indent=2)+"\n")


if __name__ == "__main__":
    main()
