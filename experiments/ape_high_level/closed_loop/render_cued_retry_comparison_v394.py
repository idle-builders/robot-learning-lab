"""Render synchronized ape-camera views of V393 owner response modes.

Orange boxes are current appearance/depth proposals. Green is the item
actually accepted by the high-level loop after its stillness gate. Labels
describe actor events, not recorded speech.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "artifacts/ape-closed-loop-cove50-v393"
OUTPUT = INPUT / "seed170-cued-retry-comparison.mp4"
MODES = ("compliant", "delayed", "ignored")
SEED = 170
SLOW_FRAMES_PER_CAPTURE = 10


def paint(mode: str, sample: dict, current_action: str) -> np.ndarray:
    with np.load(ROOT / sample["actor_packet_path"]) as packet:
        rgb = packet["exposed_rgb"].copy()
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    choice = sample["actor_choice"]
    for candidate in choice["actor_all_candidates"]:
        x1, y1, x2, y2 = map(lambda n: int(round(n)), candidate["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
    box = choice["accepted_box"]
    if box is not None:
        x1, y1, x2, y2 = map(lambda n: int(round(n)), box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (35, 210, 40), 4)
    heading = np.zeros((72, 640, 3), dtype=np.uint8)
    action = current_action.replace("please_", "").replace("_", " ")
    cv2.putText(heading, f"{mode.upper()}  t={sample['capture_time_s']:.1f}s",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .63,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heading, action[:42], (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, .54,
                (160, 230, 250), 1, cv2.LINE_AA)
    footer = np.zeros((46, 640, 3), dtype=np.uint8)
    status = ("GREEN = accepted after stillness" if box is not None else
              "ORANGE = proposals; no item accepted")
    cv2.putText(footer, status, (10, 29), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (190, 255, 190) if box is not None else
                (170, 205, 255), 1, cv2.LINE_AA)
    return np.vstack((heading, frame, footer))


def main() -> None:
    samples_by_mode = {}
    for mode in MODES:
        report = json.loads((INPUT / mode /
                             "blocked-owner-help-audit.json").read_text())
        row = next(row for row in report["owner_show_rows"]
                   if row["seed"] == SEED)
        samples_by_mode[mode] = row["response"]["actor_samples"]
    count = len(samples_by_mode[MODES[0]])
    assert all(len(samples) == count for samples in samples_by_mode.values())
    writer = cv2.VideoWriter(str(OUTPUT), cv2.VideoWriter_fourcc(*"mp4v"),
                             30., (640 * len(MODES), 598))
    assert writer.isOpened()
    actions = {mode: "awaiting first view" for mode in MODES}
    try:
        for index in range(count):
            panels = []
            for mode in MODES:
                sample = samples_by_mode[mode][index]
                events = sample["actor_choice"]["actor_social_events"]
                if events:
                    actions[mode] = events[-1]["action"]
                panels.append(paint(mode, sample, actions[mode]))
            frame = np.hstack(panels)
            for _ in range(SLOW_FRAMES_PER_CAPTURE):
                writer.write(frame)
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(OUTPUT))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    assert decoded == count * SLOW_FRAMES_PER_CAPTURE, decoded
    print(OUTPUT)
    print("decoded frames", decoded)


if __name__ == "__main__":
    main()
