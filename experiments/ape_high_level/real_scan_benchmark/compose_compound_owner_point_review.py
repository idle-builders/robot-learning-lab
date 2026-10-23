"""Compose an owner-point success/failure review from audited source media."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from run_common_policy_20_object_generalization import ROOT, relative
from run_compound_owner_point_recovery import OUT
from run_compound_owner_point_approach import OUT as APPROACH_OUT


CASE = "living"
SOURCE = ROOT / "assets/real_indoor_scenes/integrated_owner_20_compound_stress_20261015" / CASE
ARMS = ("accurate_present", "wrong_present")


def frames(path: Path) -> list[np.ndarray]:
    reader = imageio.get_reader(str(path))
    try:
        return [frame for frame in reader]
    finally:
        reader.close()


def overlay(rgb: np.ndarray, label: str) -> np.ndarray:
    out = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (640, 36), (13, 22, 30), -1)
    cv2.putText(out, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def branch(name: str) -> tuple[list[np.ndarray], list[dict]]:
    trace = json.loads((OUT / CASE / name / "trace.json").read_text())
    signal = trace["owner_signal"]
    source_video = SOURCE / "full_find_present.mp4"
    rows = []
    frames_out = []
    for index, frame in enumerate(frames(source_video)):
        frames_out.append(overlay(frame, "Initial search: identity unknown"))
        rows.append({"phase": "initial_search", "video": relative(source_video),
                     "source_frame": index})
    point_rgb = cv2.cvtColor(cv2.imread(str(ROOT / signal["rgb"])),
                             cv2.COLOR_BGR2RGB)
    click = tuple(round(v) for v in signal["click_xy"])
    cv2.circle(point_rgb, click, 13, (245, 225, 30), 3)
    label = ("Owner points to item" if name == "accurate_present"
             else "Owner points elsewhere")
    for _ in range(4):
        frames_out.append(overlay(point_rgb, label))
        rows.append({"phase": "owner_point", "rgb": signal["rgb"],
                     "depth": signal["depth"], "click_xy": signal["click_xy"]})
    fresh_video = ROOT / trace["fresh_video"]
    for index, frame in enumerate(frames(fresh_video)):
        frames_out.append(overlay(frame, "Fresh RGB-D search and owner review"))
        rows.append({"phase": "fresh_search", "video": relative(fresh_video),
                     "source_frame": index})
    return frames_out, rows


def main() -> None:
    left, left_rows = branch(ARMS[0])
    right, right_rows = branch(ARMS[1])
    count = max(len(left), len(right))
    path = OUT / CASE / "point_comparison.mp4"
    manifest = []
    with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for index in range(count):
            a = left[min(index, len(left) - 1)]
            b = right[min(index, len(right) - 1)]
            frame = np.zeros((400, 1280, 3), np.uint8)
            frame[:40, :] = (13, 22, 30)
            cv2.putText(frame, "Correct point -> owner confirmed", (12, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            cv2.putText(frame, "Wrong point -> still unknown", (652, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2)
            frame[40:, :640] = a
            frame[40:, 640:] = b
            writer.append_data(frame)
            manifest.append({"frame": index,
                "correct": left_rows[min(index, len(left_rows) - 1)],
                "wrong": right_rows[min(index, len(right_rows) - 1)]})
    meta = {"schema": "ape.compound-owner-point-review.v1",
        "video": relative(path), "frames": manifest,
        "limits": "Phase overview at 5 fps; original source traces retain observation and worker times. Correct/wrong point is scripted; owner confirmation is a separate saved event."}
    (OUT / CASE / "point_comparison.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(relative(path), len(manifest))

    approach = json.loads((APPROACH_OUT / CASE / "trace.json").read_text())
    sequence = [(frame, row) for frame, row in zip(left, left_rows)]
    for phase, ref in (("Guarded approach", approach["walk"]["video"]),
                       ("Fresh confirmation after move", json.loads(
                           (ROOT / approach["final_recheck"]).read_text())["video"])):
        for index, frame in enumerate(frames(ROOT / ref)):
            sequence.append((overlay(frame, phase),
                             {"phase": phase, "video": ref,
                              "source_frame": index}))
    full_path = OUT / CASE / "full_helped_action.mp4"
    with imageio.get_writer(str(full_path), fps=5, codec="libx264", quality=7,
                            macro_block_size=16) as writer:
        for frame, _ in sequence:
            writer.append_data(frame)
    full_meta = {"schema": "ape.compound-owner-point-full-helped-review.v1",
        "video": relative(full_path),
        "source_help_trace": relative(OUT / CASE / "accurate_present/trace.json"),
        "source_approach_trace": relative(APPROACH_OUT / CASE / "trace.json"),
        "frames": [row for _, row in sequence],
        "limits": "Phase overview; source traces preserve capture and worker timing. Scripted owner and static room; provisional kinematic body."}
    (OUT / CASE / "full_helped_action.json").write_text(
        json.dumps(full_meta, indent=2) + "\n")
    print(relative(full_path), len(sequence))


if __name__ == "__main__":
    main()
