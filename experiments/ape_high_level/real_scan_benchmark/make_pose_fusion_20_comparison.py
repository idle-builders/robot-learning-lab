"""Show the paired museum-room approach and fresh item review side by side."""

from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes/pose_fusion_20_20261015"
ROOM = BASE / "capture_nazaretian_inner_east_0p8m"
OUT = BASE / "nazaretian_motion_vs_range_full_task.mp4"
FULL = BASE / "nazaretian_full_task_range_corrected.mp4"


def read(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    rows = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rows.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not rows:
        raise ValueError(f"Empty source video: {path}")
    return rows


def label(frame: np.ndarray, heading: str, stage: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, heading, (12, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(out, (0, out.shape[0] - 30),
                  (out.shape[1], out.shape[0]), (0, 0, 0), -1)
    cv2.putText(out, stage, (12, out.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, .52, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    arms = []
    for name in ("motion_only", "motion_plus_range"):
        folder = ROOM / name
        arms.append((read(folder / "walk/first_person.mp4"),
                     read(folder / "final_view/episode.mp4")))
    headings = ("Motion only: asks for a new view",
                "Motion + range: item confirmed")
    with imageio.get_writer(str(OUT), fps=4, codec="libx264",
                            quality=7, macro_block_size=8) as writer:
        for stage_index, caption in ((0, "Approach"), (1, "Fresh item review")):
            length = max(len(arms[0][stage_index]), len(arms[1][stage_index]))
            for i in range(length):
                sides = [label(pair[stage_index][min(i, len(pair[stage_index]) - 1)],
                               headings[index], caption)
                         for index, pair in enumerate(arms)]
                writer.append_data(np.hstack(sides))
    print(OUT)
    source = (ROOT /
        "assets/real_indoor_scenes/integrated_owner_20_colour_bank_none_20261015"
        / "capture_nazaretian_inner_east_0p8m")
    corrected = ROOM / "motion_plus_range"
    stages = (
        ("Approach the owner", read(source / "owner_approach/first_person.mp4")),
        ("Owner teaches the item", [np.asarray(Image.open(
            source / "teach/rgb_000.png").convert("RGB"))] * 5),
        ("Find the taught item", read(source / "find_present/episode.mp4")),
        ("Approach the found item", read(corrected / "walk/first_person.mp4")),
        ("Fresh view and owner review", read(corrected / "final_view/episode.mp4")),
    )
    with imageio.get_writer(str(FULL), fps=5, codec="libx264",
                            quality=7, macro_block_size=8) as writer:
        for caption, frames in stages:
            for frame in frames:
                writer.append_data(label(frame, "Captured room; scripted owner", caption))
    print(FULL)


if __name__ == "__main__":
    main()
