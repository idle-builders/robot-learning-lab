"""Join representative retained owner-task phases into viewable simulations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes"
OUT = BASE / "alt_pair_full_action_20_20261015"
EPISODES = {
    "living_full_healthy": [
        BASE / "alt_pair_compound_20_20261015/living/owner_to_find_present.mp4",
        OUT / "living/healthy/fresh_before_move/episode.mp4",
        OUT / "living/healthy/guarded_walk/first_person.mp4",
        OUT / "living/healthy/final_view/episode.mp4"],
    "hotel_full_help_healthy": [
        BASE / "alt_pair_owner_help_20261015/capture_hotel_north_1m/owner_find_help_full.mp4",
        OUT / "capture_hotel_north_1m/healthy/fresh_before_move/episode.mp4",
        OUT / "capture_hotel_north_1m/healthy/guarded_walk/first_person.mp4",
        OUT / "capture_hotel_north_1m/healthy/final_view/episode.mp4"],
    "lincoln_final_range_gap": [
        BASE / "alt_pair_compound_20_20261015/capture_lincoln_drawing_north_1p5m/owner_approach/first_person.mp4",
        BASE / "alt_pair_compound_20_20261015/capture_lincoln_drawing_north_1p5m/find_present/episode.mp4",
        BASE / "alt_pair_owner_help_20261015/capture_lincoln_drawing_north_1p5m/accurate_present/fresh_search/episode.mp4",
        OUT / "capture_lincoln_drawing_north_1p5m/range_gap/fresh_before_move/episode.mp4",
        OUT / "capture_lincoln_drawing_north_1p5m/range_gap/guarded_walk/first_person.mp4",
        OUT / "capture_lincoln_drawing_north_1p5m/range_gap/final_view/episode.mp4"]}


def frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    count = 0
    while True:
        okay, image = capture.read()
        if not okay:
            break
        if image.shape[:2] != (360, 640):
            raise ValueError(f"Unexpected frame shape in {path}: {image.shape}")
        count += 1
    capture.release()
    if count == 0:
        raise ValueError(f"No decoded frames in {path}")
    return count


def main() -> None:
    rows = []
    for name, parts in EPISODES.items():
        counts = [frame_count(path) for path in parts]
        playlist = OUT / f"{name}_parts.txt"
        playlist.write_text("".join(f"file '{p}'\n" for p in parts))
        movie = OUT / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(playlist), "-c", "copy", str(movie)], check=True)
        combined = frame_count(movie)
        if combined != sum(counts):
            raise ValueError(f"Concatenated frame mismatch: {name}: {combined} != {sum(counts)}")
        rows.append({"name": name, "video": str(movie.relative_to(ROOT)),
            "frames": combined, "segments": [{"video": str(p.relative_to(ROOT)),
                "frames": n} for p, n in zip(parts, counts)]})
        playlist.unlink()
    (OUT / "full_episode_video_manifest.json").write_text(json.dumps({
        "schema": "ape.alt-pair-full-action-videos.v1", "fps": 5,
        "episodes": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
