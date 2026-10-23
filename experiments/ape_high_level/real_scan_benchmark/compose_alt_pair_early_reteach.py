"""Compose three owner-task videos from actual retained rendered stages."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "assets/real_indoor_scenes"
SOURCE = BASE / "alt_pair_compound_20_20261015"
RETEACH = BASE / "alt_pair_early_reteach_20261015"
HELP = BASE / "alt_pair_reteach_help_20261015"
ACTION_A = BASE / "alt_pair_early_reteach_action_20261015"
ACTION_B = BASE / "alt_pair_reteach_help_action_20261015"


def frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while True:
        okay, frame = cap.read()
        if not okay:
            break
        if frame.shape[:2] != (360, 640):
            raise ValueError(f"Video resolution changed: {path}")
        count += 1
    cap.release()
    if not count:
        raise ValueError(f"Empty video: {path}")
    return count


def teaching_overlay(key: str) -> Path:
    teach = RETEACH / key / "teach"
    trace = json.loads((teach / "trace.json").read_text())
    image = cv2.imread(str(teach / "rgb_000.png"))
    assert image is not None
    point = trace["owner_signal"]["click_xy"]
    box = [round(v) for v in trace["selected_box_xyxy"]]
    cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), (0, 235, 255), 2)
    cv2.circle(image, tuple(point), 5, (0, 0, 255), -1)
    cv2.putText(image, "Owner points / confirms", (8, 25),
        cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 2)
    path = RETEACH / key / "owner_reteach.mp4"
    with imageio.get_writer(str(path), fps=5, codec="libx264", quality=7,
                            macro_block_size=8) as writer:
        writer.append_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    assert frames(path) == 1
    return path


def main() -> None:
    examples = {
        "nomad_reteach_to_action": ("capture_nomad_east_1m", ACTION_A, False),
        "cabin_reteach_help_to_action": ("capture_cabin_east_1m", ACTION_B, True),
        "apartment_reteach_help_unknown": ("capture_apartment_1", None, True)}
    rows = []
    for name, (key, action, helped) in examples.items():
        parts = [SOURCE / key / "owner_approach/first_person.mp4",
            teaching_overlay(key), RETEACH / key / "find_present/episode.mp4"]
        if helped:
            parts.append(HELP / key / "accurate_present/fresh_search/episode.mp4")
        if action:
            parts += [action / key / "healthy/fresh_before_move/episode.mp4",
                action / key / "healthy/guarded_walk/first_person.mp4",
                action / key / "healthy/final_view/episode.mp4"]
        counts = [frames(part) for part in parts]
        playlist = RETEACH / f"{name}_parts.txt"
        playlist.write_text("".join(f"file '{p}'\n" for p in parts))
        movie = RETEACH / f"{name}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat",
            "-safe", "0", "-i", str(playlist), "-c", "copy", str(movie)], check=True)
        assert frames(movie) == sum(counts)
        playlist.unlink()
        rows.append({"name": name, "video": str(movie.relative_to(ROOT)),
            "frames": sum(counts), "segments": [{"video": str(p.relative_to(ROOT)),
                "frames": n} for p, n in zip(parts, counts)]})
    (RETEACH / "joined_video_manifest.json").write_text(json.dumps({
        "schema": "ape.alt-pair-early-reteach-videos.v1", "fps": 5,
        "examples": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
