"""Decode the current real-capture roster and show three views of each route.

This is an appearance audit, not a photorealism score or an owner-task run.
The contact pages make motion artifacts and repeated capture groups visible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
ROSTER = ROOT / "assets/real_indoor_scenes/real_capture_29_current_candidate_set.json"
FULL_TASK_AUDIT = ROOT / "assets/real_indoor_scenes/real_capture_20_full_task_split_audit.json"
OUTPUT = ROOT / "artifacts/ape-real-room-visual-audit-20261014"
THUMB_SIZE = (320, 180)
PAGE_ROWS = 5


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path: Path) -> list[np.ndarray]:
    stream = cv2.VideoCapture(str(path))
    if not stream.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    while True:
        okay, frame = stream.read()
        if not okay:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    stream.release()
    if len(frames) < 3:
        raise RuntimeError(f"No moving sequence in {path}")
    return frames


def main() -> None:
    roster = json.loads(ROSTER.read_text())
    completed = json.loads(FULL_TASK_AUDIT.read_text())
    known_groups = {row["scene_id"]: row["capture_group"]
                    for row in completed["rooms"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    page_paths = []
    for index, scene in enumerate(roster["scenes"]):
        video = ROOT / scene["video"]
        trace = ROOT / scene["trace"]
        frames = decode(video)
        route = json.loads(trace.read_text())
        samples = [0, len(frames) // 2, len(frames) - 1]
        if index % PAGE_ROWS == 0:
            page = Image.new("RGB", (THUMB_SIZE[0] * 3,
                                     (THUMB_SIZE[1] + 38) * PAGE_ROWS), "#17191c")
            draw = ImageDraw.Draw(page)
        y = (index % PAGE_ROWS) * (THUMB_SIZE[1] + 38)
        for column, frame_id in enumerate(samples):
            image = Image.fromarray(frames[frame_id]).resize(THUMB_SIZE)
            page.paste(image, (column * THUMB_SIZE[0], y))
            draw.text((column * THUMB_SIZE[0] + 5, y + 184),
                      f"{('start', 'middle', 'end')[column]}  {frame_id+1}/{len(frames)}",
                      fill="#d4d7de")
        draw.text((5, y + 201),
                  f"{index+1:02d} {scene['scene_id']}  |  {scene['category']}",
                  fill="#ffffff")
        rows.append({
            "scene_id": scene["scene_id"],
            "physical_space_id": scene["physical_space_id"],
            "capture_group": known_groups.get(scene["scene_id"],
                                               scene["physical_space_id"]),
            "category": scene["category"],
            "source_family": scene["source_family"],
            "source_url": scene["source_url"],
            "video": scene["video"],
            "video_sha256": sha(video),
            "trace": scene["trace"],
            "trace_sha256": sha(trace),
            "decoded_frames": len(frames),
            "trace_frames": len(route["frames"]),
            "sample_indices": samples,
            "duration_at_5fps_s": round(len(frames) / 5, 2),
            "route_m": scene["route_m"],
            "depth_frames": scene["depth_frames"],
            "frame_count_matches_trace": len(frames) == len(route["frames"]),
        })
        if index % PAGE_ROWS == PAGE_ROWS - 1 or index == len(roster["scenes"]) - 1:
            count = (index % PAGE_ROWS) + 1
            if count < PAGE_ROWS:
                page = page.crop((0, 0, THUMB_SIZE[0] * 3,
                                  count * (THUMB_SIZE[1] + 38)))
            path = OUTPUT / f"moving_triplets_{len(page_paths)+1:02d}.jpg"
            page.save(path, quality=92)
            page_paths.append(str(path.relative_to(ROOT)))
    report = {
        "schema": "ape.real-capture-moving-view-audit.v1",
        "roster": str(ROSTER.relative_to(ROOT)),
        "roster_sha256": sha(ROSTER),
        "full_task_audit": str(FULL_TASK_AUDIT.relative_to(ROOT)),
        "full_task_audit_sha256": sha(FULL_TASK_AUDIT),
        "scope": "All 29 existing provisional real-capture moving routes; decoded start/middle/end images, not a numerical photorealism claim.",
        "scene_count": len(rows),
        "capture_group_count": len({row["capture_group"] for row in rows}),
        "all_videos_decoded_and_trace_aligned": all(
            row["frame_count_matches_trace"] for row in rows),
        "contact_pages": page_paths,
        "scenes": rows,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUTPUT / "report.json")


if __name__ == "__main__":
    main()
