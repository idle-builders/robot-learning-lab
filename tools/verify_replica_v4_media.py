"""Decode retained Replica v4 videos and record integrity by run."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ROOT / "assets/real_indoor_scenes/replica_policy_v4_cached_three_20261015",
    ROOT / "assets/real_indoor_scenes/replica_v4_pose_ablation_20261015",
)
OUTPUT = ROOT / "artifacts/ape-real-room-next-captures-20261014/replica_v4_media_decode.json"


def main() -> None:
    rows = []
    for run in RUNS:
        for video in sorted(run.rglob("*.mp4")):
            result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video),
                                     "-f", "null", "-"], capture_output=True)
            rows.append({"path": str(video.relative_to(ROOT)),
                         "bytes": video.stat().st_size,
                         "decoded": result.returncode == 0,
                         "error_tail": result.stderr.decode(errors="replace")[-300:]
                                       if result.returncode else None})
    report = {"schema": "ape.replica-v4-media-decode.v1",
              "video_count": len(rows),
              "decoded_count": sum(row["decoded"] for row in rows),
              "videos": rows}
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"video_count": report["video_count"],
                      "decoded_count": report["decoded_count"]}))
    if report["decoded_count"] != report["video_count"]:
        raise RuntimeError("One or more saved videos failed to decode")


if __name__ == "__main__":
    main()
