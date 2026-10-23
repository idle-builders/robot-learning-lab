"""Decode every scored V6 pilot video and match it to its native visit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from free_object_touch_env import ROOT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    assert report["schema"] == "robot-learning-lab.cove50-observed-touch-bc-development.v6"
    rows = []
    for visit in report["records"]:
        path = ROOT / visit["video"]
        assert sha(path) == visit["video_sha256"]
        reader = imageio.get_reader(path)
        try:
            frames = [np.asarray(frame) for frame in reader]
        finally:
            reader.close()
        shape = list(frames[0].shape)
        assert shape == [120, 160, 3]
        assert len(frames) == visit["steps"] + 1 == visit["video_frames"]
        assert all(frame.std() > 1 for frame in (frames[0], frames[len(frames)//2], frames[-1]))
        rows.append({"seed": visit["seed"], "mode": visit["mode"],
                     "decoded_frames": len(frames), "shape": shape,
                     "video_sha256": visit["video_sha256"]})
    result = {"schema": "robot-learning-lab.cove50-touch-v6-video-check.v1",
              "report_sha256": sha(args.report), "videos_verified": len(rows),
              "frames_verified": sum(x["decoded_frames"] for x in rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("videos_verified", "frames_verified")}), flush=True)


if __name__ == "__main__":
    main()
