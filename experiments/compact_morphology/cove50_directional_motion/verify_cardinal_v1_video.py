"""Decode every scored cardinal MP4 and tie it to the native visit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-low-level-training-20261014/cardinal-v1"
EVIDENCE = OUT / "candidate"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = json.loads((EVIDENCE / "result.json").read_text())
    method = json.loads((OUT / "method.json").read_text())
    if result["method_sha256"] != sha(OUT / "method.json"):
        raise RuntimeError("Candidate method differs")
    rows = []
    for case in method["development_cases"]:
        for arm in ("baseline", "candidate"):
            label = f"seed{case['seed']}-{arm}"
            report_path = EVIDENCE / f"{label}.json"
            video_path = EVIDENCE / f"{label}.mp4"
            if (result["files"][video_path.name] != sha(video_path)
                    or result["files"][report_path.name] != sha(report_path)):
                raise RuntimeError(f"Source hash differs for {label}")
            visit = json.loads(report_path.read_text())
            world = visit["worlds"][0]
            if visit["seed"] != case["seed"] or visit["body_sha256"] != method["source"]["body"]["sha256"]:
                raise RuntimeError(f"Native visit differs for {label}")
            expected = math.ceil(round(world["duration_s"] / .02) / 2)
            count = 0
            minimum_std = float("inf")
            with imageio.get_reader(str(video_path), format="ffmpeg") as reader:
                for frame in reader:
                    pixel_std = float(np.std(frame))
                    if frame.shape != (480, 640, 3) or pixel_std < 1.0:
                        raise RuntimeError(f"Blank or malformed frame {count} for {label}")
                    minimum_std = min(minimum_std, pixel_std)
                    count += 1
            if count != expected:
                raise RuntimeError(f"Frame count {count} differs from {expected} for {label}")
            rows.append({"seed": case["seed"], "family": case["family"], "arm": arm,
                         "native_duration_s": world["duration_s"], "decoded_frames": count,
                         "frame_shape": [480, 640, 3],
                         "minimum_pixel_std": round(minimum_std, 3),
                         "video_sha256": sha(video_path), "report_sha256": sha(report_path)})
    report = {"schema": "robot-learning-lab.cardinal-v1.video-decode.v1",
              "source_result_sha256": sha(EVIDENCE / "result.json"),
              "decoded_videos": len(rows), "rows": rows,
              "claim_limit": "Decoded videos and native source identity; physical gates are in development-score.json."}
    (OUT / "video-decode-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"decoded_videos": len(rows),
                      "total_decoded_frames": sum(row["decoded_frames"] for row in rows)}, indent=2))


if __name__ == "__main__":
    main()
