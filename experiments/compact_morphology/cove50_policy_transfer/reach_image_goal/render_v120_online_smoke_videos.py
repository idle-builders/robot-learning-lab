"""Publish completed opened online selector replays in the local inspector."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v120"
PLAN = BASE / "online-opened-smoke-plan.json"
MATRIX = BASE / "native-opened-smoke/matrix.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V120 online matrix differs from frozen plan")
    for row in matrix["cases"]:
        report, trace = ROOT / row["result"], ROOT / row["trace"]
        video = report.with_suffix(".mp4")
        if video.is_file() or not report.is_file() or not trace.is_file():
            continue
        temp = video.with_name(video.stem + ".tmp.mp4")
        temp.unlink(missing_ok=True)
        try:
            subprocess.run([sys.executable, "-m", RENDERER,
                            "--report", str(report), "--trace", str(trace),
                            "--video", str(temp)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
            os.replace(temp, video)
            print(f"published {video.name}", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            temp.unlink(missing_ok=True)
            print(f"waiting for complete trace {row['id']}: {exc}", flush=True)


if __name__ == "__main__":
    main()
