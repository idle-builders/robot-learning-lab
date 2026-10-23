"""Render completed V115 native target-slope visits for the inspector."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v115"
PLAN = OUT / "opened-contact-slope-plan.json"
MATRIX = OUT / "native-opened-contact-slope/matrix.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v115_contact_slope"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V115 matrix does not match its frozen plan")
    for row in matrix["cases"]:
        report, trace = ROOT / row["result"], ROOT / row["trace"]
        video = report.with_suffix(".mp4")
        if video.is_file() or not report.is_file() or not trace.is_file():
            continue
        temporary = video.with_name(video.stem + ".tmp.mp4")
        temporary.unlink(missing_ok=True)
        try:
            subprocess.run([sys.executable, "-m", RENDERER,
                            "--report", str(report), "--trace", str(trace),
                            "--video", str(temporary)], cwd=ROOT, check=True,
                           capture_output=True, text=True)
            os.replace(temporary, video)
            print(f"published {video.name}", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            temporary.unlink(missing_ok=True)
            print(f"waiting for complete {row['id']} trace: {exc}", flush=True)
    subprocess.run([sys.executable, "-m", SUMMARY], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
