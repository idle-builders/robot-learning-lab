"""Render source-matched held-out selector visits for the results page."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v119"
PLAN = BASE / "selector-heldout-plan.json"
MATRIX = BASE / "native-selector-heldout/matrix.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v119_selector_holdout"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V119 native matrix differs from frozen heldout plan")
    for row in matrix["cases"]:
        seen = set()
        for report, trace in ((row["baseline_result"], row["baseline_trace"]),
                              (row["candidate_result"], row["candidate_trace"])):
            if report in seen:
                continue
            seen.add(report)
            report_path, trace_path = ROOT / report, ROOT / trace
            video = report_path.with_suffix(".mp4")
            if video.is_file() or not report_path.is_file() or not trace_path.is_file():
                continue
            temp = video.with_name(video.stem + ".tmp.mp4")
            temp.unlink(missing_ok=True)
            try:
                subprocess.run([sys.executable, "-m", RENDERER,
                                "--report", str(report_path),
                                "--trace", str(trace_path),
                                "--video", str(temp)], cwd=ROOT,
                               check=True, capture_output=True, text=True)
                os.replace(temp, video)
                print(f"published {video.name}", flush=True)
            except (OSError, subprocess.CalledProcessError) as exc:
                temp.unlink(missing_ok=True)
                print(f"waiting for complete trace {row['id']}: {exc}",
                      flush=True)
    subprocess.run([sys.executable, "-m", SUMMARY], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
