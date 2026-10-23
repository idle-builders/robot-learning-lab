"""Render locally synced untouched V101 native cases for inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v101"
MATRIX = BASE / "native-disjoint/matrix.json"
PLAN = BASE / "disjoint-reach-plan.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_available() -> bool:
    if not MATRIX.is_file():
        return False
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return False
    if matrix["plan_sha256"] != sha(PLAN):
        raise ValueError("V101 matrix does not match its frozen plan")
    cases = sorted(matrix["cases"], key=lambda case: (
        case["pass"],
        not case["automatic_alternate_gaze_specialist_applied"],
        not case["automatic_rear_contact_bias_applied"],
        case["contact_s"],
        case["seed"],
    ))
    for case in cases:
        video = MATRIX.parent / f"seed-{case['seed']}-disjoint.mp4"
        if video.is_file():
            continue
        report, trace = ROOT / case["result"], ROOT / case["trace"]
        if not report.is_file() or not trace.is_file():
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
            print(f"waiting for complete {case['id']} trace: {exc}", flush=True)
    return matrix["complete"] and all(
        (MATRIX.parent / f"seed-{case['seed']}-disjoint.mp4").is_file()
        for case in matrix["cases"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    args = parser.parse_args()
    for _ in range(240 if args.watch else 1):
        if render_available():
            return
        if args.watch:
            time.sleep(60)


if __name__ == "__main__":
    main()
