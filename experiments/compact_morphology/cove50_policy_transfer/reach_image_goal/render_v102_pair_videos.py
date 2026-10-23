"""Publish native videos as locally synced matched route visits arrive."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SUMMARY = Path(__file__).with_name("summarize_v102_pairs.py")
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def render_available(version: int) -> bool:
    matrix_path = ARTIFACTS / f"v{version}/native-opened-pairs/matrix.json"
    if not matrix_path.is_file():
        return False
    try:
        matrix = json.loads(matrix_path.read_text())
    except json.JSONDecodeError:
        return False  # The remote sync may still be replacing this snapshot.
    for case in matrix["cases"]:
        route = "automatic" if case["id"].startswith("automatic-") else "forced"
        video = matrix_path.parent / f"seed-{case['seed']}-{route}.mp4"
        if video.is_file():
            continue
        report, trace = ROOT / case["result"], ROOT / case["trace"]
        if not report.is_file() or not trace.is_file():
            continue
        temp = video.with_name(video.stem + ".tmp.mp4")
        temp.unlink(missing_ok=True)
        command = [sys.executable, "-m", RENDERER, "--report", str(report),
                   "--trace", str(trace), "--video", str(temp)]
        try:
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                           text=True)
            os.replace(temp, video)
            print(f"published {video.name}", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            temp.unlink(missing_ok=True)
            print(f"waiting for complete {case['id']} trace: {exc}", flush=True)
    try:
        subprocess.run([sys.executable, str(SUMMARY), "--version", str(version)],
                       cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError as exc:
        print(f"waiting to publish synced matrix: {exc}", flush=True)
    return matrix["complete"] and all(
        (matrix_path.parent / f"seed-{case['seed']}-"
         f"{'automatic' if case['id'].startswith('automatic-') else 'forced'}.mp4").is_file()
        for case in matrix["cases"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--version", type=int, choices=(102, 103, 104), default=102)
    args = parser.parse_args()
    for _ in range(240 if args.watch else 1):
        if render_available(args.version):
            return
        if args.watch:
            time.sleep(60)


if __name__ == "__main__":
    main()
