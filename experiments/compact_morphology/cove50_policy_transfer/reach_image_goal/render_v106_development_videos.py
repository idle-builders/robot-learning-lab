"""Render synced V106 full native development cases with their observed RGB view."""

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
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v106"
MATRIX = BASE / "native-varied-development/matrix.json"
PLAN = BASE / "varied-native-development-plan.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"


def render_available() -> bool:
    if not MATRIX.is_file():
        return False
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return False
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V106 matrix does not match its frozen plan")
    cases = sorted(matrix["cases"], key=lambda case: (
        case["pass"], case["margin_1p2_s_met"], case["contact_s"],
        case["development_role"] != "failed_score_condition_new_start",
        case["seed"],
    ))
    for case in cases:
        video = MATRIX.parent / f"seed-{case['seed']}-development.mp4"
        if video.is_file():
            continue
        report, trace = ROOT / case["result"], ROOT / case["trace"]
        if not report.is_file() or not trace.is_file():
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
            print(f"waiting for complete {case['id']} trace: {exc}", flush=True)
    return matrix["complete"] and all(
        (MATRIX.parent / f"seed-{case['seed']}-development.mp4").is_file()
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
