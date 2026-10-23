"""Render the staged weak-prior target-shift pilot for the evidence inspector."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v111"
MATRIX = BASE / "native-weak-prior-shift-pilot/matrix.json"
PLAN = BASE / "weak-prior-shift-pilot-plan.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v111_prior_shift_pilot"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V111 matrix does not match its frozen plan")
    for case in matrix["cases"]:
        video = MATRIX.parent / f"seed-{case['seed']}-{case['id']}.mp4"
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
    subprocess.run([sys.executable, "-m", SUMMARY], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
