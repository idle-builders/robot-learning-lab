"""Render opened contact-target probes with observed RGB and native traces."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v108"
MATRIX = BASE / "native-opened-goal-shift/matrix.json"
PLAN = BASE / "opened-goal-shift-plan.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
SUMMARIZER = Path(__file__).with_name("summarize_v108_contact_target.py")


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V108 matrix does not match its frozen plan")
    for case in sorted(matrix["cases"], key=lambda row: (row["pass"], row["seed"])):
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
    subprocess.run([sys.executable, str(SUMMARIZER)], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
