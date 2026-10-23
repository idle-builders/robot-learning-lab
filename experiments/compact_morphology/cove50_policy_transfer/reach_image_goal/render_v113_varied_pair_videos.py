"""Render both visits in each varied RGB training pair into the inspector."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v113"
MATRIX = BASE / "native-varied-paired-contact-data/matrix.json"
PLAN = BASE / "varied-paired-contact-data-plan.json"
RENDERER = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.render_observable_video"
SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v113_varied_pairs"
PAIR_DATASET = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.build_v118_observable_pairs"
POSTFREEZE_SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.score_v118_postfreeze_varied"
NEIGHBOR_SUMMARY = "experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v117_neighborhood"
NEIGHBOR_MATRIX = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v117/native-neighborhood-transfer/matrix.json"


def main() -> None:
    if not MATRIX.is_file():
        return
    try:
        matrix = json.loads(MATRIX.read_text())
    except json.JSONDecodeError:
        return
    if matrix["plan_sha256"] != hashlib.sha256(PLAN.read_bytes()).hexdigest():
        raise ValueError("V113 matrix does not match its frozen plan")
    for row in matrix["cases"]:
        for variant in ("baseline", "shifted"):
            report = ROOT / row[f"{variant}_result"]
            trace = ROOT / row[f"{variant}_trace"]
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
                print(f"waiting for complete {row['id']} {variant} trace: {exc}",
                      flush=True)
    subprocess.run([sys.executable, "-m", SUMMARY], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", PAIR_DATASET, "--allow-partial"],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "-m", POSTFREEZE_SUMMARY], cwd=ROOT,
                   check=True, stdout=subprocess.DEVNULL)
    if NEIGHBOR_MATRIX.is_file():
        subprocess.run([sys.executable, "-m", NEIGHBOR_SUMMARY], cwd=ROOT,
                       check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
