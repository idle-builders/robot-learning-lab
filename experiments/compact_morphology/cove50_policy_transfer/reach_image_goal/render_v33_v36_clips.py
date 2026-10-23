"""Render selected native evidence clips for the opened V33-V36 lessons."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
RENDERER = Path(__file__).with_name("render_adaptive_video_v12.py")
CASES = (
    ("v33", "forced-route-screen", "seed-10534-locked", "seed-10534-locked-rescue"),
    ("v33", "forced-route-screen", "seed-10539-specialist", "seed-10539-specialist-rescue"),
    ("v33", "forced-route-screen", "seed-10513-live", "seed-10513-live-failure"),
    ("v34", "feedback-gap", "seed-10530", "seed-10530-feedback-rescue"),
    ("v34", "feedback-gap", "seed-10538", "seed-10538-feedback-failure"),
    ("v35", "wider-rgb-screen", "seed-10538", "seed-10538-template-rescue"),
    ("v36", "actor-screen", "seed-10538", "seed-10538-combined-rescue"),
    ("v36", "actor-screen", "seed-10513", "seed-10513-still-failing"),
    ("v36", "actor-screen", "seed-10608", "seed-10608-thin-retention"),
)


def main() -> None:
    for version, cohort, stem, name in CASES:
        source = BASE / version / cohort
        video = BASE / version / "clips" / f"{name}.mp4"
        subprocess.run((sys.executable, str(RENDERER),
                        "--report", str(source / f"{stem}.json"),
                        "--trace", str(source / f"{stem}.npz"),
                        "--video", str(video)), check=True)


if __name__ == "__main__":
    main()
