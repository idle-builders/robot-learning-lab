"""Plot earlier opened RGB goal error with legal stronger gaze."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v31"


def main() -> None:
    report = json.loads((OUT / "prior-offline-report.json").read_text())
    rows = report["rows"]
    accepted = [row for row in rows if row["template_applied"]]
    x = np.arange(len(accepted))
    base = [row["baseline_error_mm_scoring_only"] for row in accepted]
    matched = [row["template_error_mm_scoring_only"] for row in accepted]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    ax.scatter(x, base, s=18, color="#9aa8b7", alpha=.65, label="Saved pixel regression")
    ax.scatter(x, matched, s=20, color="#278c7d", alpha=.8, label="Rendered template")
    ax.axhline(3, linestyle="--", linewidth=1.5, color="#b55348", label="3 mm gate")
    ax.set_ylim(0, max(10, max(base) * 1.1))
    ax.set_xlabel("Opened native view and varied toy/head/camera sample")
    ax.set_ylabel("RGB goal XY error (mm; simulator truth scores only)")
    ax.set_title("V31 · Earlier opened images also pass the visibility and goal gate",
                 loc="left", fontweight="bold")
    ax.text(.99, .97, f"{len(accepted)}/{len(rows)} visible · offline gate passed",
            transform=ax.transAxes, ha="right", va="top", color="#227962")
    ax.grid(axis="y", alpha=.18)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "prior-offline-evaluation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
