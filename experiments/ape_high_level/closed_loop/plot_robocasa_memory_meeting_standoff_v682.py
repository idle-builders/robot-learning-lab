"""Evaluator-only sketch of V680 meeting poses and later prop paths."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-memory-meeting-standoff-v680"
PRIOR = ROOT / "artifacts/ape-robocasa-memory-early-hold-select-v671"


def main() -> None:
    episodes = json.loads((OUT / "report.json").read_text())["episodes"]
    trajectory = json.loads((PRIOR / "forecast-coverage-diagnostic-v673.json").read_text())[
        "plot_data"]
    points = {}
    for branch in ("current_meeting", "standoff_meeting"):
        ep = next(ep for ep in episodes if ep["memory_return"] and
                  ep["memory_return"]["run_label"] ==
                  f"toward_seed6323_{branch}" and
                  ep["memory_return"]["source"] == "range")
        first = next(row for row in ep["trace"] if row["stage"] ==
                     "predeparture_guard")
        points[branch] = np.asarray(first["evaluator_actual_xy_m"])
    fig, ax = plt.subplots(figsize=(7.5, 5.7))
    for label, color in (("toward", "#c62828"), ("away", "#1565c0")):
        path = np.asarray(trajectory["actual_evaluator"][label])
        ax.plot(path[:, 0], path[:, 1], lw=2.6, color=color,
                label=f"later {label} prop path (evaluator)")
    for branch, color, marker, name in (
            ("current_meeting", "#7b1fa2", "o", "current owner-show pose"),
            ("standoff_meeting", "#ef6c00", "s", "30 cm south standoff")):
        xy = points[branch]
        ax.add_patch(Circle(xy, .17, fill=False, ec=color, lw=2, alpha=.7))
        ax.scatter(*xy, color=color, marker=marker, s=80, zorder=5,
                   label=name)
    ax.annotate("30 cm", xy=(1.37, -3.5), fontsize=9, color="#455a64")
    ax.set(xlabel="room x (m)", ylabel="room y (m)",
           title="The farther owner-show pose lies in the later prop corridor")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(.55, 2.05)
    ax.set_ylim(-4.15, -2.25)
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "meeting-standoff-versus-prop-paths-v682.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
