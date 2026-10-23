"""Plot the retained V673 evaluator trajectory diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-memory-early-hold-select-v671"


def main() -> None:
    diagnostic = json.loads((OUT / "forecast-coverage-diagnostic-v673.json").read_text())
    data = diagnostic["plot_data"]
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    for index, points in enumerate(data["actor_hypotheses"].values()):
        path = np.asarray(points)
        ax.plot(path[:, 0], path[:, 1], color="#999999", lw=1.2,
                ls="--", label="actor forecast" if index == 0 else None)
    for direction, color in (("toward", "#c62828"), ("away", "#1565c0")):
        path = np.asarray(data["actual_evaluator"][direction])
        ax.plot(path[:, 0], path[:, 1], color=color, lw=2.8,
                label=f"actual {direction} (evaluator)")
    ax.scatter(*data["actor_track_xy_m"], c="black", s=45,
               label="actor range track at 15.8 s")
    ax.scatter(*data["actor_selected_goal_xy_m"], c="#ef6c00",
               s=140, marker="*", label="selected waiting place")
    ax.set(xlabel="room x (m)", ylabel="room y (m)",
           title="Early route forecast misses both later prop paths")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "early-hold-forecast-versus-actual-v673.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
