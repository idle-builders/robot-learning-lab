"""Evaluator-only overview of the joint-station pilot geometry."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-robocasa-memory-joint-station-v686"
BASELINE = ROOT / "artifacts/ape-robocasa-memory-meeting-standoff-v680/report.json"
PATHS = ROOT / "artifacts/ape-robocasa-memory-early-hold-select-v671/forecast-coverage-diagnostic-v673.json"


def first_watch(ep):
    return np.asarray(next(row["evaluator_actual_xy_m"] for row in ep["trace"]
                           if row["stage"] == "predeparture_guard"))


def main() -> None:
    base = json.loads(BASELINE.read_text())["episodes"]
    new = json.loads((OUT / "report.json").read_text())["episodes"]
    paths = json.loads(PATHS.read_text())["plot_data"]["actual_evaluator"]
    positions = {
        "current": first_watch(base[4]),
        "south": first_watch(base[5]),
        "joint": first_watch(new[2]),
    }
    chosen = next(row["chosen"] for row in new[2]["trace"] if
                  row["stage"] == "choose_owner_standoff")
    refuge = np.asarray(chosen["selected_refuge"]["target_xy_est_m"])
    fig, ax = plt.subplots(figsize=(8.0, 6.3))
    for direction, color in (("toward", "#c62828"), ("away", "#1565c0")):
        trail = np.asarray(paths[direction])
        ax.plot(trail[:, 0], trail[:, 1], lw=2.8, color=color,
                label=f"later {direction} prop path (evaluator)")
    for key, name, color, marker in (
            ("current", "current owner show", "#7b1fa2", "o"),
            ("south", "30 cm south standoff", "#ef6c00", "s"),
            ("joint", "map-backed station", "#00897b", "D")):
        xy = positions[key]
        ax.add_patch(Circle(xy, .17, fill=False, ec=color, lw=2))
        ax.scatter(*xy, c=color, marker=marker, s=95, zorder=5, label=name)
    ax.scatter(*refuge, c="#00897b", marker="*", s=165, zorder=5,
               label="mapped refuge option (unused)")
    ax.plot([positions["current"][0], positions["joint"][0]],
            [positions["current"][1], positions["joint"][1]],
            color="#00897b", linestyle="--", alpha=.7)
    ax.set(xlabel="room x (m)", ylabel="room y (m)",
           title="Owner-show station lies outside both tested later prop corridors")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(.55, 2.25)
    ax.set_ylim(-4.65, -2.65)
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "joint-station-versus-prop-paths-v688.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
