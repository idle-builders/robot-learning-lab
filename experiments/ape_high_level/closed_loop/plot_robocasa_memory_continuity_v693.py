"""Show the delayed-crossing departure failure and paired continuity hold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-robocasa-memory-joint-station-heldout-v689"
OUT = ROOT / "artifacts/ape-robocasa-memory-joint-station-continuity-v691"


def pick(path: Path) -> dict:
    return next(ep for ep in json.loads(path.read_text())["episodes"]
                if ep["memory_return"]["source"] == "range" and
                ep["memory_return"]["run_label"].startswith("away_shift-0.6"))


def main() -> None:
    before = pick(BASE / "report.json")
    after = pick(OUT / "report.json")
    start = np.asarray(next(row["evaluator_actual_xy_m"] for row in before["trace"]
                            if row["stage"] == "predeparture_guard"))
    commanded = [start]
    for row in before["trace"]:
        if row["stage"] != "memory_route_decision":
            continue
        commanded.extend(np.asarray(sub["evaluator_actual_xy_m"])
                         for sub in row["motion_substeps"]
                         if not sub["stopped_before_move"])
    old_path = np.asarray(commanded)
    old_hold = [row for row in before["trace"] if row["stage"] ==
                "memory_post_stop_hold"]
    prop = np.asarray([row["evaluator_prop_xy_m"] for row in old_hold])
    held = np.asarray(old_hold[0]["evaluator_actual_xy_m"])
    new_hold = np.asarray(next(row["evaluator_actual_xy_m"] for row in
                               after["trace"] if row["stage"] ==
                               "memory_post_stop_hold"))
    breach = next(row for row in old_hold if row["evaluator_proxy_gap_m"] < .05)
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    ax.plot(old_path[:, 0], old_path[:, 1], color="#d84315", lw=3,
            label="old item-return motion, 17–19 s")
    ax.plot(prop[:, 0], prop[:, 1], color="#1e5aa8", lw=3,
            label="delayed low-prop path while ape held, 19–20.8 s")
    for xy, color, label in ((held, "#d84315", "old held pose"),
                             (new_hold, "#00897b", "new held station")):
        ax.add_patch(Circle(xy, .17, fill=False, ec=color, lw=2))
        ax.scatter(*xy, color=color, s=90, zorder=5, label=label)
    ax.scatter(*breach["evaluator_prop_xy_m"], color="#1e5aa8", marker="x",
               s=105, zorder=5, label=f"first old <5 cm at {breach['time_s']:.2f} s")
    ax.annotate("same station before departure", xy=start,
                xytext=(1.62, -4.42),
                arrowprops={"arrowstyle": "->", "color": "#455a64"}, fontsize=9)
    ax.set(xlabel="room x (m)", ylabel="room y (m)",
           title="Stale range ID allowed departure into a later crossing")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(1.05, 1.88)
    ax.set_ylim(-4.52, -3.42)
    ax.grid(alpha=.22)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "delayed-crossing-track-continuity-v693.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
