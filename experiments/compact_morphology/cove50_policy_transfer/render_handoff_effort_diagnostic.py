"""Plot paired right-ankle effort on an opened Cove stress start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    matrix = json.loads(args.matrix.read_text())
    fig, ax = plt.subplots(figsize=(10.5, 4))
    fig.patch.set_facecolor("#f7f9fa")
    ax.set_facecolor("white")
    colors = {10: "#bb5a2d", 11: "#2b719c", 12: "#338c66"}
    for case in matrix["cases"]:
        report = json.loads(Path(case["result"]).read_text())
        index = report["leg_joint_order"].index("right_ankle_pitch")
        rated = report["leg_rated_limit_nm"][index]
        frames = [frame for frame in report["timeseries"] if frame["phase"] == "reach"]
        ax.plot([frame["time_s"] for frame in frames],
                [frame["leg_effort_nm"][index] / rated for frame in frames],
                label=f"{case['rail_v']} V · {rated:.3f} N·m rating",
                color=colors[case["rail_v"]], linewidth=1.7)
    ax.axhline(1, color="#222e38", linestyle="--", linewidth=1.1, label="Catalog-rated boundary")
    ax.set(xlabel="Native simulation time (s)", ylabel="Right ankle pitch effort / rail-specific rating",
           xlim=(7.5, 16.5), ylim=(0, 1.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dce4e8", linewidth=.7)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("Opened heavy low-grip start · same seed, toy and 10 ms delay", fontsize=13,
                 fontweight="bold", y=.98)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
