"""Plot the retained paired load-gated support traces for inspection."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/ape-cove50-gated-com-20261010/trials"
OUT = ROOT / "artifacts/ape-cove50-triggered-com-20261010"
COLORS = {"baseline": "#666b74", "0.15 s after release": "#007a78", "0.30 s after release": "#b56122"}


def main() -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
    for col, side in enumerate(("right", "left")):
        sign = 1 if side == "right" else -1
        for label, folder, stem in (
            ("baseline", BASE, f"baseline-reload-{side}-nominal"),
            ("0.15 s after release", OUT / "trials", f"triggered015-{side}-nominal"),
            ("0.30 s after release", OUT / "trials", f"triggered030-{side}-nominal"),
        ):
            report = json.loads((folder / f"{stem}.json").read_text())
            rows = report["timeseries"]
            time = [row["time_s"] for row in rows]
            com = [-sign * row["com_minus_stance_world_xy_m"][1] * 1000 for row in rows]
            load = [row["stance_foot_load_fraction"] for row in rows]
            hip_id = report["leg_joint_names"].index(f"{side}_hip_roll")
            hip_effort = [row["leg_effort_nm"][hip_id] /
                          report["leg_rated_effort_nm"][hip_id] for row in rows]
            for row_idx, values in enumerate((com, load, hip_effort)):
                axes[row_idx, col].plot(time, values, color=COLORS[label], label=label,
                                        linewidth=2.2 if label != "baseline" else 1.8,
                                        linestyle="--" if label == "baseline" else "-")
        axes[0, col].set_title(f"{side.title()} stance")
        for axis in axes[:, col]:
            axis.axvline(4.1, color="#9198a0", linewidth=1, linestyle=":")
            axis.set_xlim(3.8, 5.5)
            axis.grid(alpha=.18)
        axes[1, col].axhline(.96, color="#727982", linewidth=1, linestyle=":")
        axes[2, col].axhline(1, color="#9a2828", linewidth=1, linestyle=":")
        axes[2, col].set_xlabel("Time since upright start (s)")
    axes[0, 0].set_ylabel("COM toward stance sole (mm)")
    axes[1, 0].set_ylabel("Stance foot load fraction")
    axes[2, 0].set_ylabel("Stance hip-roll effort / catalog rated")
    axes[0, 0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Cove-50 foot release: delayed damping reduces COM drift but does not hold one foot",
                 fontsize=14)
    fig.savefig(OUT / "triggered-com-phase.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
