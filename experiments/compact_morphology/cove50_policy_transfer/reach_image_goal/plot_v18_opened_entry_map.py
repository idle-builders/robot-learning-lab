"""Show why the v16 contact misses resist one entry-error threshold."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v18/opened-entry-map.png"
COLORS = {
    "v16_locked": "#5a7f9e",
    "v16_v10_locked": "#d29446",
    "v16_v10_height_locked": "#9f70bd",
    "v16_live": "#26a68c",
    "v16_specialist_locked": "#9c6471",
    "v16_bias_y_locked": "#7aab4a",
}


def main() -> None:
    matrix = json.loads((BASE / "v16/fresh-reserved/matrix.json").read_text())
    if matrix["case_count"] != 36 or matrix["pass_count"] != 30:
        raise ValueError("Expected sealed v16 fresh cohort")
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    shown = set()
    for row in matrix["cases"]:
        report = json.loads((ROOT / row["result"]).read_text())
        x, y = [1000 * value for value in report["selection_features"]["entry_error_xy_m"]]
        mode = row["selection_mode"]
        shown.add(mode)
        ax.scatter(x, y, s=88 if row["pass"] else 115,
                   marker="o" if row["pass"] else "X", color=COLORS[mode],
                   edgecolors="white" if row["pass"] else "#a22537",
                   linewidths=1.3, zorder=3)
        if not row["pass"] or row["seed"] in (10403, 10419):
            ax.annotate(str(row["seed"]), (x, y), xytext=(5, 5),
                        textcoords="offset points", fontsize=8,
                        weight="bold" if not row["pass"] else "normal")
    ax.axvline(0, color="#b9c3ca", linewidth=.8)
    ax.axhline(0, color="#b9c3ca", linewidth=.8)
    ax.set_xlabel("Observed entry X error (mm)")
    ax.set_ylabel("Observed entry Y error (mm)")
    ax.set_title("V16 fresh reach · failures interleave with passing starts in RGB entry space",
                 loc="left", fontweight="bold")
    ax.text(.02, .02, "Circles pass · X marks fail · Direct Frame rejected · 30/36 scored",
            transform=ax.transAxes, fontsize=9, color="#475569")
    ax.grid(alpha=.16)
    handles = [Line2D([0], [0], marker="o", color="none",
                      markerfacecolor=COLORS[mode], markeredgecolor="white",
                      markersize=9,
                      label=mode.replace("v16_", "").replace("_locked", ""))
               for mode in COLORS if mode in shown]
    ax.legend(handles=handles, frameon=False, title="Actual saved branch", fontsize=8)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
