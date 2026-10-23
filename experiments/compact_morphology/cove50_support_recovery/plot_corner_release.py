"""Plot native dynamic corner clearance, load and stance slip."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts/ape-cove50-corner-release-20261010"
COLORS = {"baseline-reload": "#666b74", "corner006": "#007a78", "corner010": "#b56122"}
LABELS = {"baseline-reload": "0 rad", "corner006": "+0.06 rad", "corner010": "+0.10 rad"}


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(ROOT / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml"))
    data = mujoco.MjData(model)
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True, layout="constrained")
    for col, side in enumerate(("right", "left")):
        swing = "left" if side == "right" else "right"
        geom = model.geom(f"{swing}_sole").id
        half = model.geom_size[geom]
        for case in ("baseline-reload", "corner006", "corner010"):
            stem = f"{case}-{side}-nominal"
            report = json.loads((OUT / "trials" / f"{stem}.json").read_text())
            with np.load(OUT / "trials" / f"{stem}.npz") as saved:
                qpos = saved["qpos"].copy()
            rows = report["timeseries"]
            times = [row["time_s"] for row in rows]
            corners = []
            for state in qpos:
                data.qpos[:] = state
                mujoco.mj_forward(model, data)
                rotation = data.geom_xmat[geom].reshape(3, 3)
                corners.append(min(float((data.geom_xpos[geom] + rotation @ np.array(
                    [x * half[0], y * half[1], -half[2]]))[2]) * 1000
                    for x in (-1, 1) for y in (-1, 1)))
            values = (corners,
                      [row["stance_foot_load_fraction"] for row in rows],
                      [row["stance_slip_m"] * 1000 for row in rows])
            for index, series in enumerate(values):
                axes[index, col].plot(times, series, color=COLORS[case], label=LABELS[case],
                                      linewidth=2.2 if case != "baseline-reload" else 1.8,
                                      linestyle="--" if case == "baseline-reload" else "-")
        axes[0, col].set_title(f"{side.title()} stance")
        for axis in axes[:, col]:
            axis.axvline(4.1, color="#9198a0", linewidth=1, linestyle=":")
            axis.set_xlim(4.0, 5.6)
            axis.grid(alpha=.18)
        axes[0, col].axhline(0, color="#9a2828", linewidth=1, linestyle=":")
        axes[1, col].axhline(.96, color="#727982", linewidth=1, linestyle=":")
        axes[2, col].axhline(15, color="#9a2828", linewidth=1, linestyle=":")
        axes[2, col].set_xlabel("Time since upright start (s)")
    axes[0, 0].set_ylabel("Lowest swing sole corner (mm)")
    axes[1, 0].set_ylabel("Stance foot load fraction")
    axes[2, 0].set_ylabel("Stance sole slip (mm)")
    axes[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Cove-50: fixed-pose ankle clearance reverses under native contact and load",
                 fontsize=14)
    fig.savefig(OUT / "corner-release-phase.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
