"""Plot full retained coverage traces in the isolated plotting environment."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    data = json.loads((folder / "plot-input.json").read_text())
    fig, axes = plt.subplots(
        len(data["cases"]), 2, figsize=(12, 2.8 * len(data["cases"])), squeeze=False
    )
    for row, pair in zip(data["cases"], axes, strict=True):
        m = row["measurement"]
        if "end_time_s" not in m:
            for ax in pair:
                ax.text(
                    0.5,
                    0.5,
                    f"{row['name']}\n{m['status'].upper()} · no complete trace",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_axis_off()
            continue
        values = np.loadtxt(Path(row["source"]) / row["name"] / "trace.dat", skiprows=1)
        blocks = np.array_split(values, min(1400, len(values)))
        times = np.array([b[:, 0].mean() * 1000 for b in blocks])
        for ax, col, label in zip(
            pair, (1, 3), ("Output (V)", "Inductor (A)"), strict=True
        ):
            lo = np.array([b[:, col].min() for b in blocks])
            hi = np.array([b[:, col].max() for b in blocks])
            ax.fill_between(times, lo, hi, color="#238a78", alpha=0.8)
            ax.plot(times, (lo + hi) / 2, color="#238a78", lw=0.6)
            ax.axvspan(0, m["evaluation_start_s"] * 1000, color="#888888", alpha=0.15)
            ax.set(xlabel="Time (ms)", ylabel=label, title=row["name"])
            ax.grid(alpha=0.2)
        for bound in (4.75, 5.25):
            pair[0].axhline(bound, color="#ae543c", ls="--", lw=0.8)
    fig.suptitle(
        "Camera-inclusive vendor-model probes · full traces\nWarm initialization is not physical power-up; cold load is a resistor",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(folder / "waveforms.png", dpi=150)
    plt.close(fig)
    (folder / "plot-runtime.json").write_text(
        json.dumps(
            {"matplotlib": matplotlib.__version__, "numpy": np.__version__}, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    plot(Path(sys.argv[1]))
