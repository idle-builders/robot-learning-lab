"""Static min/max envelope figure; run with the isolated Matplotlib environment."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(folder):
    report = json.loads((folder / "report.json").read_text())
    rows = [row for row in report["cases"] if "plot_envelope" in row]
    fig, axes = plt.subplots(len(rows), 2, figsize=(12, 2.8 * len(rows)), squeeze=False)
    for row, pair in zip(rows, axes, strict=True):
        data = np.array(row["plot_envelope"])
        start = row["original"]["evaluation_start_s"] * 1000
        for ax, columns, label in zip(
            pair, [(1, 2), (3, 4)], ["Output (V)", "Inductor (A)"], strict=True
        ):
            ax.fill_between(
                data[:, 0],
                data[:, columns[0]],
                data[:, columns[1]],
                color="#238a78",
                alpha=0.8,
            )
            ax.axvspan(
                0, start, color="#a9a9a9", alpha=0.18, label="Outside evaluation"
            )
            ax.set(xlabel="Time (ms)", ylabel=label, title=row["name"])
            ax.grid(alpha=0.2)
        for bound in (4.75, 5.25):
            pair[0].axhline(bound, color="#ae543c", ls="--", lw=0.8)
    fig.suptitle(
        "Retained vendor-model probes · min/max per time bin\nWarm initialization is not a power-up result",
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
