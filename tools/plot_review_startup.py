"""Render a retained SPICE transient using the isolated plotting environment."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def plot_startup(output, result):
    """Plot retained adaptive-step SPICE data, preserving extrema per time bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    path = "artifacts/logic-supply-startup-v1/startup/trace.dat"
    t, v = np.loadtxt(ROOT / path, skiprows=1, usecols=(0, 1)).T
    if not np.isclose(v.max(), result["whole_trace_peak_output_v"], atol=1e-8):
        raise ValueError("Startup plot disagrees with retained measurement")
    bins = np.minimum((t / t[-1] * 1000).astype(int), 999)
    low, high = np.full(1000, np.inf), np.full(1000, -np.inf)
    np.minimum.at(low, bins, v)
    np.maximum.at(high, bins, v)
    x = (np.arange(1000) + 0.5) * t[-1] * 1000 / 1000
    valid = np.isfinite(low)
    fig, ax = plt.subplots(figsize=(10, 3.3), layout="constrained")
    fig.patch.set_facecolor("#f4f3ee")
    ax.set_facecolor("#f4f3ee")
    ax.fill_between(x[valid], low[valid], high[valid], color="#427967", alpha=0.25)
    ax.plot(x[valid], (low[valid] + high[valid]) / 2, color="#427967", lw=1)
    ax.axvspan(8, 10, color="#b39259", alpha=0.13)
    ax.set(
        xlabel="Time (ms)",
        ylabel="Logic rail (V)",
        xlim=(0, 10),
        ylim=(-0.1, 5.5),
        title="Retained SPICE cold start · 12.6 V input · resistive load referenced to 0.6 A at 5 V",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(
        (ROOT / "artifacts/logic-supply-startup-v1/startup/report.json").read_text()
    )
    plot_startup(args.output, result)
