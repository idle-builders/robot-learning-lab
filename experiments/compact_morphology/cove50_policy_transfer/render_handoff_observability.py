"""Render a compact native trace timeline for one Cove handoff case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    frames = report["timeseries"]
    times = np.array([frame["time_s"] for frame in frames])
    distance_cm = np.array([frame["cube_to_hand_distance_m"] * 100 for frame in frames])
    strips = [
        ("Finger on cube", np.array([frame["finger_contact"] for frame in frames])),
        ("Both feet", np.array([all(frame["feet_grounded"]) for frame in frames])),
        ("Leg at rated", np.array([frame["leg_at_rated"] for frame in frames])),
        ("Self contact", np.array([frame["self_body_contact"] for frame in frames])),
        ("Hand floor ≥1 N", np.array([frame["hand_floor_force_n"] >= 1 for frame in frames])),
    ]
    fig, (distance_ax, state_ax) = plt.subplots(
        2, 1, figsize=(11, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1.55], "hspace": .12},
    )
    fig.patch.set_facecolor("#f7f9fa")
    for ax in (distance_ax, state_ax):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dce4e8", linewidth=.7)
        ax.set_xlim(0, report["duration_s"])
        if report["entry"] == "continuous":
            for boundary in (6, 7.5):
                ax.axvline(boundary, color="#8796a3", linestyle="--", linewidth=1)
    distance_ax.plot(times, distance_cm, color="#245a78", linewidth=1.6)
    distance_ax.fill_between(times, 0, distance_cm, where=strips[0][1],
                             color="#35a982", alpha=.18)
    distance_ax.set_ylabel("Hand center to cube (cm)")
    distance_ax.set_ylim(bottom=0)
    distance_ax.text(.01, .98, "Green shading = fingertip contact", transform=distance_ax.transAxes,
                     ha="left", va="top", fontsize=9, color="#235b49")
    for row, (label, values) in enumerate(strips):
        baseline = len(strips) - row - 1
        state_ax.fill_between(times, baseline, baseline + .72, where=values,
                              step="post", color="#35a982" if row < 2 else "#e29351")
    state_ax.set_yticks(np.arange(len(strips)) + .36,
                        [label for label, _ in reversed(strips)])
    state_ax.tick_params(axis="y", length=0, pad=8)
    state_ax.set_ylim(-.15, len(strips) + .15)
    state_ax.set_xlabel("Native simulation time (s)")
    contact_s = report["phases"]["reach"]["continuous_finger_contact_s"]
    fig.suptitle(
        f"Cove-50 handoff · seed {report['seed']} · {report['rail_v']:g} V / {report['delay_ms']:g} ms"
        f" · longest contact {contact_s:.2f} s",
        fontsize=13, fontweight="bold", y=.99,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
