"""Plot saved native entry/closure observables without changing their verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.result.read_text())
    frames = report["timeseries"]
    t = np.array([row["time_s"] for row in frames])
    cube = np.array([row["cube_xyz_m"] for row in frames])
    entry_cube = np.asarray(report["entry_cube_xyz_m"])
    xy_drift_mm = np.linalg.norm((cube[:, :2] - entry_cube[:2]) * 1000, axis=1)
    jaw = np.array([row["right_gripper_rad"] for row in frames])
    strips = [
        ("Finger cube", np.array([row["finger_contact"] for row in frames]), "#258c6b"),
        ("Fixed jaw cube", np.array([row["palm_contact"] for row in frames]), "#2278a0"),
        ("Both feet", np.array([all(row["feet_grounded"]) for row in frames]), "#3f9e76"),
        ("Leg rated", np.array([row["leg_at_rated"] for row in frames]), "#e89042"),
        ("Hand floor ≥1 N", np.array([row["hand_floor_force_n"] >= 1 for row in frames]), "#cf5b4e"),
        ("Self contact", np.array([row["self_body_contact"] for row in frames]), "#ac5454"),
    ]
    fig, (motion_ax, contact_ax) = plt.subplots(2, 1, figsize=(12, 6.3), sharex=True,
                                                gridspec_kw={"height_ratios": [2, 1.45],
                                                             "hspace": .13})
    fig.patch.set_facecolor("#f5f7f7")
    for ax in (motion_ax, contact_ax):
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e1e6e6", linewidth=.7)
        for boundary, label in ((6, "preposition"), (7.5, "reach"),
                                (15.5, "aperture entry"),
                                (15.5 + report["entry_duration_s"], "close"),
                                (15.5 + report["entry_duration_s"] +
                                 report["close_duration_s"], "hold")):
            ax.axvline(boundary, color="#84949b", linestyle="--", linewidth=.8)
            if ax is motion_ax:
                ax.text(boundary + .04, .99, label, transform=ax.get_xaxis_transform(),
                        va="top", fontsize=8, color="#607079", rotation=90)
    motion_ax.plot(t, xy_drift_mm, color="#315f84", linewidth=1.4,
                   label="Cube XY from entry (mm)")
    motion_ax.plot(t, (cube[:, 2] - entry_cube[2]) * 1000,
                   color="#ad6348", linewidth=1.2, label="Cube height from entry (mm)")
    motion_ax.plot(t, jaw * 20, color="#6d5b9a", linewidth=1.3,
                   label="Jaw angle × 20 (rad)")
    motion_ax.axhline(5, color="#b8c4c9", linestyle=":", linewidth=.8)
    motion_ax.set_ylabel("Cube motion / scaled jaw")
    motion_ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    for idx, (label, values, color) in enumerate(strips):
        y = len(strips) - idx - 1
        contact_ax.fill_between(t, y, y + .7, where=values, step="post", color=color)
    contact_ax.set_yticks(np.arange(len(strips)) + .35,
                          [row[0] for row in reversed(strips)])
    contact_ax.tick_params(axis="y", length=0, pad=7)
    contact_ax.set_ylim(-.1, len(strips) + .1)
    contact_ax.set_xlim(0, report["duration_s"])
    contact_ax.set_xlabel("Native simulation time (s)")
    fig.suptitle(f"Cove-50 candidate aperture entry · seed {report['seed']} · "
                 f"{report['rail_v']:g} V / {report['delay_ms']:g} ms",
                 fontsize=13, fontweight="bold")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
