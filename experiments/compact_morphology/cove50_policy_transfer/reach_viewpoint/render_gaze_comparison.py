"""Show how the same native reach changed with fast and slow head-yaw ramps."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from screen_head_mount import OUT, ROOT


def main() -> None:
    sources = [
        ("Saved no-gaze", ROOT / "artifacts/ape-cove50-squat-learned-reach-20261010/v8/conditioned-route/development/seed-9921.json", "#405a72"),
        ("5–7 s ramp", OUT / "native-gaze/dev-9921.json", "#d85f4b"),
        ("0–7 s ramp", OUT / "native-gaze/dev-9921-slow.json", "#159878"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1.25, .8], "hspace": .13})
    fig.patch.set_facecolor("#f7f9fa")
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dce4e8", linewidth=.7)
        for boundary in (6, 7.5):
            ax.axvline(boundary, color="#a5afb8", linestyle="--", linewidth=1)
        ax.set_xlim(0, 16.5)
    for label, path, color in sources:
        report = json.loads(path.read_text())
        frames = report["timeseries"]
        time = np.array([r["time_s"] for r in frames])
        pelvis = np.array([r["pelvis_xyz_m"][2] * 100 for r in frames])
        axes[0].plot(time, pelvis, label=label, color=color, lw=1.7)
        contact = np.array([r["finger_contact"] for r in frames])
        axes[2].fill_between(time, 0, 1, where=contact, step="post", color=color,
                             alpha=.35 if label == "Saved no-gaze" else .58)
        if "head_yaw_rad" in frames[0]:
            yaw = np.array([r["head_yaw_rad"] for r in frames])
            axes[1].plot(time, yaw, color=color, lw=1.7, label=label)
            feet = np.array([all(r["feet_grounded"]) for r in frames])
            if not feet.all():
                axes[0].scatter(time[~feet][::5], pelvis[~feet][::5], s=5,
                                color=color, alpha=.5)
    axes[0].set_ylabel("Pelvis height (cm)")
    axes[0].set_ylim(0, 26)
    axes[0].legend(loc="upper right", frameon=False, ncol=3, fontsize=9)
    axes[1].set_ylabel("Attained head yaw (rad)")
    axes[1].set_ylim(-1.5, .1)
    axes[2].set_ylabel("Fingertip\ncontact")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Native no-reset simulation time (s)")
    fig.suptitle("Cove-50 gaze timing · same seed 9921 and saved reach actor",
                 fontsize=14, fontweight="bold", y=.99)
    fig.text(.12, .945, "Dashed boundaries: arm preposition at 6 s; reach at 7.5 s. Dots mark lost two-foot support.",
             fontsize=9, color="#536577")
    out = OUT / "native-gaze/gaze-timing-comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
