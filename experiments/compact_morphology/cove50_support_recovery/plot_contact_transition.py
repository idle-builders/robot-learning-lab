"""Plot paired contact, effort and slip phases for the Cove-50 force probe."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-contact-transition-20261010"
COLORS = {"Open loop": "#325f78", "Load feedback": "#b9742c"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_side(side: str) -> None:
    cases = [
        ("Open loop", ROOT / f"trials/sameworker-openloop-{side}-nominal.json"),
        ("Load feedback", ROOT / f"trials/loadroll-{side}-nominal.json"),
    ]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10.5, 10.5),
                             layout="constrained")
    manifest = []
    for label, path in cases:
        data = json.loads(path.read_text())
        assert data["body_sha256"] == sha(
            REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
        rows = [row for row in data["timeseries"] if 3 <= row["time_s"] <= 6.6]
        time = [row["time_s"] for row in rows]
        color = COLORS[label]
        axes[0].plot(time, [100 * row["stance_foot_load_fraction"] for row in rows],
                     color=color, lw=2, label=label)
        axes[1].plot(time, [1000 * row["swing_sole_z_m"] for row in rows],
                     color=color, lw=2)
        side_indices = [i for i, name in enumerate(data["leg_joint_names"])
                        if name.startswith(side + "_")]
        axes[2].plot(time, [100 * max(row["leg_effort_nm"][i] /
                                       data["leg_rated_effort_nm"][i]
                                       for i in side_indices) for row in rows],
                     color=color, lw=2)
        axes[3].plot(time, [1000 * row["stance_slip_m"] for row in rows],
                     color=color, lw=2)
        axes[4].plot(time, [row.get("swing_load_feedback_roll_rad", 0) for row in rows],
                     color=color, lw=2)
        manifest.append({"case": label, "report": str(path.relative_to(REPO)),
                         "sha256": sha(path)})
    axes[0].axhline(96, color="#668054", ls="--", lw=1.1, label="Target load")
    axes[1].axhline(25, color="#668054", ls="--", lw=1.1, label="Sole-center gate")
    axes[2].axhline(100, color="#ae4d3b", ls="--", lw=1.1, label="Rated effort")
    axes[3].axhline(15, color="#ae4d3b", ls="--", lw=1.1, label="Slip gate")
    for ax in axes:
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Stance-foot load (%)")
    axes[1].set_ylabel("Swing sole center (mm)")
    axes[2].set_ylabel("Peak stance motor (% rated)")
    axes[3].set_ylabel("Stance sole slip (mm)")
    axes[4].set_ylabel("Swing-roll correction (rad)")
    axes[4].set_xlabel("Seconds from upright reset")
    axes[4].set_xlim(3, 6.6)
    axes[0].legend(ncol=3, frameon=False, fontsize=9, loc="upper left")
    for ax in (axes[1], axes[2], axes[3]):
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Cove-50 {side} support · measured load feedback and motor cost",
                 fontsize=15, weight="bold")
    out = ROOT / f"contact-transition-{side}.png"
    fig.savefig(out, dpi=170, facecolor="white")
    (ROOT / f"contact-transition-{side}-manifest.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-contact-transition-plot.v1",
        "plot": str(out.relative_to(REPO)),
        "note": "Authored selected-motor diagnostic with simulator foot-force feedback; sole-center height is not exclusive support or policy qualification.",
        "cases": manifest,
    }, indent=2) + "\n")
    print(out)


def main() -> None:
    for side in ("right", "left"):
        plot_side(side)


if __name__ == "__main__":
    main()
