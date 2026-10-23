"""Plot the frozen right-stance native COM feedback comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-phase-feedback-20261010"
CASES = [
    ("Open loop", REPO / "artifacts/ape-cove50-support-recovery-20261009/native-highload-pitch06/right-nominal.json", "#343f4d"),
    ("Edge 16 mm, gain 4", ROOT / "trials/edge16-right-nominal.json", "#217a8d"),
    ("Center, gain 4", ROOT / "trials/center0-right-nominal.json", "#bf752a"),
    ("Edge 16 mm, gain 8", ROOT / "trials/edge16fast-right-nominal.json", "#7653a4"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.8, 8.4), layout="constrained")
    manifest = []
    for label, path, color in CASES:
        d = json.loads(path.read_text())
        assert d["body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
        rows = [r for r in d["timeseries"] if 3 <= r["time_s"] <= 5.0]
        t = [r["time_s"] for r in rows]
        axes[0].plot(t, [-1000 * r["com_minus_stance_world_xy_m"][1] for r in rows], color=color, label=label, lw=2)
        axes[1].step(t, [int("right_hip_roll" in r["rated_leg_joints"]) for r in rows], color=color, where="post", lw=2)
        axes[2].plot(t, [1000 * r["swing_sole_z_m"] for r in rows], color=color, lw=2)
        manifest.append({"case": label, "report": str(path.relative_to(REPO)), "sha256": sha(path)})
    axes[0].axhline(35, color="#b33f37", ls="--", lw=1.3, label="Stance sole half-width")
    axes[2].axhline(25, color="#668054", ls="--", lw=1.3, label="Clearance gate")
    axes[0].set_ylabel("COM lateral from stance sole (mm)")
    axes[1].set_ylabel("Stance hip roll at rated effort")
    axes[1].set_yticks([0, 1], ["No", "Yes"])
    axes[2].set_ylabel("Swing sole center height (mm)")
    axes[2].set_xlabel("Seconds from upright reset")
    axes[2].set_xlim(3, 5)
    for ax in axes:
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(ncol=2, loc="upper left", fontsize=9, frameon=False)
    axes[2].legend(loc="upper left", fontsize=9, frameon=False)
    fig.suptitle("Cove-50 right foot release · nominal selected-motor replay", fontsize=15, weight="bold")
    output = ROOT / "phase-feedback-right.png"
    fig.savefig(output, dpi=170, facecolor="white")
    (ROOT / "phase-feedback-right-manifest.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-phase-feedback-plot.v1",
        "plot": str(output.relative_to(REPO)),
        "cases": manifest,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
