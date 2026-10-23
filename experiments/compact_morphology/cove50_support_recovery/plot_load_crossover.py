"""Plot selected-body right-stance support-load crossover traces."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-load-crossover-20261010"
CASES = [
    ("0.30 · ankle pitch", REPO / "artifacts/ape-cove50-support-recovery-20261009/native-midload-pitch06/right-nominal.json", "#547596"),
    ("0.33 · hip pitch", ROOT / "trials/shift033-right-nominal.json", "#23887e"),
    ("0.345 · hip roll", ROOT / "trials/shift0345-right-nominal.json", "#bd792d"),
    ("0.36 · hip roll", REPO / "artifacts/ape-cove50-support-recovery-20261009/native-highload-pitch06/right-nominal.json", "#7955a4"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.5, 8), layout="constrained")
    manifest = []
    for label, path, color in CASES:
        d = json.loads(path.read_text())
        assert d["body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
        rows = [r for r in d["timeseries"] if r["time_s"] >= 3]
        t = [r["time_s"] for r in rows]
        axes[0].plot(t, [100 * r["stance_foot_load_fraction"] for r in rows], color=color, lw=2, label=label)
        axes[1].plot(t, [abs(1000 * r["com_minus_stance_world_xy_m"][1]) for r in rows], color=color, lw=2)
        axes[2].plot(t, [1000 * r["swing_sole_z_m"] for r in rows], color=color, lw=2)
        first = next((r for r in rows if r["rated_leg_joints"]), None)
        if first:
            axes[0].scatter(first["time_s"], 100 * first["stance_foot_load_fraction"], color=color,
                            edgecolors="white", linewidths=.7, zorder=3, s=48)
        manifest.append({"case": label, "report": str(path.relative_to(REPO)), "sha256": sha(path)})
    axes[1].axhline(35, color="#a84a3e", ls="--", lw=1.2, label="Sole half-width")
    axes[2].axhline(25, color="#668054", ls="--", lw=1.2, label="Sole-center gate")
    axes[0].set_ylabel("Stance-foot load (%)")
    axes[1].set_ylabel("Horizontal COM offset (mm)")
    axes[2].set_ylabel("Swing sole center height (mm)")
    axes[2].set_xlabel("Seconds from upright reset")
    axes[2].set_xlim(3, 6.7)
    axes[0].legend(ncol=2, loc="upper left", frameon=False, fontsize=9)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    axes[2].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cove-50 right foot release · pre-lift load crossover", fontsize=15, weight="bold")
    out = ROOT / "load-crossover-right.png"
    fig.savefig(out, dpi=170, facecolor="white")
    (ROOT / "load-crossover-right-manifest.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-load-crossover-plot.v1",
        "plot": str(out.relative_to(REPO)),
        "note": "First rated-joint time is marked on the stance-load panel; joint name appears in each legend label. Authored native diagnostics, all fail full support gate.",
        "cases": manifest,
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
