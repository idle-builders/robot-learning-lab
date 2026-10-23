"""Plot right-stance load reversal and slip in the ankle-release trial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-ankle-release-20261010"
CASES = [
    ("Ankle 0", REPO / "artifacts/ape-cove50-load-crossover-20261010/trials/shift033-right-nominal.json", "#325f78"),
    ("Ankle +0.10 rad", ROOT / "trials/ankle01-right-nominal.json", "#b9742c"),
    ("Ankle +0.20 rad", ROOT / "trials/ankle02-right-nominal.json", "#7f59a0"),
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10.5, 8), layout="constrained")
    manifest = []
    for label, path, color in CASES:
        d = json.loads(path.read_text())
        assert d["body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
        rows = [r for r in d["timeseries"] if 3 <= r["time_s"] <= 5.7]
        t = [r["time_s"] for r in rows]
        axes[0].plot(t, [100 * r["stance_foot_load_fraction"] for r in rows], color=color, lw=2, label=label)
        axes[1].plot(t, [1000 * r["swing_sole_z_m"] for r in rows], color=color, lw=2)
        axes[2].plot(t, [1000 * r["stance_slip_m"] for r in rows], color=color, lw=2)
        manifest.append({"case": label, "report": str(path.relative_to(REPO)), "sha256": sha(path)})
    axes[1].axhline(25, color="#668054", ls="--", lw=1.2, label="Sole-center gate")
    axes[2].axhline(15, color="#ae4d3b", ls="--", lw=1.2, label="Slip gate")
    axes[0].set_ylabel("Stance-foot load (%)")
    axes[1].set_ylabel("Swing sole center height (mm)")
    axes[2].set_ylabel("Stance sole displacement (mm)")
    axes[2].set_xlabel("Seconds from upright reset")
    axes[2].set_xlim(3, 5.7)
    axes[0].legend(ncol=3, loc="upper left", frameon=False, fontsize=9)
    axes[1].legend(loc="upper left", frameon=False, fontsize=9)
    axes[2].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cove-50 right foot release · ankle target changes the load path", fontsize=15, weight="bold")
    out = ROOT / "ankle-release-right.png"
    fig.savefig(out, dpi=170, facecolor="white")
    (ROOT / "ankle-release-right-manifest.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-ankle-release-plot.v1",
        "plot": str(out.relative_to(REPO)),
        "note": "Authored native selected-motor replays. Sole-center height alone does not establish exclusive support. All cases fail the full gate.",
        "cases": manifest,
    }, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
