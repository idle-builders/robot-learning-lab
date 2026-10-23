"""Plot the paired support force, COM, effort and slip comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "artifacts/ape-cove50-coupled-transition-20261010"
ORDER = ("roll08_com0", "roll04_com0", "roll04_com2", "roll08_com2",
         "stance_in04", "stance_out04")
LABELS = {
    "roll08_com0": "Roll cap .08 · COM 0",
    "roll04_com0": "Roll cap .04 · COM 0",
    "roll04_com2": "Roll cap .04 · COM gain 2",
    "roll08_com2": "Roll cap .08 · COM gain 2",
    "stance_in04": "Roll cap .04 · stance −.04",
    "stance_out04": "Roll cap .04 · stance +.04",
}
COLORS = {
    "roll08_com0": "#8c969e", "roll04_com0": "#315b79",
    "roll04_com2": "#d09335", "roll08_com2": "#bd5544",
    "stance_in04": "#478563", "stance_out04": "#7455a2",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_side(side: str, cases: list[dict]) -> None:
    mirror = -1 if side == "right" else 1
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(11.5, 11.6), layout="constrained")
    manifest = []
    for name in ORDER:
        case = next(row for row in cases if row["case"] == f"{name}/{side}")
        path = REPO / case["report"]
        assert sha(path) == case["report_sha256"]
        data = json.loads(path.read_text())
        rows = [row for row in data["timeseries"] if 3 <= row["time_s"] <= 6.6]
        time = [row["time_s"] for row in rows]
        color = COLORS[name]
        axes[0].plot(time, [mirror * 1000 * row["com_minus_stance_world_xy_m"][1]
                            for row in rows], color=color, lw=1.8, label=LABELS[name])
        axes[1].plot(time, [100 * row["stance_foot_load_fraction"] for row in rows],
                     color=color, lw=1.8)
        side_indices = [i for i, joint in enumerate(data["leg_joint_names"])
                        if joint.startswith(side + "_")]
        axes[2].plot(time, [100 * max(row["leg_effort_nm"][i] /
                                       data["leg_rated_effort_nm"][i]
                                       for i in side_indices) for row in rows],
                     color=color, lw=1.8)
        axes[3].plot(time, [1000 * row["stance_slip_m"] for row in rows],
                     color=color, lw=1.8)
        axes[4].plot(time, [1000 * row["swing_sole_z_m"] for row in rows],
                     color=color, lw=1.8)
        axes[0].plot(time[-1], mirror * 1000 * rows[-1]["com_minus_stance_world_xy_m"][1],
                     marker="x", color=color, ms=7)
        manifest.append({"case": case["case"], "report": case["report"],
                         "report_sha256": case["report_sha256"]})
    axes[0].axhline(16, color="#668054", ls="--", lw=1, label="COM feedback target")
    axes[0].axhline(0, color="#838d96", ls=":", lw=.9)
    axes[1].axhline(96, color="#668054", ls="--", lw=1)
    axes[2].axhline(100, color="#ae4d3b", ls="--", lw=1)
    axes[3].axhline(15, color="#ae4d3b", ls="--", lw=1)
    axes[4].axhline(25, color="#668054", ls="--", lw=1)
    for ax in axes:
        ax.grid(alpha=.17)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("COM toward stance side (mm)")
    axes[1].set_ylabel("Stance load (%)")
    axes[2].set_ylabel("Peak stance motor (% rated)")
    axes[3].set_ylabel("Stance slip (mm)")
    axes[4].set_ylabel("Swing sole center (mm)")
    axes[4].set_xlabel("Seconds from upright reset")
    axes[4].set_xlim(3, 6.6)
    axes[0].legend(ncol=2, frameon=False, fontsize=8.4, loc="upper left")
    fig.suptitle(f"Cove-50 {side} support · coupled force, COM and stance-roll response",
                 fontsize=15, weight="bold")
    out = ROOT / f"coupled-transition-{side}.png"
    fig.savefig(out, dpi=170, facecolor="white")
    (ROOT / f"coupled-transition-{side}-manifest.json").write_text(json.dumps({
        "schema": "robot-learning-lab.cove50-coupled-transition-plot.v1",
        "plot": str(out.relative_to(REPO)),
        "note": "Positive COM ordinate points toward the stance side. Crosses mark terminated runs. COM target is a controller setpoint, not a qualification boundary; sole-center height alone is not exclusive support.",
        "cases": manifest,
    }, indent=2) + "\n")
    print(out)


def main() -> None:
    comparison = json.loads((ROOT / "comparison.json").read_text())
    assert len(comparison["cases"]) == 12
    for side in ("right", "left"):
        plot_side(side, comparison["cases"])


if __name__ == "__main__":
    main()
