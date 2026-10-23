"""Plot the V40 native screen, retention gate and optional sealed fresh score."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v40"
SCREEN = OUT / "route-stability-screen/matrix.json"
FULL = OUT / "opened-native/matrix.json"
FRESH = OUT / "fresh-reserved/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_line(ax) -> None:
    ax.axhspan(1, 1.2, color="#ffe8b9", alpha=.65, zorder=0)
    ax.axhline(1, color="#b43a49", linestyle="--", linewidth=1.5, label="1 s gate")
    ax.grid(axis="y", color="#e0e8e4", linewidth=.8)
    ax.set_ylim(-.1, max(5.8, ax.get_ylim()[1]))


def main() -> None:
    screen = json.loads(SCREEN.read_text())
    full = json.loads(FULL.read_text())
    if (not screen["complete"] or screen["case_count"] != 9 or
            full["case_count"] < 1 or full["case_count"] > 75):
        raise ValueError("Need complete nine-case screen and native retention progress")
    fresh = json.loads(FRESH.read_text()) if FRESH.is_file() else None
    if fresh and (not fresh["complete"] or fresh["case_count"] != 32):
        fresh = None
    count = 3 if fresh else 2
    fig, axes = plt.subplots(1, count, figsize=(18 if fresh else 13, 5.7),
                             gridspec_kw={"width_ratios": [1.2, 1.2, 1] if fresh else [1.2, 1.2]})

    ax = axes[0]
    rows = screen["cases"]
    x = np.arange(len(rows))
    baseline = [row["baseline_contact_s"] for row in rows]
    contact = [row["contact_s"] for row in rows]
    ax.vlines(x, baseline, contact, color="#a9b5b4", linewidth=1.8, zorder=1)
    ax.scatter(x-.06, baseline, color="#839594", s=43, label="earlier automatic", zorder=3)
    ax.scatter(x+.06, contact, color="#177f66", s=52, label="V40 automatic", zorder=4)
    regressed = next(i for i, row in enumerate(rows) if row["seed"] == 10528)
    ax.scatter([regressed], [.82], marker="x", color="#b43a49", s=75,
               linewidths=2.2, label="V39 regression", zorder=5)
    ax.set_xticks(x, [str(row["seed"]) for row in rows], rotation=45)
    ax.set_title("Nine paired opened cases · 9/9")
    ax.set_ylabel("Uninterrupted fingertip contact (s)")
    ax.legend(fontsize=8, loc="upper right")
    gate_line(ax)

    ax = axes[1]
    older = sorted([r["contact_s"] for r in full["cases"] if r["seed"] < 10600])
    recent = sorted([r["contact_s"] for r in full["cases"] if r["seed"] >= 10600])
    ax.scatter(range(len(older)), older, color="#147b9b", s=35, label=f"older {len(older)}")
    ax.scatter(range(len(recent)), recent, color="#bf7831", s=35, label=f"recent {len(recent)}")
    state = "complete" if full["complete"] else "stopped" if full["stopped_after_failed_stage"] else "running"
    ax.set_title(f"Retention {full['pass_count']}/{full['case_count']} evaluated · {state}")
    ax.set_xlabel("Starts ordered by contact within each cohort")
    ax.set_ylabel("Contact (s)")
    ax.legend(fontsize=8, loc="lower right")
    gate_line(ax)
    failures = [str(r["seed"]) for r in full["cases"] if not r["pass"]]
    if failures:
        ax.text(.02, .02, "Failed: " + ", ".join(failures), transform=ax.transAxes,
                color="#912f3d", fontsize=9)

    if fresh:
        ax = axes[2]
        strata = list(dict.fromkeys(r["stratum"] for r in fresh["cases"]))
        colors = ("#266d9f", "#9165a1", "#c67533", "#38876e")
        for i, (stratum, color) in enumerate(zip(strata, colors)):
            subset = [r for r in fresh["cases"] if r["stratum"] == stratum]
            for j, row in enumerate(subset):
                ax.scatter(i + (j - (len(subset)-1)/2)*.07, row["contact_s"],
                           s=50, color=color, edgecolors="#243e4a", linewidths=.4)
        ax.set_xticks(range(len(strata)), [s.replace("_", "\n") for s in strata])
        ax.set_title(f"Frozen fresh {fresh['pass_count']}/32")
        ax.set_ylabel("Contact (s)")
        gate_line(ax)
    fig.suptitle("Cove-50 V40 · rejected historical Direct Frame · native no-reset reach", fontsize=15)
    fig.tight_layout(rect=(0, .03, 1, .93))
    output = OUT / "native-contact-evaluation.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v40-native-contact-summary.v1",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_matrix_sha256": {"screen": sha(SCREEN), "retention": sha(FULL),
                                 **({"fresh": sha(FRESH)} if fresh else {})},
        "screen": {"pass_count": screen["pass_count"], "case_count": screen["case_count"]},
        "retention": {"pass_count": full["pass_count"], "case_count": full["case_count"],
                      "complete": full["complete"], "stopped": full["stopped_after_failed_stage"],
                      "regressions": full["regressions"],
                      "min_contact_s": min(r["contact_s"] for r in full["cases"]),
                      "thin_margin_seeds": [r["seed"] for r in full["cases"]
                                            if r["pass"] and 1 <= r["contact_s"] < 1.2]},
        "fresh": None if not fresh else {
            "pass_count": fresh["pass_count"], "case_count": fresh["case_count"],
            "selection_counts": fresh["selection_counts"],
            "calibrated_corner_route_count": fresh["calibrated_corner_route_count"],
            "min_contact_s": min(r["contact_s"] for r in fresh["cases"])},
        "figure_sha256": sha(output),
    }
    (OUT / "native-contact-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
