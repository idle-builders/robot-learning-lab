"""Visualize three paired native diagnostics that motivated the next reach candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
SOURCES = {
    "negative_sparse": BASE / "v54/opened-negative-sparse-margin/matrix.json",
    "new_entry_bias": BASE / "v55/new-entry-edge-bias/matrix.json",
    "saved_actor": BASE / "v56/opened-actor-probe/matrix.json",
}
OUT = BASE / "v57"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrices = {name: json.loads(path.read_text()) for name, path in SOURCES.items()}
    expected = {"negative_sparse": (6, 6), "new_entry_bias": (3, 2), "saved_actor": (6, 3)}
    for name, matrix in matrices.items():
        count, passes = expected[name]
        if not matrix["complete"] or (matrix["case_count"], matrix["pass_count"]) != (count, passes):
            raise ValueError(f"Complete {name} diagnostic required")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), layout="constrained")
    for seed, color in ((11103, "#1c9b75"), (11109, "#2c77b7")):
        rows = [r for r in matrices["negative_sparse"]["cases"] if r["seed"] == seed]
        axes[0].plot([100 * r["negative_sparse_blend"] for r in rows],
                     [r["contact_s"] for r in rows], "o-", label=str(seed), color=color)
        for row in rows:
            axes[0].annotate(f"{row['entry_goal_error_mm_scoring_only']:.1f} mm",
                             (100 * row["negative_sparse_blend"], row["contact_s"]),
                             xytext=(0, 8 if seed == 11103 else -15),
                             textcoords="offset points", ha="center", fontsize=8)
    axes[0].set_title("V54 · negative-camera RGB correction\n6/6 paired native passes")
    axes[0].set_xlabel("Observed template correction used (%)")
    axes[0].set_xticks([15, 25, 35])
    axes[0].legend(title="Entry seed", loc="lower right")
    rows = matrices["new_entry_bias"]["cases"]
    x = [-2] + [1000 * row["scoped_y_bias_m"] for row in rows]
    y = [.04] + [row["contact_s"] for row in rows]
    axes[1].bar(x, y, width=.72,
                color=["#cf5b58" if value < 1 else "#1c9b75" for value in y])
    axes[1].set_xticks([-5, -4, -3, -2])
    axes[1].set_title("V55 · new-entry goal bias\n2/3 paired native passes")
    axes[1].set_xlabel("Body-Y goal bias (mm); −2 is V50")
    actors = matrices["saved_actor"]["cases"]
    ids = ("positive-07", "positive-08", "positive-10")
    base = [next(r for r in actors if r["id"] == ident)["baseline_v51_contact_s"]
            for ident in ids]
    v10 = [next(r for r in actors if r["id"] == ident and r["actor_override"] == "v10")["contact_s"]
           for ident in ids]
    v15 = [next(r for r in actors if r["id"] == ident and r["actor_override"] == "v15")["contact_s"]
           for ident in ids]
    z = np.arange(3)
    axes[2].bar(z - .24, base, width=.24, label="V51 arm", color="#758ca3")
    axes[2].bar(z, v10, width=.24, label="V10 teacher", color="#1c9b75")
    axes[2].bar(z + .24, v15, width=.24, label="V15 probe", color="#cf5b58")
    axes[2].set_xticks(z, ids, rotation=20, ha="right")
    axes[2].set_title("V56 · saved arm actor\nV10 passes 3/3 opened failures")
    axes[2].legend(loc="upper left", fontsize=8)
    for axis in axes:
        axis.axhline(1, color="#34465a", linestyle="--", linewidth=1.2)
        axis.set_ylabel("Longest fingertip contact (s)")
        axis.set_ylim(0, 5.9 if axis is axes[0] else 3.5)
        axis.grid(axis="y", alpha=.15)
    fig.suptitle("Opened teacher diagnostics · rejected historical Direct Frame · no disjoint score",
                 fontsize=14, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "teacher-diagnostics.png", dpi=160)
    plt.close(fig)
    report = {"schema": "robot-learning-lab.cove50-v57-teacher-diagnostic-figure.v1",
              "role": "opened_paired_evidence_not_fresh_score",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "source_matrices_sha256": {name: sha(path) for name, path in SOURCES.items()},
              "figure": str((OUT / "teacher-diagnostics.png").relative_to(ROOT)),
              "interpretation": "Small negative-camera correction restores RGB and contact on two opened seeds; -4/-5 mm bias rescues one edge entry; forced V10 arm rescues three contact failures. No automatic combined controller is qualified."}
    (OUT / "teacher-diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["source_matrices_sha256"]))


if __name__ == "__main__":
    main()
