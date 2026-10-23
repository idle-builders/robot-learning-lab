"""Plot the source-frozen scanned-edge contact-bias diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v48"
MATRIX = OUT / "scanned-edge-bias/matrix.json"
BASELINE = OUT.parent / "v45/scan-branch-probe-c/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix = json.loads(MATRIX.read_text())
    baseline = json.loads(BASELINE.read_text())
    if not matrix["complete"] or matrix["case_count"] != 5 or baseline["cases"][0]["contact_s"] != .78:
        raise ValueError("Complete paired V48 diagnostic required")
    rows = matrix["cases"]
    values = sorted([(1000 * r["scoped_y_bias_m"], r["contact_s"], r["pass"])
                     for r in rows] + [(-2, .78, False)])
    fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
    colors = ["#1d9b75" if passed else "#d65b55" for _, _, passed in values]
    ax.bar([x for x, _, _ in values], [y for _, y, _ in values],
           width=.72, color=colors)
    ax.axhline(1, color="#3a4b5c", linestyle="--", linewidth=1.5,
               label="1 s contact gate")
    for x, y, _ in values:
        ax.text(x, y + .08, f"{y:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 3.6)
    ax.set_xticks([x for x, _, _ in values])
    ax.set_xlabel("Fixed body-frame Y goal bias (mm)")
    ax.set_ylabel("Longest continuous fingertip contact (s)")
    ax.set_title("Physical zero-pixel scan: contact margin changes with Y goal bias")
    ax.legend(loc="upper right")
    fig.savefig(OUT / "scanned-edge-bias-evaluation.png", dpi=160)
    plt.close(fig)
    summary = {
        "schema": "robot-learning-lab.cove50-v48-scanned-edge-bias-summary.v1",
        "role": "opened_paired_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(OUT / "scanned-edge-bias-plan.json"),
        "matrix_sha256": sha(MATRIX), "baseline_matrix_sha256": sha(BASELINE),
        "pass_count": matrix["pass_count"], "case_count": matrix["case_count"],
        "baseline_bias_mm": -2, "baseline_contact_s": .78,
        "values": [{"bias_mm": x, "contact_s": y, "pass": passed}
                   for x, y, passed in values],
        "entry_orange_pixels": 40, "entry_rgb_error_mm_scoring_only": 1.641,
        "interpretation": "More negative fixed body-Y goal bias restores the opened physical-scan toy-edge contact while visible pixels and scored RGB error stay identical. This is a diagnostic slope on one seed, not an automatic selector, fresh score or body qualification.",
    }
    (OUT / "scanned-edge-bias-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"matrix_sha256": summary["matrix_sha256"],
                      "pass_count": summary["pass_count"]}))


if __name__ == "__main__":
    main()
