"""Plot native contact and margins for V39 retention and optional fresh score."""

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
OUT = BASE / "v39"
BASELINE_OLD = BASE / "v21/opened-native/matrix.json"
BASELINE_RECENT = BASE / "v32/opened-native/matrix.json"
FULL = OUT / "opened-native/matrix.json"
FRESH = OUT / "fresh-reserved/matrix.json"
SEEDS = (10501, 10508, 10513, 10514, 10530, 10534, 10538, 10539, 10433)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads(BASELINE_OLD.read_text())
    recent = json.loads(BASELINE_RECENT.read_text())
    full = json.loads(FULL.read_text())
    if not full["complete"] or full["case_count"] != 75:
        raise ValueError("Need complete 75-start opened gate")
    fresh = json.loads(FRESH.read_text()) if FRESH.exists() else None
    if fresh and (not fresh["complete"] or fresh["case_count"] != 32):
        fresh = None
    full_rows = {r["seed"]: r for r in full["cases"]}
    old_rows = {r["seed"]: r for r in old["cases"]}
    count = 3 if fresh else 2
    fig, axes = plt.subplots(1, count, figsize=(17 if fresh else 13, 5.7),
                             gridspec_kw={"width_ratios": [1.3, 1.2, 1] if fresh else [1.3, 1.2]})
    ax = axes[0]
    x = np.arange(len(SEEDS))
    before = [old_rows[s]["contact_s"] for s in SEEDS]
    after = [full_rows[s]["contact_s"] for s in SEEDS]
    ax.vlines(x, before, after, color="#8dbbc7", linewidth=2, alpha=.9)
    ax.scatter(x-.06, before, s=58, color="#879aa8", label="earlier automatic", zorder=3)
    ax.scatter(x+.06, after, s=64, color="#176fb0", label="V39 automatic", zorder=4)
    ax.set_xticks(x, [str(s) for s in SEEDS], rotation=45)
    ax.set_title("Nine earlier contact misses")
    ax.set_ylabel("Longest uninterrupted fingertip contact (s)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", color="#dce7ed")
    ax.axhline(1, color="#c43e51", linestyle="--", linewidth=1.5)

    ax = axes[1]
    old_contact = sorted([r["contact_s"] for r in full["cases"] if r["seed"] in old_rows])
    recent_contact = sorted([r["contact_s"] for r in full["cases"] if r["seed"] not in old_rows])
    ax.scatter(range(len(old_contact)), old_contact, color="#176fb0", s=33, label="51 earlier", zorder=3)
    ax.scatter(range(len(recent_contact)), recent_contact, color="#d58a34", s=35,
               label="24 recent", zorder=4)
    ax.set_title(f"Opened retention: {full['pass_count']}/75")
    ax.set_xlabel("Starts ordered by contact within each cohort")
    ax.set_ylabel("Contact (s)")
    ax.grid(axis="y", color="#dce7ed")
    ax.axhline(1, color="#c43e51", linestyle="--", linewidth=1.5)
    ax.legend(fontsize=8, loc="lower right")
    if full["pass_count"] < 75:
        failures = [str(r["seed"]) for r in full["cases"] if not r["pass"]]
        ax.text(.01, -.24, "Failed: " + ", ".join(failures), transform=ax.transAxes,
                fontsize=8, color="#852c3b")

    if fresh:
        ax = axes[2]
        strata = list(dict.fromkeys(r["stratum"] for r in fresh["cases"]))
        colors = {s: c for s, c in zip(strata, ("#3e88c1", "#9770bc", "#dd8e48", "#3e9e83"))}
        for i, stratum in enumerate(strata):
            subset = [r for r in fresh["cases"] if r["stratum"] == stratum]
            for j, row in enumerate(subset):
                ax.scatter(i + (j - (len(subset)-1)/2)*.06, row["contact_s"], s=53,
                           color=colors[stratum], edgecolors="#243e4a", linewidths=.4)
        ax.set_xticks(range(len(strata)), [s.replace("_", "\n") for s in strata])
        ax.set_title(f"Frozen fresh: {fresh['pass_count']}/32")
        ax.set_ylabel("Contact (s)")
        ax.grid(axis="y", color="#dce7ed")
        ax.axhline(1, color="#c43e51", linestyle="--", linewidth=1.5)
    for ax in axes:
        ax.set_ylim(-.1, max(5.7, ax.get_ylim()[1]))
    fig.suptitle("Cove-50 reach · rejected Direct Frame native evidence", fontsize=15)
    fig.tight_layout(rect=(0, .03, 1, .93))
    output = OUT / "native-contact-evaluation.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    report = {"schema": "robot-learning-lab.cove50-v37-v39-contact-summary.v1",
              "role": "historical_body_opened_and_optional_fresh_evidence",
              "body_role": "rejected_historical_direct_frame_sandbox",
              "source_matrix_sha256": {
                  "earlier": sha(BASELINE_OLD), "recent": sha(BASELINE_RECENT),
                  "v37_forced": sha(BASE / "v37/new-actor-forced-route/matrix.json"),
                  "v38_pilot": sha(BASE / "v38/opened-native/pilot-matrix.json"),
                  "v39_screen": sha(OUT / "corner-image-screen/matrix.json"),
                  "v39_full": sha(FULL),
                  **({"v39_fresh": sha(FRESH)} if fresh else {})},
              "opened": {"pass": full["pass_count"], "count": 75,
                         "rescues": full["rescues"], "regressions": full["regressions"],
                         "min_contact_s": min(r["contact_s"] for r in full["cases"])},
              "fresh": None if not fresh else {
                  "pass": fresh["pass_count"], "count": 32,
                  "selection_counts": fresh["selection_counts"],
                  "min_contact_s": min(r["contact_s"] for r in fresh["cases"])},
              "figure_sha256": sha(output)}
    (OUT / "native-contact-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figure": str(output.relative_to(ROOT)),
                      "figure_sha256": report["figure_sha256"],
                      "opened": report["opened"], "fresh": report["fresh"]}, indent=2))


if __name__ == "__main__":
    main()
