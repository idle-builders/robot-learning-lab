"""Summarize opened route, feedback, image and actor diagnostics."""

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
PATHS = {
    "v32": BASE / "v32/prior-failure-screen/matrix.json",
    "v33": BASE / "v33/forced-route-screen/matrix.json",
    "v34": BASE / "v34/feedback-gap/matrix.json",
    "v35": BASE / "v35/wider-rgb-screen/matrix.json",
    "v36": BASE / "v36/actor-screen/matrix.json",
}
OUT = BASE / "v36"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    data = {name: json.loads(path.read_text()) for name, path in PATHS.items()}
    if any(not matrix["complete"] for matrix in data.values()):
        raise ValueError("All opened matrices must be complete")
    counts = [(data[name]["case_count"], data[name]["pass_count"]) for name in
              ("v32", "v33", "v34", "v35", "v36")]
    if counts != [(9, 2), (15, 6), (5, 3), (5, 5), (10, 7)]:
        raise ValueError("Native result counts changed")
    seeds = [10508, 10513, 10530, 10534, 10538, 10539, 10433]
    baseline = {row["seed"]: row["contact_s"] for row in data["v32"]["cases"]}
    route = {}
    for row in data["v33"]["cases"]:
        if row["route"] == "v10_x2_teacher":
            continue
        route[row["seed"]] = max(route.get(row["seed"], 0), row["contact_s"])
    feedback = {row["seed"]: row["contact_s"] for row in data["v34"]["cases"]}
    image = {row["seed"]: row["contact_s"] for row in data["v35"]["cases"]}
    trained = {row["seed"]: row["contact_s"] for row in data["v36"]["cases"]}
    colors = {"V32 automatic": "#869caf", "V33 forced route": "#b785eb",
              "V34 feedback teacher": "#ffad66", "V35 image rule": "#51c9c1",
              "V36 trained arm + image": "#70a2ff"}
    series = [("V32 automatic", baseline, -.28),
              ("V33 forced route", route, -.14),
              ("V34 feedback teacher", feedback, 0),
              ("V35 image rule", image, .14),
              ("V36 trained arm + image", trained, .28)]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for name, values, offset in series:
        xs, ys = [], []
        for i, seed in enumerate(seeds):
            if seed in values:
                xs.append(i + offset)
                ys.append(values[seed])
        ax.scatter(xs, ys, s=72, color=colors[name], edgecolors="#162333", linewidths=.5,
                   label=name, zorder=3)
    biased = next(row for row in data["v33"]["cases"] if row["seed"] == 10508)
    ax.scatter([0-.14], [biased["contact_s"]], s=90, facecolors="none",
               edgecolors="#b785eb", linewidths=2.3, label="V33 prescribed +2 mm", zorder=4)
    ax.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    ax.set_ylim(-.2, 5)
    ax.set_ylabel("Longest continuous finger contact (s)")
    ax.set_title("Earlier opened misses: contact by diagnostic")
    ax.grid(axis="y", color="#d9e1e8")
    ax.axhline(1, color="#ca394e", linestyle="--", linewidth=1.5, label="1 s gate")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.text(.01, -.25, "Hollow point uses an experimenter-prescribed goal shift; missing points were not run.",
            transform=ax.transAxes, fontsize=8, color="#405365")
    ax2 = axes[1]
    control_seeds = [10604, 10608, 10612, 10514, 10533, 10620, 10621]
    for i, seed in enumerate(control_seeds):
        for name, matrix, marker in (("V35 image rule", data["v35"], "o"),
                                     ("V36 trained arm + image", data["v36"], "s")):
            row = next((row for row in matrix["cases"] if row["seed"] == seed), None)
            if row:
                ax2.scatter(i + (-.11 if marker == "o" else .11), row["contact_s"],
                            marker=marker, s=68, color=colors[name], edgecolors="#162333",
                            linewidths=.5, zorder=3)
    ax2.set_xticks(range(len(control_seeds)), [str(seed) for seed in control_seeds], rotation=45)
    ax2.set_ylim(-.2, 5)
    ax2.set_title("Retained pass controls tested here")
    ax2.grid(axis="y", color="#d9e1e8")
    ax2.axhline(1, color="#ca394e", linestyle="--", linewidth=1.5)
    ax2.text(.01, -.25, "All shown controls pass; 10608 and 10621 have thin V36 margins.",
             transform=ax2.transAxes, fontsize=8, color="#405365")
    fig.suptitle("Cove-50 historical reach recovery · opened native diagnostics", fontsize=15)
    fig.tight_layout(rect=(0, .05, 1, .94))
    output = OUT / "opened-diagnostic-evaluation.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    report = {
        "schema": "robot-learning-lab.cove50-v33-v36-opened-summary.v1",
        "role": "opened_diagnostic_not_fresh_not_selected_body",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "matrix_sha256": {name: sha(path) for name, path in PATHS.items()},
        "actor_sha256": sha(OUT / "actor.pt"),
        "figure_sha256": sha(output),
        "counts": {name: {"cases": matrix["case_count"], "pass": matrix["pass_count"]}
                   for name, matrix in data.items()},
        "specific_findings": [
            "V33 experimenter-forced locked mode rescues 10534; saved V10/specialist rescues 10539. The old 10513 live rescue does not reproduce on the current native worker.",
            "V34 calibrated-RGB feedback teacher rescues 10530 but does not rescue 10538 or 10513.",
            "V35 image-derived template goal rescues 10538 and retains four paired passing controls; its 50-pixel entry error falls from 3.84 to 0.44 mm in scoring.",
            "V36 trained arm plus V35 image rule rescues 10530 and retains 10538; ten-case opened screen passes 7/10 with 10534, 10539 and 10513 still failing automatic routing.",
            "All results are on the rejected historical Direct Frame. Full prior retention, disjoint fresh scoring and new-body transfer remain unrun.",
        ],
    }
    (OUT / "opened-diagnostic-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"figure": str(output.relative_to(ROOT)), "figure_sha256": report["figure_sha256"],
                      "counts": report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
