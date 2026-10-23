"""Make the completed disjoint V41 score legible in the inspector."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v41"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    matrix_path = OUT / "fresh-reserved/matrix.json"
    matrix = json.loads(matrix_path.read_text())
    if not matrix["complete"] or matrix["case_count"] != 32:
        raise ValueError("Need all 32 native V41 reports before summarizing")
    rows = matrix["cases"]
    failed = [r for r in rows if not r["pass"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v41-fresh-summary.v1",
        "role": "disjoint_native_score_on_rejected_historical_body",
        "body_role": matrix["body_role"],
        "fresh_plan_sha256": sha(OUT / "fresh-challenge-plan.json"),
        "fresh_matrix_sha256": sha(matrix_path),
        "scorer_sha256": matrix["scoring_parser_sha256"],
        "plotter_sha256": sha(Path(__file__)),
        "case_count": len(rows),
        "pass_count": len(rows) - len(failed),
        "stratum_counts": dict(Counter(r["stratum"] for r in rows)),
        "stratum_pass_counts": {s: sum(r["pass"] for r in rows if r["stratum"] == s)
                                for s in sorted({r["stratum"] for r in rows})},
        "route_counts": matrix["selection_counts"],
        "calibrated_corner_route_count": matrix["calibrated_corner_route_count"],
        "failed": [{"seed": r["seed"], "stratum": r["stratum"],
                    "selection_mode": r["selection_mode"], "contact_s": r["contact_s"],
                    "entry_orange_pixels": r["entry_orange_pixels"], "faults": r["faults"]}
                   for r in failed],
        "promotion": "rejected" if failed else "check_route_exposure_before_historical_body_promotion",
        "qualification": "This is a native score on the rejected Direct Frame only. No selected mechanical body is available for control qualification.",
    }
    (OUT / "fresh-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(16, 6.5))
    fig.patch.set_facecolor("#101925")
    ax.set_facecolor("#182536")
    xs = np.arange(len(rows))
    colors = ["#e67474" if not r["pass"] else
              "#65c9f4" if r["template_used_for_control"] else "#64d6aa" for r in rows]
    ax.bar(xs, [r["contact_s"] for r in rows], color=colors, width=.76)
    ax.axhline(1, color="#f2b459", linestyle="--", linewidth=1.8, label="1 s contact gate")
    ax.set_xticks(xs, [str(r["seed"]) for r in rows], rotation=55, ha="right", color="#d9e4ef")
    ax.set_ylabel("Continuous fingertip contact (s)", color="#d9e4ef")
    ax.set_ylim(0, max(6, max(r["contact_s"] for r in rows) + .5))
    ax.tick_params(axis="y", colors="#d9e4ef")
    for spine in ax.spines.values():
        spine.set_color("#556b7f")
    ax.grid(axis="y", color="#8ca0b0", alpha=.2)
    for i, r in enumerate(rows):
        if not r["pass"]:
            if r["image_goal_missing_at_s"] is not None:
                label = "no RGB goal"
            elif any("rated leg" in fault for fault in r["faults"]):
                report = json.loads((ROOT / r["result"]).read_text())
                label = (f"leg rated {100 * report['phases']['preposition']['rated_leg_fraction']:.0f}%")
            else:
                label = f"contact {r['contact_s']:.2f}s"
            ax.annotate(label, (i, max(.08, r["contact_s"])), xytext=(0, 10),
                        textcoords="offset points", ha="center", rotation=60,
                        color="#f5c7c7", fontsize=8)
    ax.scatter([], [], color="#64d6aa", label="Calibrated RGB pass")
    ax.scatter([], [], color="#65c9f4", label="Template RGB pass")
    ax.scatter([], [], color="#e67474", label="Full-gate failure")
    ax.legend(facecolor="#182536", labelcolor="#e8f1f8", edgecolor="#556b7f",
              loc="upper right")
    fig.suptitle(f"Cove-50 V41 · frozen disjoint native score · {summary['pass_count']}/32 pass",
                 color="#f6f9fc", fontsize=17, y=.98)
    fig.text(.5, .03,
             "32 predeclared seeds · full native support, effort, contact and RGB gates · rejected historical Direct Frame",
             color="#b9cad8", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .92))
    path = OUT / "fresh-native-evaluation.png"
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path} and {OUT / 'fresh-summary.json'}")


if __name__ == "__main__":
    main()
