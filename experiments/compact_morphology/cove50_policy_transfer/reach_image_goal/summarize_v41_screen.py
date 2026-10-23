"""Plot the V41 opened native screen and its two recovered V40 regressions."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v41"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    screen_path = OUT / "opened-screen/matrix.json"
    prior_path = BASE / "v40/opened-native/matrix.json"
    screen = json.loads(screen_path.read_text())
    prior = {row["seed"]: row for row in json.loads(prior_path.read_text())["cases"]}
    if not screen["complete"] or screen["case_count"] != 17:
        raise ValueError("Need complete 17-start V41 screen")
    rows = screen["cases"]
    summary = {
        "schema": "robot-learning-lab.cove50-v41-opened-screen-summary.v1",
        "role": "opened_native_candidate_not_fresh",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "screen_plan_sha256": sha(OUT / "opened-screen-plan.json"),
        "screen_matrix_sha256": sha(screen_path),
        "v40_partial_matrix_sha256": sha(prior_path),
        "runner_sha256": sha(Path(__file__)),
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "route_counts": dict(Counter(row["selection_mode"] for row in rows)),
        "template_on_count": sum(row["template_used_for_control"] for row in rows),
        "low_visibility_boundary": [
            {"seed": row["seed"], "orange_pixels": row["entry_orange_pixels"],
             "template_used": row["template_used_for_control"], "contact_s": row["contact_s"]}
            for row in rows if row["seed"] in (10546, 10604)],
        "minimum_contact": {"seed": min(rows, key=lambda r: r["contact_s"])["seed"],
                            "seconds": min(row["contact_s"] for row in rows)},
        "qualification": "Opened development screen only; 75 unique prior starts and 32 disjoint fresh starts are separate gates.",
    }
    (OUT / "screen-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 6),
                                     gridspec_kw={"width_ratios": [1, 2.6]})
    fig.patch.set_facecolor("#101925")
    for ax in (left, right):
        ax.set_facecolor("#182536")
        ax.tick_params(colors="#d9e4ef")
        for spine in ax.spines.values():
            spine.set_color("#556b7f")
        ax.grid(axis="y", color="#8ca0b0", alpha=.2)
        ax.axhline(1.0, color="#f2b459", linestyle="--", linewidth=1.7,
                   label="1 s contact gate")
        ax.set_ylim(0, 5.8)
        ax.set_ylabel("Continuous fingertip contact (s)", color="#d9e4ef")

    focus = [next(row for row in rows if row["seed"] == seed) for seed in (10545, 10546)]
    positions = np.arange(2)
    left.bar(positions - .18, [prior[row["seed"]]["contact_s"] for row in focus],
             width=.34, color="#da7272", label="V40 regression")
    left.bar(positions + .18, [row["contact_s"] for row in focus],
             width=.34, color="#64d6aa", label="V41 recovered")
    left.set_xticks(positions, [str(row["seed"]) for row in focus], color="#d9e4ef")
    left.set_title("Recovered V40 failures", color="#f3f7fa", pad=12)
    left.legend(facecolor="#182536", labelcolor="#e8f1f8", edgecolor="#556b7f",
                loc="upper left")
    left.text(1, 5.52, "20 pixels → template off", color="#a6d8c6",
              fontsize=9, ha="center")

    xs = np.arange(len(rows))
    colors = ["#65c9f4" if row["template_used_for_control"] else "#64d6aa"
              for row in rows]
    right.vlines(xs, 0, [row["contact_s"] for row in rows], color=colors,
                 alpha=.42, linewidth=3)
    right.scatter(xs, [row["contact_s"] for row in rows], c=colors,
                  s=70, edgecolor="#eef7ff", linewidth=.6, zorder=3)
    right.set_xticks(xs, [str(row["seed"]) for row in rows], rotation=58,
                     ha="right", color="#d9e4ef")
    right.set_title("17 opened starts · all pass · five automatic route types",
                    color="#f3f7fa", pad=12)
    for seed in (10604, 10529, 10608):
        index = next(i for i, row in enumerate(rows) if row["seed"] == seed)
        row = rows[index]
        label = (f"{row['entry_orange_pixels']}px" if seed == 10604
                 else f"{row['contact_s']:.2f}s")
        right.annotate(label, (index, row["contact_s"]),
                       xytext=(0, 12), textcoords="offset points",
                       ha="center", color="#e5edf3", fontsize=9)
    right.scatter([], [], color="#64d6aa", label="calibrated RGB")
    right.scatter([], [], color="#65c9f4", label="image template")
    right.legend(facecolor="#182536", labelcolor="#e8f1f8", edgecolor="#556b7f",
                 loc="upper left")
    fig.suptitle("Cove-50 V41 · native RGB reach on rejected historical body",
                 color="#f6f9fc", fontsize=16, y=.98)
    fig.text(.5, .02,
             "Opened development evidence · same checkpoint and source hashes · fresh 108xx starts still sealed",
             color="#b9cad8", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .93), w_pad=2)
    path = OUT / "native-screen-evaluation.png"
    fig.savefig(path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path} and {OUT / 'screen-summary.json'}")


if __name__ == "__main__":
    main()
