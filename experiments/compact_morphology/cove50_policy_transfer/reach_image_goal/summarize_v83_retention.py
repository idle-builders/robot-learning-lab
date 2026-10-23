"""Summarize source-matched V83 opened retention and activation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v83"
PLAN = OUT / "opened-retention-plan.json"
MATRIX = OUT / "native-retention/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != 63 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V83 retention required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(10, 7.2), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(alpha=.18)
    for label, color in (("V70 earlier tuned routes", "#40658e"),
                         ("V73 different-anchor routes", "#16846c")):
        source = "V70" if label.startswith("V70") else "V73"
        chosen = [row for row in rows if row["baseline_source"] == source and
                  not row["automatic_thin_route_bias_applied"]]
        ax.scatter([row["baseline_contact_s"] for row in chosen],
                   [row["contact_s"] for row in chosen], s=38, alpha=.8,
                   c=[color if row["pass"] else "#bd5055" for row in chosen],
                   edgecolors="white", linewidths=.5, label=label)
    activated = [row for row in rows if row["automatic_thin_route_bias_applied"]]
    ax.scatter([row["baseline_contact_s"] for row in activated],
               [row["contact_s"] for row in activated], s=96, marker="D",
               color="#d28835", edgecolors="#51311a", linewidths=.7,
               label="V83 observed rule activated")
    for row in activated:
        ax.annotate(row["id"].replace("positive-", "P ").replace("edge-", "E "),
                    (row["baseline_contact_s"], row["contact_s"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7)
    max_contact = max(max(row["baseline_contact_s"], row["contact_s"]) for row in rows)
    ax.plot([0, max_contact + .3], [0, max_contact + .3],
            color="#98a5ad", linestyle=":", label="Same contact as prior")
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axhline(1.2, color="#c48437", linestyle="-.", label="1.2 s margin reference")
    ax.set_xlim(0, max_contact + .3)
    ax.set_ylim(0, max_contact + .3)
    ax.set_xlabel("Prior opened contact (s)")
    ax.set_ylabel("V83 native contact (s)")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("V83 opened retention across 63 routes\n"
                 "Rejected Direct Frame · six observed-rule activations", fontsize=13)
    figure = OUT / "opened-retention-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    pass_by_source = {source: sum(row["pass"] for row in rows
                                  if row["baseline_source"] == source)
                      for source in ("V70", "V73")}
    fail_rows = [{"id": row["id"], "contact_s": row["contact_s"],
                  "faults": row["faults"]} for row in rows if not row["pass"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v83-opened-retention-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "pass_by_source": pass_by_source,
        "thin_rule_use_count": matrix["thin_rule_use_count"],
        "activated_margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in activated),
        "minimum_activated_contact_s": min((row["contact_s"] for row in activated),
                                           default=None),
        "screen_gate_met": matrix["screen_gate_met"],
        "failures": fail_rows,
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Opened V70/V73 route retention on the rejected historical Direct Frame. A new-seed challenge and selected-body native transfer remain separate requirements.",
    }
    (OUT / "opened-retention-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "pass_by_source",
                       "thin_rule_use_count", "activated_margin_1p2_s_count",
                       "minimum_activated_contact_s", "screen_gate_met", "failures")}, indent=2))


if __name__ == "__main__":
    main()
