"""Summarize V83 native retention of the 75 earlier V41 opened starts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v83"
PLAN = OUT / "prior75-retention-plan.json"
MATRIX = OUT / "native-prior75-retention/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != 75 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V83 earlier-start retention required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig, ax = plt.subplots(figsize=(9.5, 7), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.set_axisbelow(True)
    ax.grid(alpha=.18)
    for label, role, color in (("Earlier failures rescued by V41", "prior_failure", "#b47b37"),
                               ("Earlier retained passing starts", "retention", "#3c708f")):
        chosen = [row for row in rows if row["prior_role"] == role]
        ax.scatter([row["baseline_contact_s"] for row in chosen],
                   [row["contact_s"] for row in chosen], s=44, alpha=.85,
                   c=[color if row["pass"] else "#bd5055" for row in chosen],
                   edgecolors="white", linewidths=.5, label=label)
    max_contact = max(max(row["baseline_contact_s"], row["contact_s"]) for row in rows)
    ax.plot([0, max_contact + .3], [0, max_contact + .3],
            color="#98a5ad", linestyle=":", label="Same contact as V41")
    ax.axhline(1, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axhline(1.2, color="#c48437", linestyle="-.", label="1.2 s margin reference")
    ax.set_xlim(0, max_contact + .3)
    ax.set_ylim(0, max_contact + .3)
    ax.set_xlabel("Earlier V41 opened contact (s)")
    ax.set_ylabel("V83 native contact (s)")
    ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("V83 retention of 75 earlier opened starts\n"
                 "Rejected Direct Frame · unchanged authored route", fontsize=13)
    figure = OUT / "prior75-retention-evaluation.png"
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    failures = [{"seed": row["seed"], "contact_s": row["contact_s"],
                 "faults": row["faults"]} for row in rows if not row["pass"]]
    summary = {
        "schema": "robot-learning-lab.cove50-v83-prior75-retention-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "thin_rule_use_count": matrix["thin_rule_use_count"],
        "minimum_contact_s": matrix["minimum_contact_s"],
        "margin_1p2_s_count": matrix["margin_1p2_s_count"],
        "screen_gate_met": matrix["screen_gate_met"],
        "all_margin_1p2_s_met": matrix["all_margin_1p2_s_met"],
        "failures": failures,
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "Native replay of formerly passing V41 opened starts on the rejected historical Direct Frame. This is neither fresh scoring nor selected-body qualification.",
    }
    (OUT / "prior75-retention-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "thin_rule_use_count",
                       "minimum_contact_s", "margin_1p2_s_count",
                       "screen_gate_met", "all_margin_1p2_s_met", "failures")}, indent=2))


if __name__ == "__main__":
    main()
