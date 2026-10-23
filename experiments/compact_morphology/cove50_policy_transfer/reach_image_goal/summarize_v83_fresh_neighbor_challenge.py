"""Chart and audit the frozen V83 fresh-seed, varied-anchor challenge."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v83"
PLAN = OUT / "fresh-neighbor-challenge-plan.json"
MATRIX = OUT / "native-fresh-neighbors/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != 26 or
            matrix["plan_sha256"] != sha(PLAN) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Complete source-matched V83 fresh challenge required")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")

    targeted = [row for row in rows if row["expected_thin_route_bias"]]
    controls = [row for row in rows if not row["expected_thin_route_bias"]]
    if len(targeted) != 12 or len(controls) != 14:
        raise ValueError("Fresh target/control split differs from frozen plan")
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.5), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    for ax, chosen, title, color in (
            (axes[0], targeted, "Observed-rule target views · 12 unused seeds", "#16846c"),
            (axes[1], controls, "No-activation controls · 14 unused seeds", "#40658e")):
        labels = [f"{row['seed']}  {row['base_route_id']}" for row in chosen]
        values = [row["contact_s"] for row in chosen]
        ax.barh(range(len(chosen)), values,
                color=[color if row["pass"] else "#bd5055" for row in chosen])
        ax.set_yticks(range(len(chosen)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
        ax.axvline(1.2, color="#c48437", linestyle="-.", label="1.2 s target margin")
        ax.set_xlim(0, max(5.6, max(values) + .2))
        ax.set_xlabel("Uninterrupted fingertip contact (s)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=.16)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("V83 varied-anchor challenge on the rejected Direct Frame\n"
                 "Fresh seeds, changed toy/motor/gaze/route inputs · all native gates required",
                 fontsize=13)
    figure = OUT / "fresh-neighbor-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)

    summary = {
        "schema": "robot-learning-lab.cove50-v83-fresh-neighbor-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "case_count": len(rows), "pass_count": matrix["pass_count"],
        "target_pass_count": sum(row["pass"] for row in targeted),
        "control_pass_count": sum(row["pass"] for row in controls),
        "target_rule_use_count": matrix["target_rule_use_count"],
        "control_rule_use_count": matrix["control_rule_use_count"],
        "target_margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in targeted),
        "minimum_target_contact_s": min(row["contact_s"] for row in targeted),
        "score_gate_met": matrix["score_gate_met"],
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": "New seeds at previously opened anchors on the rejected Direct Frame. Earlier retention, a new-anchor score and a selected mechanical body remain separate gates.",
    }
    (OUT / "fresh-neighbor-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "target_pass_count",
                       "control_pass_count", "target_rule_use_count",
                       "control_rule_use_count", "target_margin_1p2_s_count",
                       "minimum_target_contact_s", "score_gate_met", "failures")}, indent=2))


if __name__ == "__main__":
    main()
