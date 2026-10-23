"""Visualize completed, source-frozen V87 native retention and fresh scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v87"
STAGES = {
    "routes63": (63, "routes63-retention-plan.json", "native-routes63-retention/matrix.json"),
    "prior75": (75, "prior75-retention-plan.json", "native-prior75-retention/matrix.json"),
    "fresh": (34, "fresh-neighbor-challenge-plan.json", "native-fresh-neighbors/matrix.json"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    expected, plan_name, matrix_name = STAGES[args.stage]
    plan_path, matrix_path = OUT / plan_name, OUT / matrix_name
    plan, matrix = json.loads(plan_path.read_text()), json.loads(matrix_path.read_text())
    rows = matrix["cases"]
    if (not matrix["complete"] or matrix["case_count"] != expected or
            len(rows) != expected or matrix["plan_sha256"] != sha(plan_path) or
            [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
        raise ValueError("Need complete source-matched V87 native matrix")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native report, trace or log for {row['id']}")

    fig, ax = plt.subplots(figsize=(12, 9), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    if args.stage == "fresh":
        groups = (
            ("Thin-route targets", "#16846c",
             [row for row in rows if row["expected_thin_route_bias"]]),
            ("Route controls", "#40658e",
             [row for row in rows if row["route_variant"] != "center" and
              not row["expected_thin_route_bias"]]),
            ("Center-lock targets", "#c48532",
             [row for row in rows if row["route_variant"] == "center"]),
        )
        ordered = [row for _, _, group in groups for row in group]
        offset = 0
        for label, color, group in groups:
            ax.barh(range(offset, offset + len(group)),
                    [row["contact_s"] for row in group],
                    color=[color if row["pass"] else "#bd5055" for row in group],
                    label=f"{label} ({len(group)})")
            offset += len(group)
        ax.set_yticks(range(len(ordered)),
                      [f"{row['seed']} {row.get('base_route_id') or 'center'}"
                       for row in ordered], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
        ax.axvline(1.2, color="#c48437", linestyle="-.", label="1.2 s target margin")
        ax.set_xlabel("Uninterrupted fingertip contact (s)")
        ax.set_title("V87 34 new-seed variations around opened route and center views\n"
                     "Rejected historical Direct Frame", fontsize=13)
    else:
        groups = (
            ("Observed thin-route correction", "#c48532", "D",
             [row for row in rows if row["automatic_thin_route_bias_applied"]]),
            ("Observed center lock", "#a34f7e", "s",
             [row for row in rows if row["automatic_center_live_lock_applied"]]),
            ("Other retained visits", "#40658e", "o",
             [row for row in rows if not row["automatic_thin_route_bias_applied"] and
              not row["automatic_center_live_lock_applied"]]),
        )
        for label, color, marker, group in groups:
            if group:
                ax.scatter([row["baseline_contact_s"] for row in group],
                           [row["contact_s"] for row in group],
                           s=80 if marker != "o" else 37, marker=marker,
                           c=[color if row["pass"] else "#bd5055" for row in group],
                           edgecolors="white", linewidths=.5,
                           label=f"{label} ({len(group)})")
        bound = max(max(row["baseline_contact_s"], row["contact_s"]) for row in rows) + .3
        ax.plot([0, bound], [0, bound], color="#98a5ad", linestyle=":",
                label="Same contact as prior")
        ax.axhline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
        ax.axhline(1.2, color="#c48437", linestyle="-.", label="1.2 s margin")
        ax.set_xlim(0, bound)
        ax.set_ylim(0, bound)
        ax.set_xlabel("Earlier opened contact (s)")
        ax.set_ylabel("V87 native contact (s)")
        label = "63-route" if args.stage == "routes63" else "75-start older"
        ax.set_title(f"V87 {label} retention\nRejected historical Direct Frame", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    figure = OUT / f"{args.stage}-retention-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    gate = matrix.get("score_gate_met", matrix.get("screen_gate_met"))
    summary = {
        "schema": f"robot-learning-lab.cove50-v87-{args.stage}-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": expected, "pass_count": sum(row["pass"] for row in rows),
        "thin_rule_use_count": sum(row["automatic_thin_route_bias_applied"] for row in rows),
        "center_lock_use_count": sum(row["automatic_center_live_lock_applied"] for row in rows),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in rows),
        "minimum_contact_s": min(row["contact_s"] for row in rows),
        "gate_met": gate,
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Native evidence on the rejected historical Direct Frame. "
                          "No selected mechanical body, physical camera, grasp or placement is qualified."),
    }
    (OUT / f"{args.stage}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "thin_rule_use_count",
                       "center_lock_use_count", "minimum_contact_s", "gate_met", "failures")},
                     indent=2))


if __name__ == "__main__":
    main()
