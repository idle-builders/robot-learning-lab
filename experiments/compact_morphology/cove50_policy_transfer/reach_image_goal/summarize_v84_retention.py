"""Plot complete source-matched V84 retention and new-seed outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v84_native_case as native,
)


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v84"
MATRIX = {
    "routes63": "native-routes63-retention/matrix.json",
    "prior75": "native-prior75-retention/matrix.json",
    "fresh": "native-fresh-neighbors/matrix.json",
}
PLAN = {
    "routes63": "routes63-retention-plan.json",
    "prior75": "prior75-retention-plan.json",
    "fresh": "fresh-neighbor-challenge-plan.json",
}
EXPECTED = {"routes63": 63, "prior75": 75, "fresh": 34}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=MATRIX)
    parser.add_argument("--stopped", action="store_true",
                        help="Audit already completed V84 older reports after the failed gate")
    args = parser.parse_args()
    if args.stopped and args.stage != "prior75":
        raise ValueError("Stopped audit applies only to older retention")
    plan_path, matrix_path = OUT / PLAN[args.stage], OUT / MATRIX[args.stage]
    plan, matrix = json.loads(plan_path.read_text()), json.loads(matrix_path.read_text())
    if args.stopped:
        if matrix["complete"] or matrix["screen_gate_met"]:
            raise ValueError("Expected an incomplete failed older retention gate")
        dest = matrix_path.parent
        rows = []
        for case in plan["cases"]:
            prefix = dest / f"seed-{case['seed']}-{case['id']}"
            if all(prefix.with_suffix(f".{suffix}").is_file()
                   for suffix in ("json", "npz", "log")):
                rows.append(native.run_case(case, plan, dest))
        scored = {row["id"]: row for row in rows}
        if (not any(not row["pass"] for row in rows) or
                any(scored.get(row["id"]) != row for row in matrix["cases"])):
            raise ValueError("Stopped audit differs from the source-frozen native matrix")
    else:
        rows = matrix["cases"]
        if (not matrix["complete"] or matrix["case_count"] != EXPECTED[args.stage] or
                [row["id"] for row in rows] != [case["id"] for case in plan["cases"]]):
            raise ValueError("Complete source-matched V84 matrix required")
    if matrix["plan_sha256"] != sha(plan_path):
        raise ValueError("V84 plan identity changed")
    for row in rows:
        if any(not (ROOT / row[key]).is_file() for key in ("result", "trace", "log")):
            raise ValueError(f"Missing native evidence for {row['id']}")
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#f8fafc")
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    if args.stage == "fresh":
        groups = (("Thin-route targets", "#16846c",
                   [row for row in rows if row["expected_thin_route_bias"]]),
                  ("Route controls", "#40658e",
                   [row for row in rows if row["route_variant"] != "center" and
                    not row["expected_thin_route_bias"]]),
                  ("Center-lock targets", "#c48532",
                   [row for row in rows if row["route_variant"] == "center"]))
        ordered = [row for _, _, group in groups for row in group]
        offset = 0
        for label, color, group in groups:
            ax.barh(range(offset, offset + len(group)),
                    [row["contact_s"] for row in group],
                    color=[color if row["pass"] else "#bd5055" for row in group],
                    label=f"{label} ({len(group)})")
            offset += len(group)
        ax.set_yticks(range(len(ordered)),
                      [str(row["seed"]) + " " + (row.get("base_route_id") or "center")
                       for row in ordered], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
        ax.axvline(1.2, color="#c48437", linestyle="-.", label="1.2 s target margin")
        ax.set_xlabel("Uninterrupted fingertip contact (s)")
        ax.set_title("V84 new-seed variation around opened route and center views\n"
                     "Rejected historical Direct Frame", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
    else:
        groups = (("Observed thin-route correction", "#c48532", "D",
                   [row for row in rows if row["automatic_thin_route_bias_applied"]]),
                  ("Observed center lock", "#a34f7e", "s",
                   [row for row in rows if row["automatic_center_live_lock_applied"]]),
                  ("Other retained visits", "#40658e", "o",
                   [row for row in rows if not row["automatic_thin_route_bias_applied"]
                    and not row["automatic_center_live_lock_applied"]]))
        for label, color, marker, group in groups:
            if group:
                ax.scatter([row["baseline_contact_s"] for row in group],
                           [row["contact_s"] for row in group],
                           s=80 if marker != "o" else 37, marker=marker,
                           c=[color if row["pass"] else "#bd5055" for row in group],
                           edgecolors="white", linewidths=.5,
                           label=f"{label} ({len(group)})")
        if args.stopped:
            for row in rows:
                if not row["pass"]:
                    ax.annotate(str(row["seed"]),
                                (row["baseline_contact_s"], row["contact_s"]),
                                xytext=(4, 10), textcoords="offset points",
                                color="#a82f38", fontsize=9, weight="bold")
        bound = max(max(row["baseline_contact_s"], row["contact_s"]) for row in rows) + .3
        ax.plot([0, bound], [0, bound], color="#98a5ad", linestyle=":",
                label="Same contact as prior")
        ax.axhline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
        ax.axhline(1.2, color="#c48437", linestyle="-.", label="1.2 s margin")
        ax.set_xlim(0, bound)
        ax.set_ylim(0, bound)
        ax.set_xlabel("Earlier opened contact (s)")
        ax.set_ylabel("V84 native contact (s)")
        ax.set_title((f"V84 stopped older retention · {len(rows)}/75 completed native visits\n"
                      if args.stopped else
                      f"V84 {'63-route' if args.stage == 'routes63' else '75-start older'} retention\n")
                     + "Rejected historical Direct Frame", fontsize=13)
        ax.legend(loc="lower right", fontsize=8)
    name = "prior75-stopped" if args.stopped else args.stage
    figure = OUT / f"{name}-retention-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-v84-{name}-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "case_count": len(rows), "pass_count": sum(row["pass"] for row in rows),
        "planned_case_count": EXPECTED[args.stage], "stopped_after_failure": args.stopped,
        "complete": not args.stopped, "screen_gate_met": False if args.stopped else matrix.get("screen_gate_met"),
        "source_matrix_case_count": matrix["case_count"],
        "thin_rule_use_count": sum(row["automatic_thin_route_bias_applied"] for row in rows),
        "center_lock_use_count": sum(row["automatic_center_live_lock_applied"] for row in rows),
        "margin_1p2_s_count": sum(row["margin_1p2_s_met"] for row in rows),
        "minimum_contact_s": min(row["contact_s"] for row in rows),
        "gate_met": False if args.stopped else matrix.get("screen_gate_met", matrix.get("score_gate_met")),
        "failures": [{"id": row["id"], "contact_s": row["contact_s"],
                      "faults": row["faults"]} for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "cases": rows if args.stopped else None,
        "qualification": ("Native evidence on the rejected historical Direct Frame. "
                          "No selected mechanical body, physical camera, grasp or placement is qualified."),
    }
    (OUT / f"{name}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "thin_rule_use_count",
                       "center_lock_use_count", "minimum_contact_s", "gate_met", "failures")},
                     indent=2))


if __name__ == "__main__":
    main()
