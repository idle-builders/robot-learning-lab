"""Chart source-frozen V86/V87 timing replay against the exact V85 starts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("v86", "v87"))
    args = parser.parse_args()
    out = BASE / args.variant
    plan_path = out / "boundary-variance-plan.json"
    matrix_path = out / "native-boundary-variance/matrix.json"
    source_path = BASE / "v85/native-boundary-variance/matrix.json"
    plan = json.loads(plan_path.read_text())
    matrix = json.loads(matrix_path.read_text())
    source = json.loads(source_path.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 16 or
            matrix["plan_sha256"] != sha(plan_path) or
            plan["source_v85_matrix_sha256"] != sha(source_path) or
            [row["id"] for row in matrix["cases"]] !=
            [case["id"] for case in plan["cases"]] or
            not source["complete"] or source["case_count"] != 16):
        raise ValueError("Complete exact paired V85 timing replay required")
    by_id = {row["id"]: row for row in source["cases"]}
    for row in matrix["cases"]:
        if (row["id"] not in by_id or
                row["baseline_contact_s"] != by_id[row["id"]]["contact_s"] or
                any(not (ROOT / row[key]).is_file()
                    for key in ("result", "trace", "log"))):
            raise ValueError(f"Missing paired native evidence for {row['id']}")
    rows = matrix["cases"]
    fig, ax = plt.subplots(figsize=(11.5, 9), layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    y = list(range(len(rows)))
    ax.barh([v - .18 for v in y], [row["baseline_contact_s"] for row in rows],
            height=.33, color="#8999a9", label="V85 four-second approach")
    ax.barh([v + .18 for v in y], [row["contact_s"] for row in rows],
            height=.33, color=["#16846c" if row["pass"] else "#bd5055" for row in rows],
            label=f"{args.variant.upper()} observed contact")
    ax.axvline(1.0, color="#8a3b42", linestyle="--", label="1 s native gate")
    ax.axvline(1.2, color="#c48532", linestyle="-.", label="1.2 s margin")
    ax.set_yticks(y, [f"{row['seed']} · " +
                      ("height" if row["selection_mode"] == "v16_v10_height_locked"
                       else "plain") + (" · FAIL" if not row["pass"] else "")
                      for row in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Uninterrupted fingertip contact (s)")
    ax.set_title(f"{args.variant.upper()} exact 16-start timing replay · "
                 f"{matrix['pass_count']}/16 full native passes\n"
                 "Rejected historical Direct Frame")
    ax.grid(axis="x", alpha=.18)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=2, fontsize=8)
    figure = out / "boundary-variance-evaluation.png"
    fig.savefig(figure, dpi=150)
    plt.close(fig)
    summary = {
        "schema": f"robot-learning-lab.cove50-{args.variant}-boundary-variance-summary.v1",
        "role": plan["role"], "body_role": plan["body_role"],
        "plan_sha256": sha(plan_path), "matrix_sha256": sha(matrix_path),
        "source_v85_matrix_sha256": sha(source_path),
        "case_count": 16, "pass_count": matrix["pass_count"],
        "height_route_use_count": matrix["height_route_use_count"],
        "variance_gate_met": matrix["variance_gate_met"],
        "minimum_contact_s": matrix["minimum_contact_s"],
        "failed_seeds": [row["seed"] for row in rows if not row["pass"]],
        "figure": str(figure.relative_to(ROOT)), "figure_sha256": sha(figure),
        "qualification": ("Opened paired development on a rejected historical body. "
                          "Route and older retention, untouched new-seed scoring and "
                          "selected-body transfer are separate gates."),
    }
    (out / "boundary-variance-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("case_count", "pass_count", "height_route_use_count",
                       "variance_gate_met", "minimum_contact_s", "failed_seeds")}, indent=2))


if __name__ == "__main__":
    main()
