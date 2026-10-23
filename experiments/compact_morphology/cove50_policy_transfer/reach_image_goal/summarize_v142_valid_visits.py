"""Retain source-checked V142 visits and both rejected out-of-range cases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v142_pitch_low_goal_sweep as sweep,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v142"
PLAN = BASE / "opened-pitch-low-y-sweep-plan.json"
DEST = BASE / "native-opened-pitch-low-y-sweep"
MATRIX = DEST / "matrix.json"
SUMMARY = BASE / "valid-visits-and-launch-failure.json"
FIGURE = BASE / "opened-pitch-low-response.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    if (plan["source_sha256"]["runner"] != sha(Path(sweep.__file__)) or
            plan["case_count"] != 7 or
            {round(case["diagnostic_goal_shift_body_y_m"] * 1000)
             for case in plan["cases"]} != {-4, -3, -1, 1, 2, 3, 4}):
        raise ValueError("Frozen V142 diagnostic plan changed")
    valid = [case for case in plan["cases"]
             if abs(case["diagnostic_goal_shift_body_y_m"]) <= .003]
    invalid = [case for case in plan["cases"] if case not in valid]
    rejected = []
    for case in invalid:
        log = DEST / f"seed-{case['seed']}-{case['id']}.log"
        if (not log.is_file() or
                "Diagnostic goal shift must be nonzero and within ±3 mm"
                not in log.read_text()):
            raise ValueError(f"Missing exact evaluator rejection: {case['id']}")
        rejected.append({"id": case["id"],
                         "shift_body_y_m": case["diagnostic_goal_shift_body_y_m"],
                         "reason": "Evaluator only accepts nonzero shifts within ±3 mm",
                         "log": str(log.relative_to(ROOT)),
                         "log_sha256": sha(log)})
    rows = []
    for case in valid:
        report = DEST / f"seed-{case['seed']}-{case['id']}.json"
        trace = report.with_suffix(".npz")
        if not report.is_file() or not trace.is_file():
            raise ValueError(f"Missing saved CUDA visit: {case['id']}")
        row = sweep.run_case(case, plan)
        if row["entry_parity_faults"]:
            raise ValueError(f"Entry parity failed: {case['id']}")
        row["report_sha256"] = sha(report)
        row["trace_sha256"] = sha(trace)
        rows.append(row)
    matrix = {
        "schema": "robot-learning-lab.cove50-v142-partial-sweep-matrix.v1",
        "role": "post_score_opened_goal_direction_diagnostic_not_policy",
        "body_role": plan["body_role"],
        "plan_sha256": sha(PLAN),
        "status": "stopped_evaluator_range_rejection",
        "complete": False,
        "case_count": len(rows), "planned_count": 7,
        "valid_supported_count": len(rows),
        "rejected_unsupported_count": len(rejected),
        "pass_count": sum(row["pass"] for row in rows),
        "margin_count": sum(row["margin_1p2_s_met"] for row in rows),
        "entry_parity_failure_count": 0,
        "rejected_cases": rejected,
        "cases": rows,
    }
    MATRIX.write_text(json.dumps(matrix, indent=2) + "\n")
    offsets = [-2, 0] + [round(row["shift_body_y_m"] * 1000)
                          for row in rows]
    contacts = [plan["prior_neg2_contact_s"], plan["baseline_contact_s"]] + [
        row["contact_s"] for row in rows]
    points = sorted(zip(offsets, contacts))
    fig, ax = plt.subplots(figsize=(9.2, 5.4), layout="constrained")
    ax.plot([x for x, _ in points], [y for _, y in points],
            color="#9aaeb7", linewidth=1.5, zorder=1)
    for x, y in points:
        ax.scatter(x, y, s=90, color="#087e70" if y >= 1.2 else "#b74c5a",
                   edgecolor="white", linewidth=.8, zorder=2)
        ax.annotate(f"{y:.2f}s", (x, y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.axhline(1.0, linestyle=":", color="#bd5960", label="1 s full-contact gate")
    ax.axhline(1.2, linestyle="--", color="#b98224", label="1.2 s score margin")
    ax.set(xlim=(-4.5, 4.5), ylim=(0, 3.25),
           xticks=range(-4, 5), xlabel="Prescribed body-Y target shift (mm)",
           ylabel="Longest continuous fingertip contact (s)")
    ax.grid(alpha=.15)
    ax.legend(loc="upper left")
    fig.suptitle("Opened pitch-low 24119 · target direction, full native contact")
    ax.set_title("Five valid new CUDA visits; ±4 mm rejected before simulation; 0 and −2 mm are saved references",
                 fontsize=9)
    temporary = FIGURE.with_name("opened-pitch-low-response.tmp.png")
    fig.savefig(temporary, dpi=150)
    plt.close(fig)
    os.replace(temporary, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v142-valid-visits-summary.v1",
        "role": "source_checked_partial_opened_diagnostic_not_policy_or_score",
        "body_role": plan["body_role"],
        "source_sha256": {"summarizer": sha(Path(__file__)),
                          "plan": sha(PLAN), "matrix": sha(MATRIX),
                          "figure": sha(FIGURE)},
        "status": matrix["status"],
        "planned_count": 7, "valid_supported_count": len(rows),
        "rejected_unsupported_count": len(rejected),
        "references": [{"shift_body_y_m": 0.0,
                        "contact_s": plan["baseline_contact_s"],
                        "result": plan["baseline_result"],
                        "result_sha256": plan["baseline_result_sha256"]},
                       {"shift_body_y_m": -.002,
                        "contact_s": plan["prior_neg2_contact_s"],
                        "result": plan["prior_neg2_result"],
                        "result_sha256": plan["prior_neg2_result_sha256"]}],
        "valid_visits": [{"shift_body_y_m": row["shift_body_y_m"],
                          "contact_s": row["contact_s"],
                          "pass": row["pass"],
                          "margin_1p2_s_met": row["margin_1p2_s_met"],
                          "faults": row["faults"],
                          "result": row["result"],
                          "report_sha256": row["report_sha256"]}
                         for row in rows],
        "rejected_cases": rejected,
        "figure": str(FIGURE.relative_to(ROOT)),
        "interpretation": ("The supported positive target shifts improved this "
                           "one opened pitch-low start. These are experimenter "
                           "choices and cannot repair V139 or establish a "
                           "learned robust policy. The frozen seven-visit plan "
                           "was not completed because ±4 mm exceeded the "
                           "saved evaluator's accepted range."),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"V142 retained {len(rows)} valid visits; "
          f"{len(rejected)} evaluator-range rejections, plan incomplete")


if __name__ == "__main__":
    main()
