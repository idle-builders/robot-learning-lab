"""Verify and plot complete paired CUDA training variance on opened V139 misses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v144"
PLAN = BASE / "supported-opposite-direction-variance-plan.json"
MATRIX = BASE / "native-supported-opposite-direction-variance/matrix.json"
FIGURE = BASE / "paired-contact.png"
SUMMARY = BASE / "paired-contact-summary.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_report(path: str, trace: str, case: dict, expected_shift: float,
                  expected_contact: float, evaluator_sha: str) -> dict:
    report_path, trace_path = ROOT / path, ROOT / trace
    if not report_path.is_file() or not trace_path.is_file():
        raise ValueError(f"Missing native source for {case['id']}: {path}")
    report = json.loads(report_path.read_text())
    expected_toy = [case["toy_offset_x_m"], case["toy_offset_y_m"]]
    if (report["seed"] != case["seed"] or
            report["evaluator_sha256"] != evaluator_sha or
            abs(report.get("diagnostic_goal_shift_body_y_m", 0) - expected_shift) > 1e-9 or
            abs(report["phases"]["reach"]["continuous_finger_contact_s"] -
                expected_contact) > 1e-7 or
            abs(report["rail_v"] - case["rail_v"]) > 1e-9 or
            abs(report["delay_ms"] - case["delay_ms"]) > 1e-9 or
            abs(report["gaze_head_yaw_rad"] - case["gaze_head_yaw_rad"]) > 1e-9 or
            abs(report["pitch_offset_rad"] - case["pitch_offset_rad"]) > 1e-9 or
            abs(report["roll_clearance_rad"] - case["roll_clearance_rad"]) > 1e-9 or
            any(abs(a-b) > 1e-9 for a, b in
                zip(report["toy_offset_b_m"][:2], expected_toy))):
        raise ValueError(f"Report-to-plan source mismatch: {case['id']} {path}")
    return report


def main() -> None:
    plan, matrix = json.loads(PLAN.read_text()), json.loads(MATRIX.read_text())
    if (not matrix["complete"] or matrix["case_count"] != 16 or
            matrix["planned_count"] != 16 or matrix["plan_sha256"] != sha(PLAN) or
            {row["id"] for row in matrix["cases"]} !=
            {case["id"] for case in plan["cases"]}):
        raise ValueError("Complete source-matched V144 matrix required")
    source = plan["source_sha256"]
    rows = {row["id"]: row for row in matrix["cases"]}
    failures = []
    for case in plan["cases"]:
        row = rows[case["id"]]
        baseline = verify_report(row["baseline_result"], row["baseline_trace"],
                                 case, 0, row["baseline_contact_s"], source["evaluator"])
        observed = score_runner.observations.observed_features(baseline)
        if (observed != row["entry_observed_features"] or
                baseline["body_sha256"] != source["body"] or
                baseline["selected_pointer_sha256"] != source["pointer"] or
                baseline["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Baseline or RGB entry mismatch: {case['id']}")
        for label, shift in (("neg2", -.002), ("pos2", .002)):
            choice = row["choices"][label]
            report = verify_report(choice["result"], choice["trace"], case,
                                   shift, choice["contact_s"],
                                   source["shift_evaluator"])
            faults = score_runner.entry_parity(
                observed, score_runner.observations.observed_features(report))
            if (faults != choice["entry_parity_faults"] or
                    report["body_sha256"] != source["body"] or
                    report["selected_pointer_sha256"] != source["pointer"] or
                    report["selection_mode"] != row["selection_mode"] or
                    choice["pass"] != (choice["native_pass"] and not faults)):
                raise ValueError(f"Paired source or RGB entry mismatch: {case['id']} {label}")
        for label, contact, passed, margin, faults in (
                ("unchanged", row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_margin_1p2_s_met"], row["baseline_faults"]),
                *((label, choice["contact_s"], choice["pass"],
                   choice["margin_1p2_s_met"], choice["faults"])
                  for label, choice in row["choices"].items())):
            if not passed or not margin:
                failures.append({"id": row["id"], "seed": row["seed"],
                                 "target": label, "contact_s": contact,
                                 "full_gate_pass": passed,
                                 "margin_1p2_s_met": margin,
                                 "reasons": faults + (["contact below 1.2 s margin"]
                                                     if not margin else [])})
    ordered = list(reversed(plan["cases"]))
    fig, ax = plt.subplots(figsize=(12.3, 9.0), layout="constrained")
    ax.axvline(1.0, color="#b74c5a", linestyle=":", label="1 s full contact gate")
    ax.axvline(1.2, color="#b98224", linestyle="--", label="1.2 s margin")
    offsets = {"unchanged": -.22, "neg2": 0, "pos2": .22}
    colors = {"unchanged": "#53646c", "neg2": "#8156ae", "pos2": "#087e70"}
    markers = {"unchanged": "s", "neg2": "D", "pos2": "o"}
    for index, case in enumerate(ordered):
        row = rows[case["id"]]
        values = {"unchanged": row["baseline_contact_s"],
                  "neg2": row["choices"]["neg2"]["contact_s"],
                  "pos2": row["choices"]["pos2"]["contact_s"]}
        for label, value in values.items():
            ax.scatter(value, index + offsets[label], marker=markers[label],
                       color=colors[label], s=58, edgecolor="white", linewidth=.7,
                       zorder=3)
    ax.set(yticks=range(len(ordered)),
           yticklabels=[f"{c['seed']} · {c['id'].split('-')[-1]} · {c['route_variant']}"
                        for c in ordered],
           xlabel="Longest continuous fingertip contact (s)",
           xlim=(-.1, 6.4), ylim=(-.65, len(ordered)-.35))
    ax.grid(axis="x", alpha=.18)
    for label in offsets:
        ax.scatter([], [], marker=markers[label], color=colors[label], label=label)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=5,
              frameon=False, fontsize=8)
    fig.suptitle("V144 opened physical variance · prescribed target directions", fontsize=14)
    ax.set_title("16 conditions / 2 opened seeds · full native CUDA · training evidence only",
                 fontsize=10)
    temp = FIGURE.with_name("paired-contact.tmp.png")
    fig.savefig(temp, dpi=145)
    plt.close(fig)
    os.replace(temp, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v144-paired-contact-summary.v1",
        "role": "source_checked_opened_training_variance_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "source_checked_condition_count": 16,
        "source_checked_native_visit_count": 48,
        "baseline_pass_count": matrix["baseline_pass_count"],
        "neg2_pass_count": matrix["neg2_pass_count"],
        "pos2_pass_count": matrix["pos2_pass_count"],
        "neg2_rescue_count": matrix["neg2_rescue_count"],
        "pos2_rescue_count": matrix["pos2_rescue_count"],
        "neg2_regression_count": matrix["neg2_regression_count"],
        "pos2_regression_count": matrix["pos2_regression_count"],
        "entry_parity_failure_count": matrix["entry_parity_failure_count"],
        "failures_or_margin_misses": failures,
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": "Opened repeated-seed training evidence on rejected Direct Frame; V139 remains failed.",
    }
    temp_summary = SUMMARY.with_suffix(".json.tmp")
    temp_summary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temp_summary, SUMMARY)
    print("V144 source-checked 16 conditions and 48 native visits")


if __name__ == "__main__":
    main()
