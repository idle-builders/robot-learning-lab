"""Source-check and chart complete new-seed signed-direction training pairs."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal.summarize_v144_supported_variance import (
    sha, verify_report,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v147"
PLAN = BASE / "new-seed-direction-training-plan.json"
RESERVE = BASE / "untouched-score-reserve-plan.json"
MATRIX = BASE / "native-new-seed-direction-pairs/matrix.json"
FIGURE = BASE / "new-seed-paired-contact.png"
SUMMARY = BASE / "new-seed-paired-summary.json"


def verify_trace(trace: str, report: dict) -> None:
    path = ROOT / trace
    with np.load(path, allow_pickle=False) as data:
        qpos, action, phase = data["qpos"], data["action"], data["phase"]
        dt = float(data["step_dt_s"])
        if (qpos.shape != (len(report["timeseries"]), 35) or
                action.shape != (len(report["timeseries"]), 21) or
                phase.shape != (len(report["timeseries"]),) or
                abs(dt - .02) > 1e-9 or
                not np.isfinite(qpos).all() or
                not np.isfinite(action).all()):
            raise ValueError(f"Saved native trace structure differs: {trace}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-partial", action="store_true",
                        help="Check completed rows without publishing a full-run summary")
    args = parser.parse_args()
    plan = json.loads(PLAN.read_text())
    reserve = json.loads(RESERVE.read_text())
    matrix = json.loads(MATRIX.read_text())
    if ((not args.verify_partial and
         (not matrix["complete"] or matrix["case_count"] != 40)) or
            matrix["case_count"] != len(matrix["cases"]) or
            matrix["complete"] != (matrix["case_count"] == 40) or
            matrix["planned_count"] != 40 or
            matrix["plan_sha256"] != sha(PLAN) or
            plan["source_sha256"]["untouched_score_reserve"] != sha(RESERVE) or
            reserve["status"] != "sealed_no_contact_visits" or
            {case["seed"] for case in reserve["cases"]} &
            {case["seed"] for case in plan["cases"]} or
            not {row["id"] for row in matrix["cases"]} <=
            {case["id"] for case in plan["cases"]} or
            len({row["id"] for row in matrix["cases"]}) != matrix["case_count"]):
        raise ValueError("Source-matched training matrix and sealed reserve required")
    source = plan["source_sha256"]
    rows = {row["id"]: row for row in matrix["cases"]}
    failures = []
    for case in plan["cases"]:
        if case["id"] not in rows:
            continue
        row = rows[case["id"]]
        baseline = verify_report(row["baseline_result"], row["baseline_trace"],
                                 case, 0, row["baseline_contact_s"], source["evaluator"])
        verify_trace(row["baseline_trace"], baseline)
        observed = score_runner.observations.observed_features(baseline)
        if (observed != row["entry_observed_features"] or
                baseline["body_sha256"] != source["body"] or
                baseline["selected_pointer_sha256"] != source["pointer"] or
                baseline["reach_checkpoint_sha256"] != source["actor"] or
                baseline["selection_mode"] != row["selection_mode"]):
            raise ValueError(f"Baseline source or RGB entry differs: {case['id']}")
        for label, shift in (("neg2", -.002), ("pos2", .002)):
            choice = row["choices"][label]
            report = verify_report(choice["result"], choice["trace"], case,
                                   shift, choice["contact_s"],
                                   source["shift_evaluator"])
            verify_trace(choice["trace"], report)
            parity = score_runner.entry_parity(
                observed, score_runner.observations.observed_features(report))
            if (parity != choice["entry_parity_faults"] or
                    report["body_sha256"] != source["body"] or
                    report["selected_pointer_sha256"] != source["pointer"] or
                    report["reach_checkpoint_sha256"] != source["actor"] or
                    report["selection_mode"] != row["selection_mode"] or
                    choice["pass"] != (choice["native_pass"] and not parity)):
                raise ValueError(f"Paired source or RGB entry differs: {case['id']} {label}")
        for label, contact, passed, margin, faults in (
                ("unchanged", row["baseline_contact_s"], row["baseline_pass"],
                 row["baseline_margin_1p2_s_met"], row["baseline_faults"]),
                *((name, choice["contact_s"], choice["pass"],
                   choice["margin_1p2_s_met"], choice["faults"])
                  for name, choice in row["choices"].items())):
            if not passed or not margin:
                failures.append({"id": row["id"], "seed": row["seed"],
                                 "target": label, "contact_s": contact,
                                 "full_gate_pass": passed,
                                 "margin_1p2_s_met": margin,
                                 "reasons": faults +
                                            (["contact below 1.2 s margin"]
                                             if not margin else [])})
    repair_counts = {}
    for label in ("neg2", "pos2"):
        repair_counts[f"{label}_full_gate_rescue_count"] = sum(
            not row["baseline_pass"] and row["choices"][label]["pass"]
            for row in rows.values())
        repair_counts[f"{label}_margin_repair_count"] = sum(
            row["baseline_pass"] and
            not row["baseline_margin_1p2_s_met"] and
            row["choices"][label]["pass"] and
            row["choices"][label]["margin_1p2_s_met"]
            for row in rows.values())
    pitch_low_specialist_count = sum(
        row["route_variant"] == "pitch-low" and
        row["selection_mode"] == "v16_specialist_locked"
        for row in rows.values())
    if args.verify_partial:
        print(json.dumps({
            "source_checked_seed_count": len(rows),
            "source_checked_native_visit_count": 3 * len(rows),
            **repair_counts,
            "pitch_low_specialist_count": pitch_low_specialist_count,
            "observed_route_counts": dict(sorted(Counter(
                row["selection_mode"] for row in rows.values()).items())),
            "failures_or_margin_misses": failures,
            "complete": matrix["complete"],
        }))
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharex=True,
                             layout="constrained")
    for axis, route in zip(axes, ("roll-high", "pitch-low")):
        route_cases = [case for case in plan["cases"]
                       if case["route_variant"] == route]
        for index, case in enumerate(reversed(route_cases)):
            row = rows[case["id"]]
            for name, offset, color, marker, contact, passed, margin in (
                    ("unchanged", -.22, "#53646c", "s", row["baseline_contact_s"],
                     row["baseline_pass"], row["baseline_margin_1p2_s_met"]),
                    ("−2 mm", 0, "#8156ae", "D", row["choices"]["neg2"]["contact_s"],
                     row["choices"]["neg2"]["pass"],
                     row["choices"]["neg2"]["margin_1p2_s_met"]),
                    ("+2 mm", .22, "#087e70", "o", row["choices"]["pos2"]["contact_s"],
                     row["choices"]["pos2"]["pass"],
                     row["choices"]["pos2"]["margin_1p2_s_met"])):
                axis.scatter(contact, index + offset, color=color if passed else "#c04a59",
                             marker=marker if passed else "X", s=44 if passed else 65,
                             edgecolor=("#b98224" if passed and not margin else "white"),
                             linewidth=1.4 if passed and not margin else .5, zorder=3)
        axis.axvline(1.0, color="#b74c5a", linestyle=":")
        axis.axvline(1.2, color="#b98224", linestyle="--")
        axis.set(yticks=range(len(route_cases)),
                 yticklabels=[str(case["seed"]) for case in reversed(route_cases)],
                 xlabel="Continuous fingertip contact (s)", title=route,
                 xlim=(-.1, 6.4), ylim=(-.65, len(route_cases)-.35))
        axis.grid(axis="x", alpha=.18)
    for name, color, marker in (("unchanged", "#53646c", "s"),
                                ("−2 mm", "#8156ae", "D"),
                                ("+2 mm", "#087e70", "o")):
        axes[0].scatter([], [], color=color, marker=marker, label=name)
    axes[0].scatter([], [], color="#c04a59", marker="X", label="full gate failure")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -.06),
                   ncol=3, frameon=False)
    fig.suptitle("V147 new-seed paired target-direction training · not a score")
    temp = FIGURE.with_name("new-seed-paired-contact.tmp.png")
    fig.savefig(temp, dpi=145)
    plt.close(fig)
    os.replace(temp, FIGURE)
    summary = {
        "schema": "robot-learning-lab.cove50-v147-new-seed-paired-summary.v1",
        "role": "source_checked_new_seed_training_not_independent_score",
        "body_role": plan["body_role"],
        "source_sha256": sha(Path(__file__)),
        "plan_sha256": sha(PLAN), "reserve_sha256": sha(RESERVE),
        "matrix_sha256": sha(MATRIX),
        "source_checked_seed_count": 40,
        "source_checked_native_visit_count": 120,
        "baseline_pass_count": matrix["baseline_pass_count"],
        "neg2_pass_count": matrix["neg2_pass_count"],
        "pos2_pass_count": matrix["pos2_pass_count"],
        "neg2_rescue_count": matrix["neg2_rescue_count"],
        "pos2_rescue_count": matrix["pos2_rescue_count"],
        "neg2_regression_count": matrix["neg2_regression_count"],
        "pos2_regression_count": matrix["pos2_regression_count"],
        "entry_parity_failure_count": matrix["entry_parity_failure_count"],
        "observed_route_counts": dict(sorted(Counter(
            row["selection_mode"] for row in rows.values()).items())),
        "commanded_to_observed_route_counts": dict(sorted(Counter(
            f"{row['route_variant']} → {row['selection_mode']}"
            for row in rows.values()).items())),
        "baseline_margin_misses_by_observed_route": dict(sorted(Counter(
            row["selection_mode"] for row in rows.values()
            if not row["baseline_pass"] or
            not row["baseline_margin_1p2_s_met"]).items())),
        **repair_counts,
        "pitch_low_specialist_count": pitch_low_specialist_count,
        "failures_or_margin_misses": failures,
        "figure": str(FIGURE.relative_to(ROOT)), "figure_sha256": sha(FIGURE),
        "qualification": ("Varied new-seed teacher training on rejected Direct Frame. "
                          "Frozen 24-start score reserve remains unopened."),
    }
    temp_json = SUMMARY.with_suffix(".json.tmp")
    temp_json.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temp_json, SUMMARY)
    print("V147 source-checked 40 new seed groups and 120 native visits")


if __name__ == "__main__":
    main()
