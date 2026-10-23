"""Build source-checked grouped training for an observed-input signed target rule.

This is opened development data. Seed identity is retained only for whole-seed
cross-validation and is never a candidate runtime input.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    run_v139_reserved_contact as score_runner,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OLD = BASE / "v131/failure-augmented-observable-pairs.json"
SOURCES = (
    ("v144", "supported-opposite-direction-variance-plan.json",
     "native-supported-opposite-direction-variance/matrix.json",
     "paired-contact-summary.json"),
    ("v145", "other-opened-seed-direction-plan.json",
     "native-other-opened-seed-directions/matrix.json",
     "other-opened-direction-summary.json"),
    ("v147", "new-seed-direction-training-plan.json",
     "native-new-seed-direction-pairs/matrix.json",
     "new-seed-paired-summary.json"),
)
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
OUT = BASE / "v148/signed-observed-training.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_report(path: str, features: dict, seed: int) -> None:
    report = json.loads((ROOT / path).read_text())
    if (report["seed"] != seed or
            score_runner.entry_parity(
                features, score_runner.observations.observed_features(report))):
        raise ValueError(f"Paired observed entry changed: {path}")


def main() -> None:
    old = json.loads(OLD.read_text())
    reserve = json.loads(RESERVE.read_text())
    if (not old["complete"] or len(old["pairs"]) != 113 or
            reserve["status"] != "sealed_no_contact_visits" or
            reserve["case_count"] != 24):
        raise ValueError("Prior paired data or untouched score reserve incomplete")
    rows = []
    source = {"v131_dataset": sha(OLD), "v147_score_reserve": sha(RESERVE)}
    for old_row in old["pairs"]:
        for target in ("baseline", "shifted"):
            report_path = ROOT / old_row[f"{target}_result"]
            if sha(report_path) != old_row[f"{target}_result_sha256"]:
                raise ValueError(f"V131 saved report changed: {report_path}")
            check_report(old_row[f"{target}_result"], old_row["features"],
                         old_row["seed_audit_only"])
        rows.append({
            "id": f"v131-{len(rows)}", "cohort": old_row["cohort"],
            "seed_audit_only": old_row["seed_audit_only"],
            "features": old_row["features"],
            "unchanged": {"contact_s": old_row["baseline_contact_s"],
                          "pass": old_row["baseline_pass"],
                          "margin_1p2_s_met": old_row["baseline_margin_1p2_s_met"],
                          "report": old_row["baseline_result"]},
            "neg2": {"contact_s": old_row["shifted_contact_s"],
                     "pass": old_row["shifted_pass"],
                     "margin_1p2_s_met": old_row["shifted_margin_1p2_s_met"],
                     "report": old_row["shifted_result"]},
            "pos2": None,
        })
    score_plan = BASE / "v139/reserved-contact-score-plan.json"
    score_matrix = BASE / "v139/native-reserved-contact/matrix.json"
    opened_plan = BASE / "v141/opened-score-failure-plan.json"
    opened_matrix = BASE / "v141/native-opened-failure-shifts/matrix.json"
    direction_plan = BASE / "v142/opened-pitch-low-y-sweep-plan.json"
    direction_matrix = BASE / "v142/native-opened-pitch-low-y-sweep/matrix.json"
    direction_summary = BASE / "v142/valid-visits-and-launch-failure.json"
    scored = json.loads(score_matrix.read_text())
    opened = json.loads(opened_matrix.read_text())
    direction = json.loads(direction_matrix.read_text())
    direction_proof = json.loads(direction_summary.read_text())
    if (not scored["complete"] or scored["score_gate_met"] or
            scored["plan_sha256"] != sha(score_plan) or
            not opened["complete"] or opened["case_count"] != 2 or
            opened["plan_sha256"] != sha(opened_plan) or
            direction["complete"] or direction["valid_supported_count"] != 5 or
            direction["plan_sha256"] != sha(direction_plan) or
            direction_proof["source_sha256"]["matrix"] != sha(direction_matrix) or
            direction_proof["source_sha256"]["plan"] != sha(direction_plan)):
        raise ValueError("Opened score-miss sources changed or incomplete")
    for label, path in (("v139_plan", score_plan), ("v139_matrix", score_matrix),
                        ("v141_plan", opened_plan), ("v141_matrix", opened_matrix),
                        ("v142_plan", direction_plan),
                        ("v142_matrix", direction_matrix),
                        ("v142_partial_summary", direction_summary)):
        source[label] = sha(path)
    score_rows = {row["seed"]: row for row in scored["cases"]}
    positive = next(row for row in direction["cases"]
                    if row["seed"] == 24119 and row["shift_body_y_m"] == .002)
    if (sha(ROOT / positive["result"]) != positive["report_sha256"] or
            sha(ROOT / positive["trace"]) != positive["trace_sha256"]):
        raise ValueError("Opened positive target result changed")
    for diagnostic in opened["cases"]:
        seed = diagnostic["seed"]
        if seed not in (24111, 24119):
            raise ValueError("Unexpected opened score-miss seed")
        baseline = score_rows[seed]
        features = score_runner.observations.observed_features(
            json.loads((ROOT / baseline["baseline_result"]).read_text()))
        if (features != baseline["native_entry_features"] or
                diagnostic["baseline_result"] != baseline["baseline_result"] or
                diagnostic["baseline_contact_s"] != baseline["baseline_contact_s"] or
                diagnostic["entry_parity_faults"]):
            raise ValueError(f"Opened score baseline changed: {seed}")
        check_report(diagnostic["diagnostic_result"], features, seed)
        if seed == 24119:
            check_report(positive["result"], features, seed)
        rows.append({
            "id": f"v141-opened-{seed}", "cohort": "opened_score_miss",
            "seed_audit_only": seed, "features": features,
            "unchanged": {"contact_s": baseline["baseline_contact_s"],
                          "pass": baseline["baseline_native_pass"],
                          "margin_1p2_s_met": (baseline["baseline_native_pass"] and
                                               baseline["baseline_contact_s"] >= 1.2),
                          "report": baseline["baseline_result"],
                          "trace": baseline["baseline_trace"]},
            "neg2": {"contact_s": diagnostic["diagnostic_contact_s"],
                     "pass": diagnostic["diagnostic_pass"],
                     "margin_1p2_s_met": diagnostic["diagnostic_margin_1p2_s_met"],
                     "report": diagnostic["diagnostic_result"],
                     "trace": diagnostic["diagnostic_trace"]},
            "pos2": ({"contact_s": positive["contact_s"],
                      "pass": positive["pass"],
                      "margin_1p2_s_met": positive["margin_1p2_s_met"],
                      "report": positive["result"], "trace": positive["trace"]}
                     if seed == 24119 else None),
        })
    for version, plan_name, matrix_name, summary_name in SOURCES:
        folder = BASE / version
        plan_path, matrix_path = folder / plan_name, folder / matrix_name
        summary_path = folder / summary_name
        plan = json.loads(plan_path.read_text())
        matrix = json.loads(matrix_path.read_text())
        summary = json.loads(summary_path.read_text())
        count = 40 if version == "v147" else 16 if version == "v144" else 11
        expected_visits = count * 3
        checked_visits = (summary["reused_native_visit_count"] +
                          summary["new_native_visit_count"]
                          if version == "v145" else
                          summary["source_checked_native_visit_count"])
        checked_key = ("source_checked_seed_count" if version in ("v145", "v147")
                       else "source_checked_condition_count")
        if (not matrix["complete"] or matrix["case_count"] != count or
                matrix["plan_sha256"] != sha(plan_path) or
                matrix["entry_parity_failure_count"] or
                summary["plan_sha256"] != sha(plan_path) or
                summary["matrix_sha256"] != sha(matrix_path) or
                summary[checked_key] != count or
                checked_visits != expected_visits):
            raise ValueError(f"{version} source-checked paired data incomplete")
        source[f"{version}_plan"] = sha(plan_path)
        source[f"{version}_matrix"] = sha(matrix_path)
        source[f"{version}_summary"] = sha(summary_path)
        cases = {case["id"]: case for case in plan["cases"]}
        if {row["id"] for row in matrix["cases"]} != set(cases):
            raise ValueError(f"{version} matrix rows differ from frozen plan")
        for row in matrix["cases"]:
            case, features = cases[row["id"]], row["entry_observed_features"]
            if row["seed"] != case["seed"]:
                raise ValueError(f"{version} seed changed: {row['id']}")
            check_report(row["baseline_result"], features, row["seed"])
            for name in ("neg2", "pos2"):
                choice = row["choices"][name]
                if choice["entry_parity_faults"]:
                    raise ValueError(f"{version} RGB entry mismatch: {row['id']}")
                check_report(choice["result"], features, row["seed"])
            rows.append({
                "id": f"{version}-{row['id']}", "cohort": version,
                "seed_audit_only": row["seed"], "features": features,
                "unchanged": {"contact_s": row["baseline_contact_s"],
                              "pass": row["baseline_pass"],
                              "margin_1p2_s_met": row["baseline_margin_1p2_s_met"],
                              "report": row["baseline_result"],
                              "trace": row["baseline_trace"]},
                **{name: {"contact_s": row["choices"][name]["contact_s"],
                          "pass": row["choices"][name]["pass"],
                          "margin_1p2_s_met": row["choices"][name]["margin_1p2_s_met"],
                          "report": row["choices"][name]["result"],
                          "trace": row["choices"][name]["trace"]}
                   for name in ("neg2", "pos2")},
            })
    seeds = {row["seed_audit_only"] for row in rows}
    if (len(rows) != 182 or len(seeds) != 142 or
            seeds & {case["seed"] for case in reserve["cases"]}):
        raise ValueError("Unexpected grouped training size or score-seed overlap")
    for row in rows:
        for name in ("unchanged", "neg2", "pos2"):
            action = row[name]
            if action is None:
                continue
            action["report_sha256"] = sha(ROOT / action["report"])
            if action.get("trace"):
                action["trace_sha256"] = sha(ROOT / action["trace"])
    report = {
        "schema": "robot-learning-lab.cove50-v148-signed-observed-training.v1",
        "role": "opened_paired_training_not_independent_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {**source, "builder": sha(Path(__file__))},
        "pair_count": len(rows), "distinct_seed_count": len(seeds),
        "negative_pair_count": len(rows),
        "positive_pair_count": sum(row["pos2"] is not None for row in rows),
        "observed_input_contract": list(rows[0]["features"]),
        "method": ("Keep all source-checked pairs, including failures and harmful "
                   "directions. Group every variation of a seed in fitting and "
                   "cross-validation. Seed, toy truth and simulated outcome are "
                   "audit-only and may never enter runtime prediction."),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Saved V148 training dataset changed")
    OUT.write_text(encoded)
    print(f"V148 source-checked {len(rows)} pairs from {len(seeds)} seeds; "
          f"{report['positive_pair_count']} signed positive pairs; sha256={sha(OUT)}")


if __name__ == "__main__":
    main()
