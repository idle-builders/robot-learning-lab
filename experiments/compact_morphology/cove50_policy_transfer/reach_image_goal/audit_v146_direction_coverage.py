"""Audit seed-group coverage before another observed-RGB direction candidate."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    audit_v134_planar_rule as old_audit,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OLD = BASE / "v131/failure-augmented-observable-pairs.json"
V144 = BASE / "v144/native-supported-opposite-direction-variance/matrix.json"
V144_SUMMARY = BASE / "v144/paired-contact-summary.json"
V145 = BASE / "v145/native-other-opened-seed-directions/matrix.json"
V145_SUMMARY = BASE / "v145/other-opened-direction-summary.json"
OUT = BASE / "v146/direction-coverage-audit.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    old = json.loads(OLD.read_text())
    v144, s144 = json.loads(V144.read_text()), json.loads(V144_SUMMARY.read_text())
    v145, s145 = json.loads(V145.read_text()), json.loads(V145_SUMMARY.read_text())
    if (not old["complete"] or len(old["pairs"]) != 113 or
            s144["matrix_sha256"] != sha(V144) or
            s144["source_checked_native_visit_count"] != 48 or
            not v144["complete"] or v144["case_count"] != 16 or
            s145["matrix_sha256"] != sha(V145) or
            s145["source_checked_seed_count"] != 11 or
            not v145["complete"] or v145["case_count"] != 11 or
            v144["body_role"] != v145["body_role"] or
            v144["entry_parity_failure_count"] or
            v145["entry_parity_failure_count"]):
        raise ValueError("Complete source-checked opened matrices required")
    rows = list(old["pairs"])
    for row in rows:
        for name in ("baseline", "shifted"):
            if sha(ROOT / row[f"{name}_result"]) != row[f"{name}_result_sha256"]:
                raise ValueError(f"Prior paired source changed: {row['seed_audit_only']}")
    new_rows = []
    for cohort, matrix in (("v144", v144), ("v145", v145)):
        for row in matrix["cases"]:
            neg = row["choices"]["neg2"]
            new_rows.append({
                "seed_audit_only": row["seed"], "cohort": cohort,
                "features": row["entry_observed_features"],
                "baseline_pass": row["baseline_pass"],
                "baseline_margin_1p2_s_met": row["baseline_margin_1p2_s_met"],
                "baseline_contact_s": row["baseline_contact_s"],
                "shifted_pass": neg["pass"],
                "shifted_margin_1p2_s_met": neg["margin_1p2_s_met"],
                "shifted_contact_s": neg["contact_s"],
                "contact_delta_s": neg["contact_delta_s"],
            })
    rows.extend(new_rows)
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["seed_audit_only"]].append(row)
    if (len(rows) != 140 or len(groups) != 102 or
            {row["seed"] for row in v144["cases"]} &
            {row["seed"] for row in v145["cases"]}):
        raise ValueError("Expected 140 paired conditions in 102 seed groups")
    held = {}
    for seed in groups:
        training = [row for row in rows if row["seed_audit_only"] != seed]
        rule = old_audit.fit(training, "equal_seed")
        held[seed] = [old_audit.predict(rule, row["features"])
                      for row in groups[seed]]
    selected = [choice for seed in groups for choice in held[seed]]
    ordered = [row for seed in groups for row in groups[seed]]
    neg_summary = old_audit.summarize(ordered, selected)
    pitch_safe = [choice and row["features"]["pitch_offset_rad"] >= .105 - 1e-9
                  for row, choice in zip(ordered, selected)]
    pitch_safe_summary = old_audit.summarize(ordered, pitch_safe)
    tri = list(v144["cases"]) + list(v145["cases"])
    tri_summary = {}
    for name in ("neg2", "pos2"):
        tri_summary[name] = {
            "pass_count": sum(row["choices"][name]["pass"] for row in tri),
            "rescue_condition_count": sum(not row["baseline_pass"] and
                                          row["choices"][name]["pass"] for row in tri),
            "rescue_seed_count": len({row["seed"] for row in tri
                                      if not row["baseline_pass"] and
                                      row["choices"][name]["pass"]}),
            "regression_condition_count": sum(row["baseline_pass"] and
                                              not row["choices"][name]["pass"]
                                              for row in tri),
            "regression_seeds": sorted({row["seed"] for row in tri
                                        if row["baseline_pass"] and
                                        not row["choices"][name]["pass"]}),
        }
    report = {
        "schema": "robot-learning-lab.cove50-v146-direction-coverage-audit.v1",
        "role": "opened_grouped_training_coverage_not_independent_score",
        "body_role": v144["body_role"],
        "source_sha256": {"prior_pairs": sha(OLD), "v144_matrix": sha(V144),
                          "v144_summary": sha(V144_SUMMARY), "v145_matrix": sha(V145),
                          "v145_summary": sha(V145_SUMMARY),
                          "old_grouped_fitter": sha(Path(old_audit.__file__)),
                          "audit": sha(Path(__file__))},
        "negative_pair_count": len(rows), "negative_seed_group_count": len(groups),
        "tri_direction_condition_count": len(tri),
        "tri_direction_seed_group_count": len({row["seed"] for row in tri}),
        "tri_direction_baseline_failure_seed_count": len({row["seed"] for row in tri
                                                         if not row["baseline_pass"]}),
        "tri_direction": tri_summary,
        "negative_rule_leave_one_seed_out": neg_summary,
        "negative_rule_pitch_qualified_leave_one_seed_out": pitch_safe_summary,
        "method": ("Verify prior result hashes and complete V144/V145 source-checked "
                   "matrices. Hold every physical variation of a seed together. "
                   "Refit the prior observed-RGB negative-shift rule without "
                   "that seed and count held decisions, both as originally "
                   "fitted and with an observed pitch >=0.105 rad guard. "
                   "Positive shifts are "
                   "coverage only because prior paired data lacks that action. "
                   "No seed or simulator truth is an inference input."),
        "conclusion": ("The tri-direction evidence contains only one unique "
                       "full-gate failure seed. Negative-shift prior pairs "
                       "add other rescue groups, but signed direction "
                       "selection and positive-shift rescue still need "
                       "new-seed paired development and old-start retention "
                       "before a new independent score."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + "\n"
    if OUT.exists() and OUT.read_text() != encoded:
        raise ValueError("Previously saved V146 audit changed")
    OUT.write_text(encoded)
    print(f"V146 coverage: {len(rows)} pairs, {len(groups)} seed groups; "
          f"negative LOO {neg_summary['selected_rescue_seed_count']} "
          f"rescued/{neg_summary['selected_full_gate_regressions']} regressions; "
          f"pitch guard {pitch_safe_summary['selected_rescue_seed_count']} "
          f"rescued/{pitch_safe_summary['selected_full_gate_regressions']} regressions")


if __name__ == "__main__":
    main()
