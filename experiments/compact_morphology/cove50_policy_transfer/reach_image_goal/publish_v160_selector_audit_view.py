"""Publish source-checked whole-seed V160 selector decisions for inspection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
METHOD = BASE / "v160/new-seed-selector-audit-method-freeze.json"
AUDIT = BASE / "v160/new-seed-selector-grouped-audit.json"
V152_PLAN = BASE / "v152/paired-route-training-plan.json"
V152_MATRIX = BASE / "v152/native-paired-routes/matrix.json"
V152_SUMMARY = BASE / "v152/partial-source-checked-paired-route-summary.json"
V158_METHOD = BASE / "v158/new-seed-contact-selection-method-freeze.json"
V158_PLAN = BASE / "v158/new-seed-contact-plan.json"
V158_MATRIX = BASE / "v158/native-development-contact/matrix.json"
V158_SUMMARY = BASE / "v158/source-checked-development-contact-summary.json"
RESERVE = BASE / "v147/untouched-score-reserve-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"
PROJECTION = BASE / "v160/grouped-selector-audit-inspector-matrix.json"
FIGURE = BASE / "v160/grouped-selector-decisions.png"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def draw(rows: list[dict]) -> None:
    fig, axis = plt.subplots(figsize=(11, max(5, len(rows) * .42 + 2)))
    for index, row in enumerate(rows):
        selected = row["selected_route"] != "unchanged"
        color = ("#ba4a57" if row["selected_full_gate_regression"] or
                 row["selected_margin_loss"] else
                 "#da9a39" if row["selected_unresolved_alternative"] else
                 "#087e70" if row["selected_full_gate_margin_rescue"] else
                 "#53646c")
        axis.scatter(row["baseline_contact_s"], index - .13,
                     color="#53646c", marker="s", s=48)
        if selected and row["prescribed_contact_s"] is not None:
            axis.scatter(row["prescribed_contact_s"], index + .13,
                         color=color, marker="o", s=50)
        elif selected:
            axis.scatter(row["baseline_contact_s"], index + .13,
                         color="#da9a39", marker="x", s=55)
    axis.axvline(1.0, color="#ba4a57", linestyle=":", label="1 s contact gate")
    axis.axvline(1.2, color="#b98224", linestyle="--", label="1.2 s margin")
    axis.set(yticks=range(len(rows)),
             yticklabels=[f"{row['seed_audit_only']} · "
                          f"{'selected' if row['selected_route'] != 'unchanged' else 'unchanged'}"
                          for row in rows],
             xlabel="Continuous fingertip contact (s)",
             title="V160 leave-one-new-seed-out selector audit · development only")
    axis.scatter([], [], color="#53646c", marker="s", label="automatic")
    axis.scatter([], [], color="#087e70", marker="o", label="selected rescue")
    axis.scatter([], [], color="#ba4a57", marker="o", label="regression or margin loss")
    axis.scatter([], [], color="#da9a39", marker="x", label="unresolved entry pair")
    axis.legend(loc="upper center", bbox_to_anchor=(.5, -.08), ncol=3, fontsize=8)
    axis.grid(axis="x", alpha=.18)
    fig.tight_layout()
    temporary = FIGURE.with_name(FIGURE.stem + ".tmp.png")
    fig.savefig(temporary, dpi=140)
    plt.close(fig)
    os.replace(temporary, FIGURE)


def main() -> None:
    needed = (METHOD, AUDIT, V152_PLAN, V152_MATRIX, V152_SUMMARY,
              V158_METHOD, V158_PLAN, V158_MATRIX, V158_SUMMARY, RESERVE,
              POINTER)
    if any(not path.is_file() for path in needed):
        raise ValueError("Complete source-checked V160 development audit required")
    method = json.loads(METHOD.read_text())
    audit = json.loads(AUDIT.read_text())
    old_summary = json.loads(V152_SUMMARY.read_text())
    plan = json.loads(V158_PLAN.read_text())
    matrix = json.loads(V158_MATRIX.read_text())
    summary = json.loads(V158_SUMMARY.read_text())
    reserve = json.loads(RESERVE.read_text())
    refs = audit["source_sha256"]
    rows = audit["held_out_decisions"]
    if (refs["method"] != sha(METHOD) or
            refs["v152_plan"] != sha(V152_PLAN) or
            refs["v152_matrix"] != sha(V152_MATRIX) or
            refs["v152_summary"] != sha(V152_SUMMARY) or
            refs["v158_method"] != sha(V158_METHOD) or
            refs["v158_plan"] != sha(V158_PLAN) or
            refs["v158_matrix"] != sha(V158_MATRIX) or
            refs["v158_summary"] != sha(V158_SUMMARY) or
            refs["reserve"] != sha(RESERVE) or
            refs["pointer"] != sha(POINTER) or
            method["source_sha256"]["audit_source"] != sha(
                Path(__file__).with_name("audit_v160_new_seed_route_selector.py")) or
            reserve["status"] != "sealed_no_contact_visits" or
            not matrix["complete"] or
            summary["source_sha256"]["matrix"] != sha(V158_MATRIX) or
            not summary["complete"] or
            audit["historical_training_seed_count"] != 47 or
            audit["held_contact_visit_count"] != 0 or
            audit["independent_score_visit_count"] != 0 or
            len(rows) != matrix["case_count"] or
            {row["id"] for row in rows} !=
            {row["id"] for row in matrix["cases"]} or
            plan["development_count"] != len(rows) or
            sum(row["selected_full_gate_margin_rescue"] for row in rows) !=
            audit["held_out_full_gate_margin_rescue_count"] or
            sum(row["selected_full_gate_regression"] for row in rows) !=
            audit["held_out_full_gate_regression_count"] or
            sum(row["selected_margin_loss"] for row in rows) !=
            audit["held_out_margin_loss_count"] or
            sum(row["selected_unresolved_alternative"] for row in rows) !=
            audit["held_out_unresolved_count"]):
        raise ValueError("V160 audit source, held boundary or decisions changed")
    expected_gate = (
        audit["held_out_full_gate_margin_rescue_count"] >= 2 and
        set(audit["held_out_rescue_cohorts"]) == {
            "opened_roll_high_neighborhood", "opened_pitch_low_neighborhood"} and
        audit["held_out_full_gate_regression_count"] == 0 and
        audit["held_out_margin_loss_count"] == 0 and
        audit["held_out_unresolved_count"] == 0)
    if audit["development_gate_met"] != expected_gate:
        raise ValueError("V160 grouped development gate differs from decisions")
    for item in old_summary["source_files"] + summary["source_files"]:
        for key in ("report", "trace"):
            path = ROOT / item[key]
            if not path.is_file() or sha(path) != item[f"{key}_sha256"]:
                raise ValueError(f"V160 audit native source changed: {path}")
    draw(rows)
    write_json(PROJECTION, {
        "schema": "robot-learning-lab.cove50-v160-selector-audit-inspector.v1",
        "role": "whole_new_seed_development_audit_not_held_score",
        "source_sha256": {"method": sha(METHOD), "audit": sha(AUDIT),
                          "publisher": sha(Path(__file__)),
                          "figure": sha(FIGURE)},
        "case_count": len(rows), "planned_count": len(rows),
        "complete": True,
        "development_gate_met": audit["development_gate_met"],
        "held_contact_visit_count": 0,
        "independent_score_visit_count": 0,
        "cases": rows,
    })
    print(json.dumps({"development_gate_met": audit["development_gate_met"],
                      "source_checked_decision_count": len(rows),
                      "figure_sha256": sha(FIGURE)}))


if __name__ == "__main__":
    main()
