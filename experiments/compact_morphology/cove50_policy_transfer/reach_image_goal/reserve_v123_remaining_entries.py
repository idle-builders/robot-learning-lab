"""Seal every V105 RGB entry still unseen by full-native contact replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v123"
PLAN = OUT / "remaining-entry-contact-holdout-plan.json"
V105_PLAN = BASE / "v105/varied-entry-exposure-plan.json"
V105_MATRIX = BASE / "v105/native-varied-entry-exposure/matrix.json"
V113_PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
V113_MATRIX = BASE / "v113/native-varied-paired-contact-data/matrix.json"
V116_PLAN = BASE / "v116/reserved-contact-validation-plan.json"
V109_PLAN = BASE / "v109/varied-specialist-transfer-plan.json"
V119_PLAN = BASE / "v119/selector-heldout-plan.json"
POINTER = ROOT / "robot/ape-current-design.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if json.loads(POINTER.read_text())["status"] != "rejected_no_selected_mechanical_design":
        raise ValueError("V123 is only a historical Direct Frame sandbox")
    entry_plan = json.loads(V105_PLAN.read_text())
    entry_matrix = json.loads(V105_MATRIX.read_text())
    varied_plan = json.loads(V113_PLAN.read_text())
    varied_matrix = json.loads(V113_MATRIX.read_text())
    if (not entry_matrix["complete"] or entry_matrix["case_count"] != 64 or
            entry_matrix["plan_sha256"] != sha(V105_PLAN) or
            not varied_matrix["complete"] or varied_matrix["case_count"] != 24 or
            varied_matrix["plan_sha256"] != sha(V113_PLAN) or
            varied_plan["source_sha256"]["pointer"] != sha(POINTER)):
        raise ValueError("Complete entry and varied contact sources are required")
    earlier_holdout = json.loads(V116_PLAN.read_text())
    first_holdout = json.loads(V119_PLAN.read_text())
    if ({case["seed"] for case in earlier_holdout["cases"]} !=
            {case["seed"] for case in first_holdout["cases"]}):
        raise ValueError("Earlier reserved contact split changed")
    excluded = ({case["seed"] for case in varied_plan["cases"]} |
                {case["seed"] for case in earlier_holdout["cases"]} |
                {case["seed"] for case in json.loads(V109_PLAN.read_text())["cases"]})
    entries = {row["seed"]: row for row in entry_matrix["cases"]}
    cases = []
    for source in entry_plan["cases"]:
        seed = source["seed"]
        if seed in excluded:
            continue
        entry = entries[seed]
        report_path = ROOT / entry["result"]
        report = json.loads(report_path.read_text())
        if (not entry["entry_available"] or entry["fault"] or
                report["seed"] != seed or
                report["body_sha256"] != varied_plan["source_sha256"]["body"] or
                report["selected_pointer_sha256"] != sha(POINTER)):
            raise ValueError(f"V123 reserved RGB entry differs from source: {seed}")
        cases.append({**source, "id": f"remaining-{seed}",
                      "expected_selection_mode": entry["selection_mode"],
                      "source_entry_report": entry["result"],
                      "source_entry_sha256": sha(report_path)})
    if len(cases) != 14 or len({case["seed"] for case in cases}) != 14:
        raise ValueError("Expected exactly fourteen remaining distinct entries")
    plan = {
        "schema": "robot-learning-lab.cove50-v123-remaining-contact-holdout-plan.v1",
        "role": "entry_seen_contact_unseen_reserve_before_complete_pair_fit",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "source_sha256": {"runner": sha(Path(__file__)),
                          "pointer": sha(POINTER),
                          "v105_plan": sha(V105_PLAN),
                          "v105_matrix": sha(V105_MATRIX),
                          "v113_plan": sha(V113_PLAN),
                          "v113_matrix": sha(V113_MATRIX),
                          "v116_plan": sha(V116_PLAN),
                          "v109_plan": sha(V109_PLAN),
                          "v119_plan": sha(V119_PLAN)},
        "selection": ("All remaining V105 entries, excluding every V113 paired "
                      "training start, earlier V116/V119 reserved start and V109 "
                      "specialist contact start. Selection uses no contact outcome "
                      "from these fourteen starts."),
        "contact_status": "not_run",
        "case_count": len(cases), "cases": cases,
        "future_evaluation": ("Freeze a fitted candidate's decisions from these exact "
                              "RGB entries before native contact. Compare each "
                              "chosen target with a fresh unchanged replay and "
                              "retain full RGB, continuous fingertip contact, "
                              "support, rated effort, action and collision outcomes. "
                              "This is a previously seen image-entry split, not "
                              "a wholly new scene score or body qualification."),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen remaining-entry holdout or source changed")
    PLAN.write_bytes(encoded)
    print(f"frozen {PLAN} sha256={sha(PLAN)} cases={len(cases)}")


if __name__ == "__main__":
    main()
