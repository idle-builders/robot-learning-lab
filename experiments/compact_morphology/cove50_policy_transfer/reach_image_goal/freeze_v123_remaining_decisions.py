"""Freeze V123 choices on reserved RGB entries before contact replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010/v123"
RESERVED = BASE / "remaining-entry-contact-holdout-plan.json"
MODEL = BASE / "observable-selector.json"
PLAN = BASE / "remaining-entry-frozen-decisions.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    reserved = json.loads(RESERVED.read_text())
    model = selector.load(MODEL)
    if (model["model_id"] != "v123_complete_varied_pairs" or
            model["reserved_entry_plan_sha256"] != sha(RESERVED) or
            model["inference_source_sha256"] != sha(Path(selector.__file__))):
        raise ValueError("V123 selector differs from reserved entry source")
    cases = []
    for case in reserved["cases"]:
        source = ROOT / case["source_entry_report"]
        if sha(source) != case["source_entry_sha256"]:
            raise ValueError(f"Reserved RGB report changed: {case['id']}")
        report = json.loads(source.read_text())
        if report["seed"] != case["seed"] or report["selection_mode"] != case["expected_selection_mode"]:
            raise ValueError(f"Reserved route changed: {case['id']}")
        features = selector.observed_features(report)
        decision = selector.predict(model, features)
        cases.append({**case, "entry_observed_features": features,
                      "frozen_decision": decision})
    plan = {
        "schema": "robot-learning-lab.cove50-v123-remaining-frozen-decisions.v1",
        "role": "precontact_decisions_on_entry_seen_contact_unseen_starts",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "reserved_plan_sha256": sha(RESERVED),
        "model_sha256": sha(MODEL),
        "inference_source_sha256": sha(Path(selector.__file__)),
        "runner_source_sha256": sha(Path(__file__)),
        "decision_timing": ("Predictions made from retained V105 entry-only RGB "
                            "reports before any V123 native contact replay; "
                            "no V119 contact outcome entered this fit."),
        "evaluation_status": "not_run",
        "case_count": len(cases),
        "selected_shift_count": sum(case["frozen_decision"]["shift_body_y_m"] < 0
                                    for case in cases),
        "cases": cases,
    }
    encoded = (json.dumps(plan, indent=2) + "\n").encode()
    if PLAN.exists() and PLAN.read_bytes() != encoded:
        raise ValueError("Frozen V123 decisions or source changed")
    PLAN.write_bytes(encoded)
    print(f"frozen {PLAN} sha256={sha(PLAN)} "
          f"selected={plan['selected_shift_count']}/{plan['case_count']}")


if __name__ == "__main__":
    main()
