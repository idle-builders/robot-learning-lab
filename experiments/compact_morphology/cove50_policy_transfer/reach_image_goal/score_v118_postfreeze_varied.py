"""Score frozen V118 choices on V113 pairs completed after its first eight."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import (
    v118_observable_selector as selector,
)


ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
PLAN = BASE / "v113/varied-paired-contact-data-plan.json"
MATRIX = BASE / "v113/native-varied-paired-contact-data/matrix.json"
MODEL = BASE / "v118/observable-selector.json"
FIRST_EIGHT = BASE / "v118/first-eight-opened-development.json"
OUT = BASE / "v118/postfreeze-varied-development.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if not MATRIX.is_file():
        return
    native = json.loads(MATRIX.read_text())
    first = json.loads(FIRST_EIGHT.read_text())
    model = selector.load(MODEL)
    if (native["plan_sha256"] != sha(PLAN) or
            first["model_sha256"] != sha(MODEL) or
            first["case_count"] != 8 or
            [row["seed"] for row in first["cases"]] !=
            [row["seed"] for row in native["cases"][:8]] or
            model["inference_source_sha256"] != sha(Path(selector.__file__))):
        raise ValueError("V118 frozen selector or first-eight boundary changed")
    rows = []
    for row in native["cases"][8:]:
        report = json.loads((ROOT / row["baseline_result"]).read_text())
        observed = selector.observed_features(report)
        decision = selector.predict(model, observed)
        shifted = decision["shift_body_y_m"] < 0
        rows.append({
            "id": row["id"], "seed": row["seed"],
            "route_variant": row["route_variant"],
            "selection_mode": row["selection_mode"],
            "rail_v": row["rail_v"], "delay_ms": row["delay_ms"],
            "entry_orange_pixels": observed["orange_pixels"],
            "frozen_decision": decision,
            "baseline_contact_s": row["baseline_contact_s"],
            "shifted_contact_s": row["shifted_contact_s"],
            "candidate_contact_s": (row["shifted_contact_s"] if shifted
                                    else row["baseline_contact_s"]),
            "candidate_contact_delta_s": (row["contact_delta_s"] if shifted else 0.0),
            "baseline_pass": row["baseline_pass"],
            "candidate_pass": (row["shifted_pass"] if shifted else row["baseline_pass"]),
            "baseline_margin_1p2_s_met": row["baseline_margin_1p2_s_met"],
            "candidate_margin_1p2_s_met": (row["shifted_margin_1p2_s_met"]
                                           if shifted else row["baseline_margin_1p2_s_met"]),
            "baseline_result": row["baseline_result"],
            "candidate_result": (row["shifted_result"] if shifted
                                 else row["baseline_result"]),
            "candidate_faults": (row["shifted_faults"] if shifted
                                 else row["baseline_faults"]),
        })
    result = {
        "schema": "robot-learning-lab.cove50-v118-postfreeze-varied.v1",
        "role": "postfreeze_opened_development_not_sealed_contact_holdout",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "model_sha256": sha(MODEL),
        "native_plan_sha256": sha(PLAN),
        "native_matrix_sha256": sha(MATRIX),
        "first_eight_opened_model_choice_sha256": sha(FIRST_EIGHT),
        "case_count": len(rows), "planned_count": 16,
        "complete": native["complete"] and len(rows) == 16,
        "selected_shift_count": sum(row["frozen_decision"]["shift_body_y_m"] < 0
                                    for row in rows),
        "baseline_pass_count": sum(row["baseline_pass"] for row in rows),
        "candidate_pass_count": sum(row["candidate_pass"] for row in rows),
        "rescue_count": sum(not row["baseline_pass"] and row["candidate_pass"]
                            for row in rows),
        "regression_count": sum(row["baseline_pass"] and not row["candidate_pass"]
                                for row in rows),
        "margin_loss_count": sum(row["baseline_margin_1p2_s_met"] and
                                 not row["candidate_margin_1p2_s_met"]
                                 for row in rows),
        "contact_gain_total_s": round(sum(row["candidate_contact_delta_s"]
                                          for row in rows), 3),
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temp, OUT)
    print(f"V118 postfreeze {len(rows)}/16, selected {result['selected_shift_count']}, "
          f"rescue {result['rescue_count']}, regression {result['regression_count']}, "
          f"contact gain {result['contact_gain_total_s']:+.2f}s", flush=True)


if __name__ == "__main__":
    main()
