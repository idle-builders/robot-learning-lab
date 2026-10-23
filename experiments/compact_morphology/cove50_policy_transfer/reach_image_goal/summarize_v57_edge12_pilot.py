"""Expose the source-matched V57 new-entry pilot beside its V50 failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v57/edge12-trained-arm-pilot"
PLAN = OUT / "plan.json"
PILOT = OUT / "pilot.json"
PRIOR = BASE / "v50/edge-neighbor-screen/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    pilot = json.loads(PILOT.read_text())
    prior = json.loads(PRIOR.read_text())
    old = next(row for row in prior["cases"] if row["id"] == "edge-12")
    if (pilot["plan_sha256"] != sha(PLAN) or plan["v50_matrix_sha256"] != sha(PRIOR) or
            pilot["seed"] != old["seed"] or not (ROOT / pilot["result"]).is_file() or
            not (ROOT / pilot["trace"]).is_file()):
        raise ValueError("V57 edge-12 pilot evidence differs from source-matched plan")
    matrix = {
        "schema": "robot-learning-lab.cove50-v57-edge12-pilot-matrix.v1",
        "role": "opened_new_entry_actor_diagnostic_not_native_qualification",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "pilot_sha256": sha(PILOT),
        "v50_matrix_sha256": sha(PRIOR),
        "complete": True, "case_count": 1, "pass_count": int(pilot["pass"]),
        "cases": [{"id": "edge-12", "seed": pilot["seed"],
                   "baseline_v50_contact_s": old["contact_s"],
                   "baseline_v50_pass": old["pass"],
                   **{key: pilot[key] for key in (
                       "selection_mode", "contact_s", "entry_orange_pixels",
                       "entry_goal_error_mm_scoring_only", "edge_goal_bias_applied",
                       "pass", "faults", "result", "trace")}}],
    }
    (OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    print(json.dumps({"pilot_pass": pilot["pass"],
                      "baseline_contact_s": old["contact_s"],
                      "trained_contact_s": pilot["contact_s"]}))


if __name__ == "__main__":
    main()
