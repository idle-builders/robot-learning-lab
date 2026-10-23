"""Pin the opened scanned/static edge-view comparison and source hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "artifacts/ape-cove50-reach-goal-channels-20261010"
OUT = BASE / "v49"
PLAN = OUT / "opened-edge-pair-plan.json"
MATRIX = OUT / "opened-edge-pair/matrix.json"
SCANNED = BASE / "v45/scan-branch-probe-c/matrix.json"
STATIC = BASE / "v45/scan-timing-control/matrix.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan = json.loads(PLAN.read_text())
    matrix = json.loads(MATRIX.read_text())
    scan = json.loads(SCANNED.read_text())["cases"][0]
    static = json.loads(STATIC.read_text())["cases"][0]
    if (not matrix["complete"] or matrix["case_count"] != 2 or
            matrix["plan_sha256"] != sha(PLAN) or
            [r["id"] for r in matrix["cases"]] != ["scanned", "static-alternate-gaze"] or
            scan["contact_s"] != .78 or static["contact_s"] != .5):
        raise ValueError("Complete matched V49 pair required")
    report = {
        "schema": "robot-learning-lab.cove50-v49-edge-pair-summary.v1",
        "role": "opened_observable_rule_diagnostic_not_fresh_score",
        "body_role": "rejected_historical_direct_frame_sandbox",
        "plan_sha256": sha(PLAN), "matrix_sha256": sha(MATRIX),
        "scanned_v45_matrix_sha256": sha(SCANNED),
        "static_v45_matrix_sha256": sha(STATIC),
        "source_sha256": plan["source_sha256"],
        "case_count": 2, "pass_count": matrix["pass_count"],
        "pairs": [
            {"id": "scanned", "baseline_contact_s": scan["contact_s"],
             "changed_contact_s": matrix["cases"][0]["contact_s"],
             "baseline_pixels": scan["entry_orange_pixels"],
             "changed_pixels": matrix["cases"][0]["entry_orange_pixels"]},
            {"id": "static-alternate-gaze", "baseline_contact_s": static["contact_s"],
             "changed_contact_s": matrix["cases"][1]["contact_s"],
             "baseline_pixels": static["entry_orange_pixels"],
             "changed_pixels": matrix["cases"][1]["entry_orange_pixels"]},
        ],
        "qualification": "Same opened seed paired diagnosis only. Neighbor variance, route training, full retention and disjoint score are required before promotion.",
    }
    (OUT / "opened-edge-pair-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"matrix_sha256": report["matrix_sha256"],
                      "pass_count": report["pass_count"]}))


if __name__ == "__main__":
    main()
