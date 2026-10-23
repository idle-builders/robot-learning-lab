"""Compare frozen intermediate-load release probes with both endpoint baselines."""

from __future__ import annotations

import json
from pathlib import Path

from summarize_phase_feedback import (
    REPO,
    sha,
    summarize,
)


ROOT = REPO / "artifacts/ape-cove50-load-crossover-20261010"


def main() -> None:
    plan_path = ROOT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["selected_design_pointer_sha256"] == sha(REPO / "robot/ape-current-design.json")
    assert plan["selected_body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    assert plan["actuator_catalog_sha256"] == sha(REPO / "robot/actuator-catalog.yaml")
    assert plan["probe_source_sha256"] == sha(Path(__file__).with_name("probe_native_feedback.py"))
    cases = []
    for shift in ("0.30", "0.36"):
        for side in ("right", "left"):
            item = plan["baseline_reports"][f"{shift}-{side}"]
            path = REPO / item["path"]
            assert sha(path) == item["sha256"]
            row = summarize(path, f"baseline-{shift}")
            row["shift_roll_rad"] = float(shift)
            cases.append(row)
    trials = sorted((ROOT / "trials").glob("*.json"))
    assert len(trials) in (4, 6, 8), len(trials)
    for path in trials:
        row = summarize(path, path.stem.split("-")[0])
        row["shift_roll_rad"] = json.loads(path.read_text())["shift_roll_rad"]
        cases.append(row)
    report = {
        "schema": "robot-learning-lab.cove50-load-crossover-comparison.v1",
        "method": "Frozen authored selected-motor upright foot-release diagnostics, same swing path and nominal seed; baselines at 0.30 and 0.36 rad. Not a learned policy.",
        "selected_design_pointer_sha256": plan["selected_design_pointer_sha256"],
        "selected_body_sha256": plan["selected_body_sha256"],
        "actuator_catalog_sha256": plan["actuator_catalog_sha256"],
        "plan": {"path": str(plan_path.relative_to(REPO)), "sha256": sha(plan_path)},
        "cases": cases,
    }
    out = ROOT / "comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(out), "cases": len(cases),
                      "passes": [x["case"] for x in cases if x["full_gate_pass"]]}, indent=2))


if __name__ == "__main__":
    main()
