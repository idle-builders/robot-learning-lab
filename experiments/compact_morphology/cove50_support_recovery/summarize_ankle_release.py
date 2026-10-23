"""Compare paired ankle-release native replays with the exact 0.33-rad baseline."""

from __future__ import annotations

import json
from pathlib import Path

from summarize_phase_feedback import REPO, sha, summarize


ROOT = REPO / "artifacts/ape-cove50-ankle-release-20261010"


def case(path: Path, label: str) -> dict:
    raw = json.loads(path.read_text())
    row = summarize(path, label)
    row["shift_roll_rad"] = raw["shift_roll_rad"]
    row["swing_hip_pitch_rad"] = raw["swing_joint_targets_rad"]["hip_pitch"]
    row["swing_ankle_pitch_rad"] = raw["swing_joint_targets_rad"]["ankle_pitch"]
    return row


def main() -> None:
    plan_path = ROOT / "plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["selected_design_pointer_sha256"] == sha(REPO / "robot/ape-current-design.json")
    assert plan["selected_body_sha256"] == sha(REPO / "artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml")
    assert plan["actuator_catalog_sha256"] == sha(REPO / "robot/actuator-catalog.yaml")
    assert plan["probe_source_sha256"] == sha(Path(__file__).with_name("probe_native_feedback.py"))
    item = plan["static_screen"]
    assert sha(REPO / item["path"]) == item["sha256"]
    cases = []
    for side in ("right", "left"):
        item = plan["baseline_reports"][side]
        path = REPO / item["path"]
        assert sha(path) == item["sha256"]
        cases.append(case(path, "ankle0"))
    trials = sorted((ROOT / "trials").glob("*.json"))
    assert len(trials) in (4, 6, 8), len(trials)
    for path in trials:
        cases.append(case(path, path.stem.split("-")[0]))
    report = {
        "schema": "robot-learning-lab.cove50-ankle-release-comparison.v1",
        "method": "Frozen authored selected-body swing-ankle target comparison from upright native MJLab motors. Static sole-corner screen is only a hypothesis. No learned policy.",
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
