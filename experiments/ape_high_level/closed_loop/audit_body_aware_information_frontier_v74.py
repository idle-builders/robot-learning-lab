"""V74: body-aware true frontiers without mandatory target-field descent."""

from __future__ import annotations

import json
from pathlib import Path

import goal_directed_frontier

from audit_body_aware_frontier_v72 import replay
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v74"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prior = goal_directed_frontier.MIN_FIELD_PROGRESS_M
    try:
        # The only intervention: allow an information frontier even when its
        # first leg temporarily raises cost-to-target. All other candidate
        # and scoring rules remain exactly the V72 source.
        goal_directed_frontier.MIN_FIELD_PROGRESS_M = -float("inf")
        rows = [replay(seed) for seed in (85, 88)]
    finally:
        goal_directed_frontier.MIN_FIELD_PROGRESS_M = prior
    gate = all(row["body_goal_cell"] is not None for row in rows)
    output = {"scope": "Exact V72 saved actor map/selected-body inflation and known-free reachable costs; remove only positive target-field-progress requirement for true adjacent-cell frontiers. Read-only, evaluator truth never used for selection.",
              "source_sha256": sha(SOURCE),
              "replay_code_sha256": sha(Path(__file__).with_name("audit_body_aware_frontier_v72.py")),
              "selected_design_sha256": sha(ROOT / "robot/ape-current-design.json"),
              "rows": rows, "two_scene_frontier_gate_pass": gate}
    (OUT / "information-frontier-audit.json").write_text(
        json.dumps(output, indent=2) + "\n")
    (OUT / "audit_body_aware_information_frontier_v74_source.py").write_bytes(
        SOURCE.read_bytes())
    for row in rows:
        print(row["seed"], row["body_field"], row["body_goal_cell"])
    print("two-scene frontier gate:", gate)


if __name__ == "__main__":
    main()
