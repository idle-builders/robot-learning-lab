"""V261: selected-body clearance through V260's owner-question head look."""

from __future__ import annotations

import json
from pathlib import Path

import audit_terminal_look_clearance_v250 as prior
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v261"
V259_ROOT = ROOT / "artifacts/ape-closed-loop-cove50-v259"
V260 = ROOT / "artifacts/ape-closed-loop-cove50-v260/seed107-owner-clarification-audit.json"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old_root = prior.V247_ROOT
    prior.V247_ROOT = V259_ROOT
    try:
        rows = [prior.score(107, case) for case in
                ("present", "ordinary_absent", "red_lookalike_absent")]
    finally:
        prior.V247_ROOT = old_root
    report = {"scope": "V261 evaluator-only selected-body 21-position head-yaw clearance screen from V259 seed107 final pose through V260's question yaw. Includes present and paired target-absent ordinary/red-material different-mesh swaps. Actor receives no evaluator geometry; selected kinematic model, not native head dynamics or physical stop.",
              "source_sha256": sha(SOURCE), "v260_sha256": sha(V260),
              "rows": rows}
    (OUT / "seed107-owner-look-clearance-audit.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        print(row["case"], row["minimum_sweep_clearance_m"])


if __name__ == "__main__":
    main()
