"""V127: replay unchanged V121 actor under corrected full-scene scoring."""

from __future__ import annotations

import json
from pathlib import Path

import run_live_committed_course_v121 as original
from run import ROOT, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v127"
ORIGINAL = ROOT / "artifacts/ape-closed-loop-cove50-v121/live-committed-course-audit.json"


def main() -> None:
    original.OUT = OUT
    original.main()
    current = json.loads((OUT / "live-committed-course-audit.json").read_text())
    historical = json.loads(ORIGINAL.read_text())
    rows = []
    for new in current["rows"]:
        old = next(row for row in historical["rows"] if row["seed"] == new["seed"])
        rows.append({"seed": new["seed"],
                     "historical_status": old["base"]["status"],
                     "corrected_status": new["base"]["status"],
                     "historical_min_large_furniture_m": old["base"]["evaluator_min_path_clearance_m"],
                     "corrected_min_full_scene_m": new["base"]["evaluator_min_path_clearance_m"],
                     "historical_overlap": old["base"]["evaluator_model_overlap"],
                     "corrected_overlap": new["base"]["evaluator_model_overlap"],
                     "historical_moves": old["base"]["executed_moves"],
                     "corrected_moves": new["base"]["executed_moves"]})
    result = {"scope":"Exact V121 actor/goal/source-state replay with corrected all-solid-scene evaluator only; original actor, sensor thresholds and action code unchanged.",
              "source_sha256":sha(SOURCE),
              "original_actor_source_sha256":sha(original.SOURCE),
              "corrected_run_source_sha256":sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
              "historical_v121_audit_sha256":sha(ORIGINAL),
              "current_audit_sha256":sha(OUT / "live-committed-course-audit.json"),
              "rows":rows,
              "paired_live_gate":current["paired_live_committed_course_gate_pass"]}
    (OUT / "corrected-course-comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "replay_corrected_course_v127_source.py").write_bytes(SOURCE.read_bytes())
    print("corrected comparison",rows,flush=True)


if __name__ == "__main__":
    main()
