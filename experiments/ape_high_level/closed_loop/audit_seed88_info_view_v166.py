"""V166: actor view options at seed88's safe V153 information endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import audit_recovery_views_v156 as prior
from run import ROOT, sha


SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v166"
SCENES = ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
V153 = ROOT / "artifacts/ape-closed-loop-cove50-v153"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(SCENES.read_text())
    variant = next(v for row in scenes["rows"] if row["seed"] == 88
                   for v in row["passable_variants"]
                   if v["variant"].endswith("-2"))
    prior.V155 = V153
    prior.OUT = OUT
    row = prior.audit(88, variant)
    result = {"scope": "V166 read-only actor high-depth side cue and 9.5cm point-clear alternate information routes from V153 seed88 origin-information endpoint, before early target-route promotion. Evaluator full-scene path clearance and endpoint target pixels afterward only. No live action.",
              "source_sha256": sha(SOURCE),
              "v153_trace_sha256": sha(V153 / "scene88/seed88/trace.json"),
              "row": row}
    (OUT / "seed88-next-view-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print(row["actor_side_cue"],
          [(candidate["side"], candidate["actor_route"]["goal_cell"],
            None if candidate["evaluator_route"] is None else
            candidate["evaluator_route"]["min_full_scene_m"])
           for candidate in row["candidate_sides"]], flush=True)


if __name__ == "__main__":
    main()
