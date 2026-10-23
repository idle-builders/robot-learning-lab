"""V139: read-only actor target/information goal on both seed85 moved-prop layouts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import audit_moved_goal_reselection_v135 as screen
from run import ROOT, Vision, sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v139"
SEED=85


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scenes=json.loads(screen.SCENES.read_text())
    variants=next(row for row in scenes["rows"] if row["seed"]==SEED)["passable_variants"]
    exemplar=np.load(screen.source.V81 / "owner-taught-exemplar.npy")
    vision=Vision("generic")
    original_out=screen.OUT
    screen.OUT=OUT
    try:
        rows=[screen.screen(variant,vision,exemplar,seed=SEED) for variant in variants]
    finally:
        screen.OUT=original_out
    gate=all(row["fresh_correct_target"] and
             not row["matched_absent_accepted"] and
             not row["look_range_alarm"] and
             row.get("selected") is not None and
             row["selected"]["first_edge_model_min_m"]>=.05 and
             row["selected"]["endpoint_model_clearance_m"]>=.05
             for row in rows)
    result={"scope":"V139 second-room seed85 actor-only guarded goal selection at exact V134 moved-prop endpoints; matched absent and evaluator selected-edge/endpoint scored afterward. Kinematic, read-only.",
            "source_sha256":sha(SOURCE),
            "screen_runner_source_sha256":sha(screen.SOURCE),
            "scene_audit_sha256":sha(screen.SCENES),
            "rows":rows,"paired_second_scene_goal_screen_pass":gate}
    (OUT / "second-scene-goal-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    (OUT / "audit_moved_goal_reselection_v135_imported_source.py").write_bytes(screen.SOURCE.read_bytes())
    for row in rows:
        print(row["variant"],"fresh",row["fresh_correct_target"],
              "annulus",row.get("annulus",{}).get("found"),
              "selected",row.get("selected",{}).get("goal_cell"),flush=True)
    print("V139 paired second-room gate",gate,flush=True)


if __name__=="__main__":
    main()
