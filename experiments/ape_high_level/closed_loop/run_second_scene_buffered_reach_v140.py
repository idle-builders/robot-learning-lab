"""V140: unchanged V138 buffered actor approach transferred to seed85."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as approach
import run_validated_target_route_v136 as previous
from run import ROOT, sha
from run_moved_props_v134 import MovedPropRoom

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v140"
V139 = ROOT / "artifacts/ape-closed-loop-cove50-v139/second-scene-goal-audit.json"
SEED = 85
NAME = "seed85-passable-1"
ACTOR_STOP_LOWER_M = .27
ACTOR_STOP_UPPER_M = .34
PHYSICAL_STOP_UPPER_M = .40


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    original_out = previous.OUT
    original_upper = approach.MAX_TARGET_GAP_M
    previous.OUT = OUT
    approach.MAX_TARGET_GAP_M = ACTOR_STOP_UPPER_M
    try:
        previous.main(seed=SEED, name=NAME, selection_path=V139)
    finally:
        previous.OUT = original_out
        approach.MAX_TARGET_GAP_M = original_upper

    prior = json.loads((OUT / "validated-target-route-audit.json").read_text())
    live = prior["live"]
    room = MovedPropRoom({"CreamCheese": (.4, -.72),
                          "Parmesan": (.4, +.6)})
    try:
        room.reset()
        room.layout(SEED, absent=False, furnished=True)
        room.model.body_pos[room.model.body("loop_table").id] = (.40, +.46, 0.)
        room.set_table_height(.37)
        true_target = np.asarray(room.positions["TomatoSauce"][:2], dtype=float)
    finally:
        room.close()
    actual = np.asarray(live["evaluator_final_actual_xyyaw"][:2])
    actual_gap = float(np.linalg.norm(actual - true_target))
    gate = (live["status"] == "target_area_reached" and
            ACTOR_STOP_LOWER_M <= live["final_actor_target_gap_m"] <= ACTOR_STOP_UPPER_M and
            ACTOR_STOP_LOWER_M <= actual_gap <= PHYSICAL_STOP_UPPER_M and
            live["final_fresh_target_accepted"] and
            live["wrong_object_acceptances"] == 0 and
            live["evaluator_min_model_clearance_m"] >= .05 and
            not live["evaluator_model_overlap"] and
            not prior["matched_absent_accepted"])
    result = {
        "scope": "Seed85 variant1 kinematic development transfer of V138's fixed 27–34cm actor stop and V136 receding controller. Evaluator target/actual pose scored after motion only.",
        "source_sha256": sha(SOURCE),
        "v136_runner_source_sha256": sha(previous.SOURCE),
        "v111_runner_source_sha256": sha(approach.SOURCE),
        "v139_screen_sha256": sha(V139),
        "actor_stop_band_m": [ACTOR_STOP_LOWER_M, ACTOR_STOP_UPPER_M],
        "physical_scoring_band_m": [ACTOR_STOP_LOWER_M, PHYSICAL_STOP_UPPER_M],
        "evaluator_true_target_xy_m": true_target.tolist(),
        "evaluator_final_actual_xy_m": actual.tolist(),
        "evaluator_final_true_gap_m": round(actual_gap, 4),
        "actor_final_gap_m": live["final_actor_target_gap_m"],
        "executed_moves": live["executed_moves"],
        "evaluator_min_model_clearance_m": live["evaluator_min_model_clearance_m"],
        "physical_target_area_gate_pass": gate,
        "v136_replay_audit_sha256": sha(OUT / "validated-target-route-audit.json"),
        "video_sha256": live["video_sha256"],
    }
    (OUT / "second-scene-buffered-reach-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("V140 physical gate", gate, "moves", live["executed_moves"],
          "actor_gap", live["final_actor_target_gap_m"],
          "true_gap", round(actual_gap, 4), flush=True)


if __name__ == "__main__":
    main()
