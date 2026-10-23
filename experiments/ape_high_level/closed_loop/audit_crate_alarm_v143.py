"""V143: deterministic V142 seed85 replay with move alarm preserved in trace."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as approach
import run_validated_target_route_v136 as previous
from run import ROOT, sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v143"
V139=ROOT / "artifacts/ape-closed-loop-cove50-v139/second-scene-goal-audit.json"
V142=ROOT / "artifacts/ape-closed-loop-cove50-v142/scene85/seed85/trace.json"


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    original_out=previous.OUT
    original_upper=approach.MAX_TARGET_GAP_M
    previous.OUT=OUT
    approach.MAX_TARGET_GAP_M=.34
    try:
        previous.main(seed=85,name="seed85-passable-1",
                      selection_path=V139,motion_supported_scan=True)
    finally:
        previous.OUT=original_out
        approach.MAX_TARGET_GAP_M=original_upper
    trace=json.loads((OUT / "seed85/trace.json").read_text())
    old=json.loads(V142.read_text())
    new_frames=[e for e in trace["events"] if "actor_input_sha256" in e]
    old_frames=[e for e in old["events"] if "actor_input_sha256" in e]
    equivalent=(trace["status"]==old["status"] and
                trace["executed_moves"]==old["executed_moves"] and
                np.allclose(trace["evaluator_final_actual_xyyaw"],
                            old["evaluator_final_actual_xyyaw"],atol=1e-7) and
                [e["action"] for e in new_frames]==[e["action"] for e in old_frames])
    alarms=[{"frame":i,"actor_target_gap_m":e["actor_target_gap_m"],
             "evaluator_clearance_m":e["evaluator_model_clearance_m"],
             "move_alarm":e["detail"].get("move_range_alarm"),
             "look_alarm":e["detail"].get("look_range_alarm")}
            for i,e in enumerate(new_frames) if
            e["detail"].get("move_range_alarm") is not None or
            e["detail"].get("look_range_alarm") is not None]
    result={"scope":"V143 trace-only repeat of V142 seed85 with captured move alarm before guarded head look, no changed actor decision.",
            "source_sha256":sha(SOURCE),
            "v111_runner_source_sha256":sha(approach.SOURCE),
            "v136_runner_source_sha256":sha(previous.SOURCE),
            "old_trace_sha256":sha(V142),
            "behavior_equivalent":equivalent,
            "status":trace["status"],"executed_moves":trace["executed_moves"],
            "alarms":alarms,
            "new_trace_sha256":sha(OUT / "seed85/trace.json")}
    (OUT / "crate-alarm-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("V143 equivalent",equivalent,"alarms",alarms,flush=True)


if __name__=="__main__":
    main()
