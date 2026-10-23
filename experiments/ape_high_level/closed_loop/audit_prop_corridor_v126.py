"""V126: causal small-prop ablation on historical 6.5 cm route geometry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_fresh_recovery_approach_v110 import accepted_source
from run import ROOT, Room, sha

SOURCE = Path(__file__)
OUT = ROOT / "artifacts/ape-closed-loop-cove50-v126"
AUDIT = ROOT / "artifacts/ape-closed-loop-cove50-v115/exact-margin-feasibility-audit.json"
ARMS = ((), ("CreamCheese",), ("Parmesan",), ("CreamCheese", "Parmesan"))


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    legacy = json.loads(AUDIT.read_text())
    room = Room()
    rows = []
    try:
        for seed in (85,88):
            _,actual,paths,_,_ = accepted_source(seed)
            with np.load(paths[-1]) as z:
                qpos = z["actor_qpos"].copy()
            old = next(row for row in legacy["rows"] if row["seed"] == seed)
            path = [np.asarray(p) for p in old["path_xy_m"]]
            for removed in ARMS:
                room.reset()
                room.layout(seed,absent=False,furnished=True)
                room.model.body_pos[room.model.body("loop_table").id] = (
                    .40,(1 if seed%2 else -1)*.46,0.)
                room.set_table_height(.37)
                for name in removed:
                    room.model.body_pos[room.model.body(f"loop_{name}").id] = (10.,0.,.052)
                room.data.qpos[:] = qpos
                room.actual[:] = actual
                room.head = float(qpos[room.head_addr])
                samples = []
                for edge,(a,b) in enumerate(zip(path,path[1:])):
                    for fraction in np.linspace(0.,1.,21):
                        room.actual[:2] = a+fraction*(b-a)
                        room._forward()
                        dist,robot,geom = room._current_obstacle_clearance()
                        samples.append((float(dist),robot,geom,edge,float(fraction)))
                minimum = min(samples,key=lambda row:row[0])
                first = next((s for s in samples if s[0]<.065),None)
                row = {"seed":seed,"removed_props":list(removed),
                       "route_min_m":round(minimum[0],4),
                       "limiting_robot_geom":minimum[1],
                       "limiting_scene_geom":minimum[2],
                       "limiting_edge":minimum[3],
                       "first_margin_violation":None if first is None else {
                           "distance_m":round(first[0],4),
                           "scene_geom":first[2],"edge":first[3],
                           "fraction":round(first[4],2)},
                       "passes_65cm":minimum[0]>=.065}
                rows.append(row)
                print(seed,removed,row["route_min_m"],row["limiting_scene_geom"],flush=True)
    finally:
        room.close()
    result = {"scope":"Evaluator-only one/two-prop ablation on fixed legacy V115 XY paths with 21 samples per edge, same selected joints/yaw and target/furniture; no actor authority or physical prop motion.",
              "source_sha256":sha(SOURCE),"run_source_sha256":sha(ROOT / "experiments/ape_high_level/closed_loop/run.py"),
              "legacy_audit_sha256":sha(AUDIT),"rows":rows}
    (OUT / "prop-corridor-ablation.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_prop_corridor_v126_source.py").write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
