"""V131: paired live low-head-depth course planning against corrected V127."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_live_committed_course_v121 as original
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Room,Vision,sha
from run_live_local_bypass_v86 import run
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
RUNNER=SOURCE.with_name("run_live_local_bypass_v86.py")
CLEARANCE=SOURCE.with_name("two_height_clearance.py")
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v131"
BASE=ROOT / "artifacts/ape-closed-loop-cove50-v127/live-committed-course-audit.json"


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    original.OUT=OUT
    goal_audit=json.loads(original.V119.read_text())
    baseline=json.loads(BASE.read_text())
    room=Room()
    absent_room=Room()
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    exemplar=np.load(original.V81 / "owner-taught-exemplar.npy")
    vision=Vision("generic")
    rows=[]
    try:
        robot=RobotGeometry(room.model)
        for selected in (row for row in goal_audit["rows"]
                         if row["coefficient"]==.010):
            seed=int(selected["seed"])
            score=selected["score"]
            initial=original.load_initial(seed,room,mask)
            base=run(seed,{"actor_goal_cell":score["actor_goal_cell"],
                           "blocked_cell":None},room,robot,output_root=OUT,
                     video_label="V131",goal_audit_path=original.V119,
                     initial=initial,self_mask=mask,
                     allow_near_masked_depth=True,max_moves=10,
                     motion_consistency_stop=True,
                     clearance_cost_weight=.010,
                     physical_travel_cap=True,
                     committed_path_cells=score["actor_path_cells"],
                     max_committed_fallback_moves=5,
                     head_low_planning=True)
            views=None
            if base["status"]=="local_goal_reached":
                target=original.accepted_source(seed)[0]
                views=original.endpoint_views(seed,room,absent_room,vision,
                                              exemplar,target)
            rows.append({"seed":seed,
                         "base":{k:v for k,v in base.items() if k!="events"},
                         "views":views})
    finally:
        mask.close()
        room.close()
        absent_room.close()
    comparison=[]
    for row in rows:
        old=next(x for x in baseline["rows"] if x["seed"]==row["seed"])
        comparison.append({"seed":row["seed"],
                           "baseline_moves":old["base"]["executed_moves"],
                           "candidate_moves":row["base"]["executed_moves"],
                           "baseline_status":old["base"]["status"],
                           "candidate_status":row["base"]["status"],
                           "baseline_min_full_scene_m":old["base"]["evaluator_min_path_clearance_m"],
                           "candidate_min_full_scene_m":row["base"]["evaluator_min_path_clearance_m"],
                           "candidate_range_alarm":any(
                               e.get("detail",{}).get("range_alarm") is not None or
                               e.get("detail",{}).get("outcome") in
                               ("move_range_alarm","look_range_alarm")
                               for e in json.loads((OUT / f"seed{row['seed']}-repair/trace.json").read_text())["events"])})
    gate=(all(c["candidate_moves"]>=c["baseline_moves"] and
              c["candidate_min_full_scene_m"]>=.05 and
              not c["candidate_range_alarm"]
              for c in comparison) and
          any(row["seed"]==88 and row["base"]["status"]=="local_goal_reached"
              and row["views"] is not None and
              row["views"]["correct_target_reacquired"] and
              not row["views"]["wrong_acceptance"] and
              not row["views"]["matched_absent_accepted"]
              for row in rows))
    result={"scope":"Live candidate replaces 4cm route/point evidence with fresh 4–16cm self-masked head-depth occupancy while retaining independent 10Hz4/11cm stop; same V119 actor goals, starts, 6.5cm point margin, committed fallback, image identity, corrected all-object evaluator. Paired with V127 baseline; kinematic only.",
            "source_sha256":sha(SOURCE),"runner_source_sha256":sha(RUNNER),
            "clearance_source_sha256":sha(CLEARANCE),
            "baseline_audit_sha256":sha(BASE),
            "rows":rows,"comparison":comparison,
            "candidate_gate_pass":gate}
    (OUT / "live-low-depth-course-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "run_live_low_depth_course_v131_source.py").write_bytes(SOURCE.read_bytes())
    (OUT / "run_live_local_bypass_v86_imported_source.py").write_bytes(RUNNER.read_bytes())
    (OUT / "two_height_clearance_imported_source.py").write_bytes(CLEARANCE.read_bytes())
    print("V131 candidate gate",gate,"comparison",comparison,flush=True)


if __name__=="__main__":
    main()
