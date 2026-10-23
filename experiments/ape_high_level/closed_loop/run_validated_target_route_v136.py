"""V136: live full-3D actor target route from V135 seed88 variant-1 view."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_renewed_live_approach_v111 as approach
from audit_fresh_recovery_approach_v110 import accepted_source as original_source
from audit_side_look_target_v81 import absent_capture,infer
from robot_geometry import RobotGeometry
from run import H,ROOT,W,Vision,sha
from run_moved_props_v134 import MovedPropRoom
from self_depth_mask import SelfDepthMask

SOURCE=Path(__file__)
RUNNER=SOURCE.with_name("run_renewed_live_approach_v111.py")
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
V134=ROOT / "artifacts/ape-closed-loop-cove50-v134"
V135=ROOT / "artifacts/ape-closed-loop-cove50-v135/moved-goal-reselection-audit.json"
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v136"
SEED=88
NAME="seed88-passable-1"


def main(*,seed: int=SEED,name: str=NAME,
         selection_path: Path=V135,
         motion_supported_scan: bool=False,
         base_point_margin_m: float=.065) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    scene=json.loads(SCENES.read_text())
    variant=next(v for row in scene["rows"] if row["seed"]==seed
                 for v in row["passable_variants"] if v["variant"]==name)
    selection=json.loads(selection_path.read_text())
    selected=next(row for row in selection["rows"] if row["variant"]==name)
    if not (selected["fresh_correct_target"] and
            not selected["matched_absent_accepted"] and
            selected["annulus"]["found"]):
        raise AssertionError("V135 fresh image/full-graph route gate failed")
    moved={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
           "Parmesan":tuple(variant["parmesan_xy_m"])}
    original_target,_,historical_paths,_,_=original_source(seed)
    source_dir=V134 / name / f"seed{seed}-repair"
    prior=json.loads((source_dir / "trace.json").read_text())
    actual=np.asarray(prior["evaluator_final_actual_xyyaw"])
    count=sum("actor_input_sha256" in e for e in prior["events"])
    paths=list(historical_paths)
    paths.append(V134 / name / "fresh-start-actor-input.npz")
    paths.extend(source_dir / f"frame-{i}.npz" for i in range(1,count+1))
    paths.append(selection_path.parent / name / "fresh-look-actor-input.npz")
    for path in paths:
        if not path.exists():raise FileNotFoundError(path)
    target=np.asarray(selected["fresh_actor_target_xy_m"])
    original_function=approach.accepted_source
    original_out=approach.OUT
    def source_override(requested_seed: int):
        if requested_seed!=seed:raise ValueError("unexpected seed")
        return target.copy(),actual.copy(),paths,{"similarity":selected["fresh_similarity"]},selection_path
    approach.accepted_source=source_override
    approach.OUT=OUT
    exemplar_path=ROOT / "artifacts/ape-closed-loop-cove50-v81/owner-taught-exemplar.npy"
    exemplar=np.load(exemplar_path)
    vision=Vision("generic")
    room=MovedPropRoom(moved)
    absent_room=MovedPropRoom(moved)
    mask=SelfDepthMask(room.model,camera="lower_rgb",width=W,height=H)
    try:
        robot=RobotGeometry(room.model)
        live=approach.run(seed,room,robot,mask,vision,exemplar,
                          point_all_graph_edges=True,
                          motion_supported_scan=motion_supported_scan,
                          base_point_margin_m=base_point_margin_m)
        room.camera_pose_estimate("lower_rgb")
        frame={"actor_qpos":room.pose_data.qpos.copy(),
               "estimated_pose_xyyaw":room.estimated.copy(),
               "capture_time_s":room.time_s}
        absent_source=OUT / "final-matched-absent"
        absent_rgb,absent_depth,absent_camera,_=absent_capture(
            absent_room,seed,frame,room.actual.copy(),room.head,
            absent_source)
        absent=infer(vision,exemplar,absent_rgb,absent_depth,absent_camera,
                     np.asarray(live["final_actor_target_xy_m"]),None,
                     OUT / "final-matched-absent-annotated.png")
    finally:
        mask.close()
        room.close()
        absent_room.close()
        approach.accepted_source=original_function
        approach.OUT=original_out
    gate=(live["status"]=="target_area_reached" and
          .27<=live["final_actor_target_gap_m"]<=.40 and
          live["final_fresh_target_accepted"] and
          live["wrong_object_acceptances"]==0 and
          live["evaluator_min_model_clearance_m"]>=.05 and
          not live["evaluator_model_overlap"] and
          absent["accepted_box"] is None)
    result={"scope":f"Live seed{seed} {name} V133 geometrically validated moved-prop target approach from exact V134 stop and fresh image. V111 receding route uses full-graph6.5cm selected-body point checks,4/8/11cm virtual range,independent10Hz4/11cm stop,scan localization,guarded gaze and fixed learned-object identity after every step. Matched final absent render; no evaluator truth route in actor. Kinematic only.",
            "source_sha256":sha(SOURCE),"runner_source_sha256":sha(RUNNER),
            "scene_audit_sha256":sha(SCENES),"goal_screen_sha256":sha(selection_path),
            "historical_actor_input_sha256":[sha(p) for p in paths],
            "selected_design_sha256":sha(ROOT / "robot/ape-current-design.json"),
            "live":{k:v for k,v in live.items() if k!="events"},
            "matched_absent_accepted":absent["accepted_box"] is not None,
            "matched_absent_actor_input_sha256":sha(absent_source.with_suffix(".npz")),
            "target_area_gate_pass":gate}
    (OUT / "validated-target-route-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "run_validated_target_route_v136_source.py").write_bytes(SOURCE.read_bytes())
    (OUT / "run_renewed_live_approach_v111_imported_source.py").write_bytes(RUNNER.read_bytes())
    print("V136 target-area gate",gate,"status",live["status"],
          "moves",live["executed_moves"],"gap",live["final_actor_target_gap_m"],flush=True)


if __name__=="__main__":
    main()
