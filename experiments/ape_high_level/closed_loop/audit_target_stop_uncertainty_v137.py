"""V137: evaluator-only outcome audit of V136 RGB-D target-range bias."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run import ROOT,Room,locate_world,sha

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v137"
TRACE=ROOT / "artifacts/ape-closed-loop-cove50-v136/seed88/trace.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    trace=json.loads(TRACE.read_text())
    room=Room()
    try:
        room.layout(88,absent=False,furnished=True)
        truth=np.asarray(room.positions["TomatoSauce"])
    finally:
        room.close()
    frames=[event for event in trace["events"] if "actor_input_sha256" in event]
    rows=[]
    for index,event in enumerate(frames,start=1):
        frame=TRACE.parent / f"frame-{index}.npz"
        with np.load(frame) as z:
            camera=(z["camera_origin_m"].copy(),z["camera_rotation"].copy())
            depth=z["depth_m"].copy()
        box=event["accepted_box"]
        if box is None:
            continue
        center=locate_world(box,camera,depth)
        floor=locate_world(box,camera,None)
        pose=np.asarray(event["estimated_pose_xyyaw"])
        actual=np.asarray(event["evaluator_actual_pose_xyyaw"])
        chosen=np.asarray(event["actor_target_xy_m"])
        actual_gap=float(np.linalg.norm(actual[:2]-truth))
        chosen_gap=float(np.linalg.norm(pose[:2]-chosen))
        floor_gap=(None if floor is None else
                   float(np.linalg.norm(pose[:2]-np.asarray(floor[:2]))))
        rows.append({"frame":index,"action":event["action"],
                     "actor_input_sha256":sha(frame),
                     "accepted_similarity":event["accepted_similarity"],
                     "accepted_target_xy_m":chosen.tolist(),
                     "target_location_error_m":round(float(np.linalg.norm(chosen-truth)),4),
                     "pose_error_m":round(float(np.linalg.norm(pose[:2]-actual[:2])),4),
                     "actor_target_gap_m":round(chosen_gap,4),
                     "evaluator_true_target_gap_m":round(actual_gap,4),
                     "gap_underestimate_m":round(actual_gap-chosen_gap,4),
                     "rgb_box_bottom_floor_xy_m":None if floor is None else floor[:2],
                     "rgb_floor_target_error_m":None if floor is None else round(float(
                         np.linalg.norm(np.asarray(floor[:2])-truth)),4),
                     "rgb_floor_actor_gap_m":None if floor_gap is None else round(floor_gap,4),
                     "depth_center_recomputed_xy_m":None if center is None else center[:2]})
    final=rows[-1]
    result={"scope":"Evaluator-only V136 outcome audit over all accepted actor RGB-D frames. Actor box/depth location and estimated gap versus simulator target XY/actual body gap; box-bottom floor-ray alternative uses calibrated actor camera pose. No actor action changed; one correlated development episode.",
            "source_sha256":sha(SOURCE),"trace_sha256":sha(TRACE),
            "scene_audit_sha256":sha(SCENES),
            "evaluator_true_target_xy_m":truth.tolist(),
            "rows":rows,
            "summary":{"accepted_frames":len(rows),
                       "max_target_location_error_m":max(r["target_location_error_m"] for r in rows),
                       "max_pose_error_m":max(r["pose_error_m"] for r in rows),
                       "max_gap_underestimate_m":max(r["gap_underestimate_m"] for r in rows),
                       "final_actor_gap_m":final["actor_target_gap_m"],
                       "final_true_gap_m":final["evaluator_true_target_gap_m"],
                       "final_gap_underestimate_m":final["gap_underestimate_m"],
                       "final_floor_gap_m":final["rgb_floor_actor_gap_m"]}}
    (OUT / "target-stop-uncertainty.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / "audit_target_stop_uncertainty_v137_source.py").write_bytes(SOURCE.read_bytes())
    print("summary",result["summary"],flush=True)


if __name__=="__main__":
    main()
