"""V151: frozen owner-crop DINO+hue gate on new layout/light entry sweeps."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

from audit_entry_color_v150 import descriptor
from fast_range_guard import FastRangeGuard
from run import ROOT,Vision,iou,locate_world,sensed_range,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v151"
V149=ROOT / "artifacts/ape-closed-loop-cove50-v149"
V150=ROOT / "artifacts/ape-closed-loop-cove50-v150/entry-color-audit.json"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
ANGLES=(-1.2,-.6,0.,.6,1.2)
LIGHT_MULTIPLIERS=(.65,1.35)
DINO_MIN=.35
HUE_MIN=.70


def infer(vision:Vision,exemplar:np.ndarray,owner_hist:np.ndarray,
          rgb:Image.Image,depth:np.ndarray,camera_pose:tuple,
          truth:list[int]|None,true_target:np.ndarray|None) -> dict:
    proposals=vision.detect(rgb)
    if not proposals:
        return {"proposals":0,"accepted_box":None,"accepted_score":None,
                "accepted_hue_intersection":None,"accepted_world_xyz_m":None,
                "accepted_correct_evaluator_only":False,
                "accepted_location_error_evaluator_only_m":None}
    scores=vision.feature([rgb.crop(p["box"]) for p in proposals])@exemplar
    candidates=[]
    for proposal,score in zip(proposals,scores):
        color=descriptor(rgb,proposal["box"])
        overlap=float(np.minimum(owner_hist,np.asarray(color["hue_histogram"])).sum())
        if score>=DINO_MIN and overlap>=HUE_MIN:
            candidates.append((float(score),overlap,proposal["box"]))
    selected=max(candidates,key=lambda x:x[0]) if candidates else None
    world=(None if selected is None else
           locate_world(selected[2],camera_pose,depth))
    correct=(selected is not None and truth is not None and
             iou(selected[2],truth)>=.25)
    error=(None if not correct or world is None or true_target is None else
           float(np.linalg.norm(np.asarray(world[:2])-true_target)))
    return {"proposals":len(proposals),
            "accepted_box":None if selected is None else selected[2],
            "accepted_score":None if selected is None else round(selected[0],4),
            "accepted_hue_intersection":None if selected is None else round(selected[1],4),
            "accepted_world_xyz_m":world,
            "accepted_correct_evaluator_only":bool(correct),
            "accepted_location_error_evaluator_only_m":None if error is None else round(error,4)}


def episode(seed:int,variant:dict,absent:bool,light:float,
            vision:Vision,exemplar:np.ndarray,owner_hist:np.ndarray) -> dict:
    room=MovedPropRoom({"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
                        "Parmesan":tuple(variant["parmesan_xy_m"])})
    room.reset();room.layout(seed,absent=absent,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    room.model.light_diffuse[0]*=light
    room._forward()
    true_target=(None if absent else
                 np.asarray(room.positions["TomatoSauce"],dtype=float))
    branch=OUT / f"seed{seed}-light{light:.2f}-{'absent' if absent else 'present'}"
    branch.mkdir(parents=True,exist_ok=True)
    guard4=FastRangeGuard(band_m=.05);guard11=FastRangeGuard(band_m=.05)
    serial=0
    def sense(height:float):
        nonlocal serial
        serial+=1
        return sensed_range(room,seed,1510000+serial,height_m=height)
    guard4.update(sense(.04),room.estimated,room.time_s)
    guard11.update(sense(.11),room.estimated,room.time_s)
    def probe(substep:int,fraction:float):
        high=guard11.update(sense(.11),room.estimated,room.time_s)
        low=guard4.update(sense(.04),room.estimated,room.time_s)
        if high["status"]=="near_body_change" or low["status"]=="near_body_change":
            return {"high":high,"low":low,"fraction":fraction}
        return None
    rows=[]
    try:
        with imageio.get_writer(branch / "entry-sweep.mp4",fps=3,codec="libx264",quality=7) as video:
            for angle in ANGLES:
                room.look(angle,"lower_rgb",safety_probe=probe)
                alarm=room.safety_stop_observation
                rgb,depth=room.capture("lower_rgb",depth=True)
                camera=room.camera_pose_estimate("lower_rgb")
                truth=room.truth_box("TomatoSauce","lower_rgb") if not absent else None
                choice=infer(vision,exemplar,owner_hist,rgb,depth,camera,
                             truth,true_target)
                frame=branch / f"lower-angle-{angle:+.1f}.npz"
                np.savez_compressed(frame,rgb=np.asarray(rgb),depth_m=depth,
                                    camera_origin_m=camera[0],camera_rotation=camera[1],
                                    actor_qpos=room.pose_data.qpos.copy(),
                                    estimated_pose_xyyaw=room.estimated.copy(),
                                    capture_time_s=room.time_s)
                panel=Image.fromarray(np.concatenate((np.asarray(rgb),
                                          np.asarray(room.capture_external())),axis=1))
                draw=ImageDraw.Draw(panel)
                draw.rectangle((0,0,panel.width,28),fill=(0,0,0))
                draw.text((5,6),f"seed{seed} light×{light:.2f} {'absent' if absent else 'present'} yaw{angle:+.1f} accepted {choice['accepted_box'] is not None}",fill=(255,255,255))
                video.append_data(np.asarray(panel))
                rows.append({"yaw_command_rad":angle,"head_yaw_rad":round(room.head,4),
                             "time_s":round(room.time_s,3),"look_range_alarm":alarm,
                             "actor_frame_sha256":sha(frame),
                             "evaluator_truth_box":truth,**choice})
                if alarm is not None:break
    finally:room.close()
    accepted=[r for r in rows if r["accepted_box"] is not None]
    return {"seed":seed,"variant":variant["variant"],"absent":absent,
            "light_multiplier":light,"rows":rows,
            "correct_acceptances":sum(r["accepted_correct_evaluator_only"] for r in accepted),
            "wrong_acceptances":sum(not r["accepted_correct_evaluator_only"] for r in accepted),
            "max_correct_location_error_m":max((r["accepted_location_error_evaluator_only_m"]
                for r in accepted if r["accepted_correct_evaluator_only"] and
                r["accepted_location_error_evaluator_only_m"] is not None),default=None),
            "look_range_alarm":any(r["look_range_alarm"] is not None for r in rows),
            "video_sha256":sha(branch / "entry-sweep.mp4")}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    vision=Vision("generic")
    exemplar=np.load(V149 / "owner-taught-exemplar.npy")
    owner=np.asarray(json.loads(V150.read_text())["owner_color"]["hue_histogram"])
    scenes=json.loads(SCENES.read_text())
    rows=[]
    for seed in (85,88):
        variant=next(v for row in scenes["rows"] if row["seed"]==seed
                     for v in row["passable_variants"] if v["variant"].endswith("-2"))
        for light in LIGHT_MULTIPLIERS:
            for absent in (False,True):
                row=episode(seed,variant,absent,light,vision,exemplar,owner)
                rows.append(row)
                print(seed,light,"absent",absent,"correct",row["correct_acceptances"],
                      "wrong",row["wrong_acceptances"],"error",row["max_correct_location_error_m"],flush=True)
    gate=all((row["correct_acceptances"]>=1 if not row["absent"] else
              row["correct_acceptances"]==0) and
             row["wrong_acceptances"]==0 and
             not row["look_range_alarm"] and
             (row["max_correct_location_error_m"] is None or
              row["max_correct_location_error_m"]<=.12)
             for row in rows)
    result={"scope":"V151 frozen owner-crop DINO>=.35 and saturated-hue intersection>=.70 lower RGB-D identity in variant2 seed85/88 furnished rooms, light multipliers0.65/1.35, paired target present/absent guarded head sweeps. Evaluator truth only scoring; no base motion.",
            "source_sha256":sha(SOURCE),"v149_owner_exemplar_sha256":sha(V149 / "owner-taught-exemplar.npy"),
            "v150_audit_sha256":sha(V150),"scene_audit_sha256":sha(SCENES),
            "dino_min":DINO_MIN,"hue_intersection_min":HUE_MIN,
            "rows":rows,"paired_color_grounded_entry_gate_pass":gate}
    (OUT / "color-grounded-entry-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    print("V151 gate",gate,flush=True)


if __name__=="__main__":main()
