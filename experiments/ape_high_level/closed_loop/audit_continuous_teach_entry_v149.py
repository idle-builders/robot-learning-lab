"""V149: fresh owner teach and no-motion room-entry camera sweep."""

from __future__ import annotations

import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw

from fast_range_guard import FastRangeGuard
from run import ROOT,Room,Vision,iou,sensed_range,sha
from run_moved_props_v134 import MovedPropRoom

SOURCE=Path(__file__)
OUT=ROOT / "artifacts/ape-closed-loop-cove50-v149"
SCENES=ROOT / "artifacts/ape-closed-loop-cove50-v133/paired-scene-validity.json"
ANGLES=(-1.2,-.6,0.,.6,1.2)
CAMERAS=("lower_rgb","upper_rgb")
SIMILARITY=.35


def teach(room:Room,vision:Vision) -> tuple[np.ndarray,dict]:
    room.reset();room.teach()
    rgb,_=room.capture("upper_rgb",depth=False)
    truth=room.truth_box("TomatoSauce","upper_rgb")
    proposals=vision.detect(rgb)
    ranked=sorted(((iou(p["box"],truth),p) for p in proposals),
                  key=lambda x:x[0],reverse=True)
    if not ranked or ranked[0][0]<.5:
        raise AssertionError("scripted owner cannot confirm teaching can proposal")
    box=ranked[0][1]["box"]
    exemplar=vision.feature([rgb.crop(box)])[0]
    OUT.mkdir(parents=True,exist_ok=True)
    rgb.save(OUT / "owner-teach-upper-rgb.png")
    np.save(OUT / "owner-taught-exemplar.npy",exemplar)
    return exemplar,{"owner_confirmed_box":box,"owner_overlap_evaluator_only":round(ranked[0][0],4),
                     "teaching_rgb_sha256":sha(OUT / "owner-teach-upper-rgb.png"),
                     "exemplar_sha256":sha(OUT / "owner-taught-exemplar.npy")}


def setup(room:MovedPropRoom,seed:int,absent:bool) -> None:
    room.reset();room.layout(seed,absent=absent,furnished=True)
    room.model.body_pos[room.model.body("loop_table").id]=(
        .40,(1 if seed%2 else -1)*.46,0.)
    room.set_table_height(.37)
    room._forward()


def rank(vision:Vision,exemplar:np.ndarray,rgb:Image.Image,
         truth:list[int]|None) -> dict:
    proposals=vision.detect(rgb)
    if not proposals:
        return {"proposal_count":0,"top_box":None,"top_similarity":None,
                "top_correct_evaluator_only":False,"correct_visible_evaluator_only":False,
                "best_correct_similarity_evaluator_only":None}
    scores=vision.feature([rgb.crop(p["box"]) for p in proposals])@exemplar
    index=int(np.argmax(scores))
    correct=[float(s) for p,s in zip(proposals,scores)
             if truth is not None and iou(p["box"],truth)>=.25]
    return {"proposal_count":len(proposals),"top_box":proposals[index]["box"],
            "top_similarity":round(float(scores[index]),4),
            "top_accepted_appearance_only":bool(scores[index]>=SIMILARITY),
            "top_correct_evaluator_only":bool(truth is not None and
                                               iou(proposals[index]["box"],truth)>=.25),
            "correct_visible_evaluator_only":bool(correct),
            "best_correct_similarity_evaluator_only":None if not correct else
                round(max(correct),4)}


def sweep(seed:int,variant:dict,absent:bool,
          vision:Vision,exemplar:np.ndarray) -> dict:
    positions={"CreamCheese":tuple(variant["cream_cheese_xy_m"]),
               "Parmesan":tuple(variant["parmesan_xy_m"])}
    room=MovedPropRoom(positions)
    setup(room,seed,absent)
    branch=OUT / f"seed{seed}-{'absent' if absent else 'present'}"
    branch.mkdir(parents=True,exist_ok=True)
    guard4=FastRangeGuard(band_m=.05)
    guard11=FastRangeGuard(band_m=.05)
    serial=0
    def sense(height:float):
        nonlocal serial
        serial+=1
        return sensed_range(room,seed,1490000+serial,height_m=height)
    guard4.update(sense(.04),room.estimated,room.time_s)
    guard11.update(sense(.11),room.estimated,room.time_s)
    def probe(substep:int,fraction:float):
        upper=guard11.update(sense(.11),room.estimated,room.time_s)
        lower=guard4.update(sense(.04),room.estimated,room.time_s)
        if upper["status"]=="near_body_change" or lower["status"]=="near_body_change":
            return {"upper":upper,"lower":lower,"fraction":fraction}
        return None
    rows=[]
    try:
        with imageio.get_writer(branch / "entry-sweep.mp4",fps=3,codec="libx264",quality=7) as video:
            for angle in ANGLES:
                room.look(angle,"lower_rgb",safety_probe=probe)
                alarm=room.safety_stop_observation
                for camera in CAMERAS:
                    rgb,depth=room.capture(camera,depth=True)
                    truth=room.truth_box("TomatoSauce",camera) if not absent else None
                    ranked=rank(vision,exemplar,rgb,truth)
                    filename=branch / f"{camera}-angle-{angle:+.1f}.png"
                    rgb.save(filename)
                    if camera=="lower_rgb":
                        np.savez_compressed(branch / f"lower-angle-{angle:+.1f}.npz",
                                            depth_m=depth,
                                            camera_origin_m=room.camera_pose_estimate(camera)[0],
                                            camera_rotation=room.camera_pose_estimate(camera)[1],
                                            actor_qpos=room.pose_data.qpos.copy(),
                                            estimated_pose_xyyaw=room.estimated.copy(),
                                            capture_time_s=room.time_s)
                    panel=Image.fromarray(np.concatenate((np.asarray(rgb),
                                      np.asarray(room.capture_external())),axis=1))
                    draw=ImageDraw.Draw(panel)
                    draw.rectangle((0,0,panel.width,26),fill=(0,0,0))
                    draw.text((5,5),f"seed{seed} {'absent' if absent else 'present'} {camera} yaw {angle:+.1f} score {ranked['top_similarity']}",fill=(255,255,255))
                    video.append_data(np.asarray(panel))
                    rows.append({"camera":camera,"commanded_head_yaw_rad":angle,
                                 "actual_head_yaw_rad":round(room.head,4),
                                 "time_s":round(room.time_s,3),
                                 "look_range_alarm":alarm is not None,
                                 "rgb_sha256":sha(filename),
                                 "evaluator_truth_box":truth,
                                 **ranked})
                if alarm is not None:break
    finally:
        room.close()
    return {"seed":seed,"variant":variant["variant"],"absent":absent,
            "rows":rows,"video_sha256":sha(branch / "entry-sweep.mp4")}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    vision=Vision("generic")
    teacher=Room()
    try:exemplar,teaching=teach(teacher,vision)
    finally:teacher.close()
    scenes=json.loads(SCENES.read_text())
    rows=[]
    for seed in (85,88):
        variant=next(v for row in scenes["rows"] if row["seed"]==seed
                     for v in row["passable_variants"] if v["variant"].endswith("-1"))
        rows.extend(sweep(seed,variant,absent,vision,exemplar)
                    for absent in (False,True))
    result={"scope":"V149 fresh owner-confirmed upper RGB can exemplar then guarded no-base-motion lower RGB-D/upper RGB room-entry sweeps in seed85/88 variant1 present/absent furnished rooms. Appearance ranking actor-only; truth boxes evaluator labels only; no target coordinate passed.",
            "source_sha256":sha(SOURCE),"scene_audit_sha256":sha(SCENES),
            "teaching":teaching,"similarity_threshold":SIMILARITY,
            "rows":rows}
    (OUT / "continuous-teach-entry-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    (OUT / SOURCE.name).write_bytes(SOURCE.read_bytes())
    for row in rows:
        accepted=[r for r in row["rows"] if r.get("top_accepted_appearance_only")]
        correct=[r for r in accepted if r.get("top_correct_evaluator_only")]
        print(row["seed"],"absent",row["absent"],"accepted",len(accepted),
              "correct",len(correct),"looks",len(row["rows"]),flush=True)


if __name__=="__main__":main()
