"""Run independent fast range alongside direct/cooperative final image workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import multiprocessing as mp
import time

import imageio.v2 as imageio
import numpy as np

from real_capture_dynamic_scene import TexturedFloorObject
from run_common_policy_20_object_generalization import ROOT, ART, relative, sha
from run_conditional_search_rear_hazard import ANGLES
from run_real_capture_teach_find import SCENES
import run_post_confirmation_approach_v2 as approach
import run_wait_gated_final_20 as direct
import run_cooperative_return_8 as cooperative
import run_cooperative_represent_probe as presentation
import run_cross_room_inference_response as motion
import run_shared_world_return as shared


PRECOMMIT=ART / "joined_live_hazard_16_precommit.json"
REPORT=ART / "joined_live_hazard_16_report.json"
OUT=ROOT / "assets/real_indoor_scenes/joined_live_hazard_16_20261015"
ARMS=("healthy","range_outage")
OUTAGE=(.25,1.25)
PERIOD_S=.1
HAZARD="Popcorn"
HAZARD_PROP=ROOT / ".cache/ape-high-level/hope_meshes_eval/Popcorn.obj"


def placement_for(case: dict) -> dict:
    doc=json.loads((ROOT / case["screen"]).read_text())
    return next(x for x in doc["scenes"] if x["scene_key"]==case["scene_key"])


def scene_and_props(case: dict):
    placement=placement_for(case)
    raw=approach.base.make_scene(case,placement)
    cfg=SCENES[case["scene_key"]]
    owner=np.asarray(cfg["owner_xz"],float)
    if case["branch"]=="cooperative":
        scene=presentation.PersonVisibleScene(raw,owner)
        lateral=np.asarray(cfg["owner_lateral"],float)
        lateral/=np.linalg.norm(lateral)
        target=owner+presentation.OWNER_PLACE_LATERAL_M*lateral
    else:
        scene=raw
        target=np.asarray(placement["find_target_xz"],float)
    props=ROOT / ".cache/ape-high-level/hope_meshes_eval"
    base={case["target"]:(TexturedFloorObject(props /
            f"{case['target']}.obj",scale=.012),tuple(target),0.),
          case["distractor"]:(TexturedFloorObject(props /
            f"{case['distractor']}.obj",scale=.012),
            tuple(placement["find_distractor_xz"]),0.)}
    scene.set_props(base)
    return scene,base,placement,target


def actor_state(case: dict) -> dict:
    key=case["scene_key"]
    if case["branch"]=="direct":
        action=json.loads((ROOT / case["source_state"]).read_text())[
            "actions"][-1]
        return {"actual":np.asarray(action["evaluator_actual_xz_m"],float),
            "yaw":float(action["evaluator_actual_heading_deg"]),
            "actor":np.asarray(action["actor_pose_world_xz_m"],float),
            "actor_yaw":float(action["actor_heading_deg"]),
            "worker_state":action}
    route=json.loads((ROOT / case["source_state"]).read_text())
    cfg=SCENES[key]
    state={"evaluator_actual_xz_m":route["final_actual_xz_m"],
        "evaluator_actual_heading_deg":route["final_actual_yaw_deg"],
        "actor_pose_xy_m":(np.asarray(route["final_actor_xz_m"])-
            np.asarray(cfg["start_xz"])).tolist(),
        "actor_heading_deg":route["final_actor_yaw_deg"]}
    return {"actual":np.asarray(route["final_actual_xz_m"],float),
        "yaw":float(route["final_actual_yaw_deg"]),
        "actor":np.asarray(route["final_actor_xz_m"],float),
        "actor_yaw":float(route["final_actor_yaw_deg"]),
        "worker_state":state}


def sensor_process(conn,case: dict,origin_xy: list[float],yaw: float,
                   arm: str) -> None:
    try:
        scene,base,_,_=scene_and_props(case)
        hazard_prop=TexturedFloorObject(HAZARD_PROP,scale=.012)
        start=np.asarray(origin_xy,float)
        a=math.radians(yaw+case["entry_relative_deg"])
        incoming=np.array([math.sin(a),math.cos(a)])
        eye=np.array([start[0],scene.floor_y+.42,start[1]])
        baseline=scene.range_fan(eye,yaw,owner_xz=None,angles_deg=ANGLES)
        conn.send({"kind":"ready"})
        message=conn.recv()
        if message["kind"]!="start":
            raise ValueError("Missing common clock")
        origin=float(message["origin"])
        packets=[];tick=0
        alarm_capture=alarm_delivery=alarm_angle=health_lost=None
        escape=None;stop_s=None
        while True:
            target=origin+tick*PERIOD_S
            delay=target-time.perf_counter()
            if delay>0:time.sleep(delay)
            captured=time.perf_counter()-origin
            if captured>30.:
                raise TimeoutError("Slow image worker exceeded bounded trial")
            actor=motion.actor_xy(start,escape,alarm_delivery,captured)
            hazard=motion.incoming_xy(start,incoming,captured)
            scene.set_props({**base,HAZARD:(hazard_prop,tuple(hazard),0.)})
            eye=np.array([actor[0],scene.floor_y+.42,actor[1]])
            missing=arm=="range_outage" and OUTAGE[0]<=captured<OUTAGE[1]
            rays=None if missing else scene.range_fan(eye,yaw,owner_xz=None,
                                                       angles_deg=ANGLES)
            delivered=time.perf_counter()-origin
            if missing and health_lost is None:health_lost=delivered
            changed=np.zeros(len(ANGLES),bool)
            if rays is not None:
                with np.errstate(invalid="ignore"):
                    changed=(baseline[0]-rays[0]>.08)&(rays[0]<.9)
            if alarm_delivery is None and np.any(changed):
                ids=np.flatnonzero(changed)
                selected=ids[np.argmin(rays[0,ids])]
                alarm_capture,alarm_delivery=captured,delivered
                alarm_angle=float(ANGLES[selected])
                opposite=math.radians(yaw+alarm_angle+180.)
                escape=np.array([math.sin(opposite),math.cos(opposite)])
            packets.append({"tick":tick,"capture_wall_s":round(captured,6),
                "delivery_wall_s":round(delivered,6),
                "source_health":"missing" if missing else "healthy",
                "actor_xz_m":actor.tolist(),"hazard_xz_m":hazard.tolist(),
                "range_change_alarm":bool(np.any(changed)),
                "low_range_m":[float(v) if np.isfinite(v) else None
                               for v in rays[0]] if rays is not None else None})
            tick+=1
            if conn.poll():
                message=conn.recv()
                if message["kind"]=="stop":
                    stop_s=float(message["worker_end_s"])
                    break
        endpoint=motion.actor_xy(start,escape,alarm_delivery,stop_s)
        conn.send({"kind":"result","packets":packets,
            "baseline_low_range_m":[float(v) if np.isfinite(v) else None
                for v in baseline[0]],
            "alarm_capture_s":alarm_capture,"alarm_delivery_s":alarm_delivery,
            "alarm_relative_angle_deg":alarm_angle,
            "source_health_lost_s":health_lost,
            "response_endpoint_xz_m":endpoint.tolist()})
    except Exception as exc:
        conn.send({"kind":"error","message":repr(exc)})
    finally:
        conn.close()


def run_worker(scene,case,placement,target,state,out):
    if case["branch"]=="direct":
        return direct.fresh_recheck(scene,case,placement,state,out)
    return presentation.search(scene,case,placement,state,target,out)


def run_one(case: dict,arm: str,hazard_radius: float, *,
            sensor_target=None,out_root=None,trace_schema=None) -> dict:
    key=case["scene_key"]
    folder=(out_root or OUT) / key / arm
    folder.mkdir(parents=True,exist_ok=True)
    state=actor_state(case)
    start=state["actual"]
    yaw=state["yaw"]
    context=mp.get_context("spawn")
    parent,child=context.Pipe()
    process=context.Process(target=sensor_target or sensor_process,args=(child,case,
        start.tolist(),yaw,arm))
    process.start();child.close()
    if not parent.poll(45.):
        process.terminate();process.join()
        raise TimeoutError("Near-body process not ready")
    ready=parent.recv()
    if ready["kind"]!="ready":
        process.join()
        raise RuntimeError(f"Near-body process failed: {ready}")
    origin=time.perf_counter()+.2
    parent.send({"kind":"start","origin":origin})
    a=math.radians(yaw+case["entry_relative_deg"])
    incoming=np.array([math.sin(a),math.cos(a)])
    scene,base,placement,target=scene_and_props(case)
    hazard_prop=TexturedFloorObject(HAZARD_PROP,scale=.012)
    wrapped=shared.TimedWorkerScene(scene,origin,
        {"hazard_object":HAZARD},{"hazard_prop":hazard_prop},start,incoming)
    delay=origin-time.perf_counter()
    if delay>0:time.sleep(delay)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(run_worker,wrapped,case,placement,target,
                           state["worker_state"],folder / "interrupted_recheck")
        worker=future.result(timeout=30.)
    worker_end=time.perf_counter()-origin
    parent.send({"kind":"stop","worker_end_s":worker_end})
    if not parent.poll(30.):
        process.terminate();process.join()
        raise TimeoutError("Near-body process did not return")
    response=parent.recv()
    process.join(timeout=10.)
    if process.is_alive():
        process.terminate();process.join()
        raise TimeoutError("Near-body process did not exit")
    parent.close()
    if response["kind"]!="result":
        raise RuntimeError(f"Near-body process failed: {response}")
    packets=response["packets"]
    alarm=response["alarm_delivery_s"]
    escape=None
    if response["alarm_relative_angle_deg"] is not None:
        opposite=math.radians(yaw+response["alarm_relative_angle_deg"]+180.)
        escape=np.array([math.sin(opposite),math.cos(opposite)])
    gaps=[]
    for t in np.arange(0.,worker_end+.0001,.02):
        actor=motion.actor_xy(start,escape,alarm,float(t))
        hazard=motion.incoming_xy(start,incoming,float(t))
        gaps.append(motion.proxy_gap(actor,hazard,hazard_radius))
    invalidated=alarm is not None or response["source_health_lost_s"] is not None
    render,base,_,_=scene_and_props(case)
    forward=np.array([math.sin(math.radians(yaw)),-.58,
                      math.cos(math.radians(yaw))])
    ticks=[]
    with imageio.get_writer(str(folder / "monitor.mp4"),fps=5,codec="libx264",
                            quality=6,macro_block_size=16) as writer:
        for packet in packets[::2]:
            actor=np.asarray(packet["actor_xz_m"],float)
            hazard=np.asarray(packet["hazard_xz_m"],float)
            render.set_props({**base,HAZARD:(hazard_prop,tuple(hazard),0.)})
            eye=np.array([actor[0],render.floor_y+.42,actor[1]])
            rgb,_=render.observe(eye,forward,owner_xz=None,
                                 width=160,height=90)
            writer.append_data(motion.decorate(rgb,t=packet["capture_wall_s"],
                label="FINAL OWNER CHECK",actor=actor,start=start,
                hazard=hazard,gap=motion.proxy_gap(actor,hazard,hazard_radius),
                alarm=alarm is not None and packet["capture_wall_s"]>=alarm))
            ticks.append(packet["tick"])
    trace={"schema":trace_schema or "ape.joined-live-hazard-16.v1",
        "scene_key":key,"branch":case["branch"],"arm":arm,
        "source_state":case["source_state"],
        "source_confirmed_episode":case["source_confirmed_episode"],
        "entry_relative_deg":case["entry_relative_deg"],
        "start_xz_m":start.tolist(),"heading_deg":yaw,
        "worker_wall_s":worker_end,"worker_world_captures":wrapped.captures,
        "worker_recheck_trace":relative(folder /
            "interrupted_recheck/episode_trace.json"),
        "worker_recheck_video":worker["video"],
        "worker_action_before_health_gate":worker["action"],
        "usable_action":"invalidated_hold_unknown" if invalidated else worker["action"],
        "image_invalidated":invalidated,
        "monitor_packets":packets,"monitor_video":relative(folder / "monitor.mp4"),
        "monitor_video_packet_ticks":ticks,
        "baseline_low_range_m":response["baseline_low_range_m"],
        "backup_packets":response.get("backup_packets"),
        "backup_baseline_low_range_m":response.get("backup_baseline_low_range_m"),
        "alarm_source":response.get("alarm_source","primary_range" if alarm is not None else None),
        "source_health_lost_s":response["source_health_lost_s"],
        "alarm_capture_s":response["alarm_capture_s"],
        "alarm_delivery_s":alarm,
        "alarm_relative_angle_deg":response["alarm_relative_angle_deg"],
        "response_endpoint_xz_m":response["response_endpoint_xz_m"],
        "hazard_footprint_radius_m":hazard_radius,
        "min_evaluator_proxy_gap_m":min(gaps),
        "max_monitor_capture_interval_s":max((b["capture_wall_s"]-
            a["capture_wall_s"] for a,b in zip(packets,packets[1:])),default=0.),
        "limits":"Sixteen actor-gated direct/cooperative final poses; independent spawned two-height range process and actual slow image worker share a monotonic clock and timed rigid prop. Actor response is ideal delayed kinematic disk, worker camera stays at original pose, and postrun monitor video is diagnostic only. In-flight image claims are discarded on alarm/source loss. No post-interruption owner return or physical person/stop certification."}
    path=folder / "trace.json"
    path.write_text(json.dumps(trace,indent=2)+"\n")
    return {"scene_key":key,"branch":case["branch"],"arm":arm,
        "trace":relative(path),"min_gap_m":min(gaps),
        "usable_action":trace["usable_action"]}


def case_rows() -> list[dict]:
    joined=json.loads((ART / "wait_gated_with_cooperation_20_audit.json").read_text())
    cases={c["scene_key"]:c for c in json.loads((ART /
        "integrated_owner_20_conditional_precommit.json").read_text())["cases"]}
    rows=[]
    for decision in joined["rows"]:
        key=decision["scene_key"]
        if decision["outcome"]=="direct_fresh_confirmed":
            branch="direct"
            gate=json.loads((direct.OUT / key / "trace.json").read_text())
            bearings=gate["admitted_relative_entry_bearings_deg"]
            source_state=gate["source_action_trace"]
            episode=gate["recheck_trace"]
        elif decision["outcome"]=="cooperative_fresh_confirmed":
            branch="cooperative"
            route=json.loads((cooperative.OUT / key / "trace.json").read_text())
            bearings=route["admitted_entry_bearings_deg"]
            source_state=relative(cooperative.OUT / key / "trace.json")
            follow=json.loads((ROOT /
                f"assets/real_indoor_scenes/returned_owner_presentation_8_20261015/{key}/trace.json").read_text())
            episode=follow["fresh_search_trace"]
        else:
            continue
        if not bearings:
            raise ValueError("Confirmed case has no actor-admitted wait lane")
        rows.append({**cases[key],"branch":branch,
            "entry_relative_deg":bearings[0],
            "source_state":source_state,
            "source_confirmed_episode":episode})
    if len(rows)!=16:
        raise ValueError("Expected 16 confirmed actor-gated final poses")
    return rows


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-new",type=int,default=0)
    args=parser.parse_args()
    cases=case_rows()
    hazard_radius=TexturedFloorObject(HAZARD_PROP,scale=.012).footprint_radius_m
    frozen={"schema":"ape.joined-live-hazard-16-precommit.v1",
        "source_joined_audit":relative(ART /
            "wait_gated_with_cooperation_20_audit.json"),
        "source_joined_audit_sha256":sha(ART /
            "wait_gated_with_cooperation_20_audit.json"),
        "cases":cases,"arms":list(ARMS),"range_outage_s":OUTAGE,
        "monitor_period_s":PERIOD_S,"hazard_object":HAZARD,
        "hazard_footprint_radius_m":hazard_radius,
        "policy":"At each actor-admitted direct or returned-owner final pose, start an actual fresh RGB-D/owner image worker and a separate 10Hz two-height all-around near-body process on the same timed rigid entrant. Use the first actor-admitted corridor bearing, not evaluator lane selection. The healthy arm has continuous range; the outage arm loses range from 0.25 to 1.25s. A changed low return commands the same ideal delayed opposite-bearing disk response. Source loss or alarm invalidates any in-flight visual claim. Compare modeled hazard gap, room/person/prop geometry, packet timing and image/video evidence across all 16 confirmed rooms.",
        "limits":"Static scripted owner and rigid low prop; ideal kinematic response, no articulated body/IMU/joint/contact timing or post-interruption owner return."}
    if PRECOMMIT.exists():
        if json.loads(PRECOMMIT.read_text())!=json.loads(json.dumps(frozen)):
            raise ValueError("Joined hazard precommit changed")
    else:
        PRECOMMIT.write_text(json.dumps(frozen,indent=2)+"\n")
    rows=[];launched=0
    for case in cases:
        for arm in ARMS:
            path=OUT / case["scene_key"] / arm / "trace.json"
            if path.exists():
                trace=json.loads(path.read_text())
                row={"scene_key":case["scene_key"],"branch":case["branch"],
                    "arm":arm,"trace":relative(path),
                    "min_gap_m":trace["min_evaluator_proxy_gap_m"],
                    "usable_action":trace["usable_action"]}
            else:
                if args.max_new and launched>=args.max_new:
                    break
                row=run_one(case,arm,hazard_radius)
                launched+=1
            rows.append(row)
            REPORT.write_text(json.dumps({"schema":"ape.joined-live-hazard-16-report.v1",
                "precommit":relative(PRECOMMIT),"processed":len(rows),
                "rows":rows},indent=2)+"\n")
            print(case["scene_key"],case["branch"],arm,
                  round(row["min_gap_m"],3),row["usable_action"],flush=True)
        if args.max_new and launched>=args.max_new:
            break


if __name__=="__main__":
    main()
