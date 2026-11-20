"""One nominal family, four source-first arm conditions, no fitting."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
import mujoco
from adapter import ROOT,ref
from run import HERE,episode,write
from arms_profile import Arms,AUDIT,PAWS

DATA=ROOT/'data/basic-skills-retry-20261027/arms'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/arms-20261027'
CASES=['bilateral-both','independent-source-left-current-R','independent-source-right-current-L','independent-both']


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    selected=json.loads((ROOT/'robot/model/current-model.json').read_text());cfg=json.loads((HERE/'stand.json').read_text());cfg['duration_s']=10.
    cfg['starts']=[dict(seed=140001,label='fresh-nominal-paired-arm-conditions',joint_sigma_rad=0.,rpy_envelope_rad=0.,joint_velocity_sigma_rad_s=0.)]
    cfg['controllers']=CASES;cfg['actor_controllers']=CASES;cfg['scope']='Four sequential10s armconditions fromsame nominalfamily; fixed currentCAD-neutral support, no fits/Runpod.'
    # Measurement frames only: compile/forward, no native integration, body or contact edits.
    model=mujoco.MjModel.from_xml_path(str(ROOT/selected['rigid_MJCF']['path']));data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    measurement=[]
    for name in PAWS:
        g=model.geom(name).id;mesh=int(model.geom_dataid[g]);a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);v=model.mesh_vert[a:a+n];c=(v.min(0)+v.max(0))/2
        rot=np.zeros(9);mujoco.mju_quat2Mat(rot,model.geom_quat[g]);bodypoint=model.geom_pos[g]+rot.reshape(3,3)@c
        measurement.append(dict(geom=name,body=model.body(int(model.geom_bodyid[g])).name,compiled_mesh_center_local_m=c.tolist(),body_local_measurement_point_m=bodypoint.tolist(),meaning='Selected paw visual AABB-center proxy, not handcontactsite. Readonly kinematic measurement.'))
    acts=json.loads((ROOT/selected['actuators']['path']).read_text());shoulders=[a for a in acts['actuators'] if a['joint'] in ['R_shoulder_pitch','L_shoulder_pitch']]
    method=dict(schema='cove50.basic-arms.source-first.v1',selected=selected,pointer=ref(ROOT/'robot/model/current-model.json'),configuration=cfg,
        source_audit=ref(AUDIT),profiles={case:Arms(case).proof() for case in CASES},
        schedule='Binaryraise flags1 for2≤t<6,otherwise0. Bilateral72 raisesboth; independent75 exposesleftflag intwist69andrightflag72. Hidden1.5salpha is diagnostic only, initialized0forcurrentlowered-arm standingentry. Original source equalheight resetartifact not transplanted.',
        source_home_mapping='Virtualsourceabsoluteq=sourcehome+sign*currentq, henceactorrelativeq=sign*currentq. Current CAD-neutral q0 represents source fixedhome inactor; physicaltargets for20non-shoulderaxes remaincurrent0. Shoulderhome0 and sign−1 maps sourceleft(+y)→currentR, sourceright(−y)→currentL.',
        authority='Only currentR_shoulder_pitch/currentL_shoulder_pitch. Source raw21→all21EMA(alpha.2bilateral/.05independent, firstvalidbypass)→source shoulderclip±1.52rad→sign map→unchangedR3Runtimecurrent rangeclip/delay. All21filteredraw messages feednexthistory; no rawclip orzeroinginactiveheads.',
        current_motor_metadata=dict(shoulders=shoulders,source_target_clip_rad=[-1.52,1.52],current_shoulder_range_rad=[float(-np.pi/3),float(5*np.pi/6)],
            rail_V=12.,peak=False,C047_rated_cap_Nm=.980665,command_delay_s=.02,feedback_delay_s=.01,physics_dt_s=.00025,control_dt_s=.02,
            note='Source clip and current range are separate. Continuousjoint diagnostic±.35rad doesnot apply toshoulders. Gains8/.16 and catalogcurves uncalibrated, unchanged.'),
        measurement_frames=measurement,measurement_contacts='Whole R/L_forearm collisionbodyfloor/selfnormal at50Hz, includeselbow/wristgeometry; not dedicatedhandpad sensing. Full q/qvel/contact/torque/duty andrequested/accepted/applied recorded.',
        recording='Exactpreactiondelayedpacket, source72/75, all21raw/EMA/history,flags/hiddenalpha, pawproxy/worldCOM/solepositions andbodycontact. First20ms sensorbootstrapheldinside10s; no hiddensteps. Fulltails retained despitehistoricalhand-contacttermination.',
        sources=[ref(HERE/n) for n in ('arms_profile.py','arms_trial.py','run.py','adapter.py','crouch_policy.py')],
        accounting='Fourfull10sconditions,onepairednominalstartfamily140001,oneworldslot sequential,2000controls/160000physics planned,2reusedsavedactors,0newweights/optimizerupdates.',
        limitations='One nominal family, no loads/pushes/grasp/independentelbows orfullarmWBC. Zero-update bodytransfer with centeredhomeandsensorcontract differences declared; originalsource rewarddefinitions retained only asprovenance, no newreward orvalue fitting.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen arm method differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path),measurement_frames=measurement)));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Original arm visits retained')
    rows=[episode(cfg,cfg['starts'][0],case,Arms(case),OUT,method) for case in CASES]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),native_physics_steps=sum(e['native_physics_steps'] for e in rows),optimizer_updates=0,new_weight_versions=0,independent_start_families=1,world_slots=1,reused_actor_versions=2))


if __name__=='__main__':main()
