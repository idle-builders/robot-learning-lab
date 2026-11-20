"""Sequential finite native CPU basic-skill transfer, full episodes retained."""
from pathlib import Path
import argparse,json,time
import mujoco
import numpy as np
import torch
from adapter import ROOT,Posture,Runtime,ref

HERE=Path(__file__).resolve().parent
OUT=ROOT/'artifacts/cove50-basic-skills-retry/stand-20261027'
DATA=ROOT/'data/basic-skills-retry-20261027/stand'


def write(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2)+'\n')
def distribution(x):return dict(zip(('min','p10','p50','p90','p99','max'),np.quantile(x,[0,.1,.5,.9,.99,1]).tolist()))


def episode(cfg,case,controller,posture,out,method):
    model=mujoco.MjModel.from_xml_path(str(ROOT/method['selected']['rigid_MJCF']['path']))
    if model.opt.timestep!=cfg['physics_dt_s']:raise ValueError('Selected physics cadence differs')
    model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    free_joints=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE)
    root_body=int(model.jnt_bodyid[free_joints[0]])
    data=mujoco.MjData(model);rt=Runtime(model,data,rail_V=cfg['rail_V'],command_delay_s=cfg['command_delay_s'],
        feedback_delay_s=cfg['feedback_delay_s'],peak=cfg['peak_motor_envelope'],seed=case['seed'])
    home=np.zeros(22)
    if cfg.get('entry_pose')=='mapped-source-home':
        home=np.array([r['sign']*posture.offset[r['source_index']] if r['source_index'] is not None else 0. for r in posture.mapping.rows])
    rng=np.random.default_rng(case['seed']);data.qpos[rt.qids]=home+rng.normal(0,case['joint_sigma_rad'],22)
    for i,j in enumerate(rt.jids):
        if model.jnt_limited[j]:data.qpos[rt.qids[i]]=np.clip(data.qpos[rt.qids[i]],*model.jnt_range[j])
    rpy=rng.uniform(-case['rpy_envelope_rad'],case['rpy_envelope_rad'],3)
    mujoco.mju_euler2Quat(data.qpos[3:7],rpy,'xyz');data.qvel[rt.vids]=rng.normal(0,case['joint_velocity_sigma_rad_s'],22)
    mujoco.mj_forward(model,data);floor=model.geom('floor').id
    feet=[i for i in range(model.ngeom) if model.geom(i).name.startswith(('collision_R_foot_','collision_L_foot_'))]
    data.qpos[2]+=cfg['initial_floor_clearance_m']-min(mujoco.mj_geomDistance(model,data,floor,i,1.,None) for i in feet)
    mujoco.mj_forward(model,data);rt.target=data.qpos[rt.qids].copy();rt.previous_command=rt.target.copy()
    initial_qpos=data.qpos.copy();initial_qvel=data.qvel.copy();posture.reset();initial_runtime=dict(target=rt.target.tolist(),
        previous_command=rt.previous_command.tolist(),accel_bias=rt.accel_bias.tolist(),gyro_bias=rt.gyro_bias.tolist(),queues_empty=True)
    native={};controls={};force=np.zeros(6);first_event=None;physics=0;start=time.monotonic();failure=None
    initial_yaw=np.arctan2(2*(data.qpos[3]*data.qpos[6]+data.qpos[4]*data.qpos[5]),1-2*(data.qpos[5]**2+data.qpos[6]**2))
    def add(dst,**values):
        for key,value in values.items():dst.setdefault(key,[]).append(np.asarray(value).copy())
    def sample():
        nonlocal first_event
        load=np.zeros(2);other=0.;selfdepth=0.
        for i,c in enumerate(data.contact):
            if c.geom1==floor or c.geom2==floor:
                g=c.geom2 if c.geom1==floor else c.geom1;name=model.geom(g).name
                mujoco.mj_contactForce(model,data,i,force);normal=max(0.,force[0])
                if name.startswith('collision_R_foot_'):load[0]+=normal
                elif name.startswith('collision_L_foot_'):load[1]+=normal
                else:other+=normal
            else:selfdepth=max(selfdepth,-float(c.dist))
        tilt=np.arccos(np.clip(1-2*(data.qpos[4]**2+data.qpos[5]**2),-1,1))
        yaw=np.arctan2(2*(data.qpos[3]*data.qpos[6]+data.qpos[4]*data.qpos[5]),1-2*(data.qpos[5]**2+data.qpos[6]**2))
        headingerror=np.arctan2(np.sin(initial_yaw-yaw),np.cos(initial_yaw-yaw))
        if first_event is None and (tilt>1.0 or data.qpos[2]<.15 or other>1.):first_event=float(data.time)
        values=dict(time_s=float(data.time),qpos=data.qpos,qvel=data.qvel,foot_normal_N=load,other_floor_normal_N=other,
            self_contact_depth_m=selfdepth,torso_tilt_rad=tilt,truth_heading_error_rad=headingerror,
            whole_body_com_m=data.subtree_com[root_body],
            sole_positions_m=np.array([data.site(s+'_sole_contact_frame').xpos for s in ('R','L')]),
            target22_rad=rt.target,torque22_Nm=rt.torque,current22_A=rt.current,saturated22=rt.saturated)
        if hasattr(posture,"sample_native"):values.update(posture.sample_native(model,data))
        return values
    add(native,**sample())
    try:
        for control in range(round(cfg['duration_s']/cfg['control_dt_s'])):
            packet=rt.observation();obs=posture.observe(packet);raw=np.zeros(getattr(posture,'action_width',21));filtered=raw.copy();oldtarget=posture.offset.copy()
            if control==0:target=rt.target.copy()
            elif controller in cfg.get('actor_controllers',('posture-bc','walker-zero')):raw,filtered,oldtarget,target=posture.action(obs)
            else:target=home.copy()
            if control>0:rt.command(rt.names,target)
            add(controls,time_s=data.time,observation72=obs,sensor_valid=packet is not None,
                sensor_capture_time_s=packet['capture_time_s'] if packet else np.nan,sensor_imu_capture_time_s=packet['imu_capture_time_s'] if packet else np.nan,
                raw_action21=raw,filtered_raw21=filtered,source_target21_rad=oldtarget,requested22_rad=target,
                accepted22_rad=rt.previous_command,estimated_heading_error_rad=obs[71] if len(obs)>71 else 0.)
            if hasattr(posture,'control_info'):add(controls,**posture.control_info())
            sat=np.zeros(22);peak=np.zeros(22);peakother=0.;sumfoot=np.zeros(2)
            for _ in range(round(cfg['control_dt_s']/cfg['physics_dt_s'])):
                rt.update();mujoco.mj_step(model,data);physics+=1
                sat+=rt.saturated;peak=np.maximum(peak,abs(rt.torque))
            mujoco.mj_forward(model,data);add(native,**sample());add(controls,substep_saturated_fraction22=sat/80,substep_peak_abs_torque22_Nm=peak)
            if not np.isfinite(data.qpos).all():raise FloatingPointError('Nonfinite physical state')
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc)
    trace=out/(controller+'-'+str(case['seed'])+'.npz');out.mkdir(parents=True,exist_ok=True)
    arrays={k:np.asarray(v) for k,v in native.items()};arrays.update({'control_'+k:np.asarray(v) for k,v in controls.items()})
    np.savez_compressed(trace,initial_qpos=initial_qpos,initial_qvel=initial_qvel,**arrays)
    q=arrays['qpos'];tilt=arrays['torso_tilt_rad'];n=len(controls.get('time_s',[]));error=np.abs(q[1:,rt.qids]-arrays['control_requested22_rad'])
    headingdiff=np.arctan2(np.sin(arrays['control_estimated_heading_error_rad']-arrays['truth_heading_error_rad'][:-1]),np.cos(arrays['control_estimated_heading_error_rad']-arrays['truth_heading_error_rad'][:-1]))
    report=dict(id=trace.stem,controller=controller,case=case,trace=ref(trace),initial_runtime=initial_runtime,
        status='completed' if failure is None else 'interrupted',failure=failure,requested_duration_s=cfg['duration_s'],actual_duration_s=float(data.time),
        controls=n,native_physics_steps=physics,optimizer_updates=0,policy_checkpoints=1 if controller=='posture-bc' else 0,
        initial_qpos=initial_qpos.tolist(),initial_qvel=initial_qvel.tolist(),first_evaluator_physical_event_s=first_event,
        evaluator_event_semantics='First tilt>1rad or rootheight<.15m or nonfoot normal>1N; observation annotation only, never stops full episode or grades it.',
        root_height_m=distribution(q[:,2]),torso_tilt_rad=distribution(tilt),final_root_displacement_m=(q[-1,:3]-initial_qpos[:3]).tolist(),
        max_horizontal_root_drift_m=float(np.linalg.norm(q[:,:2]-initial_qpos[:2],axis=1).max()),
        heading_estimator_minus_truth_error_rad=distribution(headingdiff[1:]),
        both_feet_contact_control_fraction=float(np.mean((arrays['foot_normal_N'][1:]>0).all(1))),
        foot_load_N_source_R_L=[distribution(arrays['foot_normal_N'][1:,i]) for i in range(2)],
        other_floor_normal_N=distribution(arrays['other_floor_normal_N']),
        per_joint_physics_saturation_fraction=np.mean(arrays['control_substep_saturated_fraction22'],axis=0).tolist(),
        per_joint_peak_abs_torque_Nm=np.max(arrays['control_substep_peak_abs_torque22_Nm'],axis=0).tolist(),
        per_joint_requested_minus_accepted_max_abs_rad=np.max(abs(arrays['control_requested22_rad']-arrays['control_accepted22_rad']),axis=0).tolist(),
        joint_target_tracking_rmse_rad=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        tracking_accuracy_curve=[dict(abs_error_radius_rad=r,control_joint_fraction=float(np.mean(error<=r))) for r in (.01,.025,.05,.1,.2)],
        wall_s=time.monotonic()-start,joint_order=rt.names,
        limits='Full50Hz states/actions and motor peaks/duty aggregated from4kHz substeps; source uncalibrated motors/IMU and diagnostic hip envelope. No pushes or learned recovery test.')
    write(out/(trace.stem+'.json'),report);print(json.dumps({k:report[k] for k in ('id','status','first_evaluator_physical_event_s','max_horizontal_root_drift_m','root_height_m','wall_s')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--config',type=Path,default=HERE/'stand.json');p.add_argument('--data',type=Path,default=DATA);args=p.parse_args()
    torch.set_num_threads(1);cfg=json.loads(args.config.read_text());posture=Posture()
    pointer=ROOT/'robot/model/current-model.json';selected=json.loads(pointer.read_text())
    for key in ('rigid_MJCF','actuators','sensors','timing'):
        if ref(ROOT/selected[key]['path'])['sha256']!=selected[key]['sha256']:raise ValueError('Selected input differs: '+key)
    method=dict(schema='cove50.basic-skills.native-method.v1',configuration=cfg,config_source=ref(args.config),
        sources=[ref(HERE/n) for n in ('adapter.py','run.py')]+[ref(ROOT/'robot/model/build'/n) for n in ('runtime.py','common.py')],
        pointer=ref(pointer),selected=selected,posture=posture.audit(),
        baseline='Exact selected body fixed entry-home output-joint target through identical motor/sensor Runtime; a physical servo reference, no learned actor.',
        entry=cfg.get('entry_pose','CADneutral')+' q plus predefined noise, foot collider clearance.2mm; same complete initialq/qvel and sensor RNG per pairedcase. First20ms sensor bootstrap retained inside8s.',
        counts='8 whole episodes,4 paired independent start families,1 worldslot per sequential visit,1 saved actor,0 optimizer updates; no concurrency or Runpod.')
    args.data.mkdir(parents=True,exist_ok=True);path=args.data/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Method differs from frozen input')
    write(path,method)
    if args.freeze_only:print(json.dumps(ref(path)));return
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()):raise FileExistsError('Original native output exists; use fresh --out')
    reports=[episode(cfg,case,controller,posture,args.out,method) for case in cfg['starts'] for controller in cfg['controllers']]
    write(args.out/'report.json',dict(method=ref(path),episodes=reports,simulator_transitions=sum(r['controls'] for r in reports),
        native_physics_steps=sum(r['native_physics_steps'] for r in reports),optimizer_updates=0,independent_start_families=4,world_slots=1))


if __name__=='__main__':main()
