"""Plan-bound classic MuJoCo stop comparison, separate from CUDA originals.

Collection is opt-in CLI only. Source preparation imports this module without
constructing models, stepping, sampling sensors, or invoking its main function.
"""
import argparse,copy,json,os,sys,time
from pathlib import Path
import mujoco
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
BASIC=ROOT/'control/skills/basic-skills'
SELECTED=json.loads((ROOT/'robot/model/current-model.json').read_text())
os.environ['COVE50_MODEL_DIR']=str((ROOT/SELECTED['rigid_MJCF']['path']).parent)
sys.path.insert(0,str(BASIC))
from walker_zero import Walker,CHECKPOINT
from adapter import Runtime,ref
import runtime as scalar_runtime


def command(control,branch):
    if branch not in ('hard','ramp'):raise ValueError('Unknown frozen command profile')
    vx=.25 if control<30 or (branch=='ramp' and control==30) else 0.
    if branch=='ramp' and 30<control<45:vx=.25*(45-control)/15
    return np.array([vx,0.,0.],dtype=np.float32)


class CommandWalker(Walker):
    def observe_command(self,packet,cmd):
        # An invalid packet is zero physical sensor values, not zero actor81:
        # relative q contains -sourcehome and command is already visible.
        values=np.zeros(50) if packet is None else packet['values'][:50]
        v=torch.tensor(values[None,:],dtype=torch.float32)
        return self.mapping.observe(v[:,:22],v[:,22:44],v[:,44:47],v[:,47:50],torch.tensor(self.previous_raw[None,:]),torch.tensor(cmd[None,:]))[0].numpy()

    def decision(self,packet,cmd):
        obs=self.observe_command(packet,cmd)
        raw,filtered,source_target,decoded=self.action(obs)
        requested=decoded if packet is not None else self.mapping.target(torch.zeros(1,24))[0].numpy()
        return obs,raw,decoded,requested


def runtime_snapshot(rt):
    return dict(target=rt.target.tolist(),previous_command=rt.previous_command.tolist(),
        pending=[dict(due_s=t,target22_rad=q.tolist()) for t,q in rt.pending],
        feedback=copy.deepcopy(list(rt.feedback)),last=copy.deepcopy(rt.last),
        gravity_estimate=rt.gravity_estimate.copy(),sample=copy.deepcopy(rt.sample),next_imu=rt.next_imu,next_joint=rt.next_joint,last_imu_time=rt.last_imu_time,seq=rt.seq,
        accel_bias=rt.accel_bias.copy(),gyro_bias=rt.gyro_bias.copy(),rng_state=copy.deepcopy(rt.rng.bit_generator.state))


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def collect(plan_path,case_name,out):
    plan=json.loads(Path(plan_path).read_text());cases=[c for c in plan['cases'] if c['name']==case_name]
    if len(cases)!=1:raise ValueError('Case outside frozen local comparison')
    case=cases[0];out=Path(out)
    if out.exists():raise FileExistsError('Original attempt path already exists')
    out.mkdir(parents=True);start=time.monotonic();physics=0;controls=0;failure=None;native={};decisions={};snapshots=[]
    model=data=rt=walker=None
    report=dict(schema='cove50.classic-stop-visit.v1',case=case,plan=ref(Path(plan_path)),engine='classic MuJoCo CPU float64 physics / NumPy scalar Runtime',status='starting',optimizer_updates=0,new_policies=0,critic_inference=False,reward_evaluation=False)
    def add(dst,**values):
        for key,value in values.items():dst.setdefault(key,[]).append(np.asarray(value).copy())
    try:
        if Path(scalar_runtime.OUT).resolve()!=(ROOT/SELECTED['rigid_MJCF']['path']).parent:raise ValueError('Scalar Runtime sidecars resolve outside selected R3')
        for item in plan['inputs']+plan['implementation']:
            if ref(ROOT/item['path'])!=item:raise ValueError('Frozen source/input differs: '+item['path'])
        seal=json.loads((ROOT/plan['selected_input_seal']['path']).read_text())
        for section in ('models','meshes','contracts'):
            for item in seal[section]:
                actual=ref(ROOT/item['path'])
                if actual['sha256']!=item['sha256'] or actual['bytes']!=item['bytes']:raise ValueError('Selected native dependency differs: '+item['path'])
        walker=CommandWalker();model=mujoco.MjModel.from_xml_path(str(ROOT/SELECTED['rigid_MJCF']['path']));data=mujoco.MjData(model)
        model.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
        rt=Runtime(model,data,rail_V=12.,command_delay_s=.02,feedback_delay_s=.01,peak=False,seed=case['seed'])
        floor=model.geom('floor').id;feet=[i for i in range(model.ngeom) if model.geom(i).name.startswith(('collision_R_foot_','collision_L_foot_'))]
        if case['entry']!='mapped-source-home':raise ValueError('Only frozen fresh full-start entry is supported')
        home=walker.mapping.target(torch.zeros(1,24))[0].numpy();rng=np.random.default_rng(case['seed'])
        data.qpos[rt.qids]=home+rng.normal(0,case['joint_sigma_rad'],22)
        for i,j in enumerate(rt.jids):
            if model.jnt_limited[j]:data.qpos[rt.qids[i]]=np.clip(data.qpos[rt.qids[i]],*model.jnt_range[j])
        rpy=rng.uniform(-case['rpy_envelope_rad'],case['rpy_envelope_rad'],3);mujoco.mju_euler2Quat(data.qpos[3:7],rpy,'xyz');data.qvel[rt.vids]=rng.normal(0,case['joint_velocity_sigma_rad_s'],22)
        mujoco.mj_forward(model,data);data.qpos[2]+=.0002-min(mujoco.mj_geomDistance(model,data,floor,i,1.,None) for i in feet)
        mujoco.mj_forward(model,data);rt.target=data.qpos[rt.qids].copy();rt.previous_command=rt.target.copy();walker.reset()
        report.update(checkpoint=ref(CHECKPOINT),library_versions=dict(mujoco=mujoco.__version__,numpy=np.__version__,torch=torch.__version__),numerical_options=dict(timestep_s=float(model.opt.timestep),gravity_m_s2=model.opt.gravity.tolist(),integrator=int(model.opt.integrator),solver=int(model.opt.solver),cone=int(model.opt.cone),iterations=int(model.opt.iterations)),joint_order=rt.names,native_qpos_addresses=rt.qids.tolist(),native_qvel_addresses=rt.vids.tolist(),initial_runtime=serial(runtime_snapshot(rt)))
        initial_qpos=data.qpos.copy();initial_qvel=data.qvel.copy();force=np.zeros(6)
        def sample():
            load=np.zeros(2);other=0.
            for i,c in enumerate(data.contact):
                if floor not in (c.geom1,c.geom2):continue
                name=model.geom(c.geom2 if c.geom1==floor else c.geom1).name;mujoco.mj_contactForce(model,data,i,force);normal=max(0.,force[0])
                if name.startswith('collision_R_foot_'):load[0]+=normal
                elif name.startswith('collision_L_foot_'):load[1]+=normal
                else:other+=normal
            rotation=data.xmat[model.body('torso').id].reshape(3,3)
            return dict(time_s=data.time,qpos=data.qpos,qvel=data.qvel,root_velocity_body_m_s=rotation.T@data.qvel[:3],foot_normal_N=load,other_floor_normal_N=other,whole_body_com_m=data.subtree_com[model.body('torso').id],sole_positions_m=np.array([data.site(s+'_sole_contact_frame').xpos for s in ('R','L')]),applied_target22_rad=rt.target,torque22_Nm=rt.torque,current22_A=rt.current,saturated22=rt.saturated)
        add(native,**sample())
        for control in range(150):
            if time.monotonic()-start>plan['wall_budget_per_visit_s']:raise TimeoutError('Prospective local wall budget exhausted; retain full available tail')
            packet=rt.observation();cmd=command(control,case['branch']);before=runtime_snapshot(rt) if control in (0,29,30,31,44,45,46) else None
            obs,raw,decoded,requested=walker.decision(packet,cmd);rt.command(rt.names,requested)
            if before is not None:snapshots.append(dict(control=control,qpos=data.qpos.copy(),qvel=data.qvel.copy(),before=before,after=runtime_snapshot(rt),command=cmd,raw24=raw,actor81=obs))
            add(decisions,time_s=data.time,command3=cmd,is_standing=bool(np.all(cmd==0)),observation81=obs,raw_action24=raw,decoded22_rad=decoded,requested22_rad=requested,accepted22_rad=rt.previous_command,sensor_valid=packet is not None,sensor_capture_time_s=packet['capture_time_s'] if packet else np.nan,imu_capture_time_s=packet['imu_capture_time_s'] if packet else np.nan)
            saturation=np.zeros(22);peak=np.zeros(22)
            for _ in range(80):rt.update();mujoco.mj_step(model,data);physics+=1;saturation+=rt.saturated;peak=np.maximum(peak,np.abs(rt.torque))
            mujoco.mj_forward(model,data);add(native,**sample());add(decisions,substep_saturated_fraction22=saturation/80,substep_peak_abs_torque22_Nm=peak);controls+=1
            if not np.isfinite(data.qpos).all():raise FloatingPointError('Nonfinite state')
        report['status']='completed'
    except BaseException as exc:failure=type(exc).__name__+': '+str(exc);report.update(status='interrupted',failure=failure)
    finally:
        trace=out/'episode.npz';arrays={k:np.asarray(v) for k,v in native.items()};arrays.update({'control_'+k:np.asarray(v) for k,v in decisions.items()});
        if 'initial_qpos' in locals():arrays.update(initial_qpos=initial_qpos,initial_qvel=initial_qvel)
        np.savez_compressed(trace,**arrays)
        (out/'switch-runtime-state.json').write_text(json.dumps(serial(snapshots),indent=2,allow_nan=False)+'\n')
        report.update(physical_world_slots_constructed=int(data is not None),controls=controls,native_physics_steps=physics,trace=ref(trace),wall_s=time.monotonic()-start,full_tail_requested_s=3.,state_resolution='50Hz full states/actions,4kHz aggregated motor saturation/peak; no early termination or automatic reset',scalar_sidecars=str(scalar_runtime.OUT))
        (out/'report.json').write_text(json.dumps(serial(report),indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1);collect(a.plan,a.case,a.out)


if __name__=='__main__':main()
