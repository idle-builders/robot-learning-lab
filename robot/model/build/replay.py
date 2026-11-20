"""Matched bounded native replay of a retained learned trajectory.

This is a lossy, named trajectory retargeting diagnostic, NOT closed-loop old
actor inference on a new body. All episodes and early numerical ends retained.
"""
import argparse
import time
import mujoco
from common import *
from runtime import Runtime,capture_transform

OLD_BODY=ROOT/'artifacts/ape-frame50-supported-body/ape-frame50-hip-contact-probe.xml'
OLD_TRACE=ROOT/'artifacts/ape-frame50-supported-body/learned-replay/forward-frame50-motor-adapted-seed144.npz'

def source_mapping(model):
    old=mujoco.MjModel.from_xml_path(str(OLD_BODY));od=mujoco.MjData(old);mujoco.mj_forward(old,od)
    nd=mujoco.MjData(model);mujoco.mj_forward(model,nd)
    old_names=[mujoco.mj_id2name(old,mujoco.mjtObj.mjOBJ_JOINT,int(old.actuator_trnid[i,0]))for i in range(old.nu)]
    current=read(OUT/'actuators.json')['actuators'];mapping=[]
    for a in current:
        name=a['joint'];source=None
        if name=='neck_yaw':source='head_yaw'
        elif name.startswith(('R_','L_')):
            # Preserve spatial sides under the proper REP rotation: positive
            # modelY is legacyleft and sourceR. No anatomical reflection.
            side='left'if name.startswith('R_')else'right';axis=name[2:]
            if axis=='elbow':axis='elbow_pitch'
            candidate=side+'_'+axis
            if candidate in old_names:source=candidate
        if source:
            oj=old.joint(source).id;nj=model.joint(name).id
            dot=float(od.xaxis[oj]@nd.xaxis[nj]);sign=1. if dot>=0 else-1.
            mapping.append(dict(target=name,source=source,sign=sign,neutral_axis_alignment=abs(dot),
                source_qpos_index=int(old.jnt_qposadr[oj]),target_index=a['index']))
        else:mapping.append(dict(target=name,source=None,sign=None,target_index=a['index'],default_rad=0.,reason='New source-native degree of freedom absent in legacy trace'))
    return dict(source_body=ref(OLD_BODY),source_trace=ref(OLD_TRACE),
        checkpoint=dict(path='artifacts/policies/checkpoints/frame50_catalog_motor_walk.pt',sha256=sha(ROOT/'artifacts/policies/checkpoints/frame50_catalog_motor_walk.pt')),
        mapping=mapping,removed_source_channels=[n for n in old_names if n not in [r['source']for r in mapping]],
        limitations=['Source trace stores attained positions only, not actor targets/actions/observations. Replay treats attained positions as prescribed targets; it is not the saved actor.',
            'Hip pitch/roll ordering, joint positions, limb proportions and neutral endpoint poses changed. Axis-sign mapping is a diagnostic, not an identified policy or task-space retargeter.',
            'SourceR maps to legacy spatialleft. New neckpitch and both anklerolls heldzero; legacy grippers absent.19 shared channels,3new,2removed.',
            'Unknown hip source limits remain unknown; runtime diagnostic target envelope±0.35rad is an explicit temporary command restriction.'])

def targets(mapping,source,time_s):
    index=min(len(source)-1,time_s/.02);lo=int(index);f=index-lo
    q=source[lo]*(1-f)+source[min(lo+1,len(source)-1)]*f
    return np.array([r['sign']*q[r['source_qpos_index']]if r['source']else 0. for r in mapping['mapping']])

def episode(variant,condition,family,horizon):
    body_path=OUT/f'cove50-{variant}.xml';body_identity=ref(body_path)
    parameter_inputs=[ref(OUT/name)for name in ['mesh-transfer.json','collision-decomposition.json','actuators.json','sensors.json','timing.json']]
    executed_recipe=ref(Path(__file__))
    m=mujoco.MjModel.from_xml_path(str(body_path))
    if sha(body_path)!=body_identity['sha256']:raise ValueError('Model changed during native load')
    # Explicit flexible integrator sensitivity, retained in episode config.
    if variant=='compliant':m.opt.timestep=.000002
    if condition.get('physics_timestep_s') is not None:m.opt.timestep=condition['physics_timestep_s']
    m.opt.disableflags|=int(mujoco.mjtDisableBit.mjDSBL_AUTORESET)
    source=np.load(OLD_TRACE)['qpos'];mapping=source_mapping(m)
    d=mujoco.MjData(m);rt=Runtime(m,d,rail_V=condition['rail_V'],command_delay_s=condition['delay_s'],feedback_delay_s=.01,seed=3753+family)
    offset=family*.2;q0=targets(mapping,source,offset)
    # Same diagnostic clipping for initial state and target stream.
    rt.command(rt.names,q0);rt.pending.clear();q0=rt.previous_command.copy()
    d.qpos[rt.qids]=q0;rt.target=q0.copy();mujoco.mj_forward(m,d)
    floor=m.geom('floor').id;distance=[]
    for i in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i)or''
        if name.startswith(('collision_R_foot_','collision_L_foot_')):distance.append(mujoco.mj_geomDistance(m,d,floor,i,1.,None))
    dz=.002-min(distance);d.qpos[2]+=dz;mujoco.mj_forward(m,d)
    name=f'{variant}-{condition["name"]}-start{family}'
    arrays={k:[]for k in ['time_s','qpos','qvel','target_rad','torque_Nm','current_estimate_A','saturation','observation75','observation_age_s','root_position_m','floor_foot_normal_N','self_contact_depth_m','spring_translation_max_m','spring_rotation_max_rad','T_world_rgb','T_world_depth']}
    initial=d.qpos.copy();last_command=-1;terminal='requested_horizon';start=time.monotonic()
    sample_steps=max(1,round(.001/m.opt.timestep));steps=0
    while d.time<horizon-1e-8:
        if d.time-last_command>=.02-1e-9:
            rt.command(rt.names,targets(mapping,source,offset+d.time));last_command=d.time
        rt.update();mujoco.mj_forward(m,d)
        foot=np.zeros(2);self_depth=0.
        force=np.zeros(6)
        for ci,c in enumerate(d.contact):
            if c.geom1==floor or c.geom2==floor:
                gid=c.geom2 if c.geom1==floor else c.geom1
                gn=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(gid))or''
                mujoco.mj_contactForce(m,d,ci,force)
                for side,prefix in enumerate(['collision_R_foot_','collision_L_foot_']):
                    if gn.startswith(prefix):foot[side]+=max(0,float(force[0]))
            else:self_depth=max(self_depth,-float(c.dist))
        obs=rt.observation();flex_t=[];flex_r=[]
        for j in range(m.njnt):
            jname=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j)or''
            if jname.startswith('flex_'):
                value=abs(float(d.qpos[m.jnt_qposadr[j]]));(flex_t if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_SLIDE else flex_r).append(value)
        record=dict(time_s=d.time,qpos=d.qpos.copy(),qvel=d.qvel.copy(),target_rad=rt.target.copy(),torque_Nm=rt.torque.copy(),current_estimate_A=rt.current.copy(),
            saturation=rt.saturated.copy(),observation75=obs['values']if obs else np.zeros(75),observation_age_s=obs['age_s']if obs else -1.,
            root_position_m=d.xpos[m.body('torso').id].copy(),floor_foot_normal_N=foot,self_contact_depth_m=self_depth,
            spring_translation_max_m=max(flex_t,default=0.),spring_rotation_max_rad=max(flex_r,default=0.),
            T_world_rgb=capture_transform(m,d,'rgb_optical'),T_world_depth=capture_transform(m,d,'depth_optical'))
        for k in arrays:arrays[k].append(record[k])
        mujoco.mj_step(m,d,nstep=sample_steps);steps+=sample_steps
        if not np.all(np.isfinite(d.qpos))or not np.all(np.isfinite(d.qvel)):
            terminal='nonfinite_native_state';break
        if d.time<record['time_s']:
            terminal='native_clock_reset';break
    arrays={k:np.array(v)for k,v in arrays.items()}
    trace=OUT/'replay'/f'{name}.npz';trace.parent.mkdir(exist_ok=True);np.savez_compressed(trace,**arrays)
    error=arrays['qpos'][:,rt.qids]-arrays['target_rad'];duration=float(d.time)
    summary=dict(id=name,variant=variant,condition=condition,start_family=family,
        source_time_offset_s=offset,initial_qpos=initial.tolist(),root_height_adjustment_m=float(dz),
        requested_horizon_s=horizon,actual_duration_s=duration,terminal_reason=terminal,
        physics_steps=steps,recorded_samples=len(arrays['time_s']),sample_period_s=.001,physics_timestep_s=float(m.opt.timestep),
        command_period_s=.02,motor_envelope_update_period_s=.001,wall_seconds=time.monotonic()-start,
        trace=ref(trace),body=body_identity,parameter_inputs=parameter_inputs,executed_recipe=executed_recipe,
        joint_tracking_error_rad=dict(zip(['p10','p50','p90','p99'],np.quantile(abs(error),[.1,.5,.9,.99]).tolist())),
        rms_joint_tracking_error_rad=float(np.sqrt(np.mean(error**2))),
        root_displacement_m=(arrays['root_position_m'][-1]-arrays['root_position_m'][0]).tolist(),
        minimum_root_height_m=float(arrays['root_position_m'][:,2].min()),
        per_joint_peak_effort_Nm=np.max(abs(arrays['torque_Nm']),axis=0).tolist(),
        per_joint_rated_saturation_fraction=arrays['saturation'].mean(0).tolist(),
        peak_catalog_current_sum_A=float(arrays['current_estimate_A'].sum(1).max()),
        mean_catalog_current_sum_A=float(arrays['current_estimate_A'].sum(1).mean()),
        maximum_self_contact_proxy_depth_m=float(arrays['self_contact_depth_m'].max()),
        maximum_connector_translation_m=float(arrays['spring_translation_max_m'].max()),
        maximum_connector_rotation_rad=float(arrays['spring_rotation_max_rad'].max()),
        observation_age_s=dict(zip(['p10','p50','p90','p99'],np.quantile(arrays['observation_age_s'][arrays['observation_age_s']>=0],[.1,.5,.9,.99]).tolist())),
        requested_horizon_coverage=min(1,duration/horizon),
        command_error_accuracy_curve=[dict(error_rad=e,active_sample_fraction=float(np.mean(abs(error)<=e)),requested_horizon_fraction=float(np.mean(abs(error)<=e))*min(1,duration/horizon))for e in [0.01,.025,.05,.1,.2,.4]],
        interpretation='Native closed-loop assumed position servos follow a prescribed retargeted attained-position trajectory. Zero old actor inference, zero optimizer updates. Contact proxies, FEM linearity and unmeasured drive assumptions bound this evidence.')
    write(OUT/'replay'/f'{name}.json',summary)
    print(dict(episode=name,duration_s=duration,wall_s=summary['wall_seconds'],tracking_rms_rad=summary['rms_joint_tracking_error_rad'],terminal=terminal),flush=True)
    return summary,mapping

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=float,default=.8);args=parser.parse_args()
    conditions=[('rigid',dict(name='nominal',rail_V=12.,delay_s=.02)),('rigid',dict(name='low-rail-delay',rail_V=10.,delay_s=.04)),('compliant',dict(name='nominal',rail_V=12.,delay_s=.02))]
    summaries=[]
    for variant,condition in conditions:
        for family in [0,1]:
            summary,mapping=episode(variant,condition,family,args.horizon);summaries.append(summary)
            write(OUT/'trajectory-mapping.json',mapping)
            write(OUT/'replay-summary.json',dict(episodes=summaries,independent_start_families=2,world_slots=1,
                episodes_requested=6,episodes_completed=len(summaries),new_policy_checkpoints=0,optimizer_updates=0,
                source_trace_origin='Retained learned policy trajectory, not new policy inference',
                full_trace_scope='Every1ms native sampling station through each episode including early ends; internal microsteps counted separately, not separately stored episodes.'))

if __name__=='__main__':main()
