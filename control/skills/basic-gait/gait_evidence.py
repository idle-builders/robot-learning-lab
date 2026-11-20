"""Read-only full-state attestation and command-switch queue snapshots."""
import hashlib
import json
import numpy as np


def fingerprint(value):
    digest=hashlib.sha256()
    def visit(x):
        if hasattr(x,'detach'):
            a=x.detach().cpu().contiguous().numpy();digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=repr):digest.update(repr(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:digest.update(repr(x).encode())
    visit(value);return digest.hexdigest()


def state_attestation(runner,saved):
    rows={}
    for name,current in (('actor',runner.alg.actor.state_dict()),('critic',runner.alg.critic.state_dict()),('optimizer',runner.alg.optimizer.state_dict())):
        actual=fingerprint(current);expected=fingerprint(saved[name+'_state_dict'])
        rows[name]=dict(actual_sha256=actual,checkpoint_sha256=expected,equal=actual==expected)
    return rows


def live_rewards(env,native):
    rows={}
    for name in env.reward_manager.active_terms:
        term=env.reward_manager.get_term_cfg(name)
        rows[name]=dict(function=term.func.__module__+'.'+getattr(term.func,'__qualname__',type(term.func).__qualname__),weight=term.weight,params=native.serializable(term.params))
    return rows


class TransitionCapture:
    """Copy attained state around command edits, never advance/reset any state."""
    runtime_fields=('tick','pending_reset','target_history','target','initial_target','previous_command','torque','current','saturated','rail','gravity_estimate','sample_gyro','accel_bias','gyro_bias','next_imu','next_joint','last_imu_time','seq','packet_values','packet_capture','packet_imu_capture','last_values','capture_time_s','imu_capture_time_s','valid')

    def __init__(self,env,native,profile):
        self.env,self.native,self.profile=env,native,profile;self.rows=[];self.checks=[]
        switch=round(profile.switch_s/.02) if profile.switch_s is not None else None
        self.controls={0} if switch is None else {0,switch-1,switch,switch+1}

    def snapshot(self,control,phase,observations):
        if control not in self.controls:return None
        env=self.env;rt=self.native.runtime_package().get_runtime(env);term=env.command_manager.get_term('twist')
        def copy(v):return v.detach().cpu().numpy().copy()
        arrays={name:copy(getattr(rt,name)) for name in self.runtime_fields}
        arrays.update(qpos=copy(env.sim.data.qpos),qvel=copy(env.sim.data.qvel),native_time_s=copy(env.sim.data.time),raw_action24=copy(env.action_manager.action),previous_raw_action24=copy(env.action_manager.prev_action),runtime_rng_state=copy(rt.generator.get_state()),actor81=copy(observations['actor']),critic96=copy(observations['critic']),command_body3=copy(term.vel_command_b),command_world3=copy(term.vel_command_w),is_standing=copy(term.is_standing_env),physics_armed=np.asarray(rt.physics_armed))
        for name in ('is_world_env','is_forward_env','is_heading_env'):
            if hasattr(term,name):arrays['command_flag_'+name]=copy(getattr(term,name))
        for name in ('time_left','command_counter'):
            if hasattr(term,name):arrays['command_'+name]=copy(getattr(term,name))
        self.rows.append((control,phase,arrays));return arrays

    def compare(self,control,before,after):
        if before is None:return
        changed=[]
        for name,a in before.items():
            b=after[name]
            if name in ('command_body3','command_world3','is_standing','command_flag_is_world_env','command_flag_is_forward_env','command_flag_is_heading_env'):continue
            if name in ('actor81','critic96'):
                start=69 if name=='actor81' else 72
                a=np.concatenate((a[:,:start],a[:,start+3:]),axis=1);b=np.concatenate((b[:,:start],b[:,start+3:]),axis=1)
            if not np.array_equal(a,b,equal_nan=True):changed.append(name)
        self.checks.append(dict(control=control,unexpected_changed_state_fields=changed))
        if changed:raise RuntimeError('Command edit changed retained plant/history/queue state: '+','.join(changed))

    def save(self,directory,ref):
        if not self.rows:return None
        path=directory/'command-switch-state.npz'
        values={name:np.stack([row[2][name] for row in self.rows]) for name in self.rows[0][2]}
        np.savez_compressed(path,control=np.asarray([r[0] for r in self.rows]),phase=np.asarray([r[1] for r in self.rows]),**values)
        result=dict(schema='cove50.command-switch-state.v1',trace=ref(path),sampled_preaction_controls=sorted({x[0] for x in self.rows}),phases=['before_command','after_command'],checks=self.checks,meaning='Full runtime motor ring, delayed packet state, RNG and raw history are copied around command-only edits. No state reset/copy-back, sampling, physics or observation recompute. Full native Episodes separately retains every50Hz transition.',four_kHz_full_trajectory=False)
        (directory/'command-switch-state.json').write_text(json.dumps(result,indent=2)+'\n');return result
