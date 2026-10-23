"""Native Gymnasium/SB3 adapter for bounded grasp-retention residual learning.

The authored feedback controller owns task order, perception, bending and travel.
RL chooses gripper effort allowance and a small forearm residual during carrying.
Policy observations are sensor counterparts; simulator truth is reward/evaluation.
"""
import gymnasium as gym
import mujoco
import numpy as np
from lab.floor_pickup import FloorProbe,CleanupReference,evaluate_cleanup

PHASES=['find','approach','bend','reach','grasp','lift','carry','lower','release','settle','done']

class FloorCleanupEnv(gym.Env):
    def __init__(self,scene,capture=False):
        self.motor=FloorProbe(scene,1.)
        self.controller=None;self.capture_rows=capture
        self.action_space=gym.spaces.Box(-1.,1.,(2,),np.float32)
        self.observation_space=gym.spaces.Box(-10.,10.,(17,),np.float32)

    def observation(self):
        m,c=self.motor,self.controller;q=m.data.qpos[m.qidx];dq=m.data.qvel[m.vidx]
        p,_,gyro,x,v,_=m.motion();phase=np.zeros(len(PHASES));phase[PHASES.index(c.phase)]=1
        return np.r_[phase,q[c.grip],dq[c.grip]/4,m.electrical['applied_effort_nm'][c.grip],q[c.elbow],p,gyro[1]].astype(np.float32)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        if self.controller:self.controller.close()
        self.motor.reset(seed=seed);self.motor.data.qpos[0]=-.25
        a=self.motor.model.jnt_qposadr[self.motor.model.joint('task_block_free').id]
        self.initial=np.array([options.get('object_x',self.np_random.uniform(.013,.017)),options.get('object_y',self.np_random.uniform(.138,.142)),.015])
        self.motor.data.qpos[a:a+3]=self.initial;mujoco.mj_forward(self.motor.model,self.motor.data)
        self.motor.pack.effort_limit[self.motor.order.index('left_gripper_joint')]=.4
        self.controller=CleanupReference(self.motor);self.rows=[];self.done=False;self.previous_stage=0;self.last_lift=0.;self.previous_transport=0.;self.grasp_seen=False;self.dropped=False
        self.controller.capture();return self.observation(),{}

    def step(self,action):
        if self.done:raise RuntimeError('reset completed episode')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('two finite residual actions required')
        action=np.clip(action,-1,1);m,c=self.motor,self.controller
        effort=.22+.18*action[0];arm=.08*action[1]
        reward=0.;recent=[]
        for _ in range(10):
            if m.steps%10==0 and (c.phase=='find' or self.capture_rows):c.capture()
            # Only change gripper motor limit; never object contacts/state.
            m.pack.effort_limit[c.grip]=effort if c.phase in ['grasp','lift','carry','lower'] else .4
            target,speed,obs=c.command(arm_residual=arm);m.heading_error=c.yaw
            row=m.advance(target,speed)
            row.update(phase=c.phase,estimate=c.estimate,frame_estimate=c.frame_estimate,estimated_world_x_m=obs['world_x_m'],estimated_hand_m=obs['hand'].tolist(),heading_estimate_rad=c.yaw,
                policy_action=action.tolist(),gripper_effort_limit_nm=float(effort),forearm_residual_rad=float(arm))
            addr=m.model.jnt_dofadr[m.model.joint('task_block_free').id]
            row['block_velocity_m_s']=m.data.qvel[addr:addr+3].tolist()
            self.rows.append(row);recent.append(row)
            if c.phase=='done' or abs(row['pitch_rad'])>.28:break
        # Reward uses physics truth. It is not present in policy observations.
        row=recent[-1];contacts=row['block_contacts_n'];bilateral=all(contacts.get(n,0)>.05 for n in ['left_forearm_1','left_gripper_0'])
        lift=max(0,row['block_m'][2]-.015);self.grasp_seen|=bilateral and lift>.02
        stage=PHASES.index(c.phase);reward+=.5*max(0,stage-self.previous_stage);self.previous_stage=stage
        if self.grasp_seen and c.phase in ['lift','carry','lower']:
            reward+=.15 if bilateral else -.5
            transport=float(np.linalg.norm(np.asarray(row['block_m'][:2])-self.initial[:2]));reward+=10*(transport-self.previous_transport);self.previous_transport=transport
        reward+=10*(lift-self.last_lift);self.last_lift=lift
        reward-=.005*float(np.sum(action**2))
        fallen=abs(row['pitch_rad'])>.28
        terminal=c.phase=='done' or fallen;truncated=m.data.time>=55 or c.age>15
        info={}
        if terminal or truncated:
            evaluation=evaluate_cleanup(self.rows,initial=self.initial);info['evaluation']=evaluation
            reward+=10. if evaluation['success'] else -3.
        self.done=terminal or truncated
        return self.observation(),float(reward),bool(terminal),bool(truncated),info

    def close(self):
        if self.controller:self.controller.close();self.controller=None
        self.motor.close()
