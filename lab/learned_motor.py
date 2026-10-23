"""Reference-conditioned native PPO boundary on the existing electrical plant.

No task controller, analytic balance, gravity feedforward or object truth enters
the actor. Joint PD and velocity servos are explicit actuator primitives.
"""
import gymnasium as gym
import hashlib
import json
from dataclasses import asdict
import mujoco
import numpy as np
from lab.floor_pickup import FloorProbe

PLANT_ARRAYS = ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
               'jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_frictionloss',
               'dof_armature','geom_type','geom_pos','geom_quat','geom_size','geom_friction',
               'geom_contype','geom_conaffinity','mesh_vert','mesh_face','actuator_ctrlrange',
               'pair_friction','pair_solref','pair_solimp','qpos0']

def plant_parameters(motor):
    m = motor
    arrays={name:np.asarray(getattr(m.model,name)).copy() for name in PLANT_ARRAYS}
    settings=json.dumps({'electrical':asdict(m.pack.cfg),'kp':m.pack.kp.tolist(),
        'kd':m.pack.kd.tolist(),'effort':m.pack.effort_limit.tolist(),
        'velocity':m.limits.tolist(),'gravity':m.model.opt.gravity.tolist(),
        'timestep':m.model.opt.timestep,'cone':int(m.model.opt.cone)},sort_keys=True).encode()
    arrays['settings_json']=np.frombuffer(settings,dtype=np.uint8)
    return arrays

def plant_digest(arrays):
    digest=hashlib.sha256()
    for name in PLANT_ARRAYS:
        digest.update(name.encode());digest.update(arrays[name].tobytes())
    digest.update(arrays['settings_json'].tobytes())
    return digest.hexdigest()

def verify_portable_plant(actual,reference,expected_digest):
    """Verify attested reference bytes first, then explicit numeric tolerances."""
    if set(reference) != set(actual) or plant_digest(reference) != expected_digest:
        raise ValueError('Plant reference does not attest the trained contract')
    differences=[]
    for name in actual:
        a,b=actual[name],reference[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('Plant array layout differs: '+name)
        if np.array_equal(a,b):continue
        if not np.issubdtype(a.dtype,np.floating):
            raise ValueError('Plant discrete parameter differs: '+name)
        # Float32 compiled mesh geometry permits 1 nm absolute / 1e-7 relative;
        # native double arrays permit 1e-12 absolute / 1e-9 relative. No changed
        # topology, joint ordering, actuator limits or electrical settings pass.
        atol,rtol=(1e-9,1e-7) if a.dtype.itemsize==4 else (1e-12,1e-9)
        difference={'parameter':name,'max_absolute_difference':float(np.max(abs(a-b))),
                    'atol':atol,'rtol':rtol,'pass':bool(np.allclose(a,b,atol=atol,rtol=rtol))}
        differences.append(difference)
    return {'status':'pass' if all(d['pass'] for d in differences) else 'fail',
            'reference_digest':expected_digest,'actual_digest':plant_digest(actual),
            'differences':differences}


class MotorInterface:
    def __init__(self, motor):
        self.motor = motor
        self.positions = np.flatnonzero(~motor.wheels)
        self.low = np.array([motor.joints[i]['limits'][0] for i in self.positions] + [-.06,-.15,-.12])
        self.high = np.array([motor.joints[i]['limits'][1] for i in self.positions] + [.06,.15,.12])
        self.goal = np.r_[motor.rest[self.positions], 0.,0.,0.]
        self.previous = np.zeros(16)

    def reset(self):
        self.previous[:] = 0
        self.set_goal(np.r_[self.motor.rest[self.positions], 0.,0.,0.])

    def set_goal(self, goal):
        goal = np.asarray(goal, float)
        if goal.shape != (17,) or not np.isfinite(goal).all():
            raise ValueError('Expected 17 finite named VLA references')
        self.goal = np.clip(goal, self.low, self.high)
        self.motor.reference_x = self.motor.motion()[3] + self.goal[16]
        self.motor.command = np.array([self.goal[14], 0., self.goal[15]])
        return self.goal.copy()

    def observe(self):
        m = self.motor
        _, gravity, gyro, x, v, _ = m.motion()
        q = np.where(m.wheels, 0., m.data.qpos[m.qidx])
        return np.r_[q, m.data.qvel[m.vidx]/m.limits, gravity, gyro,
                     (m.reference_x-x)/.12, v/.06,
                     2*(self.goal-self.low)/(self.high-self.low)-1,
                     self.previous].astype(np.float32)

    def advance(self, action):
        action = np.asarray(action,float)
        if action.shape != (16,) or not np.isfinite(action).all():
            raise ValueError('Expected 16 finite learned motor actions')
        action = np.clip(action,-1,1)
        m = self.motor
        target = np.zeros(16)
        target[self.positions] = self.goal[:14] + .4*action[self.positions]
        target[m.wheels] = m.limits[m.wheels]*action[m.wheels]
        m.reference_x += self.goal[14]*.02
        row = m.advance_servos(target)  # zero feedforward; no authored balance
        self.previous = action.copy()
        row.pop('residuals',None)
        row.update(motor_policy_action=action.tolist(), motor_goal=self.goal.tolist(),
                   controller='learned PPO reference tracker; defined actuator PD/velocity servo')
        addr = m.model.jnt_dofadr[m.model.joint('task_block_free').id]
        row['block_velocity_m_s'] = m.data.qvel[addr:addr+3].tolist()
        return row

    def contract(self):
        m = self.motor
        return {'version':1, 'joint_order':m.order, 'observation_dim':73, 'action_dim':16,
                'plant_parameters_sha256':plant_digest(plant_parameters(m)),
                'goal_names':[m.order[i]+'.position_rad' for i in self.positions]+[
                    'forward_velocity_m_s','yaw_rate_rad_s','odometry_reference_error_m'],
                'goal_min':self.low.tolist(), 'goal_max':self.high.tolist(),
                'observation_fields':['q_rad_wheel_zero[16]','dq_div_velocity_limit[16]',
                    'gravity_body[3]','gyro_body_rad_s[3]','reference_minus_odometry_div_0.12',
                    'odometry_velocity_div_0.06','normalized_goal[17]','previous_actor_action[16]'],
                'servo_hz':50, 'physics_hz':500,
                'action_mapping':'nonwheel: goal position + 0.4 rad * action; wheel: velocity_limit * action',
                'analytic_balance':False, 'gravity_feedforward':False,
                'inner_control':'defined rate-limited joint PD / wheel velocity servo and electrical proxy',
                'state_excludes':['task phase','object pose','future state','episode clock']}


class MotorTrackingEnv(gym.Env):
    def __init__(self, scene, horizon=250):
        self.motor = FloorProbe(scene,1.)
        self.interface = MotorInterface(self.motor)
        self.horizon = horizon
        self.observation_space = gym.spaces.Box(-np.inf,np.inf,(73,),np.float32)
        self.action_space = gym.spaces.Box(-1.,1.,(16,),np.float32)
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        m = self.motor
        m.reset(seed=seed)
        m.data.qpos[0] = -.25
        m.pack.effort_limit[m.order.index('left_gripper_joint')] = .4
        mujoco.mj_forward(m.model,m.data)
        self.interface.reset()
        goal = self.interface.goal.copy()
        if options and 'goal' in options:
            goal = np.asarray(options['goal'])
        else:
            # Broad command conditioning, not a cleanup phase sequence.
            goal[:14] += self.np_random.uniform(-.08,.08,14)
            goal[14:16] = self.np_random.uniform([-.025,-.05],[.025,.05])
            if self.np_random.random() < .25:
                goal = self.interface.goal.copy()
        self.interface.set_goal(goal)
        self.finished = False
        self.row = None
        return self.interface.observe(),{}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset after episode termination')
        previous = self.interface.previous.copy()
        self.row = self.interface.advance(action)
        m = self.motor
        p,g,gyro,x,v,_ = m.motion()
        error = m.data.qpos[m.qidx[self.interface.positions]]-self.interface.goal[:14]
        fall = bool(g[2] > -.92)
        overspeed = bool(np.any(np.abs(m.data.qvel[m.vidx]) > m.limits*1.5))
        terminated = fall or overspeed
        truncated = m.steps >= self.horizon
        tracking = np.exp(-np.mean(error**2)/.04)
        velocity = np.exp(-((v-self.interface.goal[14])/.05)**2)
        yaw = np.exp(-((gyro[2]-self.interface.goal[15])/.2)**2)
        station = np.exp(-((x-m.reference_x)/.1)**2)
        reward = float(1 + tracking + velocity + .3*yaw + .3*station
                       -.01*np.mean((self.interface.previous-previous)**2) - 5*terminated)
        self.finished = terminated or truncated
        return self.interface.observe(),reward,terminated,truncated,{
            'fallen':fall,'overspeed':overspeed,'posture_rmse_rad':float(np.sqrt(np.mean(error**2))),
            'velocity_error_m_s':float(v-self.interface.goal[14]),'pitch_rad':float(p)}

    def close(self):
        self.motor.close()


class LearnedMotor:
    def __init__(self, motor, checkpoint, contract, plant_reference=None):
        from stable_baselines3 import PPO
        self.interface = MotorInterface(motor)
        actual=self.interface.contract()
        differing=[key for key in set(actual)|set(contract) if actual.get(key)!=contract.get(key)]
        self.validation={'status':'pass','mode':'exact_contract'}
        if differing:
            if differing != ['plant_parameters_sha256'] or plant_reference is None:
                raise ValueError('Learned motor checkpoint contract mismatch: '+','.join(differing))
            with np.load(plant_reference,allow_pickle=False) as reference:
                self.validation=verify_portable_plant(plant_parameters(motor),reference,contract['plant_parameters_sha256'])
            print('Motor plant comparison: '+json.dumps(self.validation),flush=True)
            if self.validation['status']!='pass':
                raise ValueError('Motor plant parameter mismatch; see retained comparison')
        self.policy = PPO.load(checkpoint,device='cpu')

    def reset(self):
        self.interface.reset()

    def set_goal(self, goal):
        return self.interface.set_goal(goal)

    def advance(self):
        import time
        observation=self.interface.observe()
        acquired=float(self.interface.motor.data.time)
        started=time.perf_counter()
        action = self.policy.predict(observation,deterministic=True)[0]
        latency=time.perf_counter()-started
        row=self.interface.advance(action)
        row.update(motor_observation=observation.tolist(),motor_observation_time_s=acquired,
                   motor_inference_s=latency)
        return row


if __name__=='__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Export trained-plant numeric reference for native worker portability checks')
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    env=MotorTrackingEnv(args.scene)
    try:
        env.reset(seed=1)
        if env.interface.contract()!=json.loads(args.contract.read_text()):
            raise ValueError('Local plant does not match retained training contract')
        with args.output.open('xb') as stream:
            np.savez_compressed(stream,**plant_parameters(env.motor))
    finally:env.close()
