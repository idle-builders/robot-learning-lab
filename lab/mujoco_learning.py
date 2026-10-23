"""Small native Gymnasium adapter for the sealed Pip motor experiment.

Same SI observation/action semantics as the Isaac worker, separate physics backend.
Solver/contact/reset differences are experimental variables, not parity claims.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from lab.electromechanical import ElectromechanicalConfig, ElectromechanicalPack

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'artifacts/pip-cad-gym-bundle-v4'
ISAAC_CHECKPOINT = ROOT / 'artifacts/pip-realistic-gym-v4/evidence/control/logs/rsl_rl/pip_residual_drive/2026-09-21_20-25-54/model_49.pt'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PipMotorEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(self, *, command=None, seed=23, render_mode=None, model_path=None, episode_steps=400):
        if not isinstance(episode_steps, int) or episode_steps < 1:
            raise ValueError('episode_steps must be a positive integer')
        self.episode_steps = episode_steps
        self.interface = json.loads((BUNDLE/'model/interfaces.json').read_text())
        self.config = json.loads((BUNDLE/'model/resolved-config.json').read_text())
        self.settings = json.loads((ROOT/'learning/pip-control-v1.json').read_text())
        self.order = self.interface['joint_order']
        self.joints = self.interface['joints']
        self.model = mujoco.MjModel.from_xml_path(str(model_path or BUNDLE/'model/robot.xml'))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = .002
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.qidx = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.order])
        self.vidx = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in self.order])
        self.aidx = np.array([self.model.actuator(n+'_motor').id for n in self.order])
        self.torso = self.model.body('torso').id
        self.wheels = np.array([j['continuous'] for j in self.joints])
        self.limits = np.array([j['velocity_limit_rad_s'] for j in self.joints])
        self.rest = np.array([self.settings['arm_rest_targets_rad'].get(n, 0.) for n in self.order])
        electrical = json.loads((ROOT/'learning/pip-electromechanical-v1.json').read_text())
        cfg = ElectromechanicalConfig(dt_s=.002, delay_steps=electrical['delay_steps'],
            **electrical['pack_rc'], **electrical['servo_proxy'])
        self.pack = ElectromechanicalPack(np.where(self.wheels, 0., self.config['control']['kp']),
            np.where(self.wheels, self.settings['balance']['wheel_velocity_gain_nm_s'], self.config['control']['kd']),
            [j['effort_limit_nm'] for j in self.joints], self.limits, cfg)
        self.fixed_command = None if command is None else self.validate_command(command)
        self.action_space = gym.spaces.Box(-1., 1., (4,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (47,), np.float32)
        self.render_mode = render_mode
        self.renderer = None
        self.reset(seed=seed)

    @staticmethod
    def validate_command(command):
        value = np.asarray(command, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all() or abs(value[0]) > .06 or value[1] != 0 or abs(value[2]) > .15:
            raise ValueError('Command must be [forward m/s, 0, yaw rad/s] within ±0.06 and ±0.15')
        return value.copy()

    def sample_command(self):
        self.command = self.fixed_command.copy() if self.fixed_command is not None else np.array([
            self.np_random.uniform(-.06,.06), 0., self.np_random.uniform(-.15,.15)])
        if self.fixed_command is None and self.np_random.random() < .3: self.command[:] = 0
        self.next_command = self.steps + int(self.np_random.uniform(2.,4.)/.02)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qidx] = self.rest
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.done = False
        self.raw = np.zeros(4)
        self.reference_x = 0.
        self.targets = self.rest.copy()
        self.pack.reset()
        # Match native Isaac startup: fill delay history with the first target.
        self.pack_seeded = False
        self.electrical = {'voltage_v':self.pack.voltage_v, 'total_current_a':0., 'energy_j':0., 'applied_effort_nm':np.zeros(16)}
        self.sample_command()
        return self.observe(), {}

    def motion(self):
        q, dq = self.data.qpos[self.qidx], self.data.qvel[self.vidx]
        rotation = self.data.xmat[self.torso].reshape(3,3)
        gravity = rotation.T @ np.array([0.,0.,-1.])
        spatial = np.zeros(6)
        # mj_objectVelocity's local BODY basis is the inertial/COM frame. The
        # policy expects the link frame, which differs for our full inertia tensor.
        mujoco.mj_objectVelocity(self.model,self.data,mujoco.mjtObj.mjOBJ_BODY,self.torso,spatial,0)
        gyro = rotation.T @ spatial[:3]
        pitch = np.arctan2(gravity[0],-gravity[2])
        def mean(suffix, values): return np.mean([values[self.order.index(side+suffix)] for side in ('left','right')])
        h,k,w = [mean(s,q) for s in ('_thigh_joint','_shin_joint','_wheel_joint')]
        hd,kd,wd = [mean(s,dq) for s in ('_thigh_joint','_shin_joint','_wheel_joint')]
        d=self.config['dimensions']; p=pitch; pd=gyro[1]
        x=d['wheel_radius_m']*(w+h+k+p)+d['hip_drop_m']*np.sin(p)+d['thigh_m']*np.sin(p+h)+d['shin_m']*np.sin(p+h+k)
        v=d['wheel_radius_m']*(wd+hd+kd+pd)+d['hip_drop_m']*np.cos(p)*pd+d['thigh_m']*np.cos(p+h)*(pd+hd)+d['shin_m']*np.cos(p+h+k)*(pd+hd+kd)
        return pitch, gravity, gyro, x, v, rotation.T @ self.data.qvel[:3]

    def observe(self):
        _,g,gyro,x,v,_ = self.motion()
        return np.concatenate((self.data.qpos[self.qidx],self.data.qvel[self.vidx],g,gyro,
            self.command,[x-self.reference_x,v],self.raw)).astype(np.float32)

    def step(self, action):
        if self.done: raise RuntimeError('Reset after episode termination or truncation')
        action=np.asarray(action,dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError('Expected four finite residuals; invalid action does not advance physics')
        previous_raw=self.raw.copy(); self.raw=np.clip(action,-1,1)
        p,_,gyro,x,v,_=self.motion()
        self.reference_x += self.command[0]*.02
        gains=self.settings['balance']
        speed=gains['pitch_gain_m_s']*(p-self.settings['pitch_trim_rad'])+gains['pitch_rate_gain_m']*gyro[1]+gains['position_gain_s_inv']*(x-self.reference_x)+gains['velocity_gain']*(v-self.command[0])+self.command[0]
        steering=(self.command[2]-gyro[2])*self.settings['yaw_rate_gain']
        target=self.rest.copy()
        for side,sign,idx in [('left',-1,0),('right',1,1)]:
            target[self.order.index(side+'_wheel_joint')]=speed/self.config['dimensions']['wheel_radius_m']+sign*steering+self.raw[idx]*self.settings['wheel_residual_rad_s']
            target[self.order.index(side+'_thigh_joint')]+=self.raw[2]*self.settings['leg_residual_rad']
            target[self.order.index(side+'_shin_joint')]+=self.raw[3]*self.settings['leg_residual_rad']
        for i,j in enumerate(self.joints):
            if j['continuous']: target[i]=np.clip(target[i],-self.limits[i],self.limits[i])
            else:
                target[i]=np.clip(target[i],*j['limits'])
                target[i]=self.targets[i]+np.clip(target[i]-self.targets[i],-self.limits[i]*.02,self.limits[i]*.02)
        self.targets=target
        qd=np.where(self.wheels,0.,target); dqd=np.where(self.wheels,target,0.)
        if not self.pack_seeded:
            self.pack._queue[:]=np.stack((qd,dqd,np.zeros(16)),axis=1)
            self.pack_seeded=True
        for _ in range(10):
            self.electrical=self.pack.step(self.data.qpos[self.qidx],self.data.qvel[self.vidx],qd,dqd)
            self.data.ctrl[self.aidx]=self.electrical['applied_effort_nm']
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data)
        self.steps+=1
        p,g,gyro,_,_,vel=self.motion()
        fallen=bool(g[2]>-.96)
        overspeed=bool(np.any(abs(self.data.qvel[self.vidx])>self.limits*1.25))
        terminated=fallen or overspeed
        truncated=self.steps>=self.episode_steps
        track=np.exp(-((vel[0]-self.command[0])**2/.01+(gyro[2]-self.command[2])**2/.25))
        reward=.02*(float(not terminated)+2*track+np.exp(-((p-self.settings['pitch_trim_rad'])/.1)**2)
            -.005*np.sum(self.electrical['applied_effort_nm']**2)-.01*np.sum((self.raw-previous_raw)**2)-10*terminated)
        info={'fallen':fallen,'overspeed':overspeed,'tracking_error_m_s':float(vel[0]-self.command[0])}
        if self.steps>=self.next_command: self.sample_command()
        self.done=terminated or truncated
        return self.observe(),float(reward),terminated,truncated,info

    def record(self):
        p,_,_,_,_,vel=self.motion(); e=self.electrical
        wxyz=self.data.qpos[3:7]
        return {'time_s':float(self.data.time),'q':self.data.qpos[self.qidx].tolist(),'dq':self.data.qvel[self.vidx].tolist(),
            'root_position_world_m':self.data.qpos[:3].tolist(),'root_quaternion_xyzw':[*wxyz[1:],wxyz[0]],
            'command':self.command.tolist(),'targets':self.targets.tolist(),'pitch_rad':float(p),'forward_m_s':float(vel[0]),
            'residuals':self.raw.tolist(),'electromechanical':{'bus_voltage_v':e['voltage_v'],'estimated_current_a':e['total_current_a'],
                'estimated_energy_j':e['energy_j'],'applied_effort_nm':e['applied_effort_nm'].tolist(),
                'computed_effort_nm':e.get('computed_effort_nm',np.zeros(16)).tolist()},
            'body_poses_world_xyz_xyzw':[[*self.data.xpos[i],*self.data.xquat[i,1:],self.data.xquat[i,0]] for i in range(1,self.model.nbody)]}

    def render(self):
        if self.renderer is None: self.renderer=mujoco.Renderer(self.model,height=360,width=640)
        cam=mujoco.MjvCamera();cam.lookat[:]=self.data.xpos[self.torso];cam.distance=.95;cam.azimuth=135;cam.elevation=-12
        self.renderer.update_scene(self.data,camera=cam)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None: self.renderer.close();self.renderer=None


def isaac_actor(path=ISAAC_CHECKPOINT):
    """Exact ELU/no-normalization actor from the pinned v3 Cartpole-derived config."""
    import torch
    if Path(path)==ISAAC_CHECKPOINT and digest(path)!='ed943ca88297c322215da5a1f6a8189b6d81ceab3b91ad7f48557eaf29fd0d93':
        raise ValueError('Isaac checkpoint hash mismatch')
    weights=torch.load(path,map_location='cpu',weights_only=True)['actor_state_dict']
    actor=torch.nn.Sequential(torch.nn.Linear(47,64),torch.nn.ELU(),torch.nn.Linear(64,64),torch.nn.ELU(),torch.nn.Linear(64,4))
    actor.load_state_dict({k.removeprefix('mlp.'):v for k,v in weights.items() if k.startswith('mlp.')},strict=True)
    actor.eval()
    def predict(observation):
        with torch.inference_mode(): return actor(torch.as_tensor(observation,dtype=torch.float32)).numpy()
    return predict
