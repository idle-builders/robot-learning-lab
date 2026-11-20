"""Named 22-channel policy/sensor adapter; no legacy tensor substitution.

Motor electrical/current values are catalog sensitivity estimates, not physical
telemetry. Sensor packets keep capture and delivery times separate.
"""
from collections import deque
import numpy as np
import mujoco
from common import *

class Runtime:
    def __init__(self, model, data, *, rail_V=12., command_delay_s=.02,
                 feedback_delay_s=.01, peak=False, seed=0, backlash_rad=0.):
        self.model,self.data=model,data
        self.contract=read(OUT/'actuators.json'); self.acts=self.contract['actuators']
        self.names=[a['joint']for a in self.acts]
        actual=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,int(model.actuator_trnid[i,0]))for i in range(model.nu)]
        if actual!=self.names:raise ValueError('Body/actuator ordering does not match this model contract')
        self.jids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n)for n in self.names])
        self.qids=model.jnt_qposadr[self.jids];self.vids=model.jnt_dofadr[self.jids]
        self.rail=rail_V;self.delay=command_delay_s;self.feedback_delay=feedback_delay_s;self.peak=peak
        self.backlash=backlash_rad;self.rng=np.random.default_rng(seed)
        self.pending=deque();self.feedback=deque();self.last=None;self.target=data.qpos[self.qids].copy()
        self.torque=np.zeros(22);self.current=np.zeros(22);self.saturated=np.zeros(22,dtype=bool)
        self.gravity_estimate=np.array([0.,0.,-1.]);self.last_imu_time=None;self.previous_command=self.target.copy()
        self.joint_quantum=2*np.pi/4096
        self.kp=np.array([a['simulation_kp_Nm_rad']for a in self.acts]);self.kd=np.array([a['simulation_kd_Nm_s_rad']for a in self.acts])
        self.imu=next(s for s in read(OUT/'sensors.json')['sensors']if s['name']=='imu_board')
        self.imu_rotation=np.array(self.imu['rotation_link_sensor'])
        self.accel_bias=self.rng.uniform(-.05,.05,3);self.gyro_bias=self.rng.uniform(-.003,.003,3)
        self.last_sensor_capture=-1.;self.next_imu=0.;self.next_joint=0.;self.sample=None;self.seq=0

    def command(self, names, targets_rad, *, capture_time_s=None):
        if list(names)!=self.names:raise ValueError('An action requires the exact ordered22 joint names')
        target=np.asarray(targets_rad,float)
        if target.shape!=(22,) or not np.all(np.isfinite(target)):raise ValueError('Expected22 finite output-joint radians')
        target=target.copy()
        for i,a in enumerate(self.acts):
            limits=a['source_axis']['range_deg']
            if limits is not None:target[i]=np.clip(target[i],*np.deg2rad(limits))
            else:target[i]=np.clip(target[i],-.35,.35) # Declared diagnostic command envelope, NOT a joint limit.
        t=self.data.time if capture_time_s is None else capture_time_s
        self.pending.append((t+self.delay,target));self.previous_command=target.copy()

    def update(self):
        t=self.data.time
        while self.pending and self.pending[0][0]<=t+1e-10:_,self.target=self.pending.popleft()
        q=self.data.qpos[self.qids];qd=self.data.qvel[self.vids]
        error=self.target-q
        error=np.sign(error)*np.maximum(0,abs(error)-self.backlash/2)
        raw=self.kp*error-self.kd*qd
        for i,a in enumerate(self.acts):
            voltage=min(self.rail,12.) if a['model']=='xm335'else self.rail
            scale=voltage/12.
            if a['model']=='c047':
                spec=self.contract['models']['c047'];stall=spec['stall_torque_Nm']*scale;speed=spec['no_load_speed_rad_s']*scale
                limit=stall if self.peak else spec['rated_torque_Nm']*scale
                lower=-min(limit,max(0,stall*(1+qd[i]/speed)));upper=min(limit,max(0,stall*(1-qd[i]/speed)))
                self.torque[i]=np.clip(raw[i],lower,upper)
                self.current[i]=spec['idle_current_A']+(spec['stall_current_A']-spec['idle_current_A'])*abs(self.torque[i])/max(stall,1e-9)
            else:
                curve=self.contract['models']['xm335']['performance_curve']
                speeds=np.array(curve['speed_rad_s'])*scale;torques=np.array(curve['torque_Nm'])*scale
                curve_limit=np.interp(abs(qd[i]),speeds[::-1],torques[::-1],left=torques[-1],right=0.)
                limit=curve_limit if self.peak else min(.2,curve_limit)
                self.torque[i]=np.clip(raw[i],-limit,limit)
                self.current[i]=np.interp(abs(self.torque[i]),torques,curve['current_A'],left=.017,right=curve['current_A'][-1])
            self.saturated[i]=abs(self.torque[i]-raw[i])>1e-9
        self.data.ctrl[:]=self.torque
        if t+1e-10>=self.next_imu:
            accel=self.data.sensor('imu_accel').data.copy()
            gyro=self.data.sensor('imu_gyro').data.copy()
            ns=self.imu['source_noise'];band=self.imu['simulation_noise']['equivalent_bandwidth_Hz']
            accel+=self.accel_bias+self.rng.normal(0,ns['accel_density_m_s2_sqrt_Hz']*np.sqrt(band),3)
            gyro+=self.gyro_bias+self.rng.normal(0,ns['gyro_density_rad_s_sqrt_Hz']*np.sqrt(band),3)
            a_body=self.imu_rotation@accel;g_body=self.imu_rotation@gyro
            dt=1/208 if self.last_imu_time is None else t-self.last_imu_time
            self.gravity_estimate-=np.cross(g_body,self.gravity_estimate)*dt
            if np.linalg.norm(a_body)>1e-5:
                self.gravity_estimate=.98*self.gravity_estimate+.02*(-a_body/np.linalg.norm(a_body))
            self.gravity_estimate/=np.linalg.norm(self.gravity_estimate)
            self.sample=dict(accel=accel,gyro_body=g_body,gravity=self.gravity_estimate.copy(),capture_time_s=t)
            self.last_imu_time=t;self.next_imu+=1/208
        if t+1e-10>=self.next_joint and self.sample is not None:
            packet=dict(sequence=self.seq,clock_domain='simulation_monotonic',capture_time_s=t,
                imu_capture_time_s=self.sample['capture_time_s'],host_receive_s=t+self.feedback_delay,
                q=np.round(q/self.joint_quantum)*self.joint_quantum,qd=qd.copy(),
                gyro=self.sample['gyro_body'].copy(),gravity=self.sample['gravity'].copy(),
                calibration='Simulated board-aligned die assumption; physical transform uncalibrated',health='synthetic_available')
            self.feedback.append(packet);self.seq+=1;self.next_joint+=.01
        while self.feedback and self.feedback[0]['host_receive_s']<=t+1e-10:self.last=self.feedback.popleft()

    def observation(self, command=(0.,0.,0.)):
        if self.last is None:return None
        p=self.last
        values=np.r_[p['q'],p['qd'],p['gyro'],p['gravity'],command,self.previous_command]
        return dict(values=values,layout='q22,qd22,gyro3,estimated_gravity3,command3,previous_accepted_target22',
            dimensions=75,joint_names=self.names,capture_time_s=p['capture_time_s'],
            age_s=self.data.time-p['capture_time_s'],imu_capture_time_s=p['imu_capture_time_s'],
            health=p['health'],pose_source='IMU complementary gravity estimate, no simulator root pose/velocity or object truth in actor tensor')

def capture_transform(model,data,site):
    sid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,site)
    T=np.eye(4);T[:3,:3]=data.site_xmat[sid].reshape(3,3);T[:3,3]=data.site_xpos[sid]
    return T

def metric_point(packet, point_optical, *, calibration_id, pose_at_capture, clock_epoch, now_s, max_age_s):
    """No stale/uncalibrated sample becomes a metric localization target."""
    identities = [calibration_id, clock_epoch, packet.get('calibration_id'), packet.get('clock_epoch')]
    if not all(isinstance(value, str) and value.strip() for value in identities):
        return dict(available=False, reason='Missing usable calibration or clock identity')
    if packet.get('health')!='available' or packet.get('calibration_id')!=calibration_id:
        return dict(available=False,reason='Missing sensor health or exact calibration identity')
    if packet.get('clock_epoch')!=clock_epoch or pose_at_capture is None:
        return dict(available=False,reason='Missing capture pose or clock epoch mismatch')
    try:
        capture, now, maximum_age = map(float, [packet['capture_time_s'], now_s, max_age_s])
        pose_capture = float(pose_at_capture['capture_time_s']) if isinstance(pose_at_capture, dict) else float('nan')
    except (KeyError, TypeError, ValueError):
        return dict(available=False, reason='Missing numeric capture, pose or age values')
    if not np.all(np.isfinite([capture, now, maximum_age, pose_capture])) or maximum_age < 0:
        return dict(available=False, reason='Nonfinite capture, pose or age values')
    age=now-capture
    if age<0 or age>maximum_age:return dict(available=False,reason='Sample outside caller-declared age envelope')
    if abs(pose_capture-capture)>1e-6:
        return dict(available=False,reason='Pose timestamp does not match capture')
    T=np.asarray(pose_at_capture.get('T_world_optical'),float)
    if T.shape!=(4,4) or not np.all(np.isfinite(T)):
        return dict(available=False,reason='Invalid capture transform')
    if not np.allclose(T[3], [0,0,0,1], atol=1e-8) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or abs(np.linalg.det(T[:3,:3])-1)>1e-6:
        return dict(available=False, reason='Capture transform is not a proper rigid frame')
    p=np.asarray(point_optical,float)
    if p.shape!=(3,)or not np.all(np.isfinite(p)):return dict(available=False,reason='Invalid optical point')
    return dict(available=True,point_world_m=(T@np.r_[p,1])[:3].tolist(),
                capture_time_s=packet['capture_time_s'],calibration_id=calibration_id,clock_epoch=clock_epoch)
