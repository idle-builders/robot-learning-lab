"""Camera-driven gentle-interaction proxy task over Pip's frozen motor policy.

Perception is an authored, known-size coloured marker detector, not cat recognition.
Only the evaluator/reward reads proxy truth. No robot pose or path is assigned.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import cv2
import gymnasium as gym
import mujoco
import numpy as np
from lab.mujoco_learning import PipMotorEnv, BUNDLE, isaac_actor

SKILLS = ('pause', 'approach', 'retreat')
RADIUS = .045
TARGET_DISTANCE = .50


def build_scene(path):
    root = ET.parse(BUNDLE/'model/robot.xml').getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((BUNDLE/'model'/mesh.get('file')).absolute()))
    body = ET.SubElement(root.find('worldbody'), 'body', name='interaction_proxy', mocap='true', pos='.68 0 .30')
    ET.SubElement(body, 'geom', name='proxy_marker', type='sphere', size=str(RADIUS), rgba='1 0 0.8 1', contype='1', conaffinity='1')
    ET.SubElement(body, 'geom', name='proxy_stand', type='capsule', fromto='0 0 -.23 0 0 -.06', size='.01', rgba='.3 .3 .3 1')
    ET.ElementTree(root).write(path)


def estimate_marker(rgb, fovy_deg, radius=RADIUS):
    """Metric camera estimate from RGB only, with known marker radius/intrinsics."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([140, 100, 70]), np.array([175, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 12]
    if len(contours) != 1:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])
    if min(w,h) < 4 or x <= 0 or y <= 0 or x+w >= rgb.shape[1] or y+h >= rgb.shape[0]:
        return None
    focal = .5*rgb.shape[0]/np.tan(np.deg2rad(fovy_deg)/2)
    forward = 2*radius*focal/w
    left = -(x+(w-1)/2-(rgb.shape[1]-1)/2)*forward/focal
    return {'forward_m':float(forward), 'left_m':float(left), 'bbox_xywh':[x,y,w,h],
            'range_resolution_m':float(forward/w), 'method':'known-radius colour marker; uncalibrated pinhole'}


def skill_command(action, estimate):
    if action not in range(3):
        raise ValueError('Unknown interaction skill')
    if estimate is None:
        return np.zeros(3), 'missing_vision'
    # Authored low-level steering and distance override are disclosed; learned
    # selection decides pause/approach/retreat without a timed robot trajectory.
    if estimate['forward_m'] < .32:
        return np.array([-.035,0.,0.]), 'minimum_range_retreat'
    yaw = np.clip(1.5*np.arctan2(estimate['left_m'],estimate['forward_m']),-.15,.15)
    return np.array([(0.,.05,-.035)[action],0.,yaw]), 'none'


class InteractionEnv(gym.Env):
    metadata = {'render_modes':['rgb_array'],'render_fps':5}

    def __init__(self, scene, *, horizon_s=12., capture=False):
        self.motor = PipMotorEnv(command=[0.,0.,0.], model_path=scene, episode_steps=round(horizon_s*50))
        self.actor = isaac_actor()
        self.camera = mujoco.Renderer(self.motor.model,height=180,width=320)
        self.camera_id = self.motor.model.camera('head_camera').id
        self.proxy_id = self.motor.model.body('interaction_proxy').id
        self.mocap_id = self.motor.model.body_mocapid[self.proxy_id]
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-10.,10.,(4,),np.float32)
        self.horizon_s = horizon_s
        self.capture_rows = capture
        self.rows=[]

    def capture(self):
        self.camera.update_scene(self.motor.data,camera='head_camera')
        self.rgb = self.camera.render().copy()
        if self.blackout and 4. <= self.motor.data.time < 5.:
            self.rgb[:]=0
        self.estimate = estimate_marker(self.rgb,self.motor.model.cam_fovy[self.camera_id])
        if self.estimate is not None: self.estimate['acquired_s']=float(self.motor.data.time)
        e = self.estimate
        # No time, scene seed, ground-truth pose, or desired action enters policy.
        return np.array([float(e is not None),e['forward_m'] if e else 0.,e['left_m'] if e else 0.,0.],np.float32)

    def truth(self):
        d=self.motor.data
        delta=d.xpos[self.proxy_id]-d.cam_xpos[self.camera_id]
        optical=d.cam_xmat[self.camera_id].reshape(3,3).T@delta
        return float(-optical[2]),float(-optical[0])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.motor.reset(seed=seed)
        options=options or {}
        # Balanced near/band/far starts teach retreat as well as approach.
        lo,hi=((.36,.44),(.47,.54),(.60,.78))[int(self.np_random.choice(3,p=[.4,.2,.4]))]
        self.start = np.array([options.get('forward_m',self.np_random.uniform(lo,hi))+.066,
                               options.get('left_m',self.np_random.uniform(-.08,.08)),.30])
        self.blackout=bool(options.get('blackout',False))
        self.intrusion=bool(options.get('intrusion',False))
        self.motor.data.mocap_pos[self.mocap_id]=self.start
        mujoco.mj_forward(self.motor.model,self.motor.data)
        self.rows=[];self.done=False;self.dwell_s=0.;self.ever_unsafe=False
        self.previous_error=abs(self.truth()[0]-TARGET_DISTANCE)
        return self.capture(),{}

    def step(self, action):
        if self.done: raise RuntimeError('Reset a completed task before stepping')
        if not self.action_space.contains(action):raise ValueError('Expected a discrete interaction skill')
        decision_time = float(self.motor.data.time)
        observed = None if self.estimate is None else dict(self.estimate)
        command,override=skill_command(int(action),observed)
        self.motor.fixed_command=command.copy();self.motor.command=command.copy()
        fallen=overspeed=False
        for _ in range(10):
            if self.intrusion:
                # Environmental disturbance, never a robot command.
                self.motor.data.mocap_pos[self.mocap_id]=self.start+[-.12*np.clip((self.motor.data.time-5)/2,0,1),0,0]
            motor_input=self.motor.observe()
            _,_,terminated,truncated,info=self.motor.step(self.actor(motor_input))
            fallen|=info['fallen'];overspeed|=info['overspeed']
            if terminated or truncated:break
        observation=self.capture();forward,left=self.truth()
        _,gravity,gyro,odometry_x,odometry_v,vel=self.motor.motion()
        error=abs(forward-TARGET_DISTANCE)
        band=abs(forward-TARGET_DISTANCE)<.06 and abs(left)<.08 and abs(vel[0])<.025
        self.dwell_s = self.dwell_s+.2 if band else 0.
        unsafe=forward<.30
        self.ever_unsafe |= unsafe
        reward=35*(self.previous_error-error)+(.12 if band else -.025)-.02*abs(command[0]/.05)-.03*abs(left)
        if unsafe or fallen or overspeed: reward-=5.
        self.previous_error=error
        self.done=bool(terminated or truncated or unsafe)
        info={'fallen':fallen,'overspeed':overspeed,'unsafe_separation':unsafe,'dwell_s':self.dwell_s,
              'success':bool(truncated and self.dwell_s>=1. and not self.ever_unsafe and not fallen and not overspeed)}
        if self.capture_rows:
            row=self.motor.record()
            row.update({'decision_time_s':decision_time,'motor_observation_last':motor_input.tolist(),
                        'imu_gyro_body_rad_s':gyro.tolist(),'gravity_body':gravity.tolist(),
                        'odometry_position_m':float(odometry_x),'odometry_speed_m_s':float(odometry_v),
                        'skill_id':int(action),'skill':SKILLS[int(action)],'estimate_before_action':observed,
                        'estimate':self.estimate,'truth_forward_m':forward,'truth_left_m':left,
                        'vision_error_m':self.estimate['forward_m']-forward if self.estimate else None,
                        'override':override,'reward':float(reward),'dwell_s':self.dwell_s,'success':info['success'],
                        'unsafe_separation':unsafe,'fallen':fallen,'overspeed':overspeed,
                        'proxy_world_m':self.motor.data.xpos[self.proxy_id].tolist(),
                        'yaw_rad_s':float(gyro[2]),'observation':observation.tolist()})
            self.rows.append(row)
        return observation,float(reward),bool(terminated or unsafe),bool(truncated),info

    def render(self):
        if self.motor.renderer is None:self.motor.renderer=mujoco.Renderer(self.motor.model,height=360,width=640)
        cam=mujoco.MjvCamera();cam.lookat[:]=[.3,0,.2];cam.distance=1.15;cam.azimuth=115;cam.elevation=-18
        self.motor.renderer.update_scene(self.motor.data,camera=cam)
        return self.motor.renderer.render()

    def close(self):
        self.camera.close();self.motor.close()
