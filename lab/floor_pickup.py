"""Task-specific whole-body reference probe on the existing sealed MuJoCo robot.

An authored controller, not a learned cleanup policy. Object motion is exclusively
MuJoCo contact dynamics. Encoder-based model trim/feedforward uses no object truth.
"""
from pathlib import Path
from xml.etree import ElementTree as ET
import json
import mujoco
import numpy as np
from lab.mujoco_learning import PipMotorEnv, BUNDLE


def build_scene(path, source=(.015,.140,.020), friction=.8, block_size=.040, robot_model=None):
    root=ET.parse(robot_model or BUNDLE/'model/robot.xml').getroot()
    for mesh in root.findall('./asset/mesh'):
        mesh.set('file', str((BUNDLE/'model'/mesh.get('file')).absolute()))
    world=root.find('worldbody')
    block=ET.SubElement(world,'body',name='task_block',pos=' '.join(map(str,source)))
    ET.SubElement(block,'freejoint',name='task_block_free')
    ET.SubElement(block,'geom',name='task_block_geom',type='box',size=' '.join([str(block_size/2)]*3),mass='.05',rgba='.95 .40 .05 1')
    contact=root.find('contact')
    for finger in ['left_forearm_1','left_gripper_0']:
        ET.SubElement(contact,'pair',geom1=finger,geom2='task_block_geom',condim='1' if friction==0 else '3',friction=f'{friction} {friction} 0 0 0')
    ET.SubElement(world,'geom',name='destination',type='box',size='.075 .075 .0002',pos='.365 .140 .0002',contype='0',conaffinity='0',rgba='.15 .65 .4 .5')
    ET.ElementTree(root).write(path)


class FloorProbe(PipMotorEnv):
    def __init__(self, model_path, leg_cap=.7):
        super().__init__(model_path=model_path,command=[0,0,0],episode_steps=2500)
        self.kin=mujoco.MjData(self.model)
        self.leg_cap=leg_cap
        for i,n in enumerate(self.order):
            if n.endswith(('_thigh_joint','_shin_joint')):
                self.pack.effort_limit[i]=leg_cap
                self.model.actuator_ctrlrange[self.aidx[i]]=[-leg_cap,leg_cap]
        self.block=self.model.body('task_block').id
        self.left_arm=[self.order.index('left_'+n+'_joint') for n in ['shoulder_roll','upper_arm','forearm']]
        self.max_contacts=0
        self.heading_error=0.

    def hand(self,data=None):
        d=data or self.data
        b=self.model.body('left_forearm').id
        local=np.array([0,.002,self.model.geom_pos[self.model.geom('left_forearm_1').id,2]])
        return d.xpos[b]+d.xmat[b].reshape(3,3)@local

    def model_reference(self):
        # Kinematic twin from encoders only, upright root, zero velocities.
        m,k=self.model,self.kin
        mujoco.mj_resetData(m,k)
        k.qpos[self.qidx]=self.data.qpos[self.qidx]
        mujoco.mj_forward(m,k)
        wheels=[m.body(s+'_wheel').id for s in ['left','right']]
        axle=np.mean(k.xpos[wheels],axis=0)
        com=k.subtree_com[self.torso]
        delta=com-axle
        trim=-np.arctan2(delta[0],delta[2])
        # Quasistatic support load; excludes unknown payload. Same native Jacobian
        # calculation as the retained floor preflight, not an external base force.
        force=np.zeros(m.nv)
        y0,y1=k.xpos[wheels,1]
        f=np.clip((com[1]-y1)/(y0-y1),0,1)
        for b,w in zip(wheels,[f,1-f]):
            point=k.xpos[b].copy();point[2]-=.04
            jac=np.zeros((3,m.nv));mujoco.mj_jac(m,k,jac,None,point,b)
            force+=jac[2]*m.body_subtreemass[self.torso]*9.81*w
        ff=(k.qfrc_bias-force)[self.vidx];ff[self.wheels]=0
        return trim,ff

    def advance(self,target,command=0.,residual=None):
        target=np.asarray(target,float).copy()
        self.command=np.array([float(command),0,0])
        self.raw=np.zeros(4) if residual is None else np.clip(residual,-1,1)
        p,_,gyro,x,v,_=self.motion()
        trim,ff=self.model_reference()
        self.reference_x+=command*.02
        gains=self.settings['balance']
        speed=gains['pitch_gain_m_s']*(p-trim)+gains['pitch_rate_gain_m']*gyro[1]+gains['position_gain_s_inv']*(x-self.reference_x)+gains['velocity_gain']*(v-command)+command
        for side,sign,idx in [('left',-1,0),('right',1,1)]:
            target[self.order.index(side+'_wheel_joint')]=speed/.04+sign*(np.clip(-3*self.heading_error,-.15,.15)-gyro[2])+self.raw[idx]*.5
        return self.advance_servos(target, ff, trim)

    def advance_servos(self, target, feedforward=None, trim=None):
        """Shared actuator plant; no balance law or gravity compensation here."""
        target=np.asarray(target,float).copy()
        ff=np.zeros(16) if feedforward is None else np.asarray(feedforward,float)
        if target.shape != (16,) or ff.shape != (16,) or not np.isfinite(target).all() or not np.isfinite(ff).all():
            raise ValueError('Expected finite 16-joint servo and feedforward commands')
        for i,j in enumerate(self.joints):
            if j['continuous']:target[i]=np.clip(target[i],-self.limits[i],self.limits[i])
            else:
                target[i]=np.clip(target[i],*j['limits'])
                target[i]=self.targets[i]+np.clip(target[i]-self.targets[i],-.02*self.limits[i],.02*self.limits[i])
        self.targets=target
        qd=np.where(self.wheels,0,target);dqd=np.where(self.wheels,target,0)
        if not self.pack_seeded:
            self.pack._queue[:]=np.stack((qd,dqd,ff),axis=1);self.pack_seeded=True
        for _ in range(10):
            self.electrical=self.pack.step(self.data.qpos[self.qidx],self.data.qvel[self.vidx],qd,dqd,ff)
            self.data.ctrl[self.aidx]=self.electrical['applied_effort_nm']
            mujoco.mj_step(self.model,self.data)
        mujoco.mj_forward(self.model,self.data);self.steps+=1
        info=self.record();info.update(hand_m=self.hand().tolist(),block_m=self.data.xpos[self.block].tolist(),trim_rad=None if trim is None else float(trim))
        contacts={};penetrations=[]
        for c in self.data.contact:
            names=[self.model.geom(c.geom1).name,self.model.geom(c.geom2).name]
            if c.dist<-.002:penetrations.append([*names,float(c.dist)])
        for ci,c in enumerate(self.data.contact):
            names=[self.model.geom(c.geom1).name,self.model.geom(c.geom2).name]
            if 'task_block_geom' in names:
                frc=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,ci,frc)
                other=names[1-names.index('task_block_geom')];contacts[other]=contacts.get(other,0)+float(frc[0])
        info.update(block_contacts_n=contacts,penetrations=penetrations)
        return info


def orange_floor_estimate(rgb, fovy, camera_origin, camera_rotation):
    """Known orange cube, flat floor: RGB centroid ray at known 15-mm centre height.

    No depth buffer, segmentation IDs, object pose, or environment seed is read.
    Pose derives from encoder/IMU/odometry FK. This is not semantic object vision.
    """
    import cv2
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    mask=cv2.inRange(hsv,np.array([3,125,65]),np.array([28,255,255]))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours=[c for c in contours if cv2.contourArea(c)>=10]
    if len(contours)!=1:return None
    x,y,w,h=cv2.boundingRect(contours[0]);height,width=rgb.shape[:2]
    if x<=0 or y<=0 or x+w>=width or y+h>=height:return None
    f=.5*height/np.tan(np.deg2rad(fovy)/2)
    ray=np.asarray(camera_rotation)@np.array([(x+(w-1)/2-(width-1)/2)/f,-(y+(h-1)/2-(height-1)/2)/f,-1.])
    origin=np.asarray(camera_origin)
    if ray[2]>=-.05:return None
    distance=(.015-origin[2])/ray[2]
    point=origin+distance*ray
    return {'position_m':point.tolist(),'bbox_xywh':[x,y,w,h], 'method':'RGB orange centroid ray + known cube centre plane + encoder/IMU/odometry camera pose','assumed_plane_z_m':.015}


class CleanupReference:
    """Authored, state-triggered skill order; closed-loop motor/model controller.

    Transition observations are encoders, IMU, odometry and a current-based grasp
    proxy. Ground-truth block pose/contact never enters this controller.
    """
    def __init__(self,motor,origin_x=-.25,goal_x=.365):
        self.motor=motor;self.origin_x=origin_x;self.goal_x=goal_x
        self.phase='find';self.phase_started=0.;self.estimate=None;self.yaw=0.;self.age=0.;self.hold=0.
        self.q=motor.rest.copy();self.target=self.q.copy();self.bend_fraction=0.;self.station_x=None
        source=json.loads((Path(__file__).resolve().parents[1]/'artifacts/floor-cleanup-preflight-18686e4733a4/report.json').read_text())['models']['wheels']['joints']
        self.crouch=np.array([source[n] for n in motor.order]);self.up=self.crouch.copy()
        for side in ['left','right']:
            self.up[motor.order.index(side+'_thigh_joint')]=0
            self.up[motor.order.index(side+'_shin_joint')]=0
        self.grip=motor.order.index('left_gripper_joint');self.elbow=motor.order.index('left_forearm_joint')
        self.head=motor.order.index('head_joint');self.neck=motor.order.index('neck_yaw_joint')
        self.knees=[motor.order.index(s+'_shin_joint') for s in ['left','right']]
        self.camera=mujoco.Renderer(motor.model,height=180,width=320);self.rgb=None
        self.frame_estimate=None;self.sensor_pose=mujoco.MjData(motor.model)

    def proprioception(self):
        m=self.motor;p,g,gyro,x,v,vel=m.motion()
        k=self.sensor_pose;mujoco.mj_resetData(m.model,k);k.qpos[m.qidx]=m.data.qpos[m.qidx]
        # Simulated IMU gravity supplies pitch; integrated gyro supplies heading.
        from scipy.spatial.transform import Rotation
        rotation=Rotation.from_euler('ZY',[self.yaw,p]);xyzw=rotation.as_quat()
        k.qpos[3:7]=[xyzw[3],*xyzw[:3]];k.qpos[0]=self.origin_x+x
        mujoco.mj_forward(m.model,k)
        k.qpos[2]+=.04-np.mean([k.body(s+'_wheel').xpos[2] for s in ['left','right']]);mujoco.mj_forward(m.model,k)
        return {'world_x_m':self.origin_x+x,'pitch':p,'gyro':gyro,'hand':m.hand(k),'velocity':v}

    def capture(self):
        m=self.motor;self.proprioception()
        self.camera.update_scene(m.data,camera='head_camera');self.rgb=self.camera.render().copy()
        c=m.model.camera('head_camera').id;k=self.sensor_pose
        self.frame_estimate=orange_floor_estimate(self.rgb,m.model.cam_fovy[c],k.cam_xpos[c],k.cam_xmat[c].reshape(3,3))
        if self.frame_estimate:self.frame_estimate['acquired_s']=float(m.data.time)
        if self.phase=='find' and self.frame_estimate and m.data.time>2.:
            self.estimate=dict(self.frame_estimate);self.enter('approach')
        return self.frame_estimate

    def enter(self,phase):
        self.phase=phase;self.phase_started=float(self.motor.data.time);self.hold=0.

    def command(self, grip_target=.45, arm_residual=0.):
        m=self.motor;obs=self.proprioception();self.yaw+=float(obs['gyro'][2])*.02
        self.age=float(m.data.time)-self.phase_started
        measured=m.data.qpos[m.qidx]
        desired=self.up.copy();desired[self.elbow]=-.7;speed=0.;grip=-.45;rate=.65
        if self.phase=='find':
            desired[self.head]=.5;desired[self.neck]=.65
        elif self.phase=='approach':
            error=self.estimate['position_m'][0]-.015-obs['world_x_m']
            speed=float(np.clip(2*error,-.06,.06))
            if abs(error)<.004 and abs(obs['velocity'])<.02:
                self.station_x=self.estimate['position_m'][0]-.015;self.enter('bend')
        elif self.phase=='bend':
            desired=self.crouch.copy();desired[self.elbow]=-.7;rate=.35
            if max(abs(measured[self.knees]-desired[self.knees]))<.03:self.enter('reach')
        elif self.phase=='reach':
            desired=self.crouch.copy();rate=.35
            error=self.estimate['position_m'][0]-obs['hand'][0]
            self.station_x=obs['world_x_m']+error
            if abs(measured[self.elbow]-desired[self.elbow])<.02 and abs(error)<.001 and abs(obs['velocity'])<.008:self.enter('grasp')
        elif self.phase=='grasp':
            desired=self.crouch.copy();grip=grip_target
            # A future motor-current/encoder proxy, not a deployed tactile sensor.
            obstruction=abs(m.electrical['applied_effort_nm'][self.grip])>.15 and -.05<measured[self.grip]<.25
            self.hold=self.hold+.02 if obstruction else 0
            if self.hold>=.3:self.enter('lift')
        elif self.phase in ['lift','carry']:
            desired=self.up.copy();grip=grip_target
            desired[self.elbow]+=arm_residual
            if self.phase=='lift' and max(abs(measured[self.knees]))<.04:self.enter('carry')
            if self.phase=='carry':
                error=self.goal_x-.015-obs['world_x_m']
                speed=float(np.clip(3*error,-.06,.06))
                if abs(error)<.008 and abs(obs['velocity'])<.03:
                    self.station_x=self.goal_x-.015;self.enter('lower')
        elif self.phase=='lower':
            desired=self.crouch.copy();grip=grip_target
            if obs['hand'][2]<.023:self.enter('release')
        elif self.phase in ['release','settle','done']:
            desired=self.crouch.copy();grip=-.45
            if self.phase=='release' and measured[self.grip]<-.40:self.enter('settle')
            if self.phase=='settle' and self.age>.8:self.enter('done')
        desired[self.grip]=grip
        self.target+=np.clip(desired-self.target,-rate*.02,rate*.02)
        # Faster finger closure within the retained velocity cap; no object force.
        self.target[self.grip]+=np.clip(desired[self.grip]-self.target[self.grip],-.04,.04)
        want_bend=float(self.phase in ['bend','reach','grasp','lower','release','settle','done'])
        bend_rate=.18 if self.phase in ['bend','reach','grasp'] else .35
        self.bend_fraction+=float(np.clip(want_bend-self.bend_fraction,-bend_rate*.02,bend_rate*.02))
        for side in ['left','right']:
            for part in ['thigh','shin']:
                index=m.order.index(side+'_'+part+'_joint');self.target[index]=self.crouch[index]*self.bend_fraction
        if self.station_x is not None and self.phase not in ['approach','carry']:
            m.reference_x=self.station_x-self.origin_x
        return self.target.copy(),speed,obs

    def close(self):self.camera.close()


def evaluate_cleanup(rows,goal=(.365,.14),initial=(.015,.14,.015)):
    """Independent contact/transport/release evaluator; truth never commands motors."""
    if not rows:return {'success':False,'reason':'empty_episode'}
    lifted=False;grasp_hold=0.;max_hold=0.;drop=False;placed_supported=False;release_seen=False;settle=0.;max_lift=0.;transport=0.;max_speed=0.;fall=False
    previous_t=0.
    for row in rows:
        dt=row['time_s']-previous_t;previous_t=row['time_s'];p=np.asarray(row['block_m']);contacts=row['block_contacts_n'];phase=row.get('phase','')
        bilateral=all(contacts.get(n,0)>.05 for n in ['left_forearm_1','left_gripper_0'])
        grasp_hold=grasp_hold+dt if bilateral else 0.;max_hold=max(max_hold,grasp_hold)
        max_lift=max(max_lift,float(p[2]-initial[2]));lifted|=bool(p[2]>initial[2]+.02 and max_hold>=.15)
        if lifted:transport=max(transport,float(np.linalg.norm(p[:2]-np.asarray(initial[:2]))))
        inside=bool(np.all(abs(p[:2]-goal)<.06))
        floor=contacts.get('floor',0)>.05
        if lifted and not placed_supported and not bilateral:
            # Once contact support is handed to the floor in the lower/release
            # phase, finger opening is expected. An airborne detached block is a drop.
            if not (floor and inside and phase in ['lower','release','settle','done']):drop=True
        if lifted and floor and inside and phase=='lower' and bilateral:placed_supported=True
        if placed_supported and phase in ['release','settle','done'] and not bilateral:release_seen=True
        speed=float(np.linalg.norm(row.get('block_velocity_m_s',[1,1,1])))
        settled=bool(inside and floor and speed<.02 and release_seen and not bilateral)
        settle=settle+dt if settled else 0.
        fall|=abs(row['pitch_rad'])>.28
        max_speed=max(max_speed,max(abs(np.asarray(row['dq']))))
    checks={'visible_target_acquired':any(r.get('estimate') for r in rows),'bilateral_grasp':max_hold>=.15,'lift':lifted,'transport_at_least_30cm':transport>=.30,'no_drop':not drop,'supported_set_down':placed_supported,'released_and_settled':settle>=.5,'no_fall':not fall,'joint_speed_within_limit':bool(max_speed<=4.375)}
    return {'success':all(checks.values()),'checks':checks,'max_lift_m':max_lift,'max_transport_m':transport,'max_bilateral_hold_s':max_hold,'final_settle_s':settle,'max_joint_speed_rad_s':float(max_speed)}
