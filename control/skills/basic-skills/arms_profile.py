"""Source arm-only actor, filtered21 history, fixed current CAD-neutral support."""
import json
import numpy as np
import torch
import mujoco
from adapter import ROOT,WalkingInterface,ref
from crouch_policy import mean

AUDIT=ROOT/'data/basic-skills-retry-20261027/arms-source-audit/report.json'
PAWS=('ape_body_shell__left_smooth_paw','ape_body_shell__right_smooth_paw')


class Arms:
    action_width=21
    def __init__(self,case):
        self.mapping=WalkingInterface();self.offset=self.mapping.home.numpy().copy();self.case=case
        self.audit=json.loads(AUDIT.read_text());self.info=next(c for c in self.audit['controllers'] if c['skill']==('bilateral' if case=='bilateral-both' else 'independent'))
        checkpoint=ROOT/self.info['checkpoint']['path']
        if ref(checkpoint)['sha256']!=self.info['checkpoint']['sha256']:raise ValueError('Source arm checkpoint differs')
        self.saved=torch.load(checkpoint,map_location='cpu',weights_only=False);self.policy=self.saved['actor_state_dict'];self.ema=self.info['ema_alpha'];self.native_model=None;self.reset()
    def reset(self):
        self.previous_filtered=np.zeros(21,dtype=np.float32);self.filtered=None;self.input_previous=self.previous_filtered.copy()
        self.capture=None;self.heading=0.;self.time_s=0.;self.flags=np.zeros(2);self.alpha=np.zeros(2);self.packet75=np.full(75,np.nan)
    def observe(self,packet):
        self.input_previous=self.previous_filtered.copy()
        if packet is None:return np.zeros(72 if self.info['skill']=='bilateral' else 75,dtype=np.float32)
        v=packet['values'];self.packet75=v.copy();self.time_s=round((packet['capture_time_s']+packet['age_s'])/.02)*.02;raised=float(2<=self.time_s<6)
        self.flags=np.array([raised if self.case!='independent-source-right-current-L' else 0.,raised if self.case!='independent-source-left-current-R' else 0.])
        # Hidden reward-target slew is an annotation. The actor receives only flags.
        self.alpha=np.array([min(1.,max(0.,(self.time_s-2)/1.5)) if self.time_s<=6 else max(0.,1-(self.time_s-6)/1.5) for _ in range(2)])
        self.alpha*=np.array([self.case!='independent-source-right-current-L',self.case!='independent-source-left-current-R'])
        capture=packet['imu_capture_time_s']
        if self.capture is not None and capture>self.capture:self.heading+=float(np.dot(v[44:47],-v[47:50]))*(capture-self.capture)
        self.capture=capture;heading=np.arctan2(np.sin(-self.heading),np.cos(-self.heading));history=np.r_[self.previous_filtered,np.zeros(3,dtype=np.float32)]
        obs=self.mapping.observe(torch.tensor(v[:22][None],dtype=torch.float32),torch.tensor(v[22:44][None],dtype=torch.float32),torch.tensor(v[44:47][None],dtype=torch.float32),
            torch.tensor(v[47:50][None],dtype=torch.float32),torch.tensor(history[None]),torch.tensor([[self.flags[0],0.,heading]],dtype=torch.float32))[0].numpy()
        obs[6:27]+=self.offset # source virtual home + signed current q; relative zero at CAD-neutral
        return obs[:72].copy() if self.info['skill']=='bilateral' else np.r_[obs[:72],[self.flags[1],0.,0.]].astype(np.float32)
    def action(self,obs):
        with torch.inference_mode():raw=mean(self.policy,torch.tensor(obs[None],dtype=torch.float32))[0].numpy().copy()
        self.filtered=raw.copy() if self.filtered is None else self.ema*raw+(1-self.ema)*self.filtered
        source=self.offset.copy();source[[13,18]]=np.clip(self.filtered[[13,18]],-1.52,1.52)
        target=np.zeros(22)
        for r in self.mapping.rows:
            if r['source_index'] in (13,18):target[r['current_index']]=r['sign']*source[r['source_index']]
        self.previous_filtered=self.filtered.copy();return raw,self.filtered.copy(),source,target
    def control_info(self):
        return dict(pre_action_delayed_packet75=self.packet75,actual_previous_executed_filtered21=self.input_previous,
            source_anatomical_raise_flags_L_R=self.flags,source_hidden_alpha_annotation_L_R=self.alpha,
            source_hidden_alpha_note='Initial0 standing-entry retarget; annotation only. Alpha does not enter actor. Source equal stand/sit height initialization pathology not copied from old absolute rootheight.')
    def sample_native(self,model,data):
        if self.native_model is not model:
            self.native_model=model;self.paw_geoms=[model.geom(n).id for n in PAWS];self.paw_centers=[];self.forearms=[model.body(s+'_forearm').id for s in ('R','L')]
            for geom in self.paw_geoms:
                mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);num=int(model.mesh_vertnum[mesh]);verts=model.mesh_vert[start:start+num];self.paw_centers.append((verts.min(0)+verts.max(0))/2)
        points=np.array([data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3)@c for g,c in zip(self.paw_geoms,self.paw_centers)])
        floor=model.geom('floor').id;floorforce=np.zeros(2);selfforce=np.zeros(2);force=np.zeros(6)
        for i,contact in enumerate(data.contact):
            bodies=[int(model.geom_bodyid[g]) for g in [contact.geom1,contact.geom2]]
            for side,body in enumerate(self.forearms):
                if body in bodies:
                    mujoco.mj_contactForce(model,data,i,force)
                    if floor in (contact.geom1,contact.geom2):floorforce[side]+=max(0.,force[0])
                    else:selfforce[side]+=max(0.,force[0])
        return dict(paw_visual_AABB_center_proxy_m_R_L=points,distal_forearm_floor_normal_N_R_L=floorforce,distal_forearm_self_normal_N_R_L=selfforce)
    def proof(self):
        sample=torch.randn(128,21,generator=torch.Generator().manual_seed(140000)).numpy();s=None;worst=0.
        for raw in sample:
            s=raw if s is None else self.ema*raw+(1-self.ema)*s
            physical=np.array([-np.clip(s[13],-1.52,1.52),-np.clip(s[18],-1.52,1.52)])
            worst=max(worst,float(np.max(abs(physical+np.clip(s[[13,18]],-1.52,1.52)))))
        return dict(source=self.info,actor_width=72 if self.info['skill']=='bilateral' else 75,normalizer_count=int(self.policy['obs_normalizer.count']),
            nonzero_EMA_rows=128,source_filtered_shoulder_decoder_error_rad=worst,physical_authority=2,
            history='All21 previousEMA outputs, including19zero-physical-scale heads. No raw clamp; first valid action bypassesEMA as source. No normalizer updates.',
            current_support='All20 other current axes fixed0 radians through selected Runtime. Source virtual leg homes translated to current CAD-neutral for both observation and physical support; original source absolute fixedlegpose is not applied.',
            sensor='Delayed selected packets with encoder/IMU noise, no added historical trainingcorruption; historical play used native sensors. Heading gyro-integrated approximation, no roottruth.',
            measurement='No hand site exists. Current R/L handheight is explicitly the world-transformed boundingboxcenter of selected paw appearance meshes, a kinematic proxy. Contact uses whole distalforearm collisionbody, not a fingerpad sensor.')
