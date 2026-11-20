"""Source-labelled Direct Frame posture BC on the selected R3 timed packet.

The shared walking Interface supplies named state/axis mapping only. Posture
keeps its own absolute-target normalization, authority, raw clamp and filter.
"""
from pathlib import Path
import sys
import importlib.util
import hashlib
import json
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('walk_state_mapping',ROOT/'control/skills/walker/policy/adapter.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
WalkingInterface=module.Interface
Runtime=sys.modules['runtime'].Runtime
sys.path.insert(0,str(ROOT))
from control.skills.low_level.multicommand_bc import Actor,load_student

CHECKPOINT=ROOT/'artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/student-squat-varied-history.pt'
TRACE=ROOT/'artifacts/ape-low-level-training-20261015/multicommand-cpu-pilot-v1/varied-squat-bank/squat-seed54110.npz'


def ref(path):
    path=path.absolute()
    with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=str(path.relative_to(ROOT)),sha256=digest,bytes=path.stat().st_size)


class Posture:
    def __init__(self):
        self.mapping=WalkingInterface()
        self.saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
        if ref(CHECKPOINT)['sha256']!='0b20cc71d765ce6913adb1f70587fcbf59442bcff25b7645acd64a9e643028f4':raise ValueError('Posture checkpoint identity changed')
        if self.saved['target_names']!=self.mapping.source_names:raise ValueError('Source21 order differs')
        with np.load(TRACE) as f:
            self.scale=f['action_scale'].copy();self.offset=f['action_offset'].copy()
        self.policy=load_student(CHECKPOINT,'cpu',self.mapping.source_names,torch.tensor(self.scale),torch.tensor(self.offset),'squat')
        self.reset()

    def reset(self):
        self.filtered=None;self.previous_raw=np.zeros(21,dtype=np.float32)
        self.heading=0.;self.capture=None

    def observe(self,packet):
        if packet is None:return np.zeros(72,dtype=np.float32)
        v=packet['values'];capture=packet['imu_capture_time_s']
        if self.capture is not None and capture>self.capture:
            # Angular velocity projected onto estimated world vertical gives
            # relative heading, using only the delayed measured IMU packet.
            self.heading+=float(np.dot(v[44:47],-v[47:50]))*(capture-self.capture)
        self.capture=capture
        command=torch.tensor([[0.,0.,-np.arctan2(np.sin(self.heading),np.cos(self.heading))]],dtype=torch.float32)
        obs=self.mapping.observe(torch.tensor(v[:22][None],dtype=torch.float32),torch.tensor(v[22:44][None],dtype=torch.float32),
            torch.tensor(v[44:47][None],dtype=torch.float32),torch.tensor(v[47:50][None],dtype=torch.float32),torch.zeros(1,24),command)[0,:72].numpy()
        obs[48:69]=0. # Source env masks history, independently of checkpoint flag.
        return obs

    def action(self,obs):
        with torch.inference_mode():raw=self.policy({'actor':torch.tensor(obs[None])})[0].numpy()
        clipped=np.clip(raw,-1.,1.)
        self.filtered=clipped.copy() if self.filtered is None else .2*clipped+.8*self.filtered
        self.previous_raw=raw.copy()
        oldtarget=self.offset+self.scale*self.filtered
        target=np.array([r['sign']*oldtarget[r['source_index']] if r['source_index'] is not None else 0. for r in self.mapping.rows])
        # Source shoulder clips are physical joint limits with0.05rad margin.
        for row in self.mapping.rows:
            if row['source_name'] and row['source_name'].endswith('shoulder_pitch'):
                j=self.mapping.old.joint(row['source_name']).id;lo,hi=self.mapping.old.jnt_range[j]
                value=np.clip(oldtarget[row['source_index']],lo+.05,hi-.05)
                target[row['current_index']]=row['sign']*value
        return raw,self.filtered.copy(),oldtarget,target

    def audit(self):
        # Independently reconstruct normalized absolute-target inference and
        # compare the source loader on nonzero random observations.
        c=self.saved;model=Actor(72);model.load_state_dict(c['model']);model.eval()
        x=torch.randn(128,72,generator=torch.Generator().manual_seed(137000));x[:,48:69]=0
        ident=torch.tensor([[1.,0.]]).repeat(len(x),1)
        with torch.inference_mode():
            target=model((x-c['observation_mean'])/c['observation_std'],ident)*c['target_std']+c['target_mean']
            scale=torch.tensor(self.scale);offset=torch.tensor(self.offset)
            raw=(target-offset)/torch.where(scale.abs()>1e-8,scale,1.)
            raw=torch.where(scale.abs()>1e-8,raw,0.)
            loader=self.policy({'actor':x})
        return dict(checkpoint=ref(CHECKPOINT),source_decoder=ref(ROOT/'control/skills/low_level/multicommand_bc.py'),
            source_router=ref(ROOT/'control/skills/low_level_portfolio/controller.py'),
            source_task=ref(ROOT/'control/skills/policy_transfer/train_posture_arm.py'),
            source_scale_offset_trace=ref(TRACE),source_order=self.mapping.source_names,
            source_action_scale=self.scale.tolist(),source_action_offset=self.offset.tolist(),mapping=self.mapping.rows,
            decoder_random_rows=128,source_loader_raw_max_abs_difference=float(abs(raw-loader).max()),
            command='slot69=standing slew alpha0, slot70=0, slot71=negative integrated delayed measured gyro heading relative to reset; source used privileged spawn-yaw error',
            observation='Source72: delayed gyro3, estimated gravity3, named mapped q-home21,qd21, maskedzero previousaction21, command3. Actual raw/filter state retained separately.',
            action='Source absolute target normalization -> raw=(target-offset)/scale; zero inactive raw; source clamp±1 then0.2filter(first bypass); source named scale+offset; source shoulder pitch margin.05; named signs into22; new3 axes neutral; independent R3 clipping/delay.',
            source_reward='BC actor; no new reward evaluation or optimizer. Original posture source retained, no forward reward substitution.',
            limitations=['Relative gyro heading is an approximate sensor-semantic transfer; bias/drift measured against truth only by evaluator.',
                'Neutral-axis mapping does not preserve full source poses across changed serial joints.',
                'Known source joint limits and current R3 target clips are separate; hip±.35 is diagnostic, not measured travel.'])
