"""Explicit82-observation/raw24 controller for all22 current physical targets.

The R3 walking actor is the unchanged initialization. New beta column is zero.
New upper authority is a residual relative to that frozen source actor, which
preserves its initial physical behavior while making every joint representable.
"""
from pathlib import Path
import copy
import numpy as np
import torch
from adapter import ROOT,WalkingInterface,ref

SOURCE=ROOT/'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/candidate.pt'


def mean(state,obs):
    x=(obs-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
    for layer in (0,2,4,6):
        x=torch.nn.functional.linear(x,state[f'mlp.{layer}.weight'],state[f'mlp.{layer}.bias'])
        if layer!=6:x=torch.nn.functional.elu(x)
    return x


def expanded_actor(source):
    result=copy.deepcopy(source['actor_state_dict'])
    for key,fill in [('obs_normalizer._mean',0.),('obs_normalizer._var',1.),('obs_normalizer._std',1.)]:
        result[key]=torch.cat((result[key],result[key].new_full((1,1),fill)),dim=1)
    w=result['mlp.0.weight'];result['mlp.0.weight']=torch.cat((w,w.new_zeros((len(w),1))),dim=1)
    return result


class Decoder:
    def __init__(self):
        self.mapping=WalkingInterface();self.saved=torch.load(SOURCE,map_location='cpu',weights_only=False)
        if ref(SOURCE)['sha256']!='467b12a17a71f06d951294b3d9185503ea78c6edda431ae7e7870aa648fc02bd':raise ValueError('Source R3 actor changed')
        self.source=self.saved['actor_state_dict']
        self.upper=[r for r in self.mapping.rows if r['source_index'] is not None and r['source_index']>=10]

    def features(self,packet,previous_raw,beta):
        if packet is None:return np.zeros(82,dtype=np.float32)
        v=packet['values'];obs=self.mapping.observe(torch.tensor(v[:22][None],dtype=torch.float32),torch.tensor(v[22:44][None],dtype=torch.float32),
            torch.tensor(v[44:47][None],dtype=torch.float32),torch.tensor(v[47:50][None],dtype=torch.float32),
            torch.tensor(previous_raw[None],dtype=torch.float32),torch.zeros(1,3))[0].numpy()
        return np.r_[obs,np.float32(beta)].astype(np.float32)

    def decode(self,raw,obs):
        """Same source leg/new axes; explicitly enabled9 old upper channels."""
        with torch.inference_mode():
            baseline=mean(self.source,obs[:,:81]);target=self.mapping.target(raw)
            for r in self.upper:
                target[:,r['current_index']]=r['sign']*.3*(raw[:,r['source_index']]-baseline[:,r['source_index']])
        return target

    def inverse_reference(self,target,obs):
        """Actually executed teacher raw24; computed BEFORE native stepping."""
        with torch.inference_mode():raw=mean(self.source,torch.tensor(obs[None,:81]))[0].numpy().copy()
        for r in self.mapping.rows:
            i=r['current_index'];j=r['source_index']
            if j is None:raw[self.mapping.new_rows[r['current_name']]]=target[i]/.3
            elif j<10:raw[j]=(r['sign']*target[i]-self.mapping.home[j].item())/.3
            else:raw[j]+=r['sign']*target[i]/.3
        return raw

    def proof(self):
        actor=expanded_actor(self.saved)
        x=torch.randn(128,82,generator=torch.Generator().manual_seed(138200))
        with torch.inference_mode():
            old=mean(self.source,x[:,:81]);new=mean(actor,x)
            physical=self.decode(new,x);source_target=self.mapping.target(old)
        return dict(source=ref(SOURCE),source_actor_width=81,new_actor_width=82,raw_action_width=24,physical_targets=22,
            zero_beta_first_layer_column_max_abs=float(actor['mlp.0.weight'][:,-1].abs().max()),
            nonzero_random_rows=128,initial_raw_source_max_abs_difference=float(abs(old-new).max()),
            initial_physical_target_source_max_abs_difference=float(abs(physical-source_target).max()),
            source_normalizer_count=int(self.source['obs_normalizer.count']),expanded_normalizer_count=int(actor['obs_normalizer.count']),
            features='Exact delayed source81: gyro/gravity,namedq-home21,qd21,actualpreviousraw21,velocitycommand000,newq3,newqd3,actualpreviousrawnew3; append requestedcrouchbeta atindex81. Velocitycommandsremain000.',
            decoder='Source Interface.target for10legacy legs+3newaxes. Ninephysical oldupper axes now target=sign*.3*(newraw−frozen_source_raw_on_same_preaction81). Twoabsentgripper rawheads retained inhistory. All22 physicalchannels nowrepresentable; no genericrawclip/filter added. ExactR3Runtime independentlyclips/delays.',
            teacher_history='Reference command is inverted into raw24 online before stepping, immediatelyforwarddecoded into actualrequested22 and suppliedtoRuntime. Stored previousraw24 is this executed rawrepresentation, never reconstructedfrompoststate orclippedjoints.',
            critic='Source R3critic/normalizers preserved as provenance, unused forBC and not a learnedsquat value claim.',
            optimizer='Fresh BC Adam moments; sourcePPO optimizer/counter provenance notresumed orcounted asnewupdates.')
