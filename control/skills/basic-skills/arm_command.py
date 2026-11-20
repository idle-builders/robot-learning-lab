"""Fresh causal flag-memory component; external scheduling and physical PD separate."""
import numpy as np
import torch
from adapter import WalkingInterface,ref
from arms_profile import Arms,AUDIT


class CommandNet(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gru=torch.nn.GRU(2,16,batch_first=True);self.head=torch.nn.Linear(16,2)
        torch.nn.init.uniform_(self.head.weight,-.01,.01);torch.nn.init.constant_(self.head.bias,.05)
    def forward(self,flags,hidden=None):
        latent,hidden=self.gru(flags,hidden);return self.head(latent).clamp(0.,1.),hidden


def flags_for(schedule,frame):
    return np.array([float(any(a<=frame<b for a,b in side)) for side in schedule['intervals_L_R']],dtype=np.float32)


def labels_for(schedule):
    flags=np.stack([flags_for(schedule,k) for k in range(500)]);alpha=np.zeros((500,2),dtype=np.float32);state=np.zeros(2,dtype=np.float64)
    for k in range(1,500):
        state+=np.clip(flags[k-1]-state,-.02/1.5,.02/1.5);alpha[k]=state
    return flags,alpha


def canonical(case):
    return dict(name=case,intervals_L_R=[[[100,300]] if case!='independent-source-right-current-L' else [],[[100,300]] if case!='independent-source-left-current-R' else []])


class CommandArms:
    action_width=21
    sample_native=Arms.sample_native
    def __init__(self,case,schedule,checkpoint=None):
        self.case=case;self.schedule=schedule;self.annotation_flags,self.annotation_alpha=labels_for(schedule);self.mapping=WalkingInterface();self.offset=self.mapping.home.numpy().copy();self.ema=.2 if case=='bilateral-both' else .05
        self.native_model=None;self.checkpoint=checkpoint;self.model=None
        if checkpoint is not None:
            payload=torch.load(checkpoint,map_location='cpu',weights_only=False);self.model=CommandNet();self.model.load_state_dict(payload['model_state_dict']);self.model.eval()
        self.reset()
    def reset(self):
        self.hidden=torch.zeros(1,1,16);self.input_hidden=self.hidden[0,0].numpy().copy();self.recurrent_calls=0;self.previous_filtered=np.zeros(21,dtype=np.float32);self.filtered=None;self.input_previous=self.previous_filtered.copy()
        self.packet75=np.full(75,np.nan);self.flags=np.zeros(2,dtype=np.float32);self.alpha=np.zeros(2,dtype=np.float32);self.predicted_alpha=np.zeros(2,dtype=np.float32);self.frame=0
    def observe(self,packet):
        self.input_previous=self.previous_filtered.copy();self.input_hidden=self.hidden[0,0].numpy().copy()
        if packet is None:return np.zeros(2,dtype=np.float32)
        self.packet75=packet['values'].copy();self.frame=round((packet['capture_time_s']+packet['age_s'])/.02)
        self.flags=flags_for(self.schedule,self.frame)
        # Reference/measurement annotation is separate from learned inference.
        self.alpha=self.annotation_alpha[min(self.frame,499)].copy()
        return self.flags.copy()
    def decode(self,alpha):
        raw=np.zeros(21,dtype=np.float32);raw[[13,18]]=-1.2*alpha
        self.filtered=raw.copy() if self.filtered is None else self.ema*raw+(1-self.ema)*self.filtered
        source=self.offset.copy();source[[13,18]]=np.clip(self.filtered[[13,18]],-1.52,1.52);target=np.zeros(22)
        for r in self.mapping.rows:
            if r['source_index'] in (13,18):target[r['current_index']]=r['sign']*source[r['source_index']]
        self.previous_filtered=self.filtered.copy();return raw,self.filtered.copy(),source,target
    def action(self,flags):
        if self.model is None:raise ValueError('Learned command model missing')
        with torch.inference_mode():prediction,self.hidden=self.model(torch.tensor(flags[None,None],dtype=torch.float32),self.hidden)
        self.recurrent_calls+=1;self.predicted_alpha=prediction[0,0].numpy().copy();return self.decode(self.predicted_alpha)
    def control_info(self):
        return dict(pre_action_delayed_packet75=self.packet75,actual_previous_executed_filtered21=self.input_previous,source_anatomical_raise_flags_L_R=self.flags,
            source_hidden_alpha_annotation_L_R=self.alpha,command_model_predicted_alpha_L_R=self.predicted_alpha,recurrent_calls=self.recurrent_calls,pre_action_command_hidden16=self.input_hidden,post_action_command_hidden16=self.hidden[0,0].numpy().copy(),
            profile_schema='Fresh actor observes2binaryexternalflags only; legacy observation72 key haswidth2. Heading and oldsourcehistory features notapplicable. Packet retained as instrumentation only.')
    def proof(self):
        return dict(source=dict(ema_alpha=self.ema,arm_target_source_audit=ref(AUDIT)),actor_width=2,physical_authority=2,
            checkpoint=ref(self.checkpoint) if self.checkpoint is not None else None,schedule=self.schedule,
            contract='Fresh GRU16→linear2→clamp[0,1] alpha; raw13/18=-1.2alpha,other19zero; original21EMA/shoulderclip±1.52/sign/currentneutral20axes. No actorclock/sensors/teacheralpha/history/critic.',
            reset='Hidden1×1×16zero; control0 invalid20msbootstrap hasnoGRUcall;499valid20mscalls. Schedule outside actor produces only current2flags.',
            source_alpha_timing='Annotation/reference integrates previous-frameflag at.02/1.5, alpha0 atfirst2sraiseflag andalpha1 atfirst6slowerflag. Learned action never reads annotation.')


class CommandReference(CommandArms):
    def action(self,flags):
        self.predicted_alpha=self.alpha.copy();return self.decode(self.alpha)
