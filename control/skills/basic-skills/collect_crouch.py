"""Fresh full-reference visits with actual live82/raw24 teacher interface."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from adapter import ROOT,ref
from run import HERE,episode,write
from com_crouch import Reference,AUDIT
from crouch_policy import Decoder

DATA=ROOT/'data/basic-skills-retry-20261027/squat/teacher-fresh'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/squat-20261027/teacher-fresh'


class Teacher(Reference):
    action_width=24
    def __init__(self):
        super().__init__(compensated=True);self.decoder=Decoder();self.reset()

    def reset(self):
        super().reset();self.previous_raw=np.zeros(24,dtype=np.float32);self.input_previous_raw=self.previous_raw.copy()
        self.packet75=np.full(75,np.nan);self.obs82=np.zeros(82,dtype=np.float32);self.roundtrip_error=0.

    def observe(self,packet):
        super().observe(packet);self.input_previous_raw=self.previous_raw.copy()
        self.packet75=np.full(75,np.nan) if packet is None else packet['values'].copy()
        self.obs82=self.decoder.features(packet,self.previous_raw,self.beta)
        return self.obs82

    def action(self,obs):
        desired=self.qdes+self.static_tau/self.kp
        raw=self.decoder.inverse_reference(desired,obs)
        actual=self.decoder.decode(torch.tensor(raw[None]),torch.tensor(obs[None]))[0].numpy()
        self.roundtrip_error=float(abs(actual-desired).max());self.previous_raw=raw.copy()
        return raw,raw.copy(),self.qdes.copy(),actual

    def control_info(self):
        values=super().control_info();values.update(pre_action_delayed_packet75=self.packet75,
            actual_previous_executed_raw24=self.input_previous_raw,reference_decode_roundtrip_max_abs_rad=self.roundtrip_error)
        return values


def recipe():
    cfg=json.loads((HERE/'stand.json').read_text());cfg['duration_s']=10.
    for case in cfg['starts']:case['seed']+=1200 #138201..138204
    cfg['controllers']=['live-interface-com-gravity-reference'];cfg['actor_controllers']=cfg['controllers']
    cfg['scope']='FourfreshwholeCOM-gravityreferencevisitswithliveexecuted82/raw24interface. No fitting; no new independent policy. Sensors/actionhistory capturedpreaction. These on-route scripted/model labels are not DAgger/human/correctiveexpert labels.'
    return cfg


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);cfg=recipe();profile=Teacher();selected=json.loads((ROOT/'robot/model/current-model.json').read_text())
    method=dict(schema='cove50.basic-squat.teacher-fresh.v1',selected=selected,pointer=ref(ROOT/'robot/model/current-model.json'),configuration=cfg,
        body_reference_table=ref(AUDIT),initialization_and_interface=profile.decoder.proof(),
        source_current_reference='Exact sameauditorvectors/conditionalgravityoffset ascom-aware-r2; only roundtripfloat32 rounding in explicitraw24 interface, recorded percontrol.',
        target_authority='All22 joints; upper9 newlyenabledrelative tofrozensourceactor, source10legs/new3 unchanged. This is changedcontrollercontract, not sourceposture rawmask.',
        frozen_data_split=dict(train_families=[138201,138202,138203],whole_family_validation=[138204]),
        frozen_learner_evaluation_families=[dict(case,seed=case['seed']+200) for case in cfg['starts']],
        proposed_fit=dict(method='CPU BC warmstart from R3walker actor withzerobeta expansion; value unused',optimizer='Adam freshmoments, LRsourceeffective2.25e-5',updates=600,batch_size=256,
            selection='Lastfixedbudget600-update actor, no offline-loss promotion or familyreuse asfreshvalidation'),
        sources=[ref(HERE/n) for n in ('collect_crouch.py','crouch_policy.py','com_crouch.py','run.py','adapter.py')],
        limits='Nominal/smallupright variations only; time-based conditionalstaticreference, not a recoveryexpert. Originalcritic remains sourcewalkingvalueprovenance only; no PPO ornewreward optimized.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen collection/fit split differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path),proof=method['initialization_and_interface'])));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Originalfreshreferencevisits alreadyretained')
    rows=[episode(cfg,case,cfg['controllers'][0],Teacher(),OUT,method) for case in cfg['starts']]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),native_physics_steps=sum(e['native_physics_steps'] for e in rows),
        optimizer_updates=0,independent_start_families=4,new_independent_start_families=4,world_slots=1))


if __name__=='__main__':main()
