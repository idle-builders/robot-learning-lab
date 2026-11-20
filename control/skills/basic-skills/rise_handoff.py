"""One source-first rise handoff pair from an actually attained shallow crouch."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from adapter import ROOT,WalkingInterface,ref
from run import HERE,write,episode
from crouch_policy import mean
from evaluate_command_student import CommandActor

DATA=ROOT/'data/basic-skills-retry-20261027/rise'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/rise-20261027'
AUDIT=ROOT/'data/basic-skills-retry-20261027/rise-source-audit/source-contract.json'
PROPOSAL=ROOT/'data/basic-skills-retry-20261027/rise-source-audit/proposal.json'
SOURCE=ROOT/'artifacts/ape-cove50-stand-recovery/seated-bc-v35/model_bc_seed.pt'
CROUCH_ASSESSMENT=ROOT/'data/basic-skills-retry-20261027/squat/crouch-assessment.json'


def phase_entry():
    a=json.loads(CROUCH_ASSESSMENT.read_text())
    row=next(e for c in a['cohorts'] if c['cohort']=='command-evaluation/after' for e in c['episodes'] if e['case']['seed']==138801)
    return row['raw_height_reduction_m']/.0511


class Handoff:
    action_width=21
    def __init__(self,actor=False):
        self.mapping=WalkingInterface();self.offset=self.mapping.home.numpy().copy();self.actor=actor;self.prefix=CommandActor()
        self.contract=json.loads(AUDIT.read_text());saved=torch.load(SOURCE,map_location='cpu',weights_only=False);self.policy=saved['actor_state_dict']
        if ref(SOURCE)['sha256']!='dec2b15e0b2955c284cfe41bb74cc0cab96e98aca50ff66a79d047d6b142c23b':raise ValueError('V35 source changed')
        if self.contract['source_actions']['joint_order']!=self.mapping.source_names:raise ValueError('V35 source order differs')
        self.scale=np.array(self.contract['source_actions']['scale']);self.active=set(self.contract['source_actions']['physical_authority_indices'])
        self.authority=np.array([r['source_index'] in self.active for r in self.mapping.rows]);self.auxiliary=~self.authority
        self.alpha0=phase_entry();self.reset()
    def reset(self):
        self.prefix.reset();self.previous_raw=np.zeros(21,dtype=np.float32);self.input_previous_raw=self.previous_raw.copy()
        self.heading=0.;self.capture=None;self.time_s=0.;self.alpha=0.;self.beta=0.;self.packet75=np.full(75,np.nan);self.aux=np.zeros(22);self.switched=False
    def observe(self,packet):
        command=self.prefix.observe(packet);self.beta=float(command[0]);self.input_previous_raw=self.previous_raw.copy()
        if packet is None:return np.zeros(73,dtype=np.float32)
        v=packet['values'];self.packet75=v.copy();self.time_s=round((packet['capture_time_s']+packet['age_s'])/.02)*.02
        capture=packet['imu_capture_time_s']
        if self.capture is not None and capture>self.capture:self.heading+=float(np.dot(v[44:47],-v[47:50]))*(capture-self.capture)
        self.capture=capture
        # Prefix phase denotes source-equivalent relative depth; original .5/s slew begins at handoff.
        self.alpha=self.alpha0*self.beta if self.time_s<6. else max(0.,self.alpha0-.5*(self.time_s-6.))
        history=np.r_[self.previous_raw,np.zeros(3,dtype=np.float32)]
        old=self.mapping.observe(torch.tensor(v[:22][None],dtype=torch.float32),torch.tensor(v[22:44][None],dtype=torch.float32),
            torch.tensor(v[44:47][None],dtype=torch.float32),torch.tensor(v[47:50][None],dtype=torch.float32),torch.tensor(history[None]),torch.tensor([[0.,0.,np.arctan2(np.sin(-self.heading),np.cos(-self.heading))]],dtype=torch.float32))[0].numpy()
        old[6:27]+=self.offset # centered source q-relative = sign*currentq; no old-home subtraction
        return np.r_[old[:72],np.float32(self.alpha)].astype(np.float32)
    def action(self,obs):
        _,_,_,reference=self.prefix.action(np.array([self.beta],dtype=np.float32));self.aux=reference.copy()
        self.switched=bool(self.actor and self.time_s>=6.)
        if self.switched:
            with torch.inference_mode():raw=mean(self.policy,torch.tensor(obs[None],dtype=torch.float32))[0].numpy().copy()
        else:
            raw=np.zeros(21,dtype=np.float32)
            for r in self.mapping.rows:
                j=r['source_index']
                if j in self.active:raw[j]=r['sign']*reference[r['current_index']]/self.scale[j]
        target=reference.copy()
        for r in self.mapping.rows:
            j=r['source_index']
            if j in self.active:target[r['current_index']]=r['sign']*self.scale[j]*raw[j]
        self.previous_raw=raw.copy()
        return raw,raw.copy(),self.offset+self.scale*raw,target
    def control_info(self):
        return dict(pre_action_delayed_packet75=self.packet75,actual_previous_executed_raw21=self.input_previous_raw,
            external_posture_beta=self.beta,source_rise_phase=self.alpha,source_actor_owns_active12=self.switched,
            auxiliary_owner_mask22=self.auxiliary,auxiliary_requested22_rad=self.aux,
            source_active12_mask22=self.authority)
    def proof(self):
        generator=np.random.default_rng(139000);commands=generator.normal(0,.1,(128,22));worst=0.
        for command in commands:
            raw=np.zeros(21)
            for r in self.mapping.rows:
                j=r['source_index']
                if j in self.active:raw[j]=r['sign']*command[r['current_index']]/self.scale[j]
            decoded=command.copy()
            for r in self.mapping.rows:
                j=r['source_index']
                if j in self.active:decoded[r['current_index']]=r['sign']*self.scale[j]*raw[j]
            worst=max(worst,float(abs(decoded-command).max()))
        return dict(nonzero_rows=128,prefix_active12_plus_auxiliary10_target_roundtrip_max_abs_rad=worst,
            source_checkpoint=ref(SOURCE),actor_normalizer_count=int(self.policy['obs_normalizer.count']),source_actor_width=73,raw_width=21,
            physical_source_authority=int(self.authority.sum()),physical_auxiliary_authority=int(self.auxiliary.sum()),
            no_raw_clamp_no_mask_no_EMA_after_actor=True,source_normalizer='(observation - mean)/(std + .01), no update',
            source_critic='V35 copied trained V25 source critic is provenance only, unused and not a current-body rise value claim.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    profile=Handoff();cfg=json.loads((HERE/'stand.json').read_text());cfg['duration_s']=10.
    cfg['starts']=[dict(seed=139001,label='fresh-nominal-shallow-handoff',joint_sigma_rad=0.,rpy_envelope_rad=0.,joint_velocity_sigma_rad_s=0.)]
    cfg['controllers']=['learned-shallow-return','v35-rise-handoff'];cfg['actor_controllers']=cfg['controllers']
    selected=json.loads((ROOT/'robot/model/current-model.json').read_text())
    method=dict(schema='cove50.basic-rise.source-first-handoff.v1',configuration=cfg,selected=selected,pointer=ref(ROOT/'robot/model/current-model.json'),
        source_contract=ref(AUDIT),proposal=ref(PROPOSAL),source_actor=ref(SOURCE),prefix_actor=ref(ROOT/'artifacts/cove50-basic-skills-retry/squat-20261027/command-student/candidate.pt'),
        schedule='Same process: quiet0–2s, learned shallow descent2–4s, low hold4–6s. At6s compare V35 deterministic mean vs continued learned return6–8s. Full10s retained, including all falls; no plant/motor/sensor/heading/history reset at handoff.',
        phase=dict(source_stand_minus_sit_m=.0511,frozen_current_relative_lowering_m=profile.alpha0*.0511,initial_source_equivalent_phase=profile.alpha0,
            measurement=ref(CROUCH_ASSESSMENT),measurement_family=138801,formula='Prior published nominal meanroot1–2s minus meanroot5–6s, divided by source.0511m. Fixed before this new visit; no current roottruth fed to actor.',
            source_slew_per_s=.5,first_handoff_observation='At6.00s alpha=alpha0; at6.02 alpha=alpha0−.01; reaches0at6.26s. This source-equivalent shallow phase differs from fullnormalized current beta and original deep source reset.'),
        observation='73: delayed measuredgyro3/estimatedgravity3, centerednamedqrelative21=sign*currentq, mappedqd21, actualpreviousexecutedraw21, commandstandflag0/0/gyrointegratedheadingerror, separatephase. Missinggripperstate0; no simulatortruth in actor.',
        action='12source-authoritative currenttargets=sign*source_scale*raw; all10legchannels+shoulderpitches. No source rawclamp, active8 mask, EMA or extra sourcejointclip. R3Runtime independently clips/delays using unchanged motor model.',
        composition='Duringprefix inverse source-authoritative12 targetchannels online into source raw21, zero9zero-authorityrawchannels, forwarddecode actualrequested22 with auxiliary10 fromlearnedposturestudent. Atactorhandoff all21actualactorrawoutputs feedhistory; auxiliary10 remain owned by learnedstudent return schedule. Rawhistory is causallyexecuted messages, not copiedphysical22 or reconstructedpoststate.',
        instrumentation='Fullpre-action73,raw21,packet75/phase/history/auxiliaryownership; requested/accepted/applied22,root/q/qd/COM/sole/contact/motor/saturation. Compare allprefixstate/actions up to6s and exactinitialq/qvel/sensorbiases acrosspair.',
        proof=profile.proof(),sources=[ref(HERE/n) for n in ('rise_handoff.py','run.py','adapter.py','crouch_policy.py','evaluate_command_student.py','command_student.py','com_crouch.py')],
        accounting='Twofull10s matchedvisits,1freshnominalfamily,1sequentialworldslot,1000controls/80000physics planned,0updates/0newweights. No provider compute.',
        limits='Only achieved~6.28mm shallow entry, not original51.1mm deep rise, seated/prone/fall recovery or robustness. Centeredbodymapping and auxiliary10 ownership are explicit changed compositional contracts. Source15rewardfunctions retained as provenance; no new rewards/valuefitting/PPO.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen rise method differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path),proof=method['proof'],phase=method['phase'])));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Original rise visits already retained')
    rows=[episode(cfg,cfg['starts'][0],name,Handoff(actor=name=='v35-rise-handoff'),OUT,method) for name in cfg['controllers']]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(r['controls'] for r in rows),native_physics_steps=sum(r['native_physics_steps'] for r in rows),optimizer_updates=0,new_weight_versions=0,independent_start_families=1,world_slots=1))


if __name__=='__main__':main()
