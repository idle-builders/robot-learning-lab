"""Two new current-body references: pure path vs conditional gravity offset.

No policy inference, optimizer or simulator-truth feedback drives either path.
Both interpolate the exact auditor's named22 target/torque table at50Hz.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from adapter import ROOT,WalkingInterface,ref
from run import HERE,write,episode

AUDIT=ROOT/'data/basic-skills-retry-20261027/squat-pose-audit/report.json'
DATA=ROOT/'data/basic-skills-retry-20261027/squat/com-aware-r2'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/squat-20261027/com-aware-r2'


def beta(time_s):
    t=round(time_s/.02)*.02
    return min(1.,max(0.,(t-2.)/2.)) if t<6. else max(0.,1.-(t-6.)/2.)


class Reference:
    action_width=22
    def __init__(self,compensated=False):
        self.mapping=WalkingInterface();self.offset=np.zeros(22);self.compensated=compensated
        audit=json.loads(AUDIT.read_text());rows=audit['candidate_family']
        if any(r['measurement']['target_names']!=self.mapping.current_names for r in rows):raise ValueError('Auditor target order differs from actual22')
        self.nodes=np.array([r['beta'] for r in rows]);self.q=np.array([r['measurement']['target_rad'] for r in rows])
        self.tau=np.array([r['measurement']['conditional_static_balance']['required_joint_torque_Nm'] for r in rows])
        self.kp=np.array(rows[0]['measurement']['conditional_static_balance']['assumed_kp_Nm_rad'])
        self.root=np.array([r['measurement']['root_xyz_m'] for r in rows]);self.com=np.array([r['measurement']['whole_body_com_m'] for r in rows])
        self.assumption=rows[0]['measurement']['conditional_static_balance']['assumption'];self.reset()

    def interpolate(self,values):return np.array([np.interp(self.beta,self.nodes,values[:,i]) for i in range(values.shape[1])])
    def reset(self):self.time_s=0.;self.beta=0.;self.qdes=np.zeros(22);self.static_tau=np.zeros(22)
    def observe(self,packet):
        self.time_s=packet['capture_time_s']+packet['age_s'] if packet else 0.;self.beta=beta(self.time_s)
        self.qdes=self.interpolate(self.q);self.static_tau=self.interpolate(self.tau)
        return np.zeros(75,dtype=np.float32) if packet is None else packet['values'].astype(np.float32).copy()

    def action(self,obs):
        command=self.qdes+(self.static_tau/self.kp if self.compensated else 0.)
        return command.copy(),command.copy(),self.qdes.copy(),command

    def control_info(self):
        return dict(reference_beta=self.beta,desired_q22_rad=self.qdes,conditional_static_torque22_Nm=self.static_tau,
            conditional_gravity_offset22_rad=self.static_tau/self.kp if self.compensated else np.zeros(22),
            geometric_desired_root_xyz_m=self.interpolate(self.root),geometric_desired_com_xyz_m=self.interpolate(self.com))


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(1);profile=Reference();selected=json.loads((ROOT/'robot/model/current-model.json').read_text())
    cfg=json.loads((HERE/'stand.json').read_text());cfg['duration_s']=10.
    cfg['starts']=[dict(seed=138001,label='nominal-reused-diagnostic',joint_sigma_rad=0.,rpy_envelope_rad=0.,joint_velocity_sigma_rad_s=0.)]
    cfg['controllers']=['com-position-reference','com-static-gravity-reference'];cfg['actor_controllers']=cfg['controllers']
    cfg['scope']='New references only: exact current-body COM-aware pose table, pureposition vs qdesired+conditionaltau/Kp. Same nominal138001initialstate/RNGlikepreviousdiagnostics, no new independentfamily. No fit/boundwidening.'
    method=dict(schema='cove50.basic-squat.com-aware-reference.v1',selected=selected,pointer=ref(ROOT/'robot/model/current-model.json'),configuration=cfg,
        auditor_table=ref(AUDIT),source_audit_config=json.loads(AUDIT.read_text())['config'],
        beta_recipe='0@2,1@4/6,0@8; holdthenreturn; exact50Hz phase nodes from roundednative clock, full10s including20ms sensorbootstrap.',
        interpolation='Piecewise linear interpolation between exactbeta0/.25/.5/.75/1 auditor named22 desired joint vectors and conditional torque vectors; no added control filter.',
        desired_root_lowering_m=float(profile.root[0,2]-profile.root[-1,2]),
        gravity_offset_assumption=profile.assumption,
        command='Pure qdesired vs separatelylabelled qdesired+tau/Kp onall22axes; same actual R3Runtime independently clips/delays and uses its source rated motor law. No gravityfeedback fromnative truth.',
        recording_schema='Legacy observation72/raw21/source_target21 suffixes carry actual75/22/22 physicalreference arrays here. Raw/filtered are joint-radian command diagnostics, not learnednormalized actions. Generic headingmetric notapplicable. Explicit desiredq/tau/offset/beta plus actualrequested/accepted/applied22 retained.',
        sources=[ref(HERE/n) for n in ('com_crouch.py','run.py','adapter.py')],
        limitations='Conditional static zeroacceleration/twoverticalsupportforce CoP choice is a model assumption, not measured motortracking or dynamic support. These time-based references are not learnedactors or recoveryexperts.',
        accounting='Two full10s nominalvisits,1 reusedfamily,0 learnedactors,newweights oroptimizerupdates inthiscomparison.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen COMreference method differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path),lowering_m=method['desired_root_lowering_m'])));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Original native visits retained')
    rows=[episode(cfg,cfg['starts'][0],controller,Reference(compensated=controller=='com-static-gravity-reference'),OUT,method) for controller in cfg['controllers']]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),
        native_physics_steps=sum(e['native_physics_steps'] for e in rows),optimizer_updates=0,independent_start_families=1,new_independent_start_families=0,world_slots=1))


if __name__=='__main__':main()
