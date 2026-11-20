"""Standalone beta-only actor plus separate reference annotation instrumentation."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from adapter import ROOT,WalkingInterface,ref
from run import HERE,write,episode
from command_student import model,DATA as FIT_DATA,OUT as FIT_OUT
from com_crouch import Reference,beta
from collect_crouch import Teacher

DATA=ROOT/'data/basic-skills-retry-20261027/squat/command-evaluation'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/squat-20261027/command-evaluation'


class CommandActor:
    """Deployed actor: saved small weights and external scalar beta only."""
    action_width=22
    def __init__(self):
        saved=torch.load(FIT_OUT/'candidate.pt',map_location='cpu',weights_only=False);self.policy=model();self.policy.load_state_dict(saved['actor_state_dict']);self.policy.eval()
        self.mapping=WalkingInterface();self.offset=np.zeros(22)
        if saved['joint_order']!=self.mapping.current_names:raise ValueError('Named22 decoder order differs')
        self.reset()
    def reset(self):self.command=0.;self.previous_raw=np.zeros(22);self.input_previous_raw=self.previous_raw.copy();self.packet75=np.full(75,np.nan)
    def observe(self,packet):
        # External harness command clock; no reference or geometric table query.
        time_s=packet['capture_time_s']+packet['age_s'] if packet else 0.;self.command=beta(time_s)
        self.packet75=np.full(75,np.nan) if packet is None else packet['values'].copy();self.input_previous_raw=self.previous_raw.copy()
        return np.array([self.command],dtype=np.float32)
    def action(self,obs):
        with torch.inference_mode():target=self.policy(torch.tensor(obs[None],dtype=torch.float32))[0].numpy().copy()
        self.previous_raw=target.copy();return target,target.copy(),np.zeros(22),target.copy()


class InstrumentedActor:
    """Actor outputs unchanged; truth-free reference annotations are outside it."""
    action_width=22
    def __init__(self):
        self.actor=CommandActor();self.reference=Reference(compensated=True);self.mapping=self.actor.mapping;self.offset=self.actor.offset
    def reset(self):self.actor.reset();self.reference.reset()
    def observe(self,packet):
        obs=self.actor.observe(packet);self.reference.observe(packet);return obs
    def action(self,obs):return self.actor.action(obs)
    def control_info(self):
        info=self.reference.control_info();info.update(pre_action_delayed_packet75=self.actor.packet75,actual_previous_executed_raw22=self.actor.input_previous_raw,
            annotation_role='Desiredq/tau are instrumentation only and never passed to the1-input actor.')
        return info


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['after','reference']);p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    fit=json.loads((FIT_DATA/'method.json').read_text());teacher=json.loads((ROOT/fit['teacher_method']['path']).read_text());cfg=teacher['configuration'].copy();cfg['starts']=fit['evaluation_families']
    name='reloaded-command-student' if args.stage=='after' else 'command-eval-com-gravity-reference';cfg['controllers']=[name];cfg['actor_controllers']=[name]
    method=dict(schema='cove50.shallow-crouch.command-evaluation.v1',stage=args.stage,selected=teacher['selected'],configuration=cfg,
        fit_method=ref(FIT_DATA/'method.json'),candidate=ref(FIT_OUT/'candidate.pt'),recording_schema='Actualactorinput1/requestedphysicaloutput22. Legacy observation72/raw21suffixes retained only for container reuse; allshapeandunit metadatahere authoritative. Generic headingaccuracy inapplicable.',
        actor='Standalone CommandActor loads onlysmallmodelweights, consumesonlyexternalrequestedbeta andreturnsnamed22radians. Reference object lives only in separate instrumentation wrapper; no annotation or sensortruth entersactor.',
        command='Live external0→1beta2–4s,lowhold4–6,return6–8,final8–10. Time is scheduler input, never actor input. Same Runtime/motor/sensors/delays for both branches.',
        sources=[ref(HERE/n) for n in ('evaluate_command_student.py','command_student.py','run.py','adapter.py','com_crouch.py','collect_crouch.py','crouch_policy.py')],
        limitations='Shallowpose-trajectory imitation only; no feedbackpolicy,value,WBC,DAgger or recoveryexpert claim. Fourfreshuprightfamilies, no pushes/deeploweredentries.')
    path=DATA/args.stage/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen commandreplay differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path))));return
    out=OUT/args.stage;out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Retained originals alreadyexist')
    rows=[episode(cfg,case,name,InstrumentedActor() if args.stage=='after' else Teacher(),out,method) for case in cfg['starts']]
    write(out/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),native_physics_steps=sum(e['native_physics_steps'] for e in rows),
        new_optimizer_updates=0,new_weight_versions=0,world_slots=1,independent_start_families=4))


if __name__=='__main__':main()
