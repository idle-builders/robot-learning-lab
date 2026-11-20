"""Source-target model references under the same four arm interfaces, no fit."""
from pathlib import Path
import argparse,copy,json
import numpy as np
import torch
from adapter import ROOT,ref
from run import HERE,write,episode
from arms_profile import Arms
from arms_trial import DATA as BASE_DATA,OUT as BASE_OUT,CASES

DATA=BASE_DATA/'model-reference'
OUT=BASE_OUT/'model-reference'


class ArmReference(Arms):
    def action(self,obs):
        raw=np.zeros(21,dtype=np.float32);raw[[13,18]]=-1.2*self.alpha
        self.filtered=raw.copy() if self.filtered is None else self.ema*raw+(1-self.ema)*self.filtered
        source=self.offset.copy();source[[13,18]]=np.clip(self.filtered[[13,18]],-1.52,1.52)
        target=np.zeros(22)
        for row in self.mapping.rows:
            if row['source_index'] in (13,18):target[row['current_index']]=row['sign']*source[row['source_index']]
        self.previous_filtered=self.filtered.copy();return raw,self.filtered.copy(),source,target


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    source=json.loads((BASE_DATA/'method.json').read_text());cfg=copy.deepcopy(source['configuration']);cfg['controllers']=['model-reference-'+case for case in CASES];cfg['actor_controllers']=cfg['controllers']
    cfg['scope']='Four source-target modelreferencevisits matched tooriginalnominalfamily140001. OriginalEMA/home/clip/Runtime; no neuralactorcalls, fit orbodychange.'
    method=dict(schema='cove50.basic-arms.model-reference.v1',selected=source['selected'],configuration=cfg,original_source_method=ref(BASE_DATA/'method.json'),
        profiles={name:Arms(case).proof() for name,case in zip(cfg['controllers'],CASES)},measurement_frames=source['measurement_frames'],current_motor_metadata=source['current_motor_metadata'],
        reference='Replace neuralmean with rawsource21 shoulder13/18=−1.2*alpha_L/R, other19raw0. Alpha initial0, independentflags2–6s,1.5sslew, firstflagframeat2s alpha0; sourceEMA.2bilateral/.05independent, firstvalidbypass; shoulderphysicalclip±1.52; sign−1intoR3. Current20otherphysicalaxes remain0.',
        history='All21 actualEMA messages feedsourcehistory inpreaction72/75 even thoughsource actor isnotcalled. No rawclamp, travelwidening, body/gain orsensorchanges. Samefirst20msbootstrapinside10s andsamecompleteinitialstate/sensorseed asoriginalcondition.',
        source_weights='Base profile loads savedsource metadata/weights but reference action never calls it. No learned controller, no correctiveexpert, DAgger/human ornewpolicy claim.',
        sources=[ref(HERE/n) for n in ('arms_reference.py','arms_profile.py','arms_trial.py','run.py','adapter.py','crouch_policy.py')],
        accounting='Fourfull10s,oneREUSEDnominalfamily140001,oneworldslot,2000controls/160000physics planned,zero learnedactorinferences/newweights/optimizerupdates.',
        limitations='This is prospective commanded shoulderpath feasibility under fixedneutral support, not recovery,grasp orrobustness. Ifsupportlost, keepwholefalltails anddiagnosesupport before fitting.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen reference differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path))));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Original modelreferencevisits retained')
    rows=[episode(cfg,cfg['starts'][0],name,ArmReference(case),OUT,method) for name,case in zip(cfg['controllers'],CASES)]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),native_physics_steps=sum(e['native_physics_steps'] for e in rows),optimizer_updates=0,new_weight_versions=0,independent_start_families=1,new_start_families=0,world_slots=1,learned_actor_inferences=0))


if __name__=='__main__':main()
