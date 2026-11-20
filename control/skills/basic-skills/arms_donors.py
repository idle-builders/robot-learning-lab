"""Four fresh whole families of supported arm reference visits for bounded BC."""
import argparse,copy,json
import torch
from adapter import ROOT,ref
from run import HERE,write,episode
from arms_trial import CASES
from arms_reference import ArmReference

DATA=ROOT/'data/basic-skills-retry-20261027/arms/donors'
OUT=ROOT/'artifacts/cove50-basic-skills-retry/arms-20261027/donors'


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze-only',action='store_true');args=p.parse_args();torch.set_num_threads(1)
    base=json.loads((ROOT/'data/basic-skills-retry-20261027/arms/model-reference/method.json').read_text());cfg=copy.deepcopy(base['configuration'])
    cfg['starts']=json.loads((HERE/'stand.json').read_text())['starts']
    for case in cfg['starts']:case['seed']+=3100
    cfg['controllers']=['donor-'+case for case in CASES];cfg['actor_controllers']=cfg['controllers'];cfg['scope']='Sixteenwholemodelreferencevisits,fourfreshstartfamilies/fourconditions. SameexactsourceEMA/history/authority; no neuralinference orfit.'
    method=dict(schema='cove50.basic-arms.fresh-donors.v1',selected=base['selected'],configuration=cfg,reference_method=ref(ROOT/'data/basic-skills-retry-20261027/arms/model-reference/method.json'),
        profiles={name:ArmReference(case).proof() for name,case in zip(cfg['controllers'],CASES)},
        split=dict(train_families=[140101,140102,140103],heldout_family=[140104]),evaluation_families=[dict(case,seed=case['seed']+200) for case in cfg['starts']],
        fit=dict(source='Separate unchanged72bilateral/75independent sourcearchitectures; initializefromsourceweights, fixednormalizers/EMA/history/authority.',
            updates_per_actor=1200,Adam_learning_rate=.0001,batch_size=256,fresh_moments=True,seeds=dict(bilateral=140200,independent=140201),
            loss='MSEunfilteredsourcechannels13/18 +.1*MSEother19rawchannels(reference0),MLPweights/biasesonly,finalfixedcheckpoint,criticunused.',
            sampling='Exactlyhalfminibatch fromfirst20validrows plusfirst20validrowsstartingateachactualflagtransition; halfremainingvalidrows. Wholetrainfamiliesonly; sameflags fromsavedpreactionrecords.'),
        recording='Exactpreaction72/75 andactualpreviousfiltered21; rawreference21=-1.2alphaonshoulders,other19zero; sourceEMA thenclip/sign/Runtime. Firstinvalidbootstraprowretained butexcludedfromfit. Allfull10stails retained; degradedreferencevisitsreported beforebankusage.',
        sources=[ref(HERE/n) for n in ('arms_donors.py','arms_reference.py','arms_profile.py','run.py','adapter.py','crouch_policy.py')],
        accounting='16full10sepisodes,4freshfamilies,1sequentialworldslot,8000controls/640000physicsplanned,0updates/newweights. Modelreferenceonroute labels,notDAgger/human/correctiveexperts.')
    path=DATA/'method.json'
    if path.exists() and json.loads(path.read_text())!=method:raise ValueError('Frozen donor method differs')
    write(path,method)
    if args.freeze_only:print(json.dumps(dict(method=ref(path))));return
    OUT.mkdir(parents=True,exist_ok=True)
    if any(OUT.iterdir()):raise FileExistsError('Originaldonors retained')
    rows=[episode(cfg,case,name,ArmReference(profile),OUT,method) for case in cfg['starts'] for name,profile in zip(cfg['controllers'],CASES)]
    write(OUT/'report.json',dict(method=ref(path),episodes=rows,simulator_transitions=sum(e['controls'] for e in rows),native_physics_steps=sum(e['native_physics_steps'] for e in rows),optimizer_updates=0,new_weight_versions=0,independent_start_families=4,world_slots=1))


if __name__=='__main__':main()
