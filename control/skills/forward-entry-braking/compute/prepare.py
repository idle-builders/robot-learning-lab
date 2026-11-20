"""Small new lesson manifest; reuse exact retained binary packaging/recovery."""
import argparse
import importlib.util
from pathlib import Path
from common import DATA, HERE, ROOT, ENVIRONMENT, SIDECAR, TRANSPORT, BASE_LAUNCH, PROTOCOL, read, ref, write, project

RETAINED = 'control/skills/forward-stop-learning/compute'
INDEX = 'artifacts/runs/cove50-forward-stop-20261030-r2-ee908fcf1f1f/source/input-index.json'


def metadata(review, release=None):
    cfg = read(DATA/'config.json')
    old_plan = read(ROOT/'control/skills/forward-stop-learning/setup/worker-plan.json')
    old_index = read(ROOT/INDEX)
    write(DATA/'environment-config.json', read(ROOT/'control/skills/forward-stop-learning/setup/environment-config.json'))
    source = [*HERE.glob('*.py'), HERE/'bootstrap.sh', DATA/'environment-config.json',
        DATA/'prove_lesson.py', DATA/'cpu-fixture.json', ROOT/'data/forward-stop-learning-20261030-independent/prove_dict_cache.py',
        ROOT/ENVIRONMENT, ROOT/SIDECAR, ROOT/TRANSPORT, ROOT/BASE_LAUNCH, ROOT/'infra/storage/storage_lifecycle.py',
        *[ROOT/RETAINED/n for n in ('common.py','worker.py','launch.py','prepare.py','prove.py','session.py','environment.py','attest.py','bootstrap.sh')]]
    write(DATA/'worker-seal.json',dict(schema='cove50.forward-entry-braking-worker-source-seal.v1',
        files=[ref(p) for p in sorted(set(source))],
        scope='Thin authored projections plus exact immutable retained operations, CPU fixture, isolated setup and native custody helpers.'))
    cases = []
    controls_by_profile = dict(forward30=400, forward25=400, **{'short-stop':150,'long-stop':300}, quiet=400)
    for phase in ('before','after'):
        if phase == 'after':
            cases.append(dict(name='train24', mode='train', seed=cfg['train_seed'], checkpoint='baseline200',
                planned=dict(simulator_transitions=18432,native_physics_steps=1474560,optimizer_steps=480,ppo_rounds=24)))
        for seed in cfg['family_seeds'] if phase == 'before' else cfg['after_family_seeds']:
            for profile in cfg['profiles']:
                controls = controls_by_profile[profile]
                cases.append(dict(name=f'{phase}-{profile}-{seed}',mode='replay',profile=profile,seed=seed,
                    checkpoint='baseline200' if phase == 'before' else 'challenger224',
                    planned=dict(simulator_transitions=controls,native_physics_steps=controls*80,optimizer_steps=0,ppo_rounds=0)))
    children = [dict(name='hardware',process_s=30,join_s=15,pack_s=20), dict(name='bootstrap',process_s=600,join_s=15,pack_s=20)]
    for i, case in enumerate(cases):
        budget = 600 if case['mode']=='train' else (180 if i==0 else {400:150,300:120,150:90}[case['planned']['simulator_transitions']])
        children.append(dict(name=case['name'],process_s=budget,join_s=15,pack_s=20))
    transport = 640
    slack = 7800-transport-sum(c['process_s']+c['join_s']+c['pack_s'] for c in children)
    if slack <= 0: raise ValueError('No finite worker entry slack')
    plan = dict(schema='cove50.forward-entry-braking-worker-plan.v1',job_name='cove50-forward-entry-braking-20261030-r1',
        status='source_frozen_awaiting_root_release',configuration=ref(DATA/'config.json'),source_seal=ref(DATA/'source-seal.json'),
        worker_seal=ref(DATA/'worker-seal.json'),cases=cases,children=children,hard_deadline_s=7800,transport_reserve_s=transport,slack_s=slack,
        absolute_ttl_safety_s=120,provider_policy=dict(executionTimeout=8400000,ttl=10800000),completion_sidecar=PROTOCOL,
        helpers=dict(sidecar=ref(ROOT/SIDECAR),transport=ref(ROOT/TRANSPORT)),dependency_manifest=old_plan['dependency_manifest'],
        library_sources=old_plan['library_sources'],planned=cfg['totals'],root_release_required=True,
        native_source_review=ref(review),dictionary_cache_fixture=ref(DATA/'prove_lesson.py'),CPU_fixture_result=ref(DATA/'cpu-fixture.json'),
        retained_operational_ancestry=ref(ROOT/'data/forward-stop-learning-20261030-independent/compute-source-review.json'),
        observed_R2_stage_timing=ref(ROOT/'artifacts/runs/cove50-forward-stop-20261030-r2-ee908fcf1f1f/evidence/worker-report.json'),
        budget_rationale='R2 first8s visit148.27s, later8s95–102s/6s75–79s/3s47–49s, train50=268.06s. Budgets180s first,150/120/90s by horizon,600s train24; each15s join+20s packing,640s transport and positive slack. No extra native probe or retry.',
        first_inspection_after_s=900,minimum_inspection_spacing_s=900,maximum_foreground_inspections=8,
        source_only_proof='Exact extracted self-contained remote projection and native entry,31 native routes, separate actual CPU lesson fixture and interrupted-owner check.',
        storage='One attached managed producer; work/evidence UUID late-bound, original inputs active until root consumers-complete. Local timeout interrupts wrapper rather than returning normal nonzero.')
    write(DATA/'worker-plan.json',plan)
    excluded = ('control/skills/forward-stop-learning/', 'data/forward-stop-learning-20261030/', 'data/forward-stop-learning-20261030-independent/')
    sources = {r['path']:r for r in old_index['files'] if not r['path'].startswith(excluded)}
    extras = source + [review, ROOT/INDEX, ROOT/plan['dependency_manifest']['path'], ROOT/plan['retained_operational_ancestry']['path']]
    extras += [ROOT/r['path'] for r in cfg['exact_inputs'] + read(DATA/'source-seal.json')['files']]
    extras += [DATA/n for n in ('config.json','source-seal.json','worker-seal.json','worker-plan.json')]
    if release:
        authority = read(release)
        extras += [release, ROOT/authority['independent_review']['path']]
    for path in extras: sources[ref(path)['path']] = ref(path)
    index = dict(schema='cove50.forward-entry-braking.active-input-index.v1',retained_base=ref(ROOT/INDEX),
        files=[sources[k] for k in sorted(sources)],scope='Required retained R3/body/vendor closure plus named projection sources, exact lesson/config/fixtures/reviews; no duplicate historical source forests.',
        recovery='Existing verified private SHA256+size routes and active exact input cache only; no checkpoint deserialization or live pointer mutation.')
    write(DATA/'input-index.json',index)
    write(DATA/'bundle-manifest.json',dict(schema='cove50.forward-entry-braking.capsule.v1',files=index['files'],plan=ref(DATA/'worker-plan.json')))
    return index


def package(index, work, recover):
    source = project(RETAINED+'/prepare.py','a24900c41125d9cb36fc5c0bfdd96093f192c6c6f5f7ef797a7e682c46fddcba',[])
    namespace = dict(__name__='retained_package_helper',__file__=str(HERE/'prepare.py'))
    exec(compile(source,str(HERE/'prepare.py')+':retained-package-helper','exec'),namespace)
    namespace['package'](index,work,recover)
    if recover:
        receipt = read(DATA/'bundle-receipt.json')
        receipt['regenerate']='.venv/bin/python -B control/skills/forward-entry-braking/compute/prepare.py --package --work <active-work> --review <exact-native-review> [--release <exact-root-release>]'
        write(DATA/'bundle-receipt.json',receipt)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--package',action='store_true')
    p.add_argument('--review',type=Path,required=True);p.add_argument('--release',type=Path)
    a=p.parse_args();index=metadata(a.review.resolve(),a.release.resolve() if a.release else None)
    package(index,a.work.resolve(),a.package)
