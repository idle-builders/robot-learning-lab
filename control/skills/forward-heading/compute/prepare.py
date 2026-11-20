"""Small finite lesson manifest, using retained exact binary custody helpers."""
import argparse
from pathlib import Path
from common import DATA, HERE, ROOT, ENVIRONMENT, SIDECAR, TRANSPORT, BASE_LAUNCH, PROTOCOL, read, ref, write, project

RETAINED='control/skills/forward-stop-learning/compute'
INDEX='artifacts/runs/cove50-forward-entry-braking-20261030-r1-ee67e1637e03/source/input-index.json'


def metadata(review,release=None):
    cfg=read(DATA/'config.json')
    previous=read(ROOT/'control/skills/forward-entry-braking/setup/worker-plan.json')
    active=read(DATA/'execution/preparation-active.json')
    owned=ROOT/active['managed_evidence']/'source'
    write(DATA/'environment-config.json',read(ROOT/'control/skills/forward-entry-braking/setup/environment-config.json'))
    source=[*HERE.glob('*.py'),HERE/'bootstrap.sh',DATA/'environment-config.json',DATA/'prove_lesson.py',
        ROOT/ENVIRONMENT,ROOT/SIDECAR,ROOT/TRANSPORT,ROOT/BASE_LAUNCH,ROOT/'infra/storage/storage_lifecycle.py',
        ROOT/'data/forward-stop-learning-20261030-independent/prove_dict_cache.py',
        *[ROOT/RETAINED/n for n in ('common.py','worker.py','launch.py','prepare.py','prove.py','session.py','environment.py','attest.py','bootstrap.sh')]]
    write(DATA/'worker-seal.json',dict(schema='cove50.forward-heading.worker-source-seal.v1',
        files=[ref(p)for p in sorted(set(source))],scope='Thin exact retained operation projections, attached owner, new owned baseline transfer and CPU component fixture; no new native runtime/transport framework.'))
    cases=[dict(name='train48',mode='train',seed=cfg['train_seed'],checkpoint='baseline224',
        planned=dict(simulator_transitions=36864,native_physics_steps=2949120,optimizer_steps=960,ppo_rounds=48))]
    controls=dict(forward30=400,forward25=400,**{'short-stop':150,'long-stop':300},quiet=400)
    for seed in cfg['after_family_seeds']:
        for profile in cfg['profiles']:
            n=controls[profile]
            cases.append(dict(name=f'after-{profile}-{seed}',mode='replay',profile=profile,seed=seed,checkpoint='challenger272',
                planned=dict(simulator_transitions=n,native_physics_steps=n*80,optimizer_steps=0,ppo_rounds=0)))
    children=[dict(name='hardware',process_s=30,join_s=15,pack_s=20),dict(name='bootstrap',process_s=600,join_s=15,pack_s=20)]
    for i,case in enumerate(cases):
        budget=600 if case['mode']=='train' else 180 if i==1 else {400:150,300:120,150:90}[case['planned']['simulator_transitions']]
        children.append(dict(name=case['name'],process_s=budget,join_s=15,pack_s=20))
    slack=7800-640-sum(c['process_s']+c['join_s']+c['pack_s']for c in children)
    if slack<=0:
        raise ValueError('Full finite worker ledger required')
    plan=dict(schema='cove50.forward-heading.worker-plan.v1',job_name='cove50-forward-heading-20261030-r1',status='source_frozen_awaiting_root_release',
        configuration=ref(DATA/'config.json'),source_seal=ref(DATA/'source-seal.json'),worker_seal=ref(DATA/'worker-seal.json'),
        cases=cases,children=children,hard_deadline_s=7800,transport_reserve_s=640,slack_s=slack,absolute_ttl_safety_s=120,
        provider_policy=dict(executionTimeout=8400000,ttl=10800000),completion_sidecar=PROTOCOL,
        helpers=dict(sidecar=ref(ROOT/SIDECAR),transport=ref(ROOT/TRANSPORT)),dependency_manifest=previous['dependency_manifest'],
        library_sources=previous['library_sources'],planned=cfg['totals'],root_release_required=True,native_source_review=ref(review),
        dictionary_cache_fixture=ref(DATA/'prove_lesson.py'),CPU_fixture_result=ref(DATA/'cpu-fixture.json'),
        initial_CPU_fixture_correction=ref(DATA/'cpu-fixture-metadata-correction.json'),initial_CPU_source=ref(owned/'initial-cpu-fixture-source.py'),
        baseline_ownership=ref(owned/'reference-baselines/ownership.json'),baseline_regeneration=ref(owned/'reference-baseline-regeneration.json'),
        baseline_initialization_hashes=ref(owned/'reference-baselines/initialization-field-hashes.json'),
        retained_operational_ancestry=ref(ROOT/'data/forward-stop-learning-20261030-independent/compute-source-review.json'),
        local_launcher_provenance=ref(ROOT/'data/forward-stop-learning-20261030/execution/launcher-restoration.json'),
        observed_previous_timing=ref(ROOT/'artifacts/runs/cove50-forward-entry-braking-20261030-r1-ee67e1637e03/evidence/worker-report.json'),
        budget_rationale='E3 train24=126.31s and R2train50=268.06s;600s includes fresh32slot compile+48fit. First1slot after180s, later8/6/3s visits150/120/90s;23children each15sjoin+20spack,640stransport and positive slack. No baseline reruns or new bootstrap probe.',
        first_inspection_after_s=900,minimum_inspection_spacing_s=900,maximum_foreground_inspections=8,
        source_only_proof='Actual extracted serialized selfcontained worker/21native command routes, corrected actualCPU schedule/reward fixture and interrupted-wrapper preservation.',
        storage='One attached managed source/native owner. Exact reviewed released archive+proof/rootmarker precede one submit; local consumers wait never polls provider. Timeout/error interrupts wrapper preserving live inputs.')
    write(DATA/'worker-plan.json',plan)
    excluded=('control/skills/forward-entry-braking/','data/forward-entry-braking-next-20261030/',
        'data/forward-entry-braking-next-20261030-independent/','control/skills/forward-stop-learning/',
        'data/forward-stop-learning-20261030/','data/forward-stop-learning-20261030-independent/')
    unused_checkpoints={'artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt',
        'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/evidence/fit-through-200/candidate.pt'}
    sources={r['path']:r for r in read(ROOT/INDEX)['files']if not r['path'].startswith(excluded) and r['path'] not in unused_checkpoints}
    extras=source+[review,ROOT/INDEX,ROOT/plan['dependency_manifest']['path'],ROOT/plan['retained_operational_ancestry']['path'],
        ROOT/plan['local_launcher_provenance']['path'],ROOT/plan['observed_previous_timing']['path'],
        ROOT/'control/skills/forward-entry-braking/schedule.py',ROOT/'control/skills/forward-entry-braking/compose.py',
        ROOT/'control/skills/forward-stop-learning/run.py',ROOT/'control/skills/forward-stop-learning/schedule.py']
    extras += [ROOT/r['path']for r in cfg['exact_inputs']+read(DATA/'source-seal.json')['files']]
    extras += [DATA/n for n in ('config.json','source-seal.json','worker-seal.json','worker-plan.json','baseline-config.json',
        'proposal.json','root-method-decision.json','cpu-fixture.json','cpu-fixture-metadata-correction.json')]
    extras += [owned/'initial-cpu-fixture-source.py',owned/'reference-baselines/ownership.json',owned/'reference-baseline-regeneration.json',ROOT/cfg['initial_checkpoint']['path']]
    extras += [DATA/'retain_initialization.py',owned/'reference-baselines/initialization-field-hashes.json']
    if release:
        authority=read(release);extras += [release,ROOT/authority['independent_review']['path']]
    for path in extras:
        sources[ref(path)['path']]=ref(path)
    index=dict(schema='cove50.forward-heading.active-input-index.v1',retained_base=ref(ROOT/INDEX),
        files=[sources[k]for k in sorted(sources)],scope='Exact flat retained R3/body/vendor closure plus narrow immutable source projections and candidate224 donor. Historical baseline distribution bulk remains owned in local managedsource; not used by remote worker.',
        recovery='Verified privateSHA/size routes and active owned/cache aliases; no checkpoint deserialization or live pointer changes.')
    write(DATA/'input-index.json',index)
    write(DATA/'bundle-manifest.json',dict(schema='cove50.forward-heading.capsule.v1',files=index['files'],plan=ref(DATA/'worker-plan.json')))
    return index


def package(index,work,recover):
    source=project(RETAINED+'/prepare.py','a24900c41125d9cb36fc5c0bfdd96093f192c6c6f5f7ef797a7e682c46fddcba',[])
    namespace=dict(__name__='retained_package_helper',__file__=str(HERE/'prepare.py'))
    exec(compile(source,str(HERE/'prepare.py')+':retained-packaging','exec'),namespace)
    namespace['package'](index,work,recover)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--package',action='store_true');parser.add_argument('--review',type=Path,required=True);parser.add_argument('--release',type=Path)
    args=parser.parse_args();index=metadata(args.review.resolve(),args.release.resolve()if args.release else None)
    package(index,args.work.resolve(),args.package)
