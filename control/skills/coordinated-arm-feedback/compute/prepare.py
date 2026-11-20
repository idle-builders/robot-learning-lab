"""Overlay the fresh lesson on the exact accepted teacher input closure."""
import argparse
from pathlib import Path
from common import ARM,DATA,ENVIRONMENT,FORWARD,HERE,ROOT,BASE_LAUNCH,SIDECAR,TRANSPORT,PROTOCOL,load,read,ref,write

BASE_DATA=ROOT/'data/coordinated-arm-reference-20261030'
SHARED_CACHE=ROOT/'.cache/forward-stop-source-preparation-20261030/input-objects'
TEACHER_WORK=ROOT/'.cache/work/cove50-coordinated-arm-reference-20261030-r1-ac4f4191c750'


def metadata(release=None):
    cfg=read(DATA/'config.json');base=read(BASE_DATA/'worker-plan.json')
    environment=read(BASE_DATA/'environment-config.json');write(DATA/'environment-config.json',environment)
    scientific=[p for p in HERE.parent.glob('*.py')]
    cfg['source']=[ref(p)for p in sorted(scientific)]
    cfg['status']='source frozen; exact native execution awaits root named release'
    cfg['remaining_before_release']=['independent final source/runtime/worker review','finite exact root release and exclusive provider lane']
    note=' The .03rad bound applies to decoded requested motor residual, not measured joint motion. The fixed-time impulse may be anticipated from command/own action history; this is not a broad recovery test. Pose tracking/load symmetry can compete with useful recovery lean/transfer; report that tradeoff.'
    if note not in cfg['interpretation']:cfg['interpretation']+=note
    cfg['planned']=dict(simulator_controls=12000,physics_substeps=960000,PPO_rounds=4,optimizer_updates=80,new_weight_sets=1)
    write(DATA/'config.json',cfg)
    write(DATA/'source-seal.json',dict(schema='cove50.coordinated-arm-feedback-source.v1',configuration=ref(DATA/'config.json'),files=cfg['source']))
    reused=[ROOT/FORWARD/n for n in ('launch.py','session.py','prepare.py','attest.py')]
    reused+=[ROOT/ARM/n for n in ('worker.py','environment.py','bootstrap.sh')]
    shared=[ROOT/ENVIRONMENT,ROOT/BASE_LAUNCH,ROOT/SIDECAR,ROOT/TRANSPORT,ROOT/'infra/storage/storage_lifecycle.py']
    sources=[*HERE.glob('*.py'),HERE/'bootstrap.sh',DATA/'environment-config.json',*reused,*shared]
    write(DATA/'worker-seal.json',dict(schema='cove50.coordinated-arm-feedback-worker-source.v1',files=[ref(p)for p in sorted(set(sources))]))
    cases=[dict(name='train',mode='train')]
    cases += [dict(name=f'eval-{i}-{policy}',mode='replay',case_index=i,policy=policy)for i in range(4)for policy in ('zero','candidate')]
    children=[dict(name='hardware',process_s=30,join_s=15,pack_s=20),dict(name='bootstrap',process_s=600,join_s=15,pack_s=20),
        dict(name='train',process_s=1800,join_s=30,pack_s=120)]
    children += [dict(name=c['name'],process_s=250,join_s=20,pack_s=45)for c in cases[1:]]
    deadline=sum(c['process_s']+c['join_s']+c['pack_s']for c in children)+600+230
    review=DATA.parent/'coordinated-arm-feedback-20261030-independent/source-review.json'
    plan=dict(schema='cove50.coordinated-arm-feedback-worker-plan.v1',job_name='cove50-coordinated-arm-feedback-20261030-r1',
        configuration=ref(DATA/'config.json'),source_seal=ref(DATA/'source-seal.json'),worker_seal=ref(DATA/'worker-seal.json'),
        native_source_review=ref(review)if review.exists()else None,cases=cases,children=children,
        hard_deadline_s=deadline,transport_reserve_s=600,slack_s=230,absolute_ttl_safety_s=120,
        provider_policy=dict(executionTimeout=(deadline+600)*1000,ttl=(deadline+2400)*1000),
        completion_sidecar=PROTOCOL,helpers=dict(sidecar=ref(ROOT/SIDECAR),transport=ref(ROOT/TRANSPORT)),
        dependency_manifest=base['dependency_manifest'],library_sources=base['library_sources'],planned=cfg['planned'],
        root_release_required=True,first_inspection_after_s=900,minimum_inspection_spacing_s=900,maximum_foreground_inspections=8,
        runtime_estimate='Teacher actual2000oneworld controls plus warm isolated bootstrap576s provider execution. Fit2000batchedcontrol calls4worlds plus PPO, then4000oneworld evalcalls; estimate35–55min with9nativeprocess setups. Conservative finite caps sum above; no smoke/retry/background monitor.',
        managed_paths='Worker LAB_EVIDENCE_DIR=<extracted>/evidence; LAB_WORK_DIR=<extracted>/work. Local pinned .venv-runpod attached owner waits explicit root start/consumer completion; source-only until exact release.')
    write(DATA/'worker-plan.json',plan)
    files={r['path']:r for r in read(BASE_DATA/'input-index.json')['files']}
    extras=[DATA/'config.json',DATA/'source-seal.json',DATA/'worker-seal.json',DATA/'worker-plan.json',*scientific,*sources]
    extras += [ROOT/r['path']for r in cfg['inputs']]
    if review.exists():extras.append(review)
    if release:
        authority=read(release);extras += [release,ROOT/authority['independent_review']['path']]
    for p in extras:files[ref(p)['path']]=ref(p)
    index=dict(schema='cove50.coordinated-arm-feedback-input-index.v1',files=[files[k]for k in sorted(files)],
        base=ref(BASE_DATA/'input-index.json'),scope='Exact accepted teacher1815-file input closure plus fresh source/config/evidence refs only; donor is sole input PT. Live CAD never rewritten.')
    write(DATA/'input-index.json',index);write(DATA/'bundle-manifest.json',dict(schema='cove50.coordinated-arm-feedback-capsule.v1',files=index['files'],plan=ref(DATA/'worker-plan.json')))
    return index


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--package',action='store_true');p.add_argument('--release',type=Path)
    args=p.parse_args();index=metadata(args.release.resolve()if args.release else None)
    helper=load(FORWARD+'/prepare.py','feedback_existing_packager');original=helper.available
    def available(row,cache,db):
        candidates=[SHARED_CACHE/row['sha256'],TEACHER_WORK/'preparation/input-objects'/row['sha256'],
            ROOT/'.cache/work/cove50-coordinated-arm-reference-20261030-r1-owner-b8deae287b03/active-report.json']
        for path in candidates:
            if path.exists()and path.stat().st_size==row['bytes']and ref(path)['sha256']==row['sha256']:return path,False
        return original(row,cache,db)
    helper.available=available
    # Existing active capsules may be replaced only after their exact cheap source
    # has been checked. New release overlays always have their own identity.
    archive=args.work/'input.tar.gz'
    if args.package and archive.exists():
        old=read(DATA/'bundle-receipt.json')['archive'];actual=ref(archive)
        if (old['sha256'],old['bytes'])!=(actual['sha256'],actual['bytes']):raise ValueError('Active capsule changed')
        archive.unlink()
    helper.package(index,args.work.resolve(),args.package)


if __name__=='__main__':main()
