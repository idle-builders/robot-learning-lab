"""Overlay the small arm method on the exercised exact forward input closure."""
import argparse
from pathlib import Path
import hashlib
import json
import tarfile
from common import DATA, ENVIRONMENT, FORWARD, HERE, ROOT, SIDECAR, TRANSPORT, BASE_LAUNCH, PROTOCOL, load, read, ref, write

SHARED_CACHE=ROOT/'.cache/forward-stop-source-preparation-20261030/input-objects'
R1_CAPSULE=ROOT/'.cache/forward-stop-released-preparation-20261030/input.tar.gz'
R1_SHA='153fb5b2c0c3b2f6507c36aaa70b6819fd0dc7c8053b29c8d17edc43d08e93ca'


def inherited():
    path=DATA/'base-runtime.json'
    if path.exists():return read(path)
    with R1_CAPSULE.open('rb')as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=R1_SHA:raise ValueError('Exact exercised R1 capsule required')
    with tarfile.open(R1_CAPSULE)as archive:
        def item(name):return json.load(archive.extractfile('data/forward-stop-learning-20261030/'+name))
        value=dict(schema='cove50.coordinated-arm-reference-inherited-runtime.v1',
            archive=dict(sha256=R1_SHA,bytes=R1_CAPSULE.stat().st_size),
            worker_plan=item('worker-plan.json'),manifest=item('bundle-manifest.json'),
            environment=item('environment-config.json'))
    write(path,value)
    return value


def metadata(release=None):
    cfg=read(DATA/'config.json');base=inherited();forward=base['worker_plan']
    write(DATA/'environment-config.json',base['environment'])
    reused=[ROOT/FORWARD/n for n in ('launch.py','session.py','attest.py','prepare.py')]
    sources=[*HERE.glob('*.py'),HERE/'bootstrap.sh',DATA/'environment-config.json',ROOT/ENVIRONMENT,
        ROOT/SIDECAR,ROOT/TRANSPORT,ROOT/BASE_LAUNCH,ROOT/'infra/storage/storage_lifecycle.py',DATA/'base-runtime.json',*reused]
    write(DATA/'worker-seal.json',dict(schema='cove50.coordinated-arm-reference-worker-seal.v1',
        files=[ref(p)for p in sorted(set(sources))],scope='Narrow arm caller plus unchanged exercised transport/session/environment helpers; native method sealed separately.'))
    children=[dict(name='hardware',process_s=30,join_s=15,pack_s=20),
        dict(name='bootstrap',process_s=600,join_s=15,pack_s=20),
        dict(name='teacher4',process_s=900,join_s=30,pack_s=90)]
    plan=dict(schema='cove50.coordinated-arm-reference-worker-plan.v1',
        job_name='cove50-coordinated-arm-reference-20261030-r1',status='source_frozen_awaiting_root_release',
        configuration=ref(DATA/'config.json'),source_seal=ref(DATA/'source-seal.json'),worker_seal=ref(DATA/'worker-seal.json'),
        native_source_review=ref(ROOT/'data/coordinated-arm-reference-20261030-independent/source-review.json'),
        cases=[dict(name='teacher4',mode='intact_teacher_replay',checkpoint='d2f5')],children=children,
        hard_deadline_s=2600,transport_reserve_s=640,slack_s=240,absolute_ttl_safety_s=120,
        provider_policy=dict(executionTimeout=3000000,ttl=4200000),completion_sidecar=PROTOCOL,
        helpers=dict(sidecar=ref(ROOT/SIDECAR),transport=ref(ROOT/TRANSPORT)),
        dependency_manifest=forward['dependency_manifest'],library_sources=forward['library_sources'],
        planned=cfg['planned'],root_release_required=True,first_inspection_after_s=900,
        minimum_inspection_spacing_s=900,maximum_foreground_inspections=4,
        retained_endpoint='Use only existing campaign endpoint after lead releases exclusive lane; this recipe never provisions.',
        runtime_estimate='Historical400control forward visit84.62s inclsetup. One reused world2000controls roughly430–550s plus bootstrapup to600s; first foreground inspection900s. Processcap900s thenSIGINT30s, no retry or extra native probe.',
        managed_paths='Remote children LAB_EVIDENCE_DIR=<extracted-root>/evidence and LAB_WORK_DIR=<extracted-root>/work; authenticated restore into late-bound local managed roots. Attached session stays active through consumers-complete.')
    write(DATA/'worker-plan.json',plan)
    baseline=base['manifest']
    files={}
    for row in baseline['files']:
        name=row['path']
        if name.endswith(('.pt','.npz')) or name.startswith(('control/skills/forward-stop-learning/','data/forward-stop-learning-20261030/')):
            continue
        files[name]=row
    # Existing exact bytes for shared vendors/body/runtime; donor weights and
    # command-only sources are distinct inputs. No old learned model is loaded.
    extras=[ROOT/r['path']for r in read(DATA/'source-seal.json')['source']+cfg['inputs']]
    extras += [HERE.parent/'reference.py',HERE.parent/'run.py',DATA/'config.json',DATA/'goal-table.json',
        DATA/'source-seal.json',DATA/'worker-seal.json',DATA/'worker-plan.json',ROOT/plan['native_source_review']['path'],*sources]
    files[cfg['checkpoint']['path']]=cfg['checkpoint']
    for row in cfg['physical_meshes']:files[row['path']]=row
    if release:
        authority=read(release);extras += [release,ROOT/authority['independent_review']['path']]
    for path in extras:files[ref(path)['path']]=ref(path)
    index=dict(schema='cove50.coordinated-arm-reference-input-index.v1',
        shared_forward_inputs=ref(DATA/'base-runtime.json'),
        files=[files[n]for n in sorted(files)],
        scope='Exact required noncheckpoint forward source/body/vendor bytes plus intact d2f5 and this method. Historical transitive CAD metadata remains historical; selectedR3 pointer and liveuserCAD are not changed.')
    write(DATA/'input-index.json',index)
    write(DATA/'bundle-manifest.json',dict(schema='cove50.coordinated-arm-reference-capsule.v1',files=index['files'],plan=ref(DATA/'worker-plan.json')))
    return index


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--package',action='store_true');parser.add_argument('--release',type=Path)
    args=parser.parse_args();index=metadata(args.release.resolve()if args.release else None)
    shared=load(FORWARD+'/prepare.py','coordinated_arm_existing_packager')
    original=shared.available
    def available(row,cache,db):
        path=SHARED_CACHE/row['sha256']
        if path.exists()and path.stat().st_size==row['bytes']:
            import hashlib
            if hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']:return path,False
        return original(row,cache,db)
    shared.available=available
    if args.package and (args.work/'input.tar.gz').exists():
        previous=read(DATA/'bundle-receipt.json')['archive']
        present=ref((args.work/'input.tar.gz').resolve())
        if (present['sha256'],present['bytes'])!=(previous['sha256'],previous['bytes']):
            raise ValueError('Existing declared cheap capsule changed; retain for inspection')
        # It is a verified derivation of the unchanged exact input index. The
        # release overlay is a new small source set; never reuse the old hash.
        (args.work/'input.tar.gz').unlink()
    shared.package(index,args.work.resolve(),args.package)
    if args.package:
        receipt=read(DATA/'bundle-receipt.json')
        receipt.update(schema='cove50.coordinated-arm-reference-capsule-receipt.v1',
            packager_source=ref(ROOT/FORWARD/'prepare.py'),shared_cache_reused=str(SHARED_CACHE.relative_to(ROOT)),
            regenerate='.venv/bin/python -B control/skills/coordinated-arm-reference/compute/prepare.py --package --work <active-work> [--release <exact-root-release>]')
        write(DATA/'bundle-receipt.json',receipt)


if __name__=='__main__':main()
