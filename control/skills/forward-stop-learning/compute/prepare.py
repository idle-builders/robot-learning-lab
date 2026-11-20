"""Small exact capsule recipe, deduplicated active cache, existing private recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
from common import DATA, HERE, ROOT, ENVIRONMENT, SIDECAR, TRANSPORT, BASE_LAUNCH, PROTOCOL, read, ref, write

BASE='data/forward-urdf-retry-20261026/compute/fit-bundle-r2/manifest.json'
RECENT='data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/compute/arm_short_entry_curriculum/bundle-manifest.json'
OLD_PLAN='data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/pilot-plan.json'


def metadata(release=None):
    cfg=read(DATA/'config.json');old=read(ROOT/OLD_PLAN)
    environment=read(ROOT/'data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/environment-config.json')
    write(DATA/'environment-config.json',environment)
    fixture=ROOT/'data/forward-stop-learning-20261030-independent/prove_dict_cache.py'
    source=[*HERE.glob('*.py'),HERE/'bootstrap.sh',DATA/'environment-config.json',ROOT/ENVIRONMENT,ROOT/SIDECAR,ROOT/TRANSPORT,ROOT/BASE_LAUNCH,ROOT/'infra/storage/storage_lifecycle.py',fixture]
    write(DATA/'worker-seal.json',dict(schema='cove50.forward-stop-worker-source-seal.v1',files=[ref(p)for p in sorted(source)],scope='Authored worker/local-session and unchanged reused helper functions. Native source has a separate seal.'))
    cases=[]
    for phase in ('before','after'):
        if phase=='after':
            cases.append(dict(name='train50',mode='train',seed=cfg['train_seed'],checkpoint='baseline200',planned=dict(simulator_transitions=38400,native_physics_steps=3072000,optimizer_steps=1000,ppo_rounds=50)))
        durations={'forward30':400,'forward25':400,'short-stop':150,'long-stop':300,'quiet':400}
        for seed in cfg['family_seeds']:
            for profile in cfg['profiles']:
                controls=durations[profile]
                cases.append(dict(name=phase+'-'+profile+'-'+str(seed),mode='replay',profile=profile,seed=seed,
                    checkpoint='baseline200'if phase=='before'else'challenger250',planned=dict(simulator_transitions=controls,native_physics_steps=controls*80,optimizer_steps=0,ppo_rounds=0)))
    children=[dict(name='hardware',process_s=30,join_s=15,pack_s=20),dict(name='bootstrap',process_s=600,join_s=15,pack_s=20)]
    children += [dict(name=c['name'],process_s=1800 if c['mode']=='train'else 180,join_s=15,pack_s=20)for c in cases]
    plan=dict(schema='cove50.forward-stop-worker-plan.v1',job_name='cove50-forward-stop-20261030-r2',status='source_frozen_awaiting_root_release',
        configuration=ref(DATA/'config.json'),source_seal=ref(DATA/'source-seal.json'),worker_seal=ref(DATA/'worker-seal.json'),
        cases=cases,children=children,hard_deadline_s=7800,transport_reserve_s=640,slack_s=325,
        absolute_ttl_safety_s=120,provider_policy=dict(executionTimeout=8400000,ttl=10800000),completion_sidecar=PROTOCOL,
        helpers=dict(sidecar=ref(ROOT/SIDECAR),transport=ref(ROOT/TRANSPORT)),dependency_manifest=ref(ROOT/BASE),
        library_sources=old['library_sources'],planned=cfg['totals'],root_release_required=True,
        native_source_review=ref(ROOT/'data/forward-stop-learning-20261030-independent/native-source-review.json'),
        dictionary_cache_fixture=ref(fixture),
        operational_repair_review=ref(ROOT/'data/forward-stop-learning-20261030-independent/r2-operational-repair-review.json'),
        first_inspection_after_s=900,minimum_inspection_spacing_s=900,maximum_foreground_inspections=8,
        source_only_proof='Exact extracted worker imports its own native entry/schedule and commands; no path-preloaded checkout substitute. Actual CUDA/dependency attestation later is a counted zero-native bootstrap child.',
        storage='Managed attached local session preserves active materialization through local consumers-complete; timeout/interrupt signals wrapper INTERRUPTED path. Work/evidence UUID is late-bound by infra/lab storage run.')
    write(DATA/'worker-plan.json',plan)
    sources={}
    for row in read(ROOT/BASE)['files']:
        if row['path'].startswith('control/common/compute/')and not row['path'].endswith('/requirements-worker.txt'):continue
        sources[row['path']]={k:row[k]for k in ('path','sha256','bytes')}
    # Reuse only required current-model/vendored/source inputs, not the arm
    # projection/configuration/task forests. Full body seal1470 remains exact.
    recent=read(ROOT/RECENT)['files']
    for row in recent:
        if row['path'].startswith(('artifacts/robot/model/timestep/','control/common/mjlab_ape/src/')):
            sources[row['path']]=row
    sources[cfg['initial_checkpoint']['path']]=cfg['initial_checkpoint']
    extras=[ROOT/x['path']for x in cfg['exact_inputs']]
    extras += [HERE.parent/'run.py',HERE.parent/'schedule.py',DATA/'config.json',DATA/'source-seal.json',DATA/'worker-seal.json',DATA/'worker-plan.json',DATA/'environment-config.json',ROOT/BASE]
    extras += source + [ROOT/plan['native_source_review']['path'],ROOT/plan['operational_repair_review']['path']]
    if release:
        authority=read(release);extras += [release,ROOT/authority['independent_review']['path']]
    for path in extras:sources[ref(path)['path']]=ref(path)
    index=dict(schema='cove50.forward-stop-active-input-index.v1',base_manifest=ref(ROOT/BASE),selected_body_manifest=ref(ROOT/RECENT),
        files=[sources[k]for k in sorted(sources)],scope='Exact logical worker inputs; immutable original identities plus narrow current source overlay. Binary bytes copied/hashed only, never deserialized.',
        recovery='Existing verified private blob index by SHA256+size into task-scoped cache; do not rewrite historical aliases or live design/model pointers.')
    write(DATA/'input-index.json',index)
    write(DATA/'bundle-manifest.json',dict(schema='cove50.forward-stop-capsule.v1',files=index['files'],plan=ref(DATA/'worker-plan.json')))
    return index


def candidate_paths(row):
    p=ROOT/row['path'];yield p
    if row['path'].startswith('vendor/microduck_rl/'):
        yield ROOT/'.venv-control/src/microduck_rl'/Path(row['path']).relative_to('vendor/microduck_rl')
    if row['path'].startswith('vendor/bam/'):
        yield ROOT/'.venv-control/lib/python3.12/site-packages/bam'/Path(row['path']).relative_to('vendor/bam')


def available(row,cache,db):
    for path in [*candidate_paths(row),cache/row['sha256']]:
        if path.is_file()and path.stat().st_size==row['bytes']:
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()==row['sha256']:return path,False
    cold=db.execute('SELECT f.*,p.verified_at FROM files f JOIN packs p ON p.key=f.pack_key WHERE f.sha256=? AND f.bytes=? AND p.verified_at IS NOT NULL LIMIT 1',(row['sha256'],row['bytes'])).fetchone()
    if cold is None:raise FileNotFoundError('No exact local/private source input: '+row['path'])
    return None,dict(cold)


def package(index, work, recover):
    cache=work/'input-objects';paths={};restored=[];missing=[];cold_objects={}
    with sqlite3.connect((ROOT/'data/blob-store.sqlite').as_uri()+'?mode=ro',uri=True)as db:
        db.row_factory=sqlite3.Row
        for row in index['files']:
            path,cold=available(row,cache,db)
            if path is None:missing.append(row['path'])
            else:paths[row['path']]=path
            if cold:
                restored.append(row);cold_objects[row['sha256']]=cold
    if not recover:
        print(json.dumps(dict(input_files=len(index['files']),input_bytes=sum(r['bytes']for r in index['files']),private_required_files=len(missing),private_required_bytes=sum(r['bytes']for r in restored),unrecoverable=0)))
        return
    if cold_objects:
        sys.path.insert(0,str(ROOT));from infra.storage.blob_store import restore_entry,client
        cache.mkdir(parents=True,exist_ok=True);private=client()
        def restore(row):return restore_entry(row,s3=private,destination=cache/row['sha256'])
        with ThreadPoolExecutor(max_workers=6)as pool:list(pool.map(restore,cold_objects.values()))
        for row in restored:paths[row['path']]=cache/row['sha256']
    archive=work/'input.tar.gz';temporary=work/'input.tar.gz.partial';work.mkdir(parents=True,exist_ok=True)
    manifest=DATA/'bundle-manifest.json';paths[str(manifest.relative_to(ROOT))]=manifest
    with temporary.open('wb')as output,gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0,compresslevel=6)as gz,tarfile.open(fileobj=gz,mode='w|')as tf:
        for name,path in sorted(paths.items()):
            raw=path.read_bytes();info=tarfile.TarInfo(name);info.mode=0o755 if name.endswith('.sh')else 0o644;info.size=len(raw);tf.addfile(info,io.BytesIO(raw))
    actual=dict(sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),bytes=temporary.stat().st_size)
    if archive.exists():
        if (hashlib.sha256(archive.read_bytes()).hexdigest(),archive.stat().st_size)!=(actual['sha256'],actual['bytes']):raise FileExistsError('Existing active capsule differs; inspect/remove only declared cheap archive before explicit new release packaging')
        temporary.unlink()
    else:temporary.rename(archive)
    with tarfile.open(archive)as tf:
        lookup={r['path']:r for r in index['files']}
        for member in tf:
            if not member.isfile():raise ValueError('Nonregular capsule member')
            if member.name in lookup:
                raw=tf.extractfile(member).read();expected=lookup[member.name]
                if len(raw)!=expected['bytes']or hashlib.sha256(raw).hexdigest()!=expected['sha256']:raise ValueError('Capsule full readback differs')
    receipt=dict(schema='cove50.forward-stop-capsule-receipt.v1',archive=ref(archive),manifest=ref(manifest),input_index=ref(DATA/'input-index.json'),
        source_seal=ref(DATA/'source-seal.json'),worker_seal=ref(DATA/'worker-seal.json'),regular_files=len(paths),
        private_recovered_files=len(restored),private_recovered_unique_objects=len({r['sha256']for r in restored}),
        binary_deserializations=0,NN_queries=0,models=0,physics=0,provider_calls=0,
        regenerate='.venv/bin/python -B control/skills/forward-stop-learning/compute/prepare.py --package --work <active-work> [--release <exact-root-release-if-used>]',
        private_recovery='Exact SHA256/size and verified cold pack offsets, full decompression/SHA verified by existing lab.blob_store.restore_entry; six bounded read-only workers, no archive writer.')
    write(DATA/'bundle-receipt.json',receipt);print(json.dumps(receipt))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--package',action='store_true');p.add_argument('--release',type=Path)
    a=p.parse_args();index=metadata(a.release.resolve()if a.release else None);package(index,a.work.resolve(),a.package)
