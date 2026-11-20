"""Finite next-lesson overlay on the exercised first-feedback exact capsule."""
import argparse
from pathlib import Path
from common import DATA, HERE, ROOT, FORWARD, load, read, ref, write

BASE_DATA = ROOT / 'data/coordinated-arm-feedback-20261030'
SHARED_CACHE = ROOT / '.cache/forward-stop-source-preparation-20261030/input-objects'
PRIOR_WORK = ROOT / '.cache/work/cove50-coordinated-arm-feedback-20261030-r1-1c01dc03a102'


def metadata(release=None):
    cfg = read(DATA / 'config.json')
    base = read(BASE_DATA / 'worker-plan.json')
    original = read(BASE_DATA / 'worker-seal.json')['files']
    scientific_review = DATA.parent / 'coordinated-arm-feedback-next-20261030-independent/source-review.json'
    if ref(DATA / 'config.json') != read(DATA / 'source-seal.json')['configuration']:
        raise ValueError('Reviewed scientific configuration differs; metadata never rewrites it')
    write(DATA / 'environment-config.json', read(BASE_DATA / 'environment-config.json'))
    sources = {row['path']: row for row in original}
    for path in [*HERE.glob('*.py'), HERE / 'bootstrap.sh', DATA / 'environment-config.json',
                 *[ROOT / 'control/skills/coordinated-arm-feedback/compute' / name
                   for name in ('common.py', 'worker.py', 'launch.py', 'session.py', 'prove.py', 'prepare.py')]]:
        sources[ref(path)['path']] = ref(path)
    write(DATA / 'worker-seal.json', dict(schema='cove50.coordinated-arm-feedback-next.worker-source.v1',
                                         files=[sources[name] for name in sorted(sources)]))
    cases = [dict(name='train', mode='train')]
    cases += [dict(name=f'eval-{i}-{policy}', mode='replay', case_index=i, policy=policy)
              for i in range(8) for policy in ('zero', 'candidate')]
    children = [dict(name='hardware', process_s=30, join_s=15, pack_s=20),
                dict(name='bootstrap', process_s=600, join_s=15, pack_s=20),
                dict(name='train', process_s=2400, join_s=30, pack_s=120)]
    children += [dict(name=case['name'], process_s=160, join_s=15, pack_s=30) for case in cases[1:]]
    deadline = sum(row['process_s'] + row['join_s'] + row['pack_s'] for row in children) + 500 + 170
    if deadline != 7200:
        raise ValueError('Exact finite7200s ledger differs')
    plan = dict(schema='cove50.coordinated-arm-feedback-next.worker-plan.v1',
        job_name='cove50-coordinated-arm-feedback-next-20261030-r1',
        configuration=ref(DATA / 'config.json'), source_seal=ref(DATA / 'source-seal.json'),
        worker_seal=ref(DATA / 'worker-seal.json'), native_source_review=ref(scientific_review),
        cases=cases, children=children, hard_deadline_s=deadline, transport_reserve_s=500, slack_s=170,
        absolute_ttl_safety_s=120,
        provider_policy=dict(executionTimeout=7800000, ttl=9600000),
        completion_sidecar=base['completion_sidecar'], helpers=base['helpers'],
        dependency_manifest=base['dependency_manifest'], library_sources=base['library_sources'],
        planned=cfg['planned'], root_release_required=True,
        first_inspection_after_s=900, minimum_inspection_spacing_s=900, maximum_foreground_inspections=9,
        runtime_estimate='R1actual train533s/8kcontrols4worlds and eval~120s/500controls. Next12rounds8worlds estimated~1600s fit plus~1920s16eval and bootstrap/packing:50–83min with vector scaling uncertainty. Train cap2400s and each eval160s; overall7200s, provider7800/TTL9600. No preview, retry or native smoke.',
        managed_paths='Worker LAB_EVIDENCE_DIR=<extracted>/evidence and LAB_WORK_DIR=<extracted>/work. Attached local source owner waits root-only start and consumer-release files; no provider polling in wait.',
        accounting_limits='Primary counters preserve unknown if missing reports; ancillary sets/donor queries are observed lower bounds in that case. PPO rounds come from native reports independently of Adam counts.')
    write(DATA / 'worker-plan.json', plan)
    files = {row['path']: row for row in read(BASE_DATA / 'input-index.json')['files']}
    extras = [DATA / name for name in ('config.json', 'source-seal.json', 'worker-seal.json', 'worker-plan.json')]
    extras += [ROOT / row['path'] for row in cfg['source'] + cfg['inputs']]
    extras += [ROOT / row['path'] for row in sources.values()]
    extras += [scientific_review]
    if release:
        authority = read(release)
        extras += [release, ROOT / authority['independent_review']['path']]
    for path in extras:
        files[ref(path)['path']] = ref(path)
    index = dict(schema='cove50.coordinated-arm-feedback-next.input-index.v1',
        files=[files[name] for name in sorted(files)], base=ref(BASE_DATA / 'input-index.json'),
        scope='Exact exercised first-feedback source/runtime/body closure plus next source/method and actual full07 state. Two input PT identities: d2f5 frozen donor and07 warm learner; no other learned inputs. Live CAD selection never changed.')
    write(DATA / 'input-index.json', index)
    write(DATA / 'bundle-manifest.json', dict(schema='cove50.coordinated-arm-feedback-next.capsule.v1',
                                             files=index['files'], plan=ref(DATA / 'worker-plan.json')))
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--package', action='store_true')
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    index = metadata(args.release.resolve() if args.release else None)
    helper = load(FORWARD + '/prepare.py', 'feedback_next_existing_packager')
    original = helper.available
    def available(row, cache, db):
        for path in (SHARED_CACHE / row['sha256'], PRIOR_WORK / 'preparation/input-objects' / row['sha256']):
            if path.is_file() and path.stat().st_size == row['bytes'] and ref(path)['sha256'] == row['sha256']:
                return path, False
        return original(row, cache, db)
    helper.available = available
    archive = args.work / 'input.tar.gz'
    if args.package and archive.exists():
        before, actual = read(DATA / 'bundle-receipt.json')['archive'], ref(archive)
        if (before['sha256'], before['bytes']) != (actual['sha256'], actual['bytes']):
            raise ValueError('Active source capsule changed')
        archive.unlink()
    helper.package(index, args.work.resolve(), args.package)


if __name__ == '__main__':
    main()
