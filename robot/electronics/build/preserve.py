"""Build and audit an explicit private cold-archive selection.

Content already indexed under a mutable name with different bytes is copied to
an immutable SHA-named alias. Existing verified identical bytes are reused.
This script never prunes local files or uploads by itself.
"""
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, sqlite3

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'artifacts/cove50-electronics-integration/r1-20261026'
DATA = ROOT / 'data/electronics-integration-r1-20261026'
RESULT = 'cove50-electronics-integrated-r1-20261026'

def sha(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n')

def build():
    # Final power closeout metadata belongs beside its already-preserved CAD.
    for folder in [ROOT / 'data/power-design-closeout-20261026', DATA]:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.json', '.md', '.csv', '.gz'] and not p.name.startswith('preservation-'):
                q = OUT / 'recovery/frozen-control' / p.relative_to(ROOT)
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, q)
    db = sqlite3.connect(ROOT / 'data/results.sqlite')
    blob = sqlite3.connect(ROOT / 'data/blob-store.sqlite')
    blob.row_factory = sqlite3.Row
    indexed = {r['path']: dict(r) for r in blob.execute('select * from files')}
    paths = {ROOT / p for (p,) in db.execute('select path from assets where result_id=?', (RESULT,))}
    paths.update(p for p in OUT.rglob('*') if p.is_file())
    paths.add(ROOT / 'artifacts/cove50-power-integration/selected-round-20261026/recovery/power-source.bundle')
    selected, mapping = {}, []
    for logical in sorted(paths):
        physical = logical.resolve()
        if not physical.is_relative_to(ROOT):
            raise ValueError(str(logical))
        digest = sha(physical)
        rel = str(physical.relative_to(ROOT))
        existing = indexed.get(rel)
        if existing and (existing['sha256'] != digest or existing['bytes'] != physical.stat().st_size):
            q = OUT / 'recovery/current-content' / digest / physical.name
            q.parent.mkdir(parents=True, exist_ok=True)
            if not q.exists():
                shutil.copyfile(physical, q)
            physical = q
            rel = str(q.relative_to(ROOT))
        stat = physical.stat()
        selected[rel] = dict(path=rel, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest)
        mapping.append(dict(logical_path=str(logical.relative_to(ROOT)), archive_path=rel, sha256=digest, bytes=stat.st_size))
    write(OUT / 'recovery/logical-to-archive-paths.json', dict(schema='cove50.electronics.archive-map.v1', files=mapping))
    map_path = OUT / 'recovery/logical-to-archive-paths.json'
    stat = map_path.stat()
    selected[str(map_path.relative_to(ROOT))] = dict(path=str(map_path.relative_to(ROOT)), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha(map_path))
    plan = dict(schema='robot-learning-lab.cold-selection.v1', reason='Selected integrated electronics CAD and complete source recovery; final power review/closeout metadata retained.',
                files=sorted(selected.values(), key=lambda r: r['path']), local_eviction=False)
    with gzip.open(DATA / 'preservation-selection.json.gz', 'wt') as f:
        json.dump(plan, f)
    print(json.dumps(dict(files=len(selected), bytes=sum(r['bytes'] for r in selected.values()), selection=str((DATA / 'preservation-selection.json.gz').relative_to(ROOT)))), flush=True)

def audit():
    with gzip.open(DATA / 'preservation-selection.json.gz', 'rt') as f:
        plan = json.load(f)
    db = sqlite3.connect(ROOT / 'data/blob-store.sqlite')
    db.row_factory = sqlite3.Row
    issues, packs = [], set()
    for f in plan['files']:
        row = db.execute('select files.*,packs.verified_at from files join packs on files.pack_key=packs.key where path=?', (f['path'],)).fetchone()
        if not row or row['sha256'] != f['sha256'] or row['bytes'] != f['bytes'] or not row['verified_at']:
            issues.append(f['path'])
        else:
            packs.add(row['pack_key'])
    report = dict(schema='cove50.electronics.private-preservation.v1',
                  selection='data/electronics-integration-r1-20261026/preservation-selection.json.gz',
                  files=len(plan['files']), bytes=sum(f['bytes'] for f in plan['files']),
                  verified_packs=len(packs), content_identity_issues=issues, local_eviction=False,
                  verification='Exact SHA/bytes match verified private pack index. New packs receive remote readback and actual range/decompression restoration in lab.blob_store. Identical prior archived bytes reused; mtime differences do not alter content identity. Logical-to-archive map retained.')
    write(DATA / 'preservation-receipt.json', report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['build', 'audit'])
    args = parser.parse_args()
    build() if args.operation == 'build' else audit()
