"""Declared run ownership: disposable work, retained evidence, verified offload.

Only explicitly closed runs are processed. No age-based guess can turn an active
experiment or an arbitrary cache folder into a deletion candidate.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from infra.storage import blob_store as bs

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / 'data/storage-runs'
WORK = ROOT / '.cache/work'
MIN_BYTES = 64 << 10


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def run_id(name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}', name):
        raise ValueError('Use a short run name containing letters, digits, dot, dash or underscore')
    return name + '-' + uuid.uuid4().hex[:12]


@contextmanager
def locked(key):
    folder = ROOT / '.cache/storage-locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def evidence_path(value):
    path = Path(value).absolute()
    if not path.is_relative_to(ROOT / 'artifacts') or path == ROOT / 'artifacts':
        raise ValueError('Choose one completed run below artifacts/, never the whole artifacts tree')
    if path.is_symlink() or not path.resolve().is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Run directory must be a real in-repository directory')
    return path


def seal(record):
    """Freeze the declared completed selection; subsequent writes are held local."""
    from infra.storage.regenerable_artifacts import lookup as derived
    rows = []
    evidence = evidence_path(ROOT / record['evidence'])
    for path in sorted(evidence.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(evidence)
        compact = (relative.parts[0] in {'records', 'source'} or
                   relative.as_posix() in {'report.json', 'config.json', 'resolved-config.json',
                                           'provenance.json', 'README.md', 'scratch-regeneration.json'})
        if compact and path.stat().st_size < MIN_BYTES:
            continue
        if derived(path):
            continue  # Disposable exports belong in work/, not in a native pack.
        before = path.stat()
        sha = bs.digest(path)
        if bs.identity(path.stat()) != bs.identity(before):
            raise RuntimeError('Writer still changing completed evidence: ' + str(path))
        rows.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=before.st_size,
                         mtime_ns=before.st_mtime_ns, sha256=sha))
    plan = RECORDS / (record['id'] + '.files.json.gz')
    with gzip.open(plan.with_suffix('.tmp'), 'wt') as stream:
        json.dump(dict(schema='robot-learning-lab.cold-selection.v1', files=rows), stream)
    os.replace(plan.with_suffix('.tmp'), plan)
    record.update(state='pending', closed_at=bs.stamp(), selection=plan.relative_to(ROOT).as_posix(),
                  selected_files=len(rows), selected_bytes=sum(r['bytes'] for r in rows))


def sync_one(path):
    """Resumable: any transfer/identity failure retains local originals and queue."""
    from infra.storage.reclaim_archived_storage import verify_archived_pack
    with locked(path.stem):
        record = json.loads(path.read_text())
        if record['state'] not in {'pending', 'retry'}:
            return record
        try:
            with gzip.open(ROOT / record['selection'], 'rt') as stream:
                selected = json.load(stream)['files']
            # Check the sealed hashes before handing bytes to the pack writer.
            for row in selected:
                source = ROOT / row['path']
                if source.is_file() and not source.is_symlink():
                    before = source.stat()
                    if (before.st_size != row['bytes'] or before.st_mtime_ns != row['mtime_ns']
                            or bs.digest(source) != row['sha256']
                            or bs.identity(source.stat()) != bs.identity(before)):
                        raise RuntimeError('Completed selection changed; kept local: ' + row['path'])
            bs.archive(ROOT / record['selection'], prune=True)
            # archive() intentionally skips already indexed files. Release those
            # only against the current verified remote identity and fresh hash.
            with bs.connect() as db:
                packs = {r['key']: dict(r) for r in db.execute('SELECT * FROM packs')}
            checked = set()
            s3 = None
            for expected in selected:
                row = bs.lookup(ROOT / expected['path'])
                if not row or (row['sha256'], row['bytes']) != (expected['sha256'], expected['bytes']):
                    raise RuntimeError('No exact recovery route: ' + expected['path'])
                source = ROOT / row['path']
                if source.is_file() and not source.is_symlink():
                    if row['pack_key'] not in checked:
                        s3 = s3 or bs.client()
                        verify_archived_pack(s3, packs[row['pack_key']])
                        checked.add(row['pack_key'])
                    before = source.stat()
                    if (before.st_mtime_ns != expected['mtime_ns'] or bs.digest(source) != row['sha256']
                            or bs.identity(source.stat()) != bs.identity(before)):
                        raise RuntimeError('Changed materialization kept local: ' + row['path'])
                    source.unlink()
                    with bs.connect() as db:
                        db.execute('UPDATE files SET evicted_at=? WHERE path=?', (bs.stamp(), row['path']))
            record.update(state='cold', synced_at=bs.stamp(), last_error=None)
        except Exception as error:
            record.update(state='retry', last_attempt_at=bs.stamp(), last_error=str(error))
        write(path, record)
        return record


def finish(evidence, name=None):
    """Explicit declaration that this run has no further writers or live inputs."""
    evidence = evidence_path(evidence)
    if not evidence.is_dir():
        raise FileNotFoundError(evidence)
    logical = evidence.relative_to(ROOT).as_posix()
    # Resume a stopped managed invocation rather than orphaning its scratch.
    for existing in RECORDS.glob('*.json'):
        record = json.loads(existing.read_text())
        if record['evidence'] != logical:
            continue
        with locked(record['id']):
            if record['state'] in {'running', 'interrupted'}:
                seal(record)
                record['work_disposable'] = True
                write(existing, record)
        result = sync_one(existing)
        clean()
        return result
    key = run_id(name or evidence.name)
    record = dict(schema='robot-learning-lab.storage-run.v1', id=key,
                  evidence=evidence.relative_to(ROOT).as_posix(), work=None,
                  created_at=bs.stamp(), state='closing')
    path = RECORDS / (key + '.json')
    with locked(key):
        seal(record)
        write(path, record)
    result = sync_one(path)
    clean()
    return result


def run(name, command):
    """Scripts use {work}/{evidence} argv placeholders or LAB_*_DIR variables."""
    key = run_id(name)
    work = WORK / key
    evidence = ROOT / 'artifacts/runs' / key
    work.mkdir(parents=True)
    evidence.mkdir(parents=True)
    path = RECORDS / (key + '.json')
    command = [arg.replace('{work}', str(work)).replace('{evidence}', str(evidence)) for arg in command]
    record = dict(schema='robot-learning-lab.storage-run.v1', id=key, state='running',
                  created_at=bs.stamp(), work=work.relative_to(ROOT).as_posix(),
                  evidence=evidence.relative_to(ROOT).as_posix(), command=command)
    # Preserve direct authored script/config arguments, without recursive input
    # tree copies. Producers still record their transitive native dependencies.
    sources = []
    for arg in command:
        source = ROOT / arg
        if (source.suffix in {'.py', '.sh', '.yaml', '.yml', '.json', '.toml'}
                and source.is_file() and source.resolve().is_relative_to(ROOT)):
            relative = source.resolve().relative_to(ROOT)
            target = evidence / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            sources.append(dict(path=relative.as_posix(), sha256=bs.digest(target)))
    record['direct_sources'] = sources
    with locked(key):
        write(path, record)
        environment = dict(os.environ, LAB_WORK_DIR=str(work), LAB_EVIDENCE_DIR=str(evidence), LAB_RUN_ID=key)
        try:
            result = subprocess.run(command, cwd=ROOT, env=environment)
            record['command_returncode'] = result.returncode
            # Nonzero runs can contain decisive native failures: retain evidence
            # exactly as for any other completed invocation.
            seal(record)
            record['work_disposable'] = True
        except BaseException:
            record.update(state='interrupted', interrupted_at=bs.stamp())
            write(path, record)
            raise
        write(path, record)
    synced = sync_one(path)
    clean()
    print(json.dumps(synced, indent=2))
    return result.returncode or (0 if synced['state'] == 'cold' else 2)


def sync():
    outcomes = []
    for path in sorted(RECORDS.glob('*.json')):
        if json.loads(path.read_text())['state'] not in {'pending', 'retry'}:
            continue
        try:
            outcomes.append(sync_one(path))
        except BlockingIOError:
            continue
    clean()
    return outcomes


def clean():
    """Only declared completed scratch is disposable; no recursive .cache purge."""
    removed = []
    for path in sorted(RECORDS.glob('*.json')):
        try:
            with locked(path.stem):
                record = json.loads(path.read_text())
                if not record.get('work_disposable') or record['state'] not in {'pending', 'retry', 'cold'}:
                    continue
                if record.get('command_returncode') != 0 and record['state'] != 'cold':
                    continue  # Keep failure scratch until its native evidence is safe.
                work = ROOT / record['work']
                if work.parent != WORK or work.is_symlink() or not work.resolve().is_relative_to(WORK):
                    raise ValueError('Invalid managed scratch path')
                if work.exists():
                    shutil.rmtree(work)
                    removed.append(record['work'])
                elif record.get('work_removed_at'):
                    continue
                record['work_removed_at'] = bs.stamp()
                write(path, record)
        except BlockingIOError:
            continue
    return removed
