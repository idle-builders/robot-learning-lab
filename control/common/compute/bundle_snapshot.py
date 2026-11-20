"""Preserve small exact compute inputs and regenerate a transport archive.

Versioned mesh and sealed checkpoint inputs stay in their original locations;
their identities are verified during regeneration. No provider calls.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','regenerate'))
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args()
    index = args.snapshot / 'snapshot.json'
    if args.action == 'freeze':
        args.snapshot.mkdir(parents=True,exist_ok=False)
        rows = []
        with tarfile.open(args.archive) as archive:
            for item in archive:
                if not item.isfile() or Path(item.name).is_absolute() or '..' in Path(item.name).parts:
                    raise ValueError('Unsafe input bundle member')
                payload = archive.extractfile(item).read()
                retained = item.name.startswith('vendor/') or Path(item.name).suffix.lower() not in ('.stl','.pt','.npz')
                if retained:
                    target = args.snapshot / 'inputs' / item.name
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(payload)
                else:
                    source = ROOT / item.name
                    if not source.is_file() or digest(source) != hashlib.sha256(payload).hexdigest():
                        raise ValueError('Required native input absent: ' + item.name)
                rows.append(dict(path=item.name,bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),mode=item.mode,snapshotted=retained))
        record = dict(archive_name=args.archive.name,sha256=digest(args.archive),bytes=args.archive.stat().st_size,files=rows)
        index.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'source_files_preserved':sum(row['snapshotted'] for row in rows),'sha256':record['sha256']}))
        return
    record = json.loads(index.read_text())
    if args.archive.name != record['archive_name']:
        raise ValueError('Use the original archive filename to preserve the gzip header')
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    with args.archive.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as archive:
        for row in record['files']:
            source = args.snapshot/'inputs'/row['path'] if row['snapshotted'] else ROOT/row['path']
            payload = source.read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Required input identity changed: '+str(source))
            item = tarfile.TarInfo(row['path']);item.size=len(payload);item.mtime=0;item.mode=row['mode']
            archive.addfile(item,io.BytesIO(payload))
    if digest(args.archive) != record['sha256']:
        raise ValueError('Regenerated bundle identity differs')
    print(json.dumps({'sha256':record['sha256'],'bytes':args.archive.stat().st_size}))


if __name__=='__main__':main()
