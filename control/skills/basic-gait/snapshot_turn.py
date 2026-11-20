"""Freeze only added turning inputs; regenerate exact transport after source edits."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/submitted-source/input-snapshot'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('freeze','regenerate'));p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);args=p.parse_args()
    if args.action=='freeze':
        args.snapshot.mkdir(parents=True,exist_ok=False);(args.snapshot/'objects').mkdir()
        base=json.loads((BASE/'snapshot.json').read_text());lookup={r['path']:r for r in base['files']};rows=[]
        with tarfile.open(args.archive) as archive:
            for member in archive:
                if not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:raise ValueError('Unsafe archive member')
                payload=archive.extractfile(member).read();sha=hashlib.sha256(payload).hexdigest();old=lookup.get(member.name)
                row=dict(path=member.name,sha256=sha,bytes=len(payload),mode=member.mode)
                if old and old['sha256']==sha:
                    row.update(source='base',base_snapshotted=old['snapshotted'])
                elif Path(member.name).suffix.lower() in ('.stl','.pt','.npz'):
                    source=ROOT/member.name
                    if not source.is_file() or digest(source)!=sha:raise ValueError('Required immutable native input absent: '+member.name)
                    row['source']='required_native'
                else:
                    target=args.snapshot/'objects'/sha
                    if target.exists() and target.read_bytes()!=payload:raise ValueError('CAS collision')
                    target.write_bytes(payload);row['source']='object'
                rows.append(row)
        record=dict(schema='cove50.turn-input-snapshot.v1',archive_name=args.archive.name,sha256=digest(args.archive),bytes=args.archive.stat().st_size,base_snapshot=dict(path=str(BASE.relative_to(ROOT)),sha256=digest(BASE/'snapshot.json')),files=rows)
        (args.snapshot/'snapshot.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(dict(added_source_objects=len(list((args.snapshot/'objects').iterdir())),base_files=sum(r['source']=='base' for r in rows),required_native_files=sum(r['source']=='required_native' for r in rows),sha256=record['sha256'])))
        return
    record=json.loads((args.snapshot/'snapshot.json').read_text());base=ROOT/record['base_snapshot']['path']
    if digest(base/'snapshot.json')!=record['base_snapshot']['sha256']:raise ValueError('Base snapshot index changed')
    if args.archive.name!=record['archive_name']:raise ValueError('Original filename required for gzip header')
    args.archive.parent.mkdir(parents=True,exist_ok=True)
    with args.archive.open('wb') as output,gzip.GzipFile(fileobj=output,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as archive:
        for row in record['files']:
            if row['source']=='base':source=base/'inputs'/row['path'] if row['base_snapshotted'] else ROOT/row['path']
            elif row['source']=='object':source=args.snapshot/'objects'/row['sha256']
            else:source=ROOT/row['path']
            payload=source.read_bytes()
            if len(payload)!=row['bytes'] or hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Required source changed: '+str(source))
            member=tarfile.TarInfo(row['path']);member.size=len(payload);member.mode=row['mode'];archive.addfile(member,io.BytesIO(payload))
    if digest(args.archive)!=record['sha256']:raise ValueError('Exact bundle regeneration differs')
    print(json.dumps(dict(sha256=record['sha256'],bytes=args.archive.stat().st_size)))


if __name__=='__main__':main()
