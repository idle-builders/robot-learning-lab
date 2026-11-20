"""Own only selected candidate224 baseline JSON and exact donor bytes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from common import DATA, ROOT, read, ref, write


def main(out):
    cfg=read(DATA/'baseline-config.json')
    if out.exists():
        raise FileExistsError('Fresh owned input destination required')
    temporary=Path(tempfile.mkdtemp(prefix='heading-input-recovery-',dir=ROOT/'.cache'))
    private=None
    def payload(expected):
        nonlocal private
        path=ROOT/expected['path']
        if not path.is_file():
            import sys
            sys.path.insert(0,str(ROOT))
            from infra.storage.blob_store import client,restore_entry
            with sqlite3.connect((ROOT/'data/blob-store.sqlite').as_uri()+'?mode=ro',uri=True) as db:
                db.row_factory=sqlite3.Row
                row=db.execute('SELECT f.* FROM files f JOIN packs p ON p.key=f.pack_key WHERE f.sha256=? AND f.bytes=? AND p.verified_at IS NOT NULL LIMIT 1',
                    (expected['sha256'],expected['bytes'])).fetchone()
            if row is None:
                raise FileNotFoundError('Exact source bytes unavailable: '+expected['path'])
            if private is None:
                private=client()
            path=temporary/expected['sha256']
            restore_entry(dict(row),s3=private,destination=path)
        raw=path.read_bytes()
        if len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']:
            raise ValueError('Required source identity differs: '+expected['path'])
        return raw
    try:
        summary=json.loads(payload(cfg['sources']['summary']))
        if summary['native_index']!=cfg['native_index']:
            raise ValueError('Exact baseline native index differs')
        names=set(cfg['case_names']);prefix=cfg['namespace']
        cases=[copy.deepcopy(c)for c in summary['cases']if c['name']in names]
        episodes=[copy.deepcopy(e)for e in summary['episodes']if e['case']in names]
        if len(cases)!=20 or len(episodes)!=20 or sum(e['controls']for e in episodes)!=6600:
            raise ValueError('Twenty exact complete baseline visits required')
        for case in cases:
            if (case['checkpoint']['sha256'],case['checkpoint']['bytes'])!=(cfg['checkpoint']['sha256'],cfg['checkpoint']['bytes']):
                raise ValueError('Only exact candidate224 baselines')
            case['source_case']=case['name'];case['name']=prefix+case['name']
        for episode in episodes:
            episode['source_case']=episode['case'];episode['case']=prefix+episode['case']
        common=dict(source_native_index=cfg['native_index'],checkpoint=cfg['checkpoint'],namespace=prefix,
            meaning=cfg['meaning'],new_native_work=0)
        outputs={'summary.json':dict(schema='cove50.candidate224.baseline-summary.v1',source=cfg['sources']['summary'],
            native_index=cfg['native_index'],cases=cases,episodes=episodes,**common)}
        for name in ('curves','distributions-used'):
            source=json.loads(payload(cfg['sources'][name]))
            groups={prefix+k:v for k,v in source['groups'].items() if k.split('/')[0]in names}
            outputs[name+'.json']=dict(schema='cove50.candidate224.baseline-'+name+'.v1',source=cfg['sources'][name],groups=groups,**common)
        out.mkdir(parents=True)
        for name,value in outputs.items():
            (out/name).write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        candidate=out/'candidate224.pt';candidate.write_bytes(payload(cfg['checkpoint']))
        rows=[]
        for path in [*(out/n for n in outputs),candidate]:
            row=ref(path);row['path']=path.name;rows.append(row)
        write(out/'ownership.json',dict(schema='cove50.candidate224.owned-inputs.v1',source=ref(Path(__file__)),
            configuration=ref(DATA/'baseline-config.json'),original_inputs=cfg,owned_files=rows,
            historical_visits=20,historical_controls=6600,new_native_work=0,NPZ_reads=0,checkpoint_deserializations=0,
            NN=0,physics=0,provider=0,namespace_note='Only case labels/group keys are prefixed baseline224/; all original measurements and trace source references are retained.'))
        print(json.dumps(dict(ownership=ref(out/'ownership.json'),files=len(rows),new_native_work=0)))
    finally:
        shutil.rmtree(temporary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
