"""Regenerate the proven native closure and add frozen turning inputs only."""
import argparse,gzip,hashlib,io,json,subprocess,tarfile,tempfile
from pathlib import Path
from replay import ROOT,ref

HERE=Path(__file__).parent
SNAPSHOT=ROOT/'artifacts/cove50-walk-resume/runpod-fit-r2-20261026/submitted-source/input-snapshot'
TURN=ROOT/'data/basic-skills-retry-20261027/turn'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sources={}
    with tempfile.TemporaryDirectory(prefix='turn-native-inputs-') as temporary:
        base=Path(temporary)/'cove50-r3-forward-inputs.tar.gz'
        subprocess.run([str(ROOT/'.venv-control/bin/python'),'-B',str(ROOT/'control/common/compute/bundle_snapshot.py'),
            'regenerate','--snapshot',str(SNAPSHOT),'--archive',str(base)],check=True)
        with tarfile.open(base) as archive:
            for member in archive:
                if not member.isfile():raise ValueError('Nonregular original input')
                sources[member.name]=archive.extractfile(member).read()
    def add(path):
        path=Path(path);logical=str(path.relative_to(ROOT));payload=path.read_bytes()
        if logical in sources and sources[logical]!=payload:raise ValueError('Original native source differs: '+logical)
        sources[logical]=payload
    method_paths=[TURN/'method-r1.json',TURN/'source-comparison-r1/method.json']
    for method_path in method_paths:
        method=json.loads(method_path.read_text());add(method_path)
        for item in method['inputs']+method['implementation']:add(ROOT/item['path'])
    selected=json.loads((ROOT/'robot/model/current-model.json').read_text())
    seal_path=(ROOT/selected['rigid_MJCF']['path']).with_name('evidence-seal.json');add(seal_path)
    seal=json.loads(seal_path.read_text())
    for section in ('models','meshes','contracts'):
        for item in seal[section]:add(ROOT/item['path'])
    for name in ('cuda-batch.json','worker_turn.py','bundle_turn.py','launch_turn.py','turn_transport.py'):add(HERE/name)
    rows=[dict(path=k,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for k,v in sorted(sources.items())]
    manifest=dict(schema='cove50.turn-gpu-input-closure.v1',files=rows,
        original_snapshot=ref(SNAPSHOT/'snapshot.json'),batch=ref(HERE/'cuda-batch.json'),
        selected_model=selected['rigid_MJCF'],regenerate='.venv-control/bin/python -B control/skills/basic-gait/bundle_turn.py --out data/basic-skills-retry-20261027/turn/compute/turn-inputs.tar.gz',
        exclusions='No provider credentials, caches, training episode bulk or CAD assemblies; inherited proven native dependency source closure retained.')
    logical='data/basic-skills-retry-20261027/turn/compute/bundle-manifest.json'
    sources[logical]=(json.dumps(manifest,indent=2)+'\n').encode()
    with args.out.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as archive:
        for name,payload in sorted(sources.items()):
            info=tarfile.TarInfo(name);info.size=len(payload);info.mtime=0;info.mode=0o755 if name.endswith('.sh') else 0o644
            archive.addfile(info,io.BytesIO(payload))
    with tarfile.open(args.out) as archive:
        for item in archive:
            if not item.isfile() or hashlib.sha256(archive.extractfile(item).read()).hexdigest()!=hashlib.sha256(sources[item.name]).hexdigest():raise ValueError('Archive verification differs')
    out=TURN/'compute';out.mkdir(exist_ok=True)
    (out/'bundle-manifest.json').write_bytes(sources[logical])
    receipt=dict(archive=ref(args.out),regular_files=len(sources),input_bytes=sum(map(len,sources.values())),manifest=ref(out/'bundle-manifest.json'),
        bootstrap='control/common/compute/bootstrap.sh',regeneration=manifest['regenerate'],provider_calls=0,native_episodes=0)
    (out/'bundle-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
