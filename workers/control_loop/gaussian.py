"""Pinned NVIDIA PLY -> ParticleField export in an isolated USD environment."""
import argparse,hashlib,json,os,subprocess,sys,urllib.request
from pathlib import Path

COMMIT='24f78996ce3064b25d599cc99b3e74fbb6c1b8fd'
REV='af235475fc2ef60e1add282f044bc64bf58cbebb'
LICENSE_REV='bb2ec30fd17ffb65e24ab40e713517d0c65de41b'
LICENSE_SHA='498bfa5d93d8a4d1a1ed5685d60614a0059ad181465116f945b5a389204c1eb1'
URL=f'https://huggingface.co/datasets/promc/reconstruction-scenes/resolve/{REV}/chili-salmon-bowl/gsplat/chili-salmon-bowl.ply?download=true'
EXPECTED='e99f6e51bc3074c8a239e4e683d2d61aabde4dc054a6617dda64cb84f1b5953a'

def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    cache=Path('/tmp/robot-lab-cache');repo=cache/'3dgrut';venv=cache/'3dgrut-export-env'
    def run(cmd):subprocess.run(cmd,check=True,timeout=180)
    if not repo.exists():run(['git','clone','--filter=blob:none','https://github.com/nv-tlabs/3dgrut.git',str(repo)])
    run(['git','-C',str(repo),'checkout',COMMIT])
    if not venv.exists():run(['uv','venv','--python',sys.executable,str(venv)])
    python=str(venv/'bin/python')
    run(['uv','pip','install','--python',python,'--index-url','https://download.pytorch.org/whl/cpu','torch==2.11.0','torchvision==0.26.0'])
    # Install the pinned project's maintained dependency closure, constrained by
    # the successful CPU environment; exporter imports also traverse datasets.
    run(['uv','pip','install','--python',python,'--no-sources','--constraint',str(Path(__file__).with_name('gaussian-constraints.txt')),str(repo),'usd-core==26.5','nvidia-ncore==19.8.0'])
    source=output/'scene.ply'
    source_cache=cache/'gaussian-inputs'/EXPECTED
    source_cache.parent.mkdir(exist_ok=True)
    candidates=[source_cache,*cache.glob('control-runs/*/evidence/control/gaussian/scene.ply')]
    previous=next((p for p in candidates if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==EXPECTED),None)
    if previous is not None:
        os.link(previous,source)
    else:
        with urllib.request.urlopen(URL,timeout=90) as response,source.open('xb') as dest:
            while chunk:=response.read(1024*1024):dest.write(chunk)
    if hashlib.sha256(source.read_bytes()).hexdigest()!=EXPECTED:raise ValueError('Dataset hash mismatch')
    if not source_cache.exists():os.link(source,source_cache)
    license_url=f'https://huggingface.co/datasets/promc/reconstruction-scenes/raw/{LICENSE_REV}/README.md'
    license_bytes=urllib.request.urlopen(license_url,timeout=30).read()
    if hashlib.sha256(license_bytes).hexdigest()!=LICENSE_SHA:raise ValueError('License README hash mismatch')
    (output/'dataset-README.md').write_bytes(license_bytes)
    env=dict(os.environ,PYTHONPATH=str(repo))
    subprocess.run([python,'-m','threedgrut.export.scripts.transcode',str(source),'-o',str(output/'scene.usdz'),'--format','lightfield'],cwd=repo,env=env,check=True,timeout=180)
    # Read bounds with the maintained PLY parser, in the exporter environment.
    code="from plyfile import PlyData;import numpy as np,json,sys;v=PlyData.read(sys.argv[1])['vertex'];p=np.stack([v[k] for k in ('x','y','z')],axis=1);print(json.dumps({'min':p.min(0).tolist(),'max':p.max(0).tolist(),'count':len(p)}))"
    bounds=json.loads(subprocess.check_output([python,'-c',code,str(source)],env=env))
    inventory=subprocess.check_output(['uv','pip','freeze','--python',python])
    (output/'packages.txt').write_bytes(inventory)
    report=dict(status='pass',exporter_commit=COMMIT,dataset_url=URL,dataset_sha256=EXPECTED,license='CC0-1.0',license_revision=LICENSE_REV,license_readme_sha256=LICENSE_SHA,bounds=bounds,scene_sha256=hashlib.sha256((output/'scene.usdz').read_bytes()).hexdigest(),scope='Precomputed tabletop Gaussian export; no reconstruction training, no room capture or collision evidence.')
    (output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.output.resolve())
