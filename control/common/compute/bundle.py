"""Build a dereferenced, source-hashed R3 forward compute input closure.

No provider/API calls or GPU execution. Generated archive is an active transport
input, reproducible from retained local sources with this script invocation.
"""
import argparse
import gzip
import hashlib
from importlib import metadata
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[3]
REL = Path('control/common/compute')
DATA = Path('data/forward-urdf-retry-20261026/compute/fit-bundle-r2')


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def dependency_pins():
    env = default_environment()
    env.update(sys_platform='linux', platform_system='Linux', platform_machine='x86_64',
        python_version='3.12', python_full_version='3.12.12', extra='')
    queue = ['mjlab', 'rsl-rl-lib', 'mujoco', 'mujoco-warp', 'warp-lang', 'torch', 'torchvision', 'PyYAML', 'scipy', 'colorama']
    seen = {}
    while queue:
        name = canonicalize_name(queue.pop())
        if name in seen or name.startswith('nvidia-') or name == 'triton': continue
        dist = metadata.distribution(name)
        seen[name] = dist.version
        for line in dist.requires or []:
            dep = Requirement(line)
            if dep.marker is None or dep.marker.evaluate(env): queue.append(dep.name)
    return dict(sorted(seen.items()))


BOOTSTRAP = r'''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,12), "The source task requires Python3.12; select a compatible pinned image/PYTHON_BIN"'
"$PYTHON_BIN" -m venv --system-site-packages "$ROOT/.venv-worker"
PY="$ROOT/.venv-worker/bin/python"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.pip-cache}"
export WANDB_MODE=offline WANDB_SILENT=true
# Reuse the image's matching CUDA Torch if present. Otherwise install the exact
# public release; Torch metadata pins its LinuxCUDA12.8 NVIDIA dependencies.
if ! "$PY" -c 'import torch,torchvision; assert torch.__version__.split("+")[0]=="2.9.1" and torchvision.__version__.split("+")[0]=="0.24.1" and torch.version.cuda'; then
  "$PY" -m pip install --prefer-binary 'torch==2.9.1' 'torchvision==0.24.1' --index-url https://download.pytorch.org/whl/cu128
fi
# Explicit local-version pins, without re-solving optional hardware dependency
# groups from Microduck/BAM. Their exact clean source trees are vendored below.
"$PY" -m pip install --only-binary=:all: --no-deps -r "$SCRIPT_DIR/requirements-worker.txt"
"$PY" - "$ROOT" <<'PY'
from pathlib import Path
import site,sys
root=Path(sys.argv[1]);target=Path(site.getsitepackages()[0])/'cove50-worker-source.pth'
target.write_text('\n'.join(str(root/p) for p in ['vendor','vendor/microduck_rl/src','control/common/mjlab_ape/src'])+'\n')
PY
"$PY" "$SCRIPT_DIR/verify_inputs.py"
"$PY" "$SCRIPT_DIR/attest.py" > "$ROOT/worker-attestation.json"
"$PY" -m pip freeze > "$ROOT/worker-pip-freeze.txt"
printf '%s\n' "$PY" > "$ROOT/worker-python.txt"
'''


VERIFY = r'''from pathlib import Path
import hashlib,json
root=Path.cwd();manifest=json.loads((root/'data/forward-urdf-retry-20261026/compute/fit-bundle-r2/manifest.json').read_text())
for item in manifest['files']:
 p=root/item['path']
 if p.is_symlink() or not p.is_file(): raise ValueError('Required regular input absent: '+str(p))
 with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
 if actual!=item['sha256']: raise ValueError('Input identity changed: '+str(p))
print(json.dumps(dict(verified_regular_files=len(manifest['files']),selected_model=manifest['selected_model'],source_checkpoint=manifest['source_checkpoint'])))
'''


ATTEST = r'''import contextlib,importlib.metadata as md,json,platform,subprocess,sys
import torch,warp,mujoco
from pathlib import Path
root=Path.cwd();manifest=json.loads((root/'data/forward-urdf-retry-20261026/compute/fit-bundle-r2/manifest.json').read_text())
versions={name:md.version(name) for name in manifest['dependency_versions']}
for name,expected in manifest['dependency_versions'].items():
 if versions[name].split('+')[0]!=expected: raise ValueError(f'{name}: {versions[name]} != {expected}')
if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA is unavailable; no simulator/training claim')
sys.path.insert(0,str(root/'control/skills/walker/task'))
with contextlib.redirect_stdout(sys.stderr):
 from config import source_forward_cfg,morphology_cfg
 from boundary import bind,runtime_package
 from adapter import CHECKPOINT,expand_checkpoint
 from prepare import mlp
 from rsl_rl.models import MLPModel
 from tensordict import TensorDict
 cfg=bind(morphology_cfg(runtime_package().robot_cfg(),play=True))
saved=torch.load(CHECKPOINT,map_location='cuda:0',weights_only=False)
expanded=expand_checkpoint(saved)
obs=TensorDict({'actor':torch.randn(128,81,device='cuda:0'),'critic':torch.randn(128,96,device='cuda:0')},batch_size=[128])
groups={'actor':['actor'],'critic':['critic']}
actor=MLPModel(obs,groups,'actor',24,hidden_dims=(512,256,128),obs_normalization=True,
 distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar')).to('cuda:0')
critic=MLPModel(obs,groups,'critic',1,hidden_dims=(512,256,128),obs_normalization=True).to('cuda:0')
actor.load_state_dict(expanded['actor_state_dict'],strict=True)
critic.load_state_dict(expanded['critic_state_dict'],strict=True)
with torch.inference_mode():
 actor_delta=float((actor(obs)[:,:21]-mlp(saved['actor_state_dict'],obs['actor'][:,:72])).abs().max())
 critic_delta=float((critic(obs)-mlp(saved['critic_state_dict'],obs['critic'][:,:87])).abs().max())
torch.cuda.synchronize()
device_proof=dict(device='cuda:0',rows=128,actor_prefix_max_abs_difference=actor_delta,
 critic_max_abs_difference=critic_delta,source_effective_lr=saved['optimizer_state_dict']['param_groups'][0]['lr'],
 expanded_tensor_devices=sorted({str(t.device) for group in ('actor_state_dict','critic_state_dict') for t in expanded[group].values()}),
 strict_actor_and_value_state_loading=True,simulator_transitions=0,optimizer_steps=0)
print(json.dumps(dict(python=sys.version,platform=platform.platform(),cuda=torch.version.cuda,
 gpu=torch.cuda.get_device_name(0),device_count=torch.cuda.device_count(),versions=versions,
 vendor_sources=manifest['vendor_sources'],task_observation='actor81/critic96/raw24/physical22',
 physics_dt_s=cfg.sim.mujoco.timestep,decimation=cfg.decimation,reward_terms=list(cfg.rewards),cuda_source_expansion=device_proof,
 qualification='CUDA visibility, version/source identity and static task construction only; zero simulator transitions or optimizer updates.'),indent=2))
'''


LAUNCH = r'''#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?use smoke, baseline, train, or adapted}"
ROOT="$(pwd)"
PY="$ROOT/.venv-worker/bin/python"
export WANDB_MODE=offline WANDB_SILENT=true
case "$MODE" in
 smoke) exec "$PY" -u control/skills/walker/task/run.py --mode smoke --device cuda:0 --worlds 1 --steps 4 --seed 135726 --out artifacts/worker/smoke ;;
 baseline) exec "$PY" -u control/skills/walker/task/run.py --mode replay --device cuda:0 --worlds 1 --seed 136101 --velocity-x .30 --out artifacts/worker/baseline ;;
 train) exec "$PY" -u control/skills/walker/task/run.py --mode train --device cuda:0 --worlds 512 --updates 100 --seed 136200 --out artifacts/worker/train ;;
 adapted) exec "$PY" -u control/skills/walker/task/run.py --mode replay --device cuda:0 --worlds 1 --seed 136101 --velocity-x .30 --checkpoint artifacts/worker/train/candidate.pt --out artifacts/worker/adapted ;;
 *) printf 'Unknown stage: %s\n' "$MODE" >&2; exit 2 ;;
esac
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / DATA / 'cove50-r3-forward-inputs.tar.gz')
    args = p.parse_args(); sources = {}
    def add(source, logical=None):
        source = Path(source)
        if not source.is_absolute(): source = ROOT / source
        logical = Path(logical) if logical else source.relative_to(ROOT)
        if logical.is_absolute() or '..' in logical.parts: raise ValueError('Unsafe logical path')
        if not source.is_file(): raise FileNotFoundError(source)
        sources[str(logical)] = source
    def tree(source, logical=None):
        source=Path(source);logical=Path(logical) if logical else source.relative_to(ROOT)
        for file in source.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts and file.suffix not in ('.pyc','.pyo'):
                add(file, logical/file.relative_to(source))
    for section in ('task','policy','runtime'):
        tree(ROOT/'control/skills/walker'/section)
    for name in ('robot/model/current-model.json','control/skills/forward-walk/contracts.py',
        'robot/model/build/runtime.py','robot/model/build/common.py',
        'artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml',
        'artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt',
        'artifacts/ape-cove50-recovery-20261008/ankleonly100-model844.pt',
        'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml',
        'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/agent.yaml',
        'artifacts/ape-basics-v3-20260929/flamingo-stage2/upright-endpoint-screen-v5-no-contact.json',
        'artifacts/ape-upright-foot-hold-20260930/upright-left-mirror-endpoint.json'):
        add(name)
    for name in ('bundle.py','worker_fit.py','recipe.json'):
        add(REL/name)
    pointer=json.loads((ROOT/'robot/model/current-model.json').read_text())
    for field in ('rigid_MJCF','actuators','sensors','timing'):
        entry=pointer[field];add(entry['path'])
        if digest(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Selected input changed: '+field)
    model_path=ROOT/pointer['rigid_MJCF']['path']
    for asset in ET.parse(model_path).getroot().findall('.//mesh'):add(model_path.parent/asset.attrib['file'])
    tree(ROOT/'control/common/mjlab_ape/src')
    microduck=ROOT/'.venv-control/src/microduck_rl'
    commit=subprocess.check_output(['git','-C',str(microduck),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(microduck),'status','--porcelain'],text=True).strip():raise ValueError('Microduck source is dirty; source snapshot needs explicit review')
    tree(microduck/'src','vendor/microduck_rl/src')
    for name in ('LICENSE','pyproject.toml'):add(microduck/name,Path('vendor/microduck_rl')/name)
    bam=metadata.distribution('better-actuator-models')
    tree(Path(bam.locate_file('bam')),'vendor/bam')
    for entry in bam.files or []:
        if str(entry).endswith('/licenses/LICENSE'):
            add(bam.locate_file(entry),'vendor/bam-LICENSE');break
    versions=dependency_pins()
    pins=''.join(f'{name}=={version}\n'for name,version in versions.items() if name not in ('torch','torchvision'))
    generated={str(REL/'bootstrap.sh'):BOOTSTRAP, str(REL/'launch.sh'):LAUNCH,
        str(REL/'verify_inputs.py'):VERIFY,str(REL/'attest.py'):ATTEST,str(REL/'requirements-worker.txt'):pins}
    files=[]
    for logical,path in sorted(sources.items()):files.append(dict(path=logical,sha256=digest(path),bytes=path.stat().st_size,source_was_symlink=path.is_symlink()))
    for logical,content in sorted(generated.items()):files.append(dict(path=logical,sha256=hashlib.sha256(content.encode()).hexdigest(),bytes=len(content.encode()),generated=True))
    manifest=dict(schema='cove50.forward-worker-inputs.v1',files=files,selected_model=pointer['rigid_MJCF'],
        source_checkpoint=next(f for f in files if f['path'].endswith('/bc-best.pt')),
        dependency_versions=versions,vendor_sources=dict(microduck=dict(commit=commit,dirty=False),bam=json.loads(bam.read_text('direct_url.json'))),
        python='3.12',bootstrap=str(REL/'bootstrap.sh'),launcher=str(REL/'launch.sh'),
        fit_worker=str(REL/'worker_fit.py'),fit_recipe=json.loads((ROOT/REL/'recipe.json').read_text()),
        excludes='No CAD assemblies, URDF visual repair bundle, unrelated run history, secrets, provider credentials, venv binaries, caches or symlinks.',
        package_scope='Exact local CPU public package versions; CUDA wheel build necessarily differs. Torch2.9.1 CUDA12.8 wheel metadata pins NVIDIA dependency versions. Vendored Microduck/BAM avoid unused hardware extras.',
        regeneration='.venv-control/bin/python control/common/compute/bundle.py')
    generated[str(DATA/'manifest.json')]=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as gz,tarfile.open(fileobj=gz,mode='w|') as tar:
        for logical in sorted(set(sources)|set(generated)):
            payload=generated[logical].encode() if logical in generated else sources[logical].read_bytes()
            info=tarfile.TarInfo(logical);info.size=len(payload);info.mtime=0;info.mode=0o755 if logical.endswith('.sh') else 0o644
            tar.addfile(info,io.BytesIO(payload))
    # Verify transport bytes, ordinary files and every source hash after gzip/tar.
    lookup={f['path']:f for f in files}
    with tarfile.open(args.out,'r:gz') as tar:
        for member in tar:
            if not member.isfile():raise ValueError('Archive contains nonregular input')
            if member.name in lookup:
                if hashlib.sha256(tar.extractfile(member).read()).hexdigest()!=lookup[member.name]['sha256']:raise ValueError('Archive member hash differs')
    (ROOT/DATA/'manifest.json').write_text(generated[str(DATA/'manifest.json')])
    receipt=dict(schema='cove50.forward-worker-bundle-receipt.v1',archive=str(args.out.resolve()),sha256=digest(args.out),bytes=args.out.stat().st_size,
        regular_files=len(files)+1,input_bytes=sum(f['bytes']for f in files),resolved_symlink_sources=sum(f.get('source_was_symlink',False)for f in files),
        manifest=str(DATA/'manifest.json'),bootstrap=manifest['bootstrap'],launcher=manifest['launcher'],python_path='.venv-worker/bin/python',
        stages=['gpuSourceActorValueExpansionAttestation','gpuSmoke1world2controlsZeroOptimizer',
            'capacity256/512/1024worlds48controlsZeroOptimizer','pairedBefore8sSeeds136101/136201',
            'measured200to400roundsIn100roundNativeAdamContinuationChunks','pairedAfter8sSeeds136101/136201'],
        verification='All archive entries regular; every archived input SHA verified after packing. No GPU launched.')
    (ROOT/DATA/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':main()
