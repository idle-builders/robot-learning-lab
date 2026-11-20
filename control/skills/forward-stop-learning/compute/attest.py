"""Pinned installed-source/CUDA visibility; no task/model/NN/physics construction."""
import importlib.metadata
from importlib.machinery import PathFinder
import hashlib
import json
import platform
import sys
from pathlib import Path
from common import DATA, ROOT, verify

plan = verify(manifest=False)  # Bootstrap moved only exact generated checkout metadata.
base = json.loads((ROOT / plan['dependency_manifest']['path']).read_text())
versions = {n: importlib.metadata.version(n) for n in base['dependency_versions']}
for name, expected in base['dependency_versions'].items():
    if versions[name].split('+')[0] != expected: raise ValueError('Dependency version differs: ' + name)
for row in plan['library_sources']:
    parts = Path(row['module']).parts; spec = PathFinder.find_spec(parts[0])
    if spec is None or not spec.origin: raise ValueError('Missing installed source: ' + row['module'])
    path = Path(spec.origin).resolve().parent.joinpath(*parts[1:])
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(),len(raw)) != (row['sha256'],row['bytes']):
        raise ValueError('Installed source differs: ' + row['module'])
import torch
if torch.__version__.split('+')[0] != '2.9.1' or torch.version.cuda != '12.8' or not torch.cuda.is_available():
    raise ValueError('Exact Torch/cu128 CUDA runtime unavailable')
print(json.dumps(dict(schema='cove50.forward-stop-environment.v1', python=sys.version, platform=platform.platform(),
    versions=versions,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),installed_sources=len(plan['library_sources']),
    task_imports=0,checkpoint_loads=0,NN_queries=0,models=0,physics=0,optimizer_updates=0),indent=2))
