"""Restore an evicted optional environment from its exact installed package list.

.venv/bin/python -B infra/storage/restore_local_environment.py .venv-learning
Optional --destination creates a scratch verification environment instead.
"""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('environment');p.add_argument('--destination',type=Path);a=p.parse_args()
record=ROOT/'data/storage-cleanup-20261029/lean/environments'/a.environment
metadata=json.loads((record/'runtime.json').read_text())
target=a.destination or ROOT/a.environment
if target.exists():raise SystemExit('Destination already exists: '+str(target))
subprocess.run([metadata['base_executable'],'-m','venv',str(target)],check=True)
subprocess.run([str(target/'bin/python'),'-m','pip','install','--no-deps','-r',str(record/'requirements.txt')],check=True,cwd=ROOT)
subprocess.run([str(target/'bin/python'),'-c','import sys; print(sys.version)'],check=True)
