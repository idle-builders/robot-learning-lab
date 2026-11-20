"""Finite fresh-feedback campaign packet, shared durable worker protocol."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
DATA=ROOT/'data/coordinated-arm-feedback-20261030'
ARM='control/skills/coordinated-arm-reference/compute'
FORWARD='control/skills/forward-stop-learning/compute'
BASE_LAUNCH='control/common/compute/launch.py'
SIDECAR='data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/sidecar.py'
ENVIRONMENT='data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/environment.py'
TRANSPORT='control/common/compute/transport.py'
PROTOCOL=json.loads((ROOT/'control/skills/coordinated-arm-reference/setup/worker-plan.json').read_text())['completion_sidecar']


def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def ref(p):
    p=Path(p).resolve()
    with p.open('rb')as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=h,bytes=p.stat().st_size)
def load(p,name):
    spec=importlib.util.spec_from_file_location(name,ROOT/p);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def operational_json(value):
    return load(HERE.parent/'method.py','feedback_structural_json').operational_json(value)


def verify(manifest=True):
    plan=read(DATA/'worker-plan.json');cfg=read(DATA/'config.json')
    if plan['configuration']!=ref(DATA/'config.json')or plan['source_seal']!=ref(DATA/'source-seal.json')or plan['worker_seal']!=ref(DATA/'worker-seal.json'):
        raise ValueError('Exact feedback scientific/worker binding differs')
    for row in read(DATA/'source-seal.json')['files']+read(DATA/'worker-seal.json')['files']+cfg['inputs']:
        if ref(ROOT/row['path'])!=row:raise ValueError('Exact feedback source/input differs: '+row['path'])
    if plan['job_name']!='cove50-coordinated-arm-feedback-20261030-r1'or plan['completion_sidecar']!=PROTOCOL:
        raise ValueError('Exact named finite feedback lesson required')
    if sum(c['process_s']+c['join_s']+c['pack_s']for c in plan['children'])+plan['transport_reserve_s']+plan['slack_s']!=plan['hard_deadline_s']:
        raise ValueError('Finite worker ledger differs')
    if manifest:
        for row in read(DATA/'bundle-manifest.json')['files']:
            if ref(ROOT/row['path'])!=row:raise ValueError('Portable exact source differs: '+row['path'])
    return plan


def command(case,checkpoint,evidence,work,python=None):
    args=[str(python or ROOT/'.venv-worker/bin/python'),'-B','-u',str(HERE.parent/'run.py'),'--execute',
        '--method',str(DATA/'config.json'),'--out',str(evidence/case['name']),'--work',str(work/case['name']),'--device','cuda:0']
    if case['mode']=='train':return args+['--mode','train']
    args+=['--mode','replay','--case-index',str(case['case_index'])]
    return args+(['--zero-residual']if case['policy']=='zero'else['--checkpoint',str(evidence/'train/candidate.pt')])


def completed(case,directory):
    native=read(directory/'report.json')
    if native['status']!='completed':raise ValueError('Native child incomplete; no retry')
    expected=dict(simulator_controls=8000,physics_substeps=640000,PPO_rounds=4,optimizer_updates=80,new_weight_sets=1)if case['mode']=='train'else dict(simulator_controls=500,physics_substeps=40000,PPO_rounds=0,optimizer_updates=0,new_weight_sets=0)
    if native['counts']!=expected:raise ValueError('Observed completed accounting differs from finite child')


def activation(plan,value):
    if value.get('authorized')is not True or value.get('job_name')!=plan['job_name']or value.get('worker_plan')!=ref(DATA/'worker-plan.json')or value.get('source_seal')!=ref(DATA/'source-seal.json'):
        raise PermissionError('Lead release for exact feedback job required')
    if ref(ROOT/value['independent_review']['path'])!=value['independent_review']:raise ValueError('Exact source review differs')
