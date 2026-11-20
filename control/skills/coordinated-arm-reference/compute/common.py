"""One named, zero-update whole22 reference screen; stdlib only."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/coordinated-arm-reference-20261030'
FORWARD = 'control/skills/forward-stop-learning/compute'
ENVIRONMENT = 'data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/environment.py'
SIDECAR = 'data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/sidecar.py'
TRANSPORT = 'control/common/compute/transport.py'
BASE_LAUNCH = 'control/common/compute/launch.py'
PROTOCOL = dict(schema='robot-learning-lab.completion-protocol.v1', completion_bytes_max=1048576,
    final_pack_bytes_max=1073741824, final_members_max=4096, endgame_operations=3,
    net_added_operations=2, additional_reserve_s=240, sidecar_conditional_create=True,
    maximum_automatic_retries=0)


def read(path): return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
def ref(path):
    path = Path(path).resolve()
    with path.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha, bytes=path.stat().st_size)
def load(relative, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def verify(manifest=True):
    plan = read(DATA / 'worker-plan.json'); cfg = read(DATA / 'config.json')
    seal = read(DATA / 'source-seal.json')
    for row in seal['source'] + read(DATA / 'worker-seal.json')['files'] + cfg['inputs']:
        if ref(ROOT / row['path']) != row: raise ValueError('Exact source differs: ' + row['path'])
    if (plan['configuration'], plan['source_seal'], plan['worker_seal']) != (
            ref(DATA/'config.json'),ref(DATA/'source-seal.json'),ref(DATA/'worker-seal.json')):
        raise ValueError('Exact native/worker binding differs')
    if plan['job_name'] != 'cove50-coordinated-arm-reference-20261030-r1' or plan['completion_sidecar'] != PROTOCOL:
        raise ValueError('Exact named single screen required')
    if plan['planned'] != cfg['planned'] or len(plan['cases']) != 1 or plan['cases'][0]['name'] != 'teacher4':
        raise ValueError('Only four intact whole visits in one native child')
    if sum(c['process_s']+c['join_s']+c['pack_s']for c in plan['children'])+plan['transport_reserve_s']+plan['slack_s'] != plan['hard_deadline_s']:
        raise ValueError('Finite budget ledger differs')
    if manifest:
        for row in read(DATA/'bundle-manifest.json')['files']:
            if ref(ROOT/row['path']) != row: raise ValueError('Capsule differs: ' + row['path'])
    return plan


def command(case, checkpoint, evidence, work, python=None):
    if case['name'] != 'teacher4' or checkpoint != ROOT/read(DATA/'config.json')['checkpoint']['path']:
        raise ValueError('Exact intact d2f5 teacher and one child required')
    return [str(python or ROOT/'.venv-worker/bin/python'),'-B','-u',str(HERE.parent/'run.py'),
        '--execute','--config',str(DATA/'config.json'),'--seal',str(DATA/'source-seal.json'),
        '--out',str(evidence/'teacher4'),'--work',str(work/'teacher4'),'--device','cuda:0']


def completed(case, directory):
    report = read(directory/'report.json')
    if report.get('status') != 'completed' or report.get('close_error'):
        raise ValueError('Original native child incomplete; no retry')
    if report['counts'] != dict(simulator_controls=2000,physics_substeps=160000,optimizer_updates=0,new_weight_sets=0):
        raise ValueError('Actual native accounting differs')
    if len(report['cases']) != 4 or any(c.get('controls') != 500 for c in report['cases']) or len(report['original_episodes']) != 4:
        raise ValueError('Four whole visit originals required; no motion scoring')
    return None


def activation(plan, value):
    if (value.get('authorized') is not True or value.get('job_name') != plan['job_name']
            or value.get('worker_plan') != ref(DATA/'worker-plan.json')
            or value.get('source_seal') != ref(DATA/'source-seal.json')):
        raise PermissionError('Root release for this exact named screen required')
    if not value.get('independent_review') or ref(ROOT/value['independent_review']['path']) != value['independent_review']:
        raise ValueError('Exact independent release review required')
