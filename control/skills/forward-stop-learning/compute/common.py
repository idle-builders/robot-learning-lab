"""Finite forward lesson identities and commands; stdlib only."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/forward-stop-learning-20261030'
SIDECAR = 'data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/sidecar.py'
TRANSPORT = 'control/common/compute/transport.py'
ENVIRONMENT = 'data/basic-skills-retry-20261029/arm-short-entry-curriculum-compute-r2/environment.py'
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
    for row in read(DATA / 'source-seal.json')['files'] + read(DATA / 'worker-seal.json')['files'] + cfg['exact_inputs']:
        if ref(ROOT / row['path']) != row: raise ValueError('Frozen source differs: ' + row['path'])
    if (plan['configuration'] != ref(DATA / 'config.json') or plan['source_seal'] != ref(DATA / 'source-seal.json')
            or plan['worker_seal'] != ref(DATA / 'worker-seal.json')):
        raise ValueError('Plan configuration/seal differs')
    if plan['job_name'] != 'cove50-forward-stop-20261030-r2' or plan['completion_sidecar'] != PROTOCOL:
        raise ValueError('Exact named one-round protocol required')
    if (cfg['train']['rounds'], cfg['train']['controls'], cfg['train']['adam_calls'],
            cfg['evaluation']['whole_visits'], cfg['totals']['controls'], cfg['totals']['physics']) != (50,38400,1000,20,45000,3600000):
        raise ValueError('Frozen native accounting domains differ')
    if sum(r['process_s'] + r['join_s'] + r['pack_s'] for r in plan['children']) + plan['transport_reserve_s'] + plan['slack_s'] != plan['hard_deadline_s']:
        raise ValueError('Finite worker budget ledger differs')
    expected = ['before-' + p + '-' + str(s) for s in cfg['family_seeds'] for p in cfg['profiles']]
    expected += ['train50'] + ['after-' + p + '-' + str(s) for s in cfg['family_seeds'] for p in cfg['profiles']]
    if [x['name'] for x in plan['cases']] != expected: raise ValueError('Exact baseline/train/after order required')
    if manifest:
        for row in read(DATA / 'bundle-manifest.json')['files']:
            if ref(ROOT / row['path']) != row: raise ValueError('Capsule member differs: ' + row['path'])
    return plan


def command(case, checkpoint, evidence, work, python=None):
    argv = [str(python or ROOT / '.venv-worker/bin/python'), '-B', '-u',
        str(HERE.parent / 'run.py'), '--mode', case['mode'], '--config', str(DATA / 'config.json'),
        '--checkpoint', str(checkpoint), '--seed', str(case['seed']), '--out', str(evidence / case['name']),
        '--work', str(work / case['name']), '--device', 'cuda:0']
    if case['mode'] == 'replay': argv += ['--profile', case['profile']]
    return argv


def completed(case, directory):
    outer = read(directory / 'report.json'); native = outer.get('native', {})
    expected = case['planned']
    observed = {k: native.get(k) for k in expected}
    if outer.get('status') != 'completed' or native.get('status') != 'completed' or observed != expected:
        raise ValueError('Original native stage incomplete; retain without retry: ' + case['name'])
    if case['mode'] == 'train':
        restored = outer['restored_state']
        if (restored['adam_parameters'], restored['adam_step_counts'], restored['completed_rounds'], restored['common_counter']) != (17,[4000],200,25200):
            raise ValueError('Exact learner initialization differs')
        if native['adam_parameter_step_counts'] != [5000] or outer['final_common_counter'] != 26400:
            raise ValueError('Exact learner terminal counters differ')
        candidate = directory / 'native/candidate.pt'
        if ref(candidate) != native['candidate']: raise ValueError('Same-run candidate differs')
        return candidate
    return None


def activation(plan, value):
    if value.get('authorized') is not True or value.get('job_name') != plan['job_name'] or value.get('worker_plan') != ref(DATA / 'worker-plan.json') or value.get('source_seal') != ref(DATA / 'source-seal.json'):
        raise PermissionError('Explicit root release for this exact named job required')
    if not value.get('independent_review'): raise PermissionError('Root release must bind independent review')
    if ref(ROOT / value['independent_review']['path']) != value['independent_review']:
        raise ValueError('Released independent review differs')
