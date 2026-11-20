"""Named source/current walking contracts; no optimizer or mechanical changes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
MODEL_POINTER = Path(os.environ.get('COVE50_MODEL_POINTER', ROOT / 'robot/model/current-model.json'))
SELECTED_MODEL = json.loads(MODEL_POINTER.read_text())
MODEL_DIR = (ROOT / SELECTED_MODEL['rigid_MJCF']['path']).parent
OUT = Path(os.environ.get('COVE50_FORWARD_OUT', ROOT / 'artifacts/cove50-forward-walk' / SELECTED_MODEL['version']))
OLD_BODY = ROOT / 'artifacts/ape-cove50-transfer-20261008/cove50-direct-training.xml'
CHECKPOINT = ROOT / 'artifacts/ape-low-level-training-20261015/shared-forward-turn/native/bc/bc-best.pt'
TEACHER_TRACE = ROOT / 'artifacts/ape-low-level-training-20261015/shared-forward-turn/native/score/development/forward-62101-bc.npz'
FITTER = ROOT / 'control/skills/shared_locomotion/fit_balanced_bc.py'
METHOD = ROOT / 'artifacts/ape-low-level-training-20261015/shared-forward-turn/method.json'
sys.path.insert(0, str(ROOT / 'robot/model/build'))
# The frozen runtime is parameterized by this source model directory. Do not
# edit r1 runtime/common or silently pair r2 geometry with r1 sidecars.
os.environ['COVE50_MODEL_DIR'] = str(MODEL_DIR)
from runtime import Runtime
from common import ref, read, write


def names(model):
    return [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]


def old_home(old_names):
    return np.array([-.2 if n.endswith(('hip_pitch', 'ankle_pitch')) else .4 if n.endswith('knee') else 0. for n in old_names])


def checked_sources():
    expected = [(OLD_BODY, '9841b2e9f57dafe7842873e80c43431f506721d729cef98d92c574ca52dc1d0c'),
        (CHECKPOINT, '0e8c6862108bd880e330fd8c07d19bf012d4a034f8be6a2ef808de190e5f301f')]
    for p, digest in expected:
        if ref(p)['sha256'] != digest: raise ValueError(f'Frozen walking source changed: {p}')
    method = read(METHOD)
    for key in ['body', 'bc_fitter', 'actuator_catalog']:
        entry = method['sources'][key]
        if ref(ROOT / entry['path'])['sha256'] != entry['sha256']:
            raise ValueError(f'Legacy recipe source changed: {key}')
    pointer = read(MODEL_POINTER)
    if ref(MODEL_DIR / 'cove50-rigid.xml')['sha256'] != pointer['rigid_MJCF']['sha256']:
        raise ValueError('Selected learning body differs from current model pointer')
    seal = read(MODEL_DIR / 'evidence-seal.json')
    for section in ['models', 'meshes', 'contracts']:
        for entry in seal[section]:
            actual = ref(ROOT / entry['path'])
            if actual['sha256'] != entry['sha256'] or actual['bytes'] != entry['bytes']:
                raise ValueError(f'Selected native input differs: {entry["path"]}')


def mapping(old, current):
    od, nd = mujoco.MjData(old), mujoco.MjData(current)
    mujoco.mj_forward(old, od); mujoco.mj_forward(current, nd)
    old_names, current_names = names(old), names(current)
    rows = []
    for i, name in enumerate(current_names):
        source = None
        if name == 'neck_yaw': source = 'head_yaw'
        elif name.startswith(('R_', 'L_')):
            side = 'left' if name.startswith('R_') else 'right'
            axis = 'elbow_pitch' if name[2:] == 'elbow' else name[2:]
            candidate = side + '_' + axis
            if candidate in old_names: source = candidate
        if source:
            oi = old_names.index(source)
            oj, nj = old.joint(source).id, current.joint(name).id
            alignment = float(od.xaxis[oj] @ nd.xaxis[nj])
            rows.append(dict(current_index=i, current_name=name, source_index=oi, source_name=source,
                sign=1. if alignment >= 0 else -1., neutral_axis_alignment=abs(alignment),
                source_axis_world=od.xaxis[oj].tolist(), current_axis_world=nd.xaxis[nj].tolist(),
                source_origin_world_m=od.xanchor[oj].tolist(), current_origin_world_m=nd.xanchor[nj].tolist(),
                source_zero_rad=float(old.qpos0[old.jnt_qposadr[oj]]), current_zero_rad=float(current.qpos0[current.jnt_qposadr[nj]])))
        else:
            rows.append(dict(current_index=i, current_name=name, source_index=None, source_name=None,
                default_rad=0., reason='New current-body axis absent from source walking actor'))
    return rows


def to_current(values, rows):
    return np.array([row['sign'] * values[row['source_index']] if row['source_index'] is not None else row['default_rad'] for row in rows])


def source_actor():
    spec = importlib.util.spec_from_file_location('frozen_shared_forward_bc', FITTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    actor = module.Actor(saved['actor_state_dict']).eval()
    for parameter in actor.parameters(): parameter.requires_grad_(False)
    return actor


def build_contract():
    checked_sources()
    old = mujoco.MjModel.from_xml_path(str(OLD_BODY))
    current = mujoco.MjModel.from_xml_path(str(MODEL_DIR / 'cove50-rigid.xml'))
    rows = mapping(old, current)
    contract = dict(schema='cove50.forward-walk-transfer-contract.v1',
        source_checkpoint=ref(CHECKPOINT), source_body=ref(OLD_BODY), source_recipe=ref(METHOD),
        source_actor_implementation=ref(FITTER), source_leg_target_trace=ref(TEACHER_TRACE),
        current_model=read(MODEL_POINTER), current_model_pointer=ref(MODEL_POINTER), current_model_evidence_seal=ref(MODEL_DIR / 'evidence-seal.json'),
        source_joint_order=names(old), current_joint_order=names(current), named_mapping=rows,
        removed_source_channels=[n for n in names(old) if n not in [r['source_name'] for r in rows]],
        source_home_rad=old_home(names(old)).tolist(), current_mapped_home_rad=to_current(old_home(names(old)), rows).tolist(),
        source_mass_kg=float(old.body_mass.sum()), current_mass_kg=float(current.body_mass.sum()),
        source_observation72='gyro3,projected_gravity3,q_minus_home21,qd21,previous_raw_actor_action21,command3. Physical leg target=source_home+.3*raw_action(first10); upper11actionscale0. Frozen actor normalizer std+.01.',
        current_observation75='Timed Runtime packet q22,qd22,gyro3,estimated_gravity3,command3,previous_accepted_physical_target22. This is not the same observation or history as source72.',
        adapter_semantics='Map delayed current q/qd by named axis signs into source21, fill absentgripper states0, subtractsourcehome, use delayed current IMU gyro/estimatedgravity, append retainedpreviouslegacyraw21 andcommand. Retainedlegacyraw21 is EXPLICIT additional controllerstate; it cannot generally be recovered from current75 previousacceptedtargets because sourceupper action scales0.',
        source_timing=dict(physics_s=.002, control_s=.02, command_delay_s=0., feedback='Instantaneous native sensors/joints in historical play runner'),
        current_timing=dict(physics_s=.001, control_s=.02, command_delay_s=.02, feedback_delay_s=.01, imu_Hz=208, joint_Hz=100),
        body_differences=['SourceR is spatialleft under proper CAD rotation; preserve current source labels.',
            'Hip roll/pitch serial order and separated rotation centres differ. Neutral-axis signs are not a whole-pose kinematic map.',
            'Selected lower knee/ankle positions, foot geometry/COM/contact proxies and mass distribution differ.',
            'Source hip/knee rated1.569064Nm and gains30/.8; currentC047rated.980665Nm and gains8/.16, with lower no-load speed.',
            'Six current sourcehip travel limits remainunknown. Runtime +/-0.35rad is a diagnostic command envelope; known limits clamp independently.',
            'Two sourcegrippers removed; current2anklerolls+neckpitch are new. Shoulder yaw neutralaxis alignment is about.986, with different serialaxes/geometry.'],
        claim='Zero-update transfer/model diagnosis only. Historical walking and high-duty source scores are provenance, not current-body qualification.')
    path = OUT / 'transfer-contract.json'
    if path.exists() and read(path) != contract:
        raise FileExistsError('Existing transfer contract differs; use a fresh COVE50_FORWARD_OUT directory')
    write(path, contract)
    return old, current, rows, contract


if __name__ == '__main__':
    _, _, _, contract = build_contract()
    print(json.dumps(dict(mapped_channels=sum(r['source_index'] is not None for r in contract['named_mapping']),
        removed=contract['removed_source_channels'], source_mass_kg=contract['source_mass_kg'], current_mass_kg=contract['current_mass_kg'])))
