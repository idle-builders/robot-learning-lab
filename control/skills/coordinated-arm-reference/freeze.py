"""Seal compact source/config/goal-table only; no scientific imports or inference."""
import ast
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/coordinated-arm-reference-20261030'


def ref(path):
    payload = path.read_bytes()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def write(path, value):
    raw = json.dumps(value, indent=2) + '\n'
    if path.exists() and path.read_text() != raw:
        raise FileExistsError('Preserve frozen source; explicitly revise before replacing ' + str(path))
    path.write_text(raw)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    proposal_path = ROOT / 'data/arm-support-method-20261030/proposal.json'
    proposal = json.loads(proposal_path.read_text())
    audit_path = ROOT / 'data/basic-skills-retry-20261027/squat-pose-audit/report.json'
    audit = json.loads(audit_path.read_text())
    rows = audit['candidate_family']
    names = rows[0]['measurement']['target_names']
    if any(r['measurement']['target_names'] != names for r in rows):
        raise ValueError('Geometric table joint-order differs across nodes')
    goal_path = DATA / 'goal-table.json'
    goals = dict(schema='cove50.command-owned-geometric-goals.v1', source=ref(audit_path),
        joint_order=names, beta=[r['beta'] for r in rows],
        q_pose_goal22_rad=[r['measurement']['target_rad'] for r in rows],
        geometric_root_xyz_m=[r['measurement']['root_xyz_m'] for r in rows],
        geometric_com_xyz_m=[r['measurement']['whole_body_com_m'] for r in rows],
        meaning='Exact com_crouch.Reference interpolation nodes for geometric desiredq; no tau/Kp motor offsets. Root/COM are instrumentation only; no solver or native corrective label is queried.')
    write(goal_path, goals)
    zero = [0.] * 22
    mild = zero.copy()
    for name, value in {'R_hip_roll': .002, 'L_hip_roll': .002,
                        'R_ankle_roll': -.002, 'L_ankle_roll': -.002}.items():
        mild[names.index(name)] = value
    rng = random.Random(240104)
    wider = [rng.uniform(-.008, .008) for _ in names]
    wider_qd = [rng.uniform(-.01, .01) for _ in names]
    cases = [dict(name='nominal', seed=240101, root_rpy_rad=[0., 0., 0.],
                  joint_offsets22_rad=zero, joint_velocities22_rad_s=zero),
             dict(name='mild-positive-roll', seed=240102, root_rpy_rad=[.004, 0., 0.],
                  joint_offsets22_rad=mild, joint_velocities22_rad_s=zero),
             dict(name='mild-negative-roll', seed=240103, root_rpy_rad=[-.004, 0., 0.],
                  joint_offsets22_rad=[-v for v in mild], joint_velocities22_rad_s=zero),
             dict(name='wider-documented-range', seed=240104, root_rpy_rad=[.015, -.006, .004],
                  joint_offsets22_rad=wider, joint_velocities22_rad_s=wider_qd)]
    target = proposal['selected_learning_model']
    pointer = json.loads((ROOT / 'robot/model/current-model.json').read_text())
    physical_seal = ROOT / pointer['evidence_seal']['path']
    declared = json.loads(physical_seal.read_text())
    known_meshes = {r['path']: r for r in declared['meshes']}
    model_path = ROOT / target['rigid_MJCF']['path']
    physical_meshes = []
    for mesh in ET.parse(model_path).getroot().findall('./asset/mesh'):
        name = (model_path.parent / mesh.attrib['file']).relative_to(ROOT).as_posix()
        physical_meshes.append(known_meshes[name])
    inputs = [proposal_path, audit_path, goal_path, physical_seal,
        ROOT / 'robot/model/current-model.json']
    inputs += [ROOT / target[k]['path'] for k in ('rigid_MJCF', 'actuators', 'sensors', 'timing')]
    inputs += [ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml',
               ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/agent.yaml']
    cfg = dict(schema='cove50.coordinated-arm-reference-method.v1', task='T-7y9vsf',
        status='source_only_no_native_release', proposal=ref(proposal_path),
        checkpoint=proposal['useful_retained_reference']['candidate'], goal_table=ref(goal_path),
        physical_source_seal=ref(physical_seal), physical_meshes=physical_meshes, joint_order=names,
        root_position_m=[0., 0., .30201], cases=cases,
        planned=dict(visits=4, world_slots=1, controls_per_visit=500, controls=2000,
                     physics_substeps=160000, optimizer_updates=0, new_weight_sets=0),
        semantics=dict(actor='Exact d2f5 float32 CPU1→32Tanh→32Tanh→22Linear; one beta input; output already physical named22 motor radians. No second scale/sign/home transform, output clamp or EMA.',
            controller='One fixed model reused across four actual resets of one native world;500 queries per visit, including bootstrap query whose output is recorded but held pending sensor validity. No critic, learner, optimizer or new checkpoint.',
            goals='Separate exact source desiredq interpolation; conditional tau/Kp is already learned into donor motor outputs, never reapplied. Geometry/root/COM never enter actor or physics correction.',
            command='0–2beta0,2–4linear rise,4–6hold1,6–8linear return,8–10settle0; integer50Hz control clock; no command-triggered state/queue resets.',
            entry='Current MJLab neutral default plus frozen named offsets/qd, native softlimit.9 clamp and declared root .30201. Actual requested/clamped q/qd, root/sole states and runtime biases/queues retained. No clearance solve or after-reset physical repair. Classic donor used hardlimit-neutral knees and.2mm collider clearance; this difference remains visible.',
            bootstrap='Invalid sensor ticks hold actual reset q22, recorded separately from donor output. Current runtime still applies exact hardlimits,20ms motor delay,10ms feedback delay,12V rated motor law,4kHz physics and50Hz control. Noise stream uses current Torch runtime, explicitly reseeded before each actual reset.',
            endings='Native source termination/timeouts are annotations;auto_resetFalse and explicit pendingflag clearance preserve whole finite10s continuations/floor tails. Nonfinite states preserve partial originals. No reset can supply predecessor return.',
            contacts='Control-rate feet, distal forearm and all-nonfoot ground forces/found retained; no claim of continuous contact measurement. Motor clipping fraction and peak torque/current aggregate all80 runtime substeps per control.',
            judgment='Describe initial entry, supported excursion, hold and return separately from measured pose/motor errors. Fresh development families, not held-out robustness or independent-arm/larger-amplitude/recovery evidence.'),
        storage=dict(run='infra/lab storage run coordinated-arm-reference-20261030-r1 -- COMMAND',
            evidence='LAB_EVIDENCE_DIR', work='LAB_WORK_DIR',
            active_ownership='Keep foreground managed invocation active until report, movie, review and publication consumers finish; then explicit release to shared offload. Never normal-return/close while originals remain active.'),
        inputs=[ref(p) for p in sorted(set(inputs))])
    config_path = DATA / 'config.json'
    write(config_path, cfg)
    sources = list(HERE.glob('*.py'))
    sources += list((ROOT / 'control/skills/walker/task').glob('*.py'))
    sources += list((ROOT / 'control/common/plant').glob('*.py'))
    sources += [ROOT / p for p in ('control/skills/walker/policy/adapter.py',
        'control/skills/forward-walk/contracts.py',
        'control/skills/basic-skills/command_student.py',
        'control/skills/basic-skills/com_crouch.py',
        'control/skills/basic-skills/evaluate_command_student.py')]
    for path in HERE.glob('*.py'): ast.parse(path.read_text())
    seal_path = DATA / 'source-seal.json'
    write(seal_path, dict(schema='cove50.coordinated-arm-reference-source-seal.v1',
        configuration=ref(config_path), source=[ref(p) for p in sorted(set(sources))],
        activity='AST/source/compact JSON only; zero PT/NPZ deserializations, NN, native models, FK, physics, provider or database mutations'))
    print(json.dumps(dict(configuration=ref(config_path), goal_table=ref(goal_path),
                         source_seal=ref(seal_path), planned=cfg['planned']), indent=2))


if __name__ == '__main__': main()
