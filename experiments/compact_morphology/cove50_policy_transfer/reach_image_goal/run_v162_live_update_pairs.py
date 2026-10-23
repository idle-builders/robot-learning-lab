"""Opened V162 native pairs; preserve V158 holds and independent score."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from experiments.compact_morphology.cove50_policy_transfer.reach_image_goal import run_v83_thin_route_pilot as v83

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts/ape-cove50-reach-goal-channels-20261010/v162-live-rgb-diagnostic'
METHOD = BASE / 'method.json'
EVALUATOR = Path(__file__).with_name('evaluate_live_goal_v162.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(report):
    return next(x for x in report['image_goal_capture_events']
                if abs(x['time_s'] - report['reach_start_s']) < 1e-6)


def gap(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run(case, mode):
    sources = v83.v16.SOURCES
    p = case['parameters']
    stem = BASE / f"seed-{case['seed']}-{mode}"
    report_path, trace_path, log_path = (stem.with_suffix('.' + suffix)
                                         for suffix in ('json', 'npz', 'log'))
    command = [sys.executable, str(EVALUATOR),
               '--body-xml', str(sources['body']),
               '--calibration-model', str(sources['calibration']),
               '--squat-checkpoint', str(sources['squat']),
               '--reach-checkpoint', str(v83.v57.ACTOR),
               '--alternate-reach-checkpoint', str(sources['alternate_actor']),
               '--specialist-reach-checkpoint', str(sources['specialist_actor']),
               '--reach-controller', 'learned-residual',
               '--reach-reference', str(sources['reference']),
               '--geometry', str(sources['geometry']),
               '--entry', 'continuous', '--seed', str(case['seed']),
               '--rail-v', str(p['rail_v']), '--delay-ms', str(p['delay_ms']),
               '--toy-offset-x-m', str(p['toy_offset_x_m']),
               '--toy-offset-y-m', str(p['toy_offset_y_m']),
               '--pitch-offset-rad', str(p['pitch_offset_rad']),
               '--roll-clearance-rad', str(p['roll_clearance_rad']),
               '--terminal-roll-offset-rad', '-.08',
               '--terminal-roll-mode', 'entry-error-x',
               '--reach-duration-s', '9',
               '--gaze-head-yaw-rad', str(p['gaze_head_yaw_rad']),
               '--gaze-start-s', '0', '--gaze-ramp-s', '7',
               '--capture-stride-steps', '5', '--max-goal-age-s', '9',
               '--out', str(report_path), '--trace', str(trace_path)]
    if mode == 'live_update':
        command.append('--live-update-diagnostic')
    if any(path.exists() for path in (report_path, trace_path, log_path)):
        raise ValueError(f'Original V162 visit already exists: {stem}')
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
        [str(ROOT), str(ROOT / 'experiments/compact_morphology/mjlab_ape/src'),
         '/root/microduck_rl/src', '/workspace/microduck_rl/src'] +
        ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    with log_path.open('w') as stream:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, check=False).returncode
    if code:
        raise RuntimeError(f'V162 {mode} exited {code}; inspect {log_path}')
    report = json.loads(report_path.read_text())
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos, action, phase = trace['qpos'], trace['action'], trace['phase']
        features, steps = trace['reach_actor_feature'], trace['reach_feature_step']
        age = trace['reach_goal_age_s']
        if (len(qpos) != len(action) or len(qpos) != len(phase) or
                features.shape != (450, 11) or steps.shape != (450,) or
                age.shape != (450,) or not np.all(phase[steps] == 'reach') or
                not np.isfinite(features).all() or not np.isfinite(action).all()):
            raise ValueError(f'Pre-action feature alignment fault: {trace_path}')
        first_qpos = qpos[int(steps[0])].tolist()
        first_action = action[int(steps[0])].tolist()
    if (report['evaluator_sha256'] != sha(EVALUATOR) or
            report['saved_prestep_reach_feature_count'] != 450 or
            report['live_update_diagnostic'] != (mode == 'live_update') or
            report['selection_mode'] != case['route']):
        raise ValueError(f'Source/route mismatch: {report_path}')
    faults = v83.gate(report)
    return {'mode': mode, 'seed': case['seed'], 'route': case['route'],
            'report': str(report_path.relative_to(ROOT)), 'report_sha256': sha(report_path),
            'trace': str(trace_path.relative_to(ROOT)), 'trace_sha256': sha(trace_path),
            'log': str(log_path.relative_to(ROOT)), 'log_sha256': sha(log_path),
            'entry_goal_xyz_m': entry(report)['estimated_goal_world_xyz_m'],
            'first_reach_qpos': first_qpos, 'first_reach_action': first_action,
            'used_goal_updates': report['image_goal_used_update_count'],
            'accepted_goal_captures': report['image_goal_accepted_capture_count'],
            'max_goal_age_s': float(np.max(age)),
            'contact_s': report['phases']['reach']['continuous_finger_contact_s'],
            'full_gate': not faults, 'margin_1p2_s':
                report['phases']['reach']['continuous_finger_contact_s'] >= 1.2,
            'faults': faults}


def main():
    method = json.loads(METHOD.read_text())
    old_base = ROOT / 'artifacts/ape-cove50-reach-goal-channels-20261010/v158'
    plan_path = old_base / 'new-seed-contact-plan.json'
    matrix_path = old_base / 'source-checked-development-contact-inspector-matrix.json'
    if (sha(EVALUATOR) != method['evaluator_sha256'] or
            sha(Path(__file__)) != method['runner_sha256'] or
            sha(plan_path) != method['v158_plan_sha256'] or
            sha(matrix_path) != method['v158_matrix_sha256'] or
            sha(ROOT / 'robot/ape-current-design.json') != method['pointer_sha256'] or
            sha(ROOT / 'robot/actuator-catalog.yaml') != method['motor_catalog_sha256'] or
            sha(ROOT / 'artifacts/ape-cove50-reach-viewpoint-20261010/heldout-freeze.json') !=
            method['camera_freeze_sha256'] or
            sha(v83.v16.SOURCES['body']) != method['body_sha256']):
        raise ValueError('Frozen V162 source changed')
    old_cases = json.loads(matrix_path.read_text())['cases']
    rows = []
    for case in method['cases']:
        old = next(x for x in old_cases if x['seed'] == case['seed'])
        old_report_path = ROOT / case['old_automatic_report']
        old_trace_path = ROOT / case['old_automatic_trace']
        if (sha(old_report_path) != case['old_automatic_report_sha256'] or
                sha(old_trace_path) != case['old_automatic_trace_sha256'] or
                not old['automatic_native_pass'] or not old['automatic_margin_1p2_s_met']):
            raise ValueError('Original V158 source changed')
        old_report = json.loads(old_report_path.read_text())
        locked = run(case, 'locked_baseline')
        with np.load(old_trace_path, allow_pickle=False) as original:
            index = int(np.flatnonzero(original['phase'] == 'reach')[0])
            old_qpos, old_action = original['qpos'][index], original['action'][index]
        parity = {'reach_qpos_max_abs_gap': gap(locked['first_reach_qpos'], old_qpos),
                  'first_reach_action_max_abs_gap': gap(locked['first_reach_action'], old_action),
                  'entry_goal_max_abs_gap_m': gap(locked['entry_goal_xyz_m'],
                                                entry(old_report)['estimated_goal_world_xyz_m']),
                  'contact_s_gap': abs(locked['contact_s'] - old['automatic_contact_s'])}
        if (max(parity[k] for k in ('reach_qpos_max_abs_gap',
                                   'first_reach_action_max_abs_gap',
                                   'entry_goal_max_abs_gap_m')) > 1e-4 or
                not locked['full_gate'] or not locked['margin_1p2_s']):
            rows.append({'seed': case['seed'], 'status': 'baseline_parity_failure',
                         'locked': locked, 'locked_parity': parity})
            break
        live = run(case, 'live_update')
        live_parity = {'reach_qpos_max_abs_gap': gap(live['first_reach_qpos'],
                                                  locked['first_reach_qpos']),
                       'first_reach_action_max_abs_gap': gap(live['first_reach_action'],
                                                           locked['first_reach_action']),
                       'entry_goal_max_abs_gap_m': gap(live['entry_goal_xyz_m'],
                                                      locked['entry_goal_xyz_m'])}
        status = 'valid_pair' if max(live_parity.values()) <= 1e-4 else 'invalid_entry_parity'
        rows.append({'seed': case['seed'], 'status': status,
                     'locked': locked, 'live': live,
                     'locked_parity': parity, 'live_entry_parity': live_parity})
        (BASE / 'partial.json').write_text(json.dumps({'cases': rows}, indent=2) + '\n')
        print(json.dumps({'seed': case['seed'], 'status': status,
                          'locked_contact_s': locked['contact_s'],
                          'live_contact_s': live['contact_s'],
                          'live_used_goal_updates': live['used_goal_updates']}), flush=True)
        if status != 'valid_pair':
            break
    report = {'schema': 'robot-learning-lab.reach-live-rgb-diagnostic.v1',
              'method_sha256': sha(METHOD), 'planned_pairs': len(method['cases']),
              'completed_pairs': sum(x['status'] == 'valid_pair' for x in rows),
              'cases': rows,
              'claim_limit': 'Opened development diagnostic only; no new learner or score'}
    (BASE / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
