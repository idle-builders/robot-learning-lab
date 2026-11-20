"""Aggregate complete native continuation evidence without success thresholds."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np

from assess_trial import distribution, render


def assess(run):
    result = {'schema': 'cove50.actor-value-fit-assessment.v1', 'stages': {},
              'measurement_note': 'Native control snapshots every20ms; all episode files retained. Error curves include fallen tails. Training reset fragments and requested-horizon replay are distinct.'}
    worker = run / 'evidence/worker-report.json'
    if worker.exists():
        result['worker'] = json.loads(worker.read_text())
    progress = run / 'progress.json'
    live = json.loads(progress.read_text()) if progress.exists() else {}
    result['last_pushed_state'] = {key:live.get(key) for key in ('status','active_stage','completed_ppo_rounds','wall_s')}
    for folder in sorted((run / 'evidence').iterdir()):
        if not folder.is_dir() or not (folder / 'report.json').exists():
            continue
        report = json.loads((folder / 'report.json').read_text())
        manifest = folder / 'episodes/episodes.jsonl'
        episode_rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
        values = defaultdict(list)
        row_metrics = []
        for row in episode_rows:
            trace = folder / 'episodes' / row['trace']
            with np.load(trace) as data:
                qpos = data['qpos']
                error = data['root_velocity_body_m_s'][:, 0] - data['command3'][:, 0]
                term = np.flatnonzero(data['source_terminated'])
                first = int(term[0]) if len(term) else len(error)
                values['absolute_forward_error_m_s'].append(abs(error))
                values['pre_termination_absolute_forward_error_m_s'].append(abs(error[:first]))
                values['torque_abs_Nm'].append(abs(data['torque22_Nm']).ravel())
                values['current_abs_A'].append(abs(data['current22_A']).ravel())
                values['motor_effort_clipping'].append(data['saturated22'].ravel())
                values['position_target_clip_abs_rad'].append(abs(data['requested_target22_rad'] - data['accepted_target22_rad']).ravel())
                values['feet_contact_fraction'].append((data['feet_found']>0).astype(float).ravel())
                values['feet_contact_count'].append(data['feet_found'].ravel())
                metrics = dict(trace=str(trace), controls=len(error), duration_s=len(error)*.02,
                    first_physical_termination_s=(first+1)*.02 if len(term) else None,
                    delta_xy_m=(qpos[-1,:2] - data['initial_qpos'][:2]).tolist(),
                    minimum_root_height_m=float(qpos[:,2].min()),
                    forward_error_m_s=distribution(abs(error)),
                    pre_termination_error_m_s=distribution(abs(error[:first])),
                    motor_effort_clipping_fraction=float(data['saturated22'].mean()),
                    terminal_reason=row['terminal_reason'])
                if report['mode'] == 'replay':
                    metrics['requested_duration_s'] = 8.
                    metrics['requested_controls'] = 400
                    metrics['coverage_fraction'] = len(error)/400
                    metrics['command_error_accuracy_curve'] = [dict(radius_m_s=radius,
                        fraction_of_requested_controls=float((abs(error)<=radius).sum())/400,
                        pre_termination_prefix_fraction_of_requested_controls=float((abs(error[:first])<=radius).sum())/400)
                        for radius in (.05,.1,.2,.3,.5,1.)]
                    metrics['initial_qpos'] = data['initial_qpos'].tolist()
                    metrics['initial_qvel'] = data['initial_qvel'].tolist()
                row_metrics.append(metrics)
        summary = {key:report.get(key) for key in ('mode','seed','world_slots_constructed',
            'independent_start_families','simulator_transitions','native_physics_steps','optimizer_steps',
            'ppo_rounds','complete_or_terminated_episodes','censored_episodes','wall_s',
            'status','transfer','final_effective_learning_rate','adam_parameter_step_counts')}
        summary.update(episodes_retained=len(row_metrics), terminal_reasons=dict(Counter(row['terminal_reason'] for row in episode_rows)),
            duration_s=distribution([row['duration_s'] for row in row_metrics]),
            delta_x_m=distribution([row['delta_xy_m'][0] for row in row_metrics]))
        for key, arrays in values.items():
            array = np.concatenate(arrays) if arrays else np.array([])
            summary[key] = float(array.mean()) if key in ('motor_effort_clipping','feet_contact_fraction') and len(array) else distribution(array)
        if report['mode'] == 'replay':
            summary['episodes'] = row_metrics
        elif row_metrics:
            ordered = sorted(row_metrics, key=lambda row:row['delta_xy_m'][0])
            summary['representative_trajectories_by_delta_x'] = [ordered[i] for i in sorted({0,len(ordered)//2,len(ordered)-1})]
        result['stages'][folder.name] = summary
    result['paired_replay_initial_state'] = {}
    for name, stage in result['stages'].items():
        if not name.startswith('before-'):
            continue
        after = result['stages'].get(name.replace('before-','after-',1))
        if after and stage.get('episodes') and after.get('episodes'):
            left, right = stage['episodes'][0], after['episodes'][0]
            result['paired_replay_initial_state'][name.removeprefix('before-')] = {
                'max_qpos_abs_difference':float(np.max(np.abs(np.array(left['initial_qpos'])-right['initial_qpos']))),
                'max_qvel_abs_difference':float(np.max(np.abs(np.array(left['initial_qvel'])-right['initial_qvel'])))}
    result['totals'] = {}; result['known_lower_bounds'] = {}; result['unknown_counter_stages'] = {}
    for key in ('simulator_transitions','native_physics_steps','optimizer_steps','ppo_rounds'):
        unknown = [name for name,stage in result['stages'].items() if stage.get(key) is None]
        active = live.get('active_stage')
        if not worker.exists() and active and active not in result['stages'] and active not in ('hardware','bootstrap'):
            if key not in ('optimizer_steps','ppo_rounds') or active.startswith('fit-through-'):
                unknown.append(active)
        lower = sum(stage[key] for stage in result['stages'].values() if stage.get(key) is not None)
        result['known_lower_bounds'][key] = result.get('worker',{}).get('known_lower_bounds',{}).get(key,lower)
        result['unknown_counter_stages'][key] = result.get('worker',{}).get('unknown_counter_stages',{}).get(key,unknown)
        result['totals'][key] = result.get('worker',{}).get('totals',{}).get(key,None if unknown else lower)
    result['training_lineages'] = (1 if result['known_lower_bounds']['optimizer_steps']>0 else
        None if result['unknown_counter_stages']['optimizer_steps'] else 0)
    result['saved_trained_checkpoints'] = sum(int(stage['mode']=='train' and (stage['optimizer_steps'] or 0)>0) for stage in result['stages'].values())
    (run / 'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--render-stage')
    args = parser.parse_args()
    run = args.run.resolve()
    if args.render_stage:
        render(run,args.render_stage)
    else:
        result = assess(run)
        print(json.dumps({'totals':result['totals'],'stages':list(result['stages']),
                          'paired_replay_initial_state':result['paired_replay_initial_state']}))
