"""Source-check V162 opened pairs and publish diagnostic, clips and cost."""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'artifacts/ape-cove50-reach-goal-channels-20261010/v162-live-rgb-diagnostic'
CREATE = '2026-10-15T05:06:33.135Z'
STOP = '2026-10-15T05:30:51Z'
RATE = 0.89


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (BASE / name).write_text(json.dumps(value, indent=2) + '\n')


def precontact(report):
    rows = report['timeseries']
    events = [x for x in report['image_goal_capture_events'] if x['used_for_control']]
    first_touch = next((x['time_s'] for x in rows
                        if x['phase'] == 'reach' and x['finger_contact']), None)
    cutoff = (first_touch if first_touch is not None else
              min((x for x in rows if x['phase'] == 'reach'),
                  key=lambda x: x['cube_to_hand_distance_m'])['time_s'])
    used = [x for x in events if x['time_s'] <= cutoff]
    origin = np.asarray(events[0]['estimated_goal_world_xyz_m'][:2])
    cube0 = np.asarray(rows[0]['cube_xyz_m'][:2])
    at_cutoff = min(rows, key=lambda x: abs(x['time_s'] - cutoff))
    return {'first_finger_contact_s': first_touch,
            'precontact_cutoff_s': cutoff,
            'max_precontact_rgb_truth_error_mm': round(max(
                x['xy_error_truth_mm_scoring_only'] for x in used), 3),
            'max_precontact_goal_shift_from_entry_mm': round(max(
                np.linalg.norm(np.asarray(x['estimated_goal_world_xyz_m'][:2]) - origin) * 1000
                for x in used), 3),
            'cube_shift_at_cutoff_mm': round(float(np.linalg.norm(
                np.asarray(at_cutoff['cube_xyz_m'][:2]) - cube0) * 1000), 3)}


def main():
    method = json.loads((BASE / 'method.json').read_text())
    raw = json.loads((BASE / 'report.json').read_text())
    if (raw['completed_pairs'], raw['planned_pairs'], raw['method_sha256']) != (4, 4, sha(BASE / 'method.json')):
        raise ValueError('Incomplete or changed V162 paired source')
    cases = []
    videos = []
    for case in raw['cases']:
        if case['status'] != 'valid_pair':
            raise ValueError('Invalid V162 pair')
        records = {}
        for mode in ('locked', 'live'):
            item = case[mode]
            report_path, trace_path, log_path = (ROOT / item[k] for k in ('report', 'trace', 'log'))
            if any(sha(path) != item[field + '_sha256'] for path, field in
                   ((report_path, 'report'), (trace_path, 'trace'), (log_path, 'log'))):
                raise ValueError('Source file changed')
            report = json.loads(report_path.read_text())
            if (report['body_sha256'] != method['body_sha256'] or
                    report['evaluator_sha256'] != method['evaluator_sha256'] or
                    report['saved_prestep_reach_feature_count'] != 450):
                raise ValueError('V162 body/evaluator/feature source mismatch')
            with np.load(trace_path, allow_pickle=False) as trace:
                if (trace['reach_actor_feature'].shape != (450, 11) or
                        trace['reach_feature_step'].shape != (450,) or
                        len(trace['qpos']) != len(report['timeseries']) or
                        not np.isfinite(trace['reach_actor_feature']).all()):
                    raise ValueError('Unaligned V162 actor feature trace')
            records[mode] = {**{k: item[k] for k in ('contact_s', 'full_gate',
                                 'margin_1p2_s', 'faults', 'used_goal_updates')},
                             'report': item['report'], 'trace': item['trace'],
                             **(precontact(report) if mode == 'live' else {})}
            video = report_path.with_name(report_path.stem + '-aligned.mp4')
            if video.is_file():
                reader = imageio.get_reader(video)
                metadata = reader.get_meta_data()
                count = sum(1 for _ in reader)
                reader.close()
                if metadata['size'] != (1280, 608) or count != 350:
                    raise ValueError('Incomplete saved-state video')
                videos.append({'seed': case['seed'], 'mode': mode,
                               'path': str(video.relative_to(ROOT)),
                               'sha256': sha(video), 'decoded_frames': count,
                               'dimensions_px': list(metadata['size']),
                               'report_sha256': sha(report_path),
                               'trace_sha256': sha(trace_path),
                               'role': 'rendered_after_native_visit_from_saved_poststep_pose'})
        cases.append({'seed': case['seed'], 'route': case['locked']['route'],
                      'locked': records['locked'], 'live': records['live'],
                      'locked_source_parity': case['locked_parity'],
                      'matched_entry_parity': case['live_entry_parity']})
    start = datetime.fromisoformat(CREATE.replace('Z', '+00:00'))
    stop = datetime.fromisoformat(STOP.replace('Z', '+00:00'))
    hours = (stop - start).total_seconds() / 3600
    cost = {'pod_id': 'e1ubswfxpg5z5p', 'status': 'stopped_by_user',
            'created_at_utc': CREATE, 'stopped_at_utc': STOP,
            'gpu': 'RTX 4090', 'listed_gpu_rate_usd_per_hour': RATE,
            'listed_gpu_hours': round(hours, 6),
            'listed_rental_estimate_usd': round(hours * RATE, 5),
            'provider_posted_pod_charge_usd': None,
            'serverless_charge_allocated_usd': 0,
            'other_concurrent_pods_excluded': ['pwwc57r56ujmu4'],
            'optimizer_updates': 0}
    assessment = {'schema': 'robot-learning-lab.reach-live-rgb-diagnostic-assessment.v1',
                  'method_sha256': sha(BASE / 'method.json'),
                  'native_pair_report_sha256': sha(BASE / 'report.json'),
                  'source_sha256': sha(Path(__file__)),
                  'claim_limit': 'Opened Direct Frame camera/controller diagnostic; no new learned reach actor, fresh score or physical-body transfer',
                  'planned_pairs': 4, 'valid_pairs': 4,
                  'locked_full_gate_and_margin': sum(c['locked']['full_gate'] and c['locked']['margin_1p2_s'] for c in cases),
                  'live_full_gate_and_margin': sum(c['live']['full_gate'] and c['live']['margin_1p2_s'] for c in cases),
                  'locked_contact_s': [c['locked']['contact_s'] for c in cases],
                  'live_contact_s': [c['live']['contact_s'] for c in cases],
                  'live_used_goal_updates': [c['live']['used_goal_updates'] for c in cases],
                  'interpretation': 'Moving-camera RGB goal drift before contact defeats the current entry-route actors; these live trajectories are negative examples, not corrective imitation teachers.',
                  'cases': cases}
    write('assessment.json', assessment)
    write('cost.json', cost)
    write('video-manifest.json', {'schema': 'robot-learning-lab.reach-live-rgb-diagnostic-videos.v1',
                                  'role': 'saved-state visualizations, not native scoring',
                                  'source_sha256': {'assessment': sha(BASE / 'assessment.json'),
                                                    'renderer': sha(Path(__file__).with_name('render_observable_video_poststep.py'))},
                                  'videos': videos})
    print(json.dumps({'valid_pairs': 4, 'locked_passes': assessment['locked_full_gate_and_margin'],
                      'live_passes': assessment['live_full_gate_and_margin'],
                      'listed_gpu_hours': cost['listed_gpu_hours'],
                      'listed_rental_usd': cost['listed_rental_estimate_usd'],
                      'videos': len(videos)}))


if __name__ == '__main__':
    main()
