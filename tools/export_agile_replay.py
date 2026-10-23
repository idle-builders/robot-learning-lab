"""Project native AGILE Parquet into Rerun using the retained MJCF/CAD.

Only visualization: no policy execution, new dynamics, inferred targets or
fabricated sensors. Link transforms use FK from recorded root/joint state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def recorded_pose(row, names, model, data, quaternion_order):
    import numpy as np
    data.qpos[:3] = [row[f'root_pos_{i}'] for i in range(3)]
    q = np.array([row[f'root_rot_{i}'] for i in range(4)])
    if not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('Invalid recorded root quaternion')
    data.qpos[3:7] = q[[3, 0, 1, 2]] if quaternion_order == 'xyzw' else q
    for i, name in enumerate(names):
        data.qpos[model.joint(name).qposadr] = row[f'joint_pos_{i}']
    if not np.isfinite(data.qpos).all():
        raise ValueError('Nonfinite recorded joint/root state')


def export(episode, bundle, output, quaternion_order):
    import mujoco
    import pyarrow.parquet as pq
    import rerun as rr
    import rerun.blueprint as rb
    from lab.rerun_scene import export_scene, log_static_scene, log_body_poses
    from lab.artifact_store import snapshot_file, retain_dependency

    metadata_path = episode.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text())
    contract = json.loads((bundle / 'model/contract.json').read_text())
    names = metadata['joint_names']
    if len(names) != len(set(names)) or set(names) != set(contract['joint_order']):
        raise ValueError('Native joint metadata differs from retained robot')
    rows = pq.read_table(episode).to_pylist()
    if not rows or len({r['episode_id'] for r in rows}) != 1:
        raise ValueError('Select exactly one nonempty native episode')
    times = [float(r['timestep']) for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Native episode timestamps are not strictly increasing')
    model_path = bundle / 'model/robot.xml'
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    output.mkdir(parents=True, exist_ok=False)
    scope = ('Native AGILE learned motor evaluation; recorded root/joint states projected through retained MJCF forward kinematics. '
             'Detailed CAD is visual geometry, not collision qualification. This is offline replay, not a new simulation or live control. '
             'No camera frames, per-step electrical telemetry, applied torques or processed actuator targets were recorded in this native dataset. '
             'Action plots are raw policy outputs. Motor evaluation is not household-task success.')
    cross_sim = metadata.get('robotlab', {}).get('action_semantics') == 'processed_joint_targets'
    action_group = 'processed_target' if cross_sim else 'raw_policy_action'
    if cross_sim:
        scope = metadata['robotlab']['scope'] + (' Offline CAD replay from recorded root/joint states. '
            'Electrical values are simulated estimates, torques are MuJoCo actuator efforts; no camera frames. '
            'The native success placeholder is ignored. Later motion after the first fall is not successful control.')
    rec = rr.RecordingStream('Pip native AGILE evaluation')
    rec.save(output / 'episode.rrd')
    rec.log('about', rr.TextDocument(scope), static=True)
    rec.log('scene', rr.ViewCoordinates.FLU, static=True)
    scene = export_scene(model_path, output / "scene.json")
    log_static_scene(rec, scene)
    origin = [rows[0][f'root_pos_{i}'] for i in range(3)]
    rec.send_blueprint(rb.Blueprint(rb.Vertical(rb.TextDocumentView(origin='about'), rb.Horizontal(
        rb.Spatial3DView(origin='scene', eye_controls=rb.EyeControls3D(
            position=[origin[0] + .9, origin[1] - 1.1, .65], look_target=origin, eye_up=[0, 0, 1])),
        rb.Tabs(rb.TimeSeriesView(origin='motion'), rb.TimeSeriesView(origin='joint_position_rad'),
                rb.TimeSeriesView(origin='joint_velocity_rad_s'), rb.TimeSeriesView(origin=action_group),
                *([rb.TimeSeriesView(origin='electrical'), rb.TimeSeriesView(origin='torque_nm')] if cross_sim else []))),
        row_shares=[.15, .85])))
    for row in rows:
        recorded_pose(row, names, model, data, quaternion_order)
        mujoco.mj_kinematics(model, data)
        rec.set_time('simulation', duration=float(row['timestep']))
        bodies = {model.body(i).name or 'world': {'position_m': data.xpos[i].tolist(),
            'quaternion_xyzw': data.xquat[i][[1, 2, 3, 0]].tolist()} for i in range(model.nbody)}
        log_body_poses(rec, bodies)
        for i, name in enumerate(names):
            rec.log('joint_position_rad/' + name, rr.Scalars(row[f'joint_pos_{i}']))
            rec.log('joint_velocity_rad_s/' + name, rr.Scalars(row[f'joint_vel_{i}']))
            if cross_sim:
                rec.log('torque_nm/' + name, rr.Scalars(row[f'joint_torque_{i}']))
        for i in range(6):
            rec.log(f'{action_group}/channel_{i}', rr.Scalars(row[f'actions_{i}']))
        if cross_sim:
            for key in ['voltage_v', 'total_current_a', 'energy_j']:
                rec.log('electrical/' + key, rr.Scalars(row[key]))
            rec.log('motion/tilt_rad', rr.Scalars(row['tilt_rad']))
        for name, key in [('forward_m_s', 'root_lin_vel_robot_0'), ('requested_forward_m_s', 'commands_0'),
                          ('height_m', 'root_pos_2'), ('requested_height_m', 'commands_3')]:
            rec.log('motion/' + name, rr.Scalars(row[key]))
    rec.flush()
    rec.disconnect()
    for source, name in [(episode, 'episode.parquet'), (metadata_path, 'metadata.json'),
                         (Path(__file__), 'export_agile_replay.py')]:
        snapshot_file(source, output / 'source' / name)
    native_run = ROOT / 'artifacts' / episode.absolute().relative_to(ROOT / 'artifacts').parts[0]
    report = {'status': 'exported', 'scope': scope, 'frames': len(rows), 'last_time_s': times[-1],
              'recorded_is_success': sorted(set(bool(r['is_success']) for r in rows if 'is_success' in r)),
              'first_orientation_failure_s': next((r['timestep'] for r in rows if r.get('tilt_rad', 0) > .28), None),
              'root_quaternion_order': quaternion_order, 'joint_names': names,
              'native_episode': str(episode), 'robot_bundle': str(bundle),
              'xml_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'dependency_manifests': {'native_run': retain_dependency(native_run), 'robot_bundle': retain_dependency(bundle)}}
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('episode', type=Path)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--root-quaternion-order', choices=['xyzw', 'wxyz'], required=True,
                   help='Explicit source convention; the pinned Isaac Lab 3.0 beta2 run uses xyzw.')
    a = p.parse_args()
    print(json.dumps(export(a.episode.absolute(), a.bundle.absolute(), a.output.absolute(), a.root_quaternion_order), indent=2))
