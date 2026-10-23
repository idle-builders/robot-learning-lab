"""Invoke AGILE's unchanged sim2mujoco CLI with Pip's embodiment adapters.

Requires the pinned upstream checkout and .venv-learning. Forward ordinary
sim2mujoco CLI flags after --bundle; output directories must be new.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / '.cache/upstream/WBC-AGILE'
sys.path[:0] = [str(ROOT), str(UPSTREAM)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--leapp-yaml', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != '6830cf995714e81c91ce63247e8e016d36e28f14':
        raise ValueError('Revalidate adapter against changed upstream revision')
    contract = json.loads((args.bundle / 'model/contract.json').read_text())
    electrical = json.loads((args.bundle / 'model/electromechanical.json').read_text())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from workers.agile.mujoco_adapter import PipSimulation, PipPolicyController
    from agile.sim2mujoco.command_provider import VelocityCommandProvider
    from agile.sim2mujoco.commands import CommandManager
    from agile.sim2mujoco.data_logger import Sim2MuJoCoDataLogger
    from lab.artifact_store import snapshot_file, retain_dependency
    import numpy as np

    spec = importlib.util.spec_from_file_location('robotlab_native_sim2mujoco', UPSTREAM / 'scripts/sim2mujoco_eval.py')
    native = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = native
    spec.loader.exec_module(native)
    instances = []
    def simulation(*a, **kw):
        sim = PipSimulation(*a, contract=contract, electrical=electrical, **kw)
        instances.append(sim)
        return sim
    def provider(desc, device, config=None):
        manager = CommandManager(device, defaults={'height': contract['root_position_m'][2]})
        manager.height_range = (.18, contract['root_position_m'][2])
        manager.linear_x_range, manager.linear_y_range, manager.angular_z_range = (-.04, .04), (0., 0.), (-.1, .1)
        return VelocityCommandProvider(manager, 4)

    class Logger(Sim2MuJoCoDataLogger):
        def record_step(self, state, command, actions, *a, **kw):
            super().record_step(state, command, actions, *a, **kw)
            row = self._rows[-1]
            # Preserve the upstream placeholder under an unmistakable name.
            row['upstream_placeholder_success'] = row.pop('is_success')
            row['timestep'] = float(instances[0].mj_data.time)
            q = state.root_quat.cpu().numpy()
            for i, value in enumerate(q):
                row[f'root_rot_{i}'] = float(value)  # WXYZ
            row['tilt_rad'] = float(np.arccos(np.clip(1 - 2 * (q[1]**2 + q[2]**2), -1, 1)))
            for i, value in enumerate(command.velocity.cpu().numpy()):
                row[f'joint_vel_des_{i}'] = float(value)
            telemetry = instances[0].electrical_telemetry
            for field in ('voltage_v', 'total_current_a', 'energy_j'):
                row[field] = float(telemetry[field])

    native.MuJocoSimulation = simulation
    native.LeappPolicyController = PipPolicyController
    native.create_leapp_command_provider = provider
    native.Sim2MuJoCoDataLogger = Logger
    report = {'status': 'started', 'upstream_revision': revision, 'task_success': 'not_run',
        'scope': 'Native LEAPP and AGILE MuJoCo loop with Pip root pose, passive dynamics, mixed targets, command ranges and existing electrical proxy. Processed action targets are not raw policy outputs. Upper body holds exported defaults. No physical validation.',
        'dependencies': {'robot': retain_dependency(args.bundle),
            'policy_export': retain_dependency(ROOT / 'artifacts' / args.leapp_yaml.absolute().relative_to(ROOT / 'artifacts').parts[0])}}
    for p in [Path(__file__), ROOT / 'workers/agile/mujoco_adapter.py', ROOT / 'lab/electromechanical.py']:
        snapshot_file(p, args.output_dir / 'source' / p.name)
    try:
        sys.argv = ['sim2mujoco_eval.py', '--mjcf', str(args.bundle / 'model/robot.xml'),
            '--leapp-yaml', str(args.leapp_yaml),
            '--output-dir', str(args.output_dir), '--no-viewer', '--no-real-time', '--save-data', *remaining]
        native.main()
        import pyarrow.parquet as pq
        rows = pq.read_table(args.output_dir / 'trajectories/episode_000.parquet').to_pylist()
        report.update(status='executed', frames=len(rows), last_time_s=rows[-1]['timestep'],
            first_orientation_failure_s=next((r['timestep'] for r in rows if r['tilt_rad'] > .28), None),
            max_tilt_rad=max(r['tilt_rad'] for r in rows))
        meta = args.output_dir / 'trajectories/metadata.json'
        d = json.loads(meta.read_text())
        d['robotlab'] = {'scope': report['scope'], 'action_semantics': 'processed_joint_targets', 'root_quaternion_order': 'wxyz'}
        meta.write_text(json.dumps(d, indent=2))
    except Exception as exc:
        report.update(status='fail', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
