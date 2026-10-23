"""Execute native Isaac conversion and our AGILE environment before training."""
import argparse
import json
import os
from pathlib import Path
import traceback

# Official Kit Python bootstrap flags for the authorized headless simulator.
os.environ['OMNI_KIT_ACCEPT_EULA'] = 'YES'
os.environ['OMNI_KIT_ALLOW_ROOT'] = '1'

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
report = {'status': 'fail', 'robot_rl': 'not_run', 'task_success': 'not_run'}
app = None
env = None
try:
    app = AppLauncher(args).app
    import agile.isaaclab_extras.monkey_patches  # same compatibility setup as native train/eval
    from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg
    from pxr import Usd, UsdGeom, UsdPhysics
    contract = json.loads((args.inputs / 'model/contract.json').read_text())
    converted = MjcfConverter(MjcfConverterCfg(
        asset_path=str(args.inputs / 'model/robot.xml'),
        usd_dir=str(args.output / 'converted'), usd_file_name='robot.usd',
        fix_base=False, force_usd_conversion=True, link_density=0.0,
        self_collision=True, make_instanceable=False))
    source = Usd.Stage.Open(converted.usd_path)
    name = source.GetDefaultPrim().GetName()
    stage = Usd.Stage.CreateNew(str(args.output / 'articulation.usda'))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    prim = stage.DefinePrim('/' + name, 'Xform')
    prim.GetReferences().AddReference(os.path.relpath(converted.usd_path, args.output))
    stage.SetDefaultPrim(prim)
    floors = [p for p in stage.Traverse() if p.GetName() == 'floor']
    if not floors:
        raise RuntimeError('Imported floor not found; refuse to embed a second ground in each robot')
    for floor in floors:
        floor.SetActive(False)
    stage.GetRootLayer().Save()
    os.environ.update(ROBOTLAB_USD=str(args.output / 'articulation.usda'),
        ROBOTLAB_CONTRACT=str(args.inputs / 'model/contract.json'),
        ROBOTLAB_ELECTRICAL=str(args.inputs / 'model/electromechanical.json'))
    from pip_cfg import PipEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    import torch
    cfg = PipEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = 'cuda:0'
    env = ManagerBasedRLEnv(cfg=cfg)
    obs, _ = env.reset()
    robot = env.scene['robot']
    if set(robot.joint_names) != set(contract['joint_order']):
        raise RuntimeError('Loaded articulation joint set differs from source robot')
    def tensor(x):
        return x.torch if hasattr(x, 'torch') else x
    masses = tensor(robot.data.default_mass)[0]
    measured_mass = float(masses.sum().cpu())
    if abs(measured_mass - contract['robot_mass_kg']) > 1e-4:
        raise RuntimeError(f'Converted mass mismatch {measured_mass} vs {contract["robot_mass_kg"]}')
    report.update(joint_names=robot.joint_names, body_names=robot.body_names,
        loaded_mass_kg=measured_mass, expected_mass_kg=contract['robot_mass_kg'],
        action_dim=env.action_manager.total_action_dim,
        policy_observation_shape=list(obs['policy'].shape))
    # Exercise native processing, without applying extreme targets to physics.
    # The same action terms run in training and native evaluation.
    target_checks = {}
    for name in ('legs', 'wheels'):
        term = env.action_manager.get_term(name)
        limits = list(term.cfg.clip.values())
        samples = {}
        for value, column in ((-100., 0), (100., 1)):
            term.process_actions(torch.full_like(term.raw_actions, value))
            expected = torch.tensor([v[column] for v in limits], device=env.device)
            torch.testing.assert_close(term.processed_actions, expected.expand_as(term.processed_actions))
            samples[str(value)] = term.processed_actions[0].detach().cpu().tolist()
        target_checks[name] = samples
    report['physical_target_clipping'] = {'status': 'pass', 'samples': target_checks}
    env.reset()
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(actions)
        if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
            raise RuntimeError('Nonfinite native environment observation/reward')
    report.update(status='pass', scope='Native asset, actuator and manager execution only; zero actions are not learned control or a stability qualification.',
        actuator=robot.actuators['shared_pack'].telemetry(0),
        robot_usd=str(args.output / 'articulation.usda'))
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    if env is not None:
        env.close()
    if app is not None:
        app.close()
