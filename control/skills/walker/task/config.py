"""Old forward MJLab lesson with explicit morphology selectors.

Only the supplied R3 boundary may drive this task. The older generic
DcMotorActuator adapter is not a substitute for its timing/sensor contract.
"""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
import re
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'control/common/mjlab_ape/src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'policy'))
from adapter import Interface
from ape_walk import task as source
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.utils.spec_config import CollisionCfg

SAVED_ENV = ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml'


def typed(value):
    """Decode only YAML scalar/list/dict data, never its Python object tags."""
    if isinstance(value, dict): return {k: typed(v) for k, v in value.items()}
    if isinstance(value, list): return [typed(v) for v in value]
    if value == 'null': return None
    if value in ('true', 'false'): return value == 'true'
    try: return float(value) if any(c in value for c in '.eE') else int(value)
    except (ValueError, TypeError): return value


def serializable(value):
    if is_dataclass(value): return {k: serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict): return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serializable(v) for v in value]
    if isinstance(value, slice): return dict(slice=[value.start, value.stop, value.step])
    if callable(value): return value.__module__ + '.' + value.__qualname__
    if isinstance(value, Path): return str(value)
    return value


def reward_recipe(cfg):
    return {name: dict(function=term.func.__module__ + '.' + term.func.__qualname__,
        weight=term.weight, params=serializable(term.params)) for name, term in cfg.rewards.items()}


def remap_scene(value, interface):
    if isinstance(value, SceneEntityCfg):
        value = deepcopy(value)
        if value.body_names:
            value.body_names = tuple('torso' if n == 'pelvis' else n for n in value.body_names)
        if value.site_names:
            sites = {'left_sole_center': 'R_sole_contact_frame', 'right_sole_center': 'L_sole_contact_frame'}
            value.site_names = tuple(sites.get(n, n) for n in value.site_names)
        if value.joint_names:
            patterns = (value.joint_names,) if isinstance(value.joint_names, str) else value.joint_names
            value.joint_names = tuple(r['current_name'] for r in interface.rows
                if r['source_name'] and any(re.fullmatch(p, r['source_name']) for p in patterns))
            value.preserve_order = True
        return value
    if isinstance(value, dict): return {k: remap_scene(v, interface) for k, v in value.items()}
    return value


def source_forward_cfg(play=False):
    # The retained checkpoint's resolved forward YAML is authoritative. The
    # current task factory changed since that run (notably yaw tracking and
    # torque reward), so reconstruct its allowlisted terms from saved data.
    saved = typed(yaml.load(SAVED_ENV.read_text(), Loader=yaml.BaseLoader))
    cfg = source.env_cfg(play=play, elbowed=True, motion='forward')
    from mjlab.managers import RewardTermCfg
    cfg.rewards['joint_torques_l2'] = RewardTermCfg(func=source.microduck_mdp.joint_torques_l2,
        weight=saved['rewards']['joint_torques_l2']['weight'], params={'asset_cfg': SceneEntityCfg('robot')})
    if set(cfg.rewards) != set(saved['rewards']): raise ValueError('Saved reward terms differ from known reconstruction')
    for name, term in cfg.rewards.items():
        term.weight = saved['rewards'][name]['weight']
        for key, value in saved['rewards'][name]['params'].items():
            if key != 'asset_cfg': term.params[key] = value
    cmd = cfg.commands['twist']
    for key, value in saved['commands']['twist'].items():
        if key == 'ranges':
            for axis, ranges in value.items(): setattr(cmd.ranges, axis, tuple(ranges) if isinstance(ranges, list) else ranges)
        elif key != 'viz': setattr(cmd, key, tuple(value) if isinstance(value, list) else value)
    for name in ('reset_base', 'reset_robot_joints'):
        for key, value in saved['events'][name]['params'].items():
            if key != 'asset_cfg': cfg.events[name].params[key] = value
    if 'action_rate_weight' not in cfg.curriculum:
        cfg.curriculum['action_rate_weight'] = deepcopy(source.env_cfg(elbowed=True).curriculum['action_rate_weight'])
    cfg.curriculum['action_rate_weight'].params = saved['curriculum']['action_rate_weight']['params']
    cfg.episode_length_s = saved['episode_length_s']
    return cfg


def morphology_cfg(robot_cfg, *, play=False):
    """Copy source rewards and task; attach caller's exact R3 entity/runtime.

    Action and measured actor observation terms are supplied by boundary.py.
    This factory deliberately cannot silently instantiate approximate motors.
    """
    interface = Interface()
    cfg = source_forward_cfg(play=play)
    cfg.scene.entities = {'robot': robot_cfg}
    floor = interface.current.geom('floor').id
    cfg.scene.terrain.collisions = (CollisionCfg(geom_names_expr=(r'^terrain$',),
        friction=tuple(interface.current.geom_friction[floor]),
        solref=tuple(interface.current.geom_solref[floor]),
        solimp=tuple(interface.current.geom_solimp[floor]),
        contype=int(interface.current.geom_contype[floor]),
        conaffinity=int(interface.current.geom_conaffinity[floor]),
        condim=int(interface.current.geom_condim[floor]),
        priority=int(interface.current.geom_priority[floor]),
        margin=float(interface.current.geom_margin[floor]),
        gap=float(interface.current.geom_gap[floor]),
        solmix=float(interface.current.geom_solmix[floor])),)
    cfg.viewer.body_name = 'torso'
    cfg.sim.mujoco.timestep = .00025
    opt = interface.current.opt
    cfg.sim.mujoco.integrator = 'implicitfast'
    cfg.sim.mujoco.cone = 'elliptic'
    cfg.sim.mujoco.iterations = int(opt.iterations)
    cfg.sim.mujoco.gravity = tuple(opt.gravity)
    cfg.sim.mujoco.impratio = float(opt.impratio)
    cfg.sim.mujoco.tolerance = float(opt.tolerance)
    cfg.sim.mujoco.ls_iterations = int(opt.ls_iterations)
    cfg.sim.mujoco.ls_tolerance = float(opt.ls_tolerance)
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 2048
    cfg.decimation = 80
    cfg.episode_length_s = 8.
    for term in cfg.rewards.values(): term.params = remap_scene(term.params, interface)
    pose = cfg.rewards['pose'].params
    for phase in ('std_standing', 'std_walking', 'std_running'):
        old = pose[phase]
        pose[phase] = {r['current_name']: next(v for p, v in old.items() if re.fullmatch(p, r['source_name']))
            for r in interface.rows if r['source_name'] and r['source_index'] < 10}
    # Foot body net force combines every current foot collider, producing two
    # channels in source left/right order (R then L). Contact geometry is R3.
    feet = ContactSensorCfg(name='feet_ground_contact',
        primary=ContactMatch(mode='body', pattern=r'^(R|L)_foot$', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='terrain'),
        fields=('found', 'force'), reduce='netforce', num_slots=1, track_air_time=True)
    # The target has no separate hand/gripper links. Distal forearm collision
    # contact is the documented morphology equivalent of old hand contact.
    hands = ContactSensorCfg(name='hand_ground_contact',
        primary=ContactMatch(mode='body', pattern=r'^(R|L)_forearm$', entity='robot'),
        secondary=ContactMatch(mode='body', pattern='terrain'),
        fields=('found', 'force'), reduce='netforce', num_slots=1)
    height = deepcopy(cfg.scene.sensors[2])
    height.frame = tuple(ObjRef(type='site', name=n, entity='robot')
        for n in ('R_sole_contact_frame', 'L_sole_contact_frame'))
    cfg.scene.sensors = (feet, hands, height)
    # Source numerical constants, commands, termination formulae, reset noise,
    # reward weights and curriculum remain unchanged. Only root base position
    # and mapped neutral posture are morphology-dependent.
    home = interface.target(torch.zeros(1, 24))[0].tolist()
    cfg.scene.entities['robot'].init_state.joint_pos = dict(zip(interface.current_names, home))
    return cfg
