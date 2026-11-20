"""Sustained E3 command interface plus a separate moving-only yaw-rate cost."""
from dataclasses import asdict, is_dataclass
import hashlib
import json
from compose import ROOT, project

_source = project('control/skills/forward-entry-braking/schedule.py',
    'd877c4f278f50ee44bf191c856d507fbd852be6973659fcb7fb1d7e7e0335707', [
        ('"cfg.episode_length_s = 4. if profile', '"cfg.episode_length_s = 6. if profile', 1),
        ('[0]*8 + [1]*8 + [2]*4 + [3]*4 + [4]*4 + [5]*4', '[0]*12 + [1]*12 + [2]*8', 1),
        ('self.velocity[:8] = .25; self.velocity[16:24] = .25', 'self.velocity[:12] = .25', 1),
        ('self.switch[16:20] = 60; self.switch[20:24] = 100\\n            self.switch[24:28] = 60; self.switch[28:] = 100',
         'self.switch[24:] = 100', 1),
    ])
exec(compile(_source, __file__ + ':retained-E3-schedule', 'exec'), globals())
_configure_E3 = configure
_verify_E3 = verify_live


def moving_yaw_rate_l2(env, command_name, moving_threshold):
    command = env.command_manager.get_command(command_name)
    actual = env.scene['robot'].data.root_link_ang_vel_b[:, 2]
    moving = command[:, :2].norm(dim=1) + command[:, 2].abs() > moving_threshold
    return moving * (actual - command[:, 2]).square()


def configure(cfg, profile):
    from mjlab.managers import RewardTermCfg
    cfg = _configure_E3(cfg, profile)
    if len(cfg.rewards) != 12 or 'moving_yaw_rate_l2' in cfg.rewards:
        raise ValueError('Expected unchanged twelve-term E3 base')
    cfg.rewards['moving_yaw_rate_l2'] = RewardTermCfg(func=moving_yaw_rate_l2,
        weight=-2., params=dict(command_name='twist', moving_threshold=.01))
    return cfg


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, slice):
        return dict(slice=[value.start, value.stop, value.step])
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unexpected reward parameter type: '+str(type(value)))


def _function_id(func):
    kind = func if hasattr(func, '__qualname__') else type(func)
    return kind.__module__+'.'+kind.__qualname__


def verify_live(env):
    result = _verify_E3(env)
    cfg = json.loads((ROOT/'data/forward-heading-next-20261030/config.json').read_text())
    expected = cfg['expected_original_rewards']
    if set(env.reward_manager.active_terms) != set(expected) | {'moving_yaw_rate_l2'}:
        raise ValueError('Expected exact thirteen live reward terms')
    actual = {}
    for name, row in expected.items():
        term = env.reward_manager.get_term_cfg(name)
        actual[name] = dict(function=_function_id(term.func), weight=term.weight, params=_plain(term.params))
        if actual[name] != row:
            raise ValueError('An original E3 live reward changed: '+name)
    term = env.reward_manager.get_term_cfg('moving_yaw_rate_l2')
    if term.func is not moving_yaw_rate_l2 or term.weight != -2. or term.params != dict(command_name='twist', moving_threshold=.01):
        raise ValueError('Live moving-only yaw cost differs')
    actual['moving_yaw_rate_l2'] = dict(function=_function_id(term.func), weight=term.weight, params=term.params)
    result.update(total_live_terms=13, original_E3_terms_verified=12, yaw_weight=-2.,
        yaw_gate='issued command XY norm + abs(yaw) > .01; zero otherwise',
        yaw_units='weighted squared(rad/s) rate; total control reward scales by.02s',
        full_live_recipe_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest())
    return result
