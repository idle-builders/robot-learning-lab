"""Four-second startup/braking lesson on the exact retained command interface."""
from compose import project

_source = project('control/skills/forward-stop-learning/schedule.py',
    '27d5a7df6e319341776047fa105ba21ab94c00ccc991647fe8d1e78f34d2a4fb', [
        ("cfg.episode_length_s = 8. if profile == 'train' else PROFILES[profile]['duration_s']",
         "cfg.episode_length_s = 4. if profile == 'train' else PROFILES[profile]['duration_s']", 1),
        ("self.cohort[16:24] = 1; self.cohort[24:] = 2\n            self.velocity[16:24] = .25\n            self.switch[16:24] = 30; self.switch[24:] = 100",
         "self.cohort[:] = torch.tensor([0]*8 + [1]*8 + [2]*4 + [3]*4 + [4]*4 + [5]*4, device=env.device)\n            self.velocity[:8] = .25; self.velocity[16:24] = .25\n            self.switch[16:20] = 60; self.switch[20:24] = 100\n            self.switch[24:28] = 60; self.switch[28:] = 100", 1),
    ])
exec(compile(_source, __file__ + ':retained-schedule-projection', 'exec'), globals())
_configure = configure


def velocity_tracking(env, command_name, std_active, std_zero, asset_cfg=None):
    """Original bounded velocity tracking with command-conditioned widths."""
    import torch
    command = env.command_manager.get_command(command_name)
    actual = env.scene['robot' if asset_cfg is None else asset_cfg.name].data.root_link_lin_vel_b
    squared_error = (command[:, :2] - actual[:, :2]).square().sum(dim=1) + actual[:, 2].square()
    active = command[:, :2].norm(dim=1) + command[:, 2].abs() > .01
    std = torch.where(active, std_active, std_zero)
    return torch.exp(-squared_error / std.square())


def configure(cfg, profile):
    cfg = _configure(cfg, profile)
    # The compatible loader computes the curriculum at common25200. Removing
    # this term before manager construction prevents that restore from writing−.6.
    if 'action_rate_weight' not in cfg.curriculum:
        raise ValueError('Expected retained action-rate curriculum')
    cfg.curriculum.pop('action_rate_weight')
    cfg.rewards['action_rate_l2'].weight = -.2
    term = cfg.rewards['track_linear_velocity']
    term.func = velocity_tracking
    term.params.pop('std')
    term.params.update(std_active=.20, std_zero=.15)
    return cfg


def verify_live(env):
    """Check the actual manager after restoration and around every PPO update."""
    if 'action_rate_weight' in env.cfg.curriculum or 'action_rate_weight' in env.curriculum_manager.active_terms:
        raise ValueError('Disabled action-rate curriculum returned')
    if env.reward_manager.get_term_cfg('action_rate_l2').weight != -.2:
        raise ValueError('Fixed action-rate weight changed after restoration')
    term = env.reward_manager.get_term_cfg('track_linear_velocity')
    if term.func is not velocity_tracking or term.weight != 2. or term.params['std_active'] != .2 or term.params['std_zero'] != .15 or 'std' in term.params:
        raise ValueError('Live command-conditioned velocity reward differs')
    return dict(action_rate_weight=env.reward_manager.get_term_cfg('action_rate_l2').weight,
        action_rate_curriculum_present=False, velocity_weight=term.weight,
        std_active_m_s=term.params['std_active'], std_zero_m_s=term.params['std_zero'],
        semantics='Native unscaled weighted reward rates; total control reward is sum(rates)*.02s.')
