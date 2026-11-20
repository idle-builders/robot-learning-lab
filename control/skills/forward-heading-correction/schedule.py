"""Unchanged E4 anchors and instantaneous rewards, plus causal cycle credit."""
import json
from compose import ROOT, project

_source = project('control/skills/forward-heading/schedule.py',
    'f246d2bac10e4d96172dec28f821d76fd9fce111abc78c9588048d628644f739', [
        ('data/forward-heading-next-20261030/config.json', 'control/skills/forward-heading-correction/setup/config.json', 1),
        ("set(expected) | {'moving_yaw_rate_l2'}", "set(expected) | {'moving_yaw_rate_l2', 'moving_cycle_yaw_mean_l2'}", 1),
    ])
exec(compile(_source, __file__ + ':retained-E4-schedule', 'exec'), globals())
_configure_E4 = configure
_verify_E4 = verify_live
_recorder_E4 = recorder_class


class MovingCycleYawMean:
    """One update per reward-time control. Snapshot/logging never calls this update."""
    def __init__(self, cfg, env):
        import torch
        self.window = cfg.params['window_controls']
        self.command_name = cfg.params['command_name']
        self.threshold = cfg.params['moving_threshold']
        n = env.num_envs
        self.ring = torch.zeros((n, self.window), device=env.device)
        self.count = torch.zeros(n, dtype=torch.long, device=env.device)
        self.cursor = self.count.clone()
        self.last_common = torch.full_like(self.count, -1)
        self.total = torch.zeros(n, device=env.device)
        self.mean = self.total.clone()
        self.before_mean = self.total.clone()
        self.error = self.total.clone()
        self.reward_body_wz = self.total.clone()
        self.reward_command3 = torch.zeros((n, 3), device=env.device)
        self.raw = self.total.clone()
        self.active = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.rows = torch.arange(n, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.ring, self.count, self.cursor, self.total, self.mean,
                      self.before_mean, self.error, self.raw, self.active,
                      self.reward_body_wz, self.reward_command3):
            value[ids] = 0
        self.last_common[ids] = -1

    def __call__(self, env, command_name, moving_threshold, window_controls):
        import torch
        if (command_name, moving_threshold, window_controls) != (self.command_name, self.threshold, self.window):
            raise ValueError('Changed causal filter recipe')
        common = int(env.common_step_counter)
        if bool((self.last_common == common).any()):
            raise RuntimeError('Duplicate reward update without a new reached control')
        command = env.command_manager.get_command(command_name)
        self.before_mean.copy_(self.mean)
        self.reward_body_wz.copy_(env.scene['robot'].data.root_link_ang_vel_b[:, 2])
        self.reward_command3.copy_(command)
        self.error.copy_(self.reward_body_wz - self.reward_command3[:, 2])
        old = self.ring[self.rows, self.cursor].clone()
        self.ring[self.rows, self.cursor] = self.error
        self.total.add_(self.error - old)
        self.count.add_(1).clamp_(max=self.window)
        self.cursor.add_(1).remainder_(self.window)
        self.mean.copy_(self.total / self.count)
        self.active.copy_(command[:, :2].norm(dim=1) + command[:, 2].abs() > moving_threshold)
        self.raw.copy_(torch.where(self.active, self.mean.square(), 0.))
        self.last_common.fill_(common)
        return self.raw

    def snapshot(self):
        """Cloned original state, including nonfinite evidence and rolling drift."""
        import torch
        recomputed = self.ring.sum(dim=1)
        finite = (torch.isfinite(self.ring).all(dim=1) & torch.isfinite(self.total)
                  & torch.isfinite(self.mean) & torch.isfinite(self.error) & torch.isfinite(self.raw)
                  & torch.isfinite(self.reward_body_wz) & torch.isfinite(self.reward_command3).all(dim=1))
        values = dict(ring33=self.ring, count=self.count, sum=self.total,
            cursor=self.cursor, mean=self.mean, before_mean=self.before_mean,
            issued_error=self.error, reward_body_wz=self.reward_body_wz,
            reward_command3=self.reward_command3, active=self.active, raw_squared_mean=self.raw,
            last_common=self.last_common, recomputed_sum=recomputed,
            rolling_minus_recomputed_sum=self.total-recomputed, finite=finite)
        return {name: value.detach().clone() for name, value in values.items()}


def cycle_term(env):
    term = env.reward_manager.get_term_cfg('moving_cycle_yaw_mean_l2').func
    if not isinstance(term, MovingCycleYawMean):
        raise TypeError('Expected live causal reward class instance')
    return term


def configure(cfg, profile):
    from mjlab.managers import RewardTermCfg
    cfg = _configure_E4(cfg, profile)
    if len(cfg.rewards) != 13:
        raise ValueError('Expected exact thirteen-term E4 base')
    cfg.rewards['moving_cycle_yaw_mean_l2'] = RewardTermCfg(func=MovingCycleYawMean,
        weight=-4., params=dict(command_name='twist', moving_threshold=.01, window_controls=33))
    return cfg


def verify_live(env):
    result = _verify_E4(env)
    cfg = json.loads((ROOT/'control/skills/forward-heading-correction/setup/config.json').read_text())
    actual = {}
    for name, row in cfg['expected_E4_rewards'].items():
        term = env.reward_manager.get_term_cfg(name)
        actual[name] = dict(function=_function_id(term.func), weight=term.weight, params=_plain(term.params))
        if actual[name] != row:
            raise ValueError('An E4 live reward changed: ' + name)
    term = env.reward_manager.get_term_cfg('moving_cycle_yaw_mean_l2')
    expected = dict(command_name='twist', moving_threshold=.01, window_controls=33)
    if term.weight != -4. or term.params != expected:
        raise ValueError('Changed cycle-mean reward recipe')
    cycle_term(env)
    actual['moving_cycle_yaw_mean_l2'] = dict(function=_function_id(term.func), weight=term.weight, params=expected)
    result.update(total_live_terms=14, E4_terms_verified=13,
        retained_E4_recipe_sha256=result.pop('full_live_recipe_sha256'),
        full_live_recipe_sha256=hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest(),
        cycle_window_controls=33, cycle_weight=-4., cycle_state_in_actor_or_critic=False)
    return result


def recorder_class(original):
    class Episodes(_recorder_E4(original)):
        def cache_inputs(self, value):
            super().cache_inputs(value)
            self.inputs.update({'input_cycle_'+key: item.cpu().numpy().copy()
                for key, item in cycle_term(self._env).snapshot().items()})

        def capture(self, ids):
            # Native terminal capture happens here before true reset. Ordinary
            # capture follows next-command compute, so use the cached reward state.
            state = {key: item[ids].cpu().numpy().copy()
                     for key, item in cycle_term(self._env).snapshot().items()}
            super().capture(ids)
            for j, i in enumerate(ids.cpu().tolist()):
                for key, values in state.items():
                    self.buffers[i]['cycle_'+key].append(values[j])
    return Episodes
