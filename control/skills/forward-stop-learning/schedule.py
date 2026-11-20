"""Reset-local forward/stop requests, before native next-observation capture."""
import json


PROFILES = {
    'forward30': dict(duration_s=8., velocity=.30, switch_control=None),
    'forward25': dict(duration_s=8., velocity=.25, switch_control=None),
    'short-stop': dict(duration_s=3., velocity=.25, switch_control=30),
    'long-stop': dict(duration_s=6., velocity=.30, switch_control=100),
    'quiet': dict(duration_s=8., velocity=0., switch_control=None),
}


def configure(cfg, profile):
    command = cfg.commands['twist']
    command.heading_command = False
    command.ranges.heading = None
    for name in ('rel_standing_envs', 'rel_heading_envs', 'rel_world_envs', 'rel_forward_envs'):
        setattr(command, name, 0.)
    if command.init_velocity_prob != 0.:
        raise ValueError('This lesson retains saved init_velocity_prob=0')
    command.resampling_time_range = (9., 9.)
    command.ranges.lin_vel_x = (.30, .30)
    command.ranges.lin_vel_y = (0., 0.)
    command.ranges.ang_vel_z = (0., 0.)
    cfg.episode_length_s = 8. if profile == 'train' else PROFILES[profile]['duration_s']
    return cfg


class Schedule:
    def __init__(self, env, profile, directory):
        import torch
        self.env = env
        self.profile = profile
        self.journal = directory / 'command-events.jsonl'
        n = env.num_envs
        if profile == 'train' and n != 32:
            raise ValueError('Frozen training lesson has exactly32 worlds')
        self.cohort = torch.zeros(n, dtype=torch.long, device=env.device)
        self.velocity = torch.full((n,), PROFILES[profile]['velocity'] if profile != 'train' else .30, device=env.device)
        self.switch = torch.full((n,), -1, dtype=torch.long, device=env.device)
        if profile == 'train':
            self.cohort[16:24] = 1; self.cohort[24:] = 2
            self.velocity[16:24] = .25
            self.switch[16:24] = 30; self.switch[24:] = 100
        elif PROFILES[profile]['switch_control'] is not None:
            self.switch[:] = PROFILES[profile]['switch_control']
        self.generation = torch.full((n,), -1, dtype=torch.long, device=env.device)
        self.previous_k = torch.full_like(self.generation, -1)
        self.previous_request = torch.full_like(self.velocity, float('nan'))
        self.term = env.command_manager.get_term('twist')
        original = self.term._update_command
        def update():
            original()
            self.apply()
        self.term._update_command = update
        self.apply()
        # Initial packet already exists. Change only the three direct command
        # columns; never push sensing/history queues a second time.
        observations = {name: value.clone() for name, value in env.obs_buf.items()}
        for name, width, start in (('actor', 81, 69), ('critic', 96, 72)):
            if observations[name].shape != (n, width):
                raise ValueError('Unexpected original observation layout')
            observations[name][:, start:start + 3] = self.term.vel_command_b
        env.obs_buf = observations

    def apply(self):
        import torch
        k = self.env.episode_length_buf
        # A physical ending in the very first control can return0 after0;
        # compare the actual reset-local zero, not only a decreasing counter.
        reset = (self.previous_k < 0) | (k == 0)
        self.generation += reset.long()
        active = (self.switch < 0) | (k < self.switch)
        request = torch.where(active, self.velocity, 0.)
        self.term.vel_command_b.zero_()
        self.term.vel_command_b[:, 0] = request
        self.term.vel_command_w.copy_(self.term.vel_command_b)
        self.term.is_standing_env[:] = request == 0.
        for name in ('is_world_env', 'is_forward_env', 'is_heading_env'):
            getattr(self.term, name)[:] = False
        changed = reset | (request != self.previous_request)
        ids = changed.nonzero(as_tuple=False).flatten().cpu().tolist()
        if ids:
            fields = [x.detach().cpu().tolist() for x in (k, self.generation, self.cohort, request)]
            with self.journal.open('a') as stream:
                for i in ids:
                    stream.write(json.dumps(dict(world_slot=i, local_control=fields[0][i],
                        reset_generation=fields[1][i], cohort=fields[2][i], vx_m_s=fields[3][i],
                        reason='genuine_reset_or_initial' if bool(reset[i]) else 'command_switch',
                        common_step_counter=self.env.common_step_counter)) + '\n')
        self.previous_k.copy_(k)
        self.previous_request.copy_(request)


def recorder_class(original):
    class Episodes(original):
        def cache_inputs(self, value):
            super().cache_inputs(value)
            env = self._env
            schedule = env._forward_stop_schedule
            term = schedule.term
            for name, start in (('actor', 69), ('critic', 72)):
                import torch
                if not torch.equal(env.obs_buf[name][:, start:start + 3], term.vel_command_b):
                    raise ValueError('Cached issued command differs from schedule')
            fields = dict(input_command3=term.vel_command_b,
                input_standing_flag=term.is_standing_env,
                input_heading_flag=term.is_heading_env, input_world_flag=term.is_world_env,
                input_forward_flag=term.is_forward_env, cohort=schedule.cohort,
                reset_generation=schedule.generation, input_local_control=env.episode_length_buf)
            self.inputs.update({k: v.detach().cpu().numpy().copy() for k, v in fields.items()})
    return Episodes
