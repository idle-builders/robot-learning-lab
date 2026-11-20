"""Whole-episode signed turn schedules; command updates never reset state."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    duration_s: float
    initial: tuple[float, float, float]
    successor: tuple[float, float, float] | None = None
    switch_s: float | None = None

    def command(self, control: int, dt: float = .02):
        if control < 0:
            raise ValueError('Control index must be nonnegative')
        if self.switch_s is not None and control >= round(self.switch_s / dt):
            return self.successor
        return self.initial

    def configure(self, cfg):
        """Freeze initial sampling, leaving the original command class intact."""
        cmd = cfg.commands['twist']
        for name, value in zip(('lin_vel_x', 'lin_vel_y', 'ang_vel_z'), self.initial):
            setattr(cmd.ranges, name, (value, value))
        cmd.heading_command = False
        for name in ('rel_standing_envs', 'rel_heading_envs', 'rel_world_envs', 'rel_forward_envs'):
            setattr(cmd, name, 0.)
        cmd.resampling_time_range = (self.duration_s + 1., self.duration_s + 1.)
        cfg.episode_length_s = self.duration_s
        cfg.auto_reset = False

    def apply(self, term, control: int):
        """Only command buffers/flags change; no robot/history/queue access."""
        values = self.command(control)
        for index, value in enumerate(values):
            term.vel_command_b[:, index] = value
            term.vel_command_w[:, index] = value
        term.is_standing_env[:] = not any(values)
        for name in ('is_world_env', 'is_forward_env', 'is_heading_env'):
            if hasattr(term, name):
                getattr(term, name)[:] = False
        return values


PROFILES = {}
for side, sign in (('left', 1.), ('right', -1.)):
    for profile in (
        Profile(f'moving-{side}', 4., (.25, 0., sign * .3)),
        Profile(f'pure-{side}', 4., (0., 0., sign * .4)),
        Profile(f'forward-to-{side}', 3., (.25, 0., 0.), (0., 0., sign * .3), .9),
    ):
        PROFILES[profile.name] = profile
