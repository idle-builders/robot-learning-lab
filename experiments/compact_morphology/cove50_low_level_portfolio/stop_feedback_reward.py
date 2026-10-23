"""Stop/settle PPO rewards over actual post-gait physics states.

These are training signals. Full walk→stop→squat→arms support, contact and
rated-duty gates stay in the independent native evaluator.
"""

from __future__ import annotations

import math

import torch


def _phase(env):
    t = env.episode_length_buf * env.step_dt
    return (t >= env.stop_feedback_walk_end) & (t < env.stop_feedback_stop_end)


def _upright_and_feet(env):
    robot = env.scene["robot"]
    feet = env.scene.sensors["feet_ground_contact"].data.found.bool()
    quat = robot.data.root_link_pose_w[:, 3:7]
    up = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
    return (feet[:, 0] & feet[:, 1] &
            (robot.data.root_link_pos_w[:, 2] >= 0.20) &
            (up >= math.cos(math.radians(15))))


def _bad_contact(env):
    sensors = env.scene.sensors
    hand = sensors["hand_ground_contact"].data.force.norm(dim=-1).amax(dim=-1) >= 1.0
    body = sensors["handoff_body_ground_contact"].data.found.any(dim=-1)
    self_contact = sensors["handoff_self_contact"].data.found.any(dim=-1)
    return hand | body | self_contact


def _rated_leg(env):
    force = env.scene["robot"].data.actuator_force[:, :10].abs()
    rating = torch.cat([act.force_limit for act in env.scene["robot"].actuators[:10]], dim=1)
    return (force >= 0.995 * rating).any(dim=1)


def supported_settle(env):
    """Dense upright two-foot signal only after the command asks for a stop."""
    return (_phase(env) & _upright_and_feet(env) & ~_bad_contact(env)).float()


def low_settle_speed(env, target_m_s: float = 0.06):
    robot = env.scene["robot"]
    speed = robot.data.root_link_lin_vel_w[:, :2].norm(dim=-1)
    valid = _phase(env) & _upright_and_feet(env) & ~_bad_contact(env)
    return valid.float() * torch.exp(-speed.square() / (target_m_s ** 2))


def rated_leg_penalty(env):
    return (_phase(env) & _rated_leg(env)).float()


def forbidden_contact_penalty(env):
    return (_phase(env) & _bad_contact(env)).float()


def residual_magnitude(env):
    return env.stop_feedback_residual[:, :10].square().sum(dim=-1)


class SettledStopTerminal:
    """Pay once for the evaluator's last stop-actor second after real gait.

    This intentionally requires travel inherited from the gait warmup so a
    stationary shortcut cannot collect the terminal training proxy.
    """

    def __init__(self, cfg, env):
        del cfg
        n = env.num_envs
        self.window_steps = torch.zeros(n, dtype=torch.int32, device=env.device)
        self.support_steps = torch.zeros_like(self.window_steps)
        self.rated_steps = torch.zeros_like(self.window_steps)
        self.bad_contact = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.start_xy = torch.zeros((n, 2), device=env.device)
        self.start_saved = torch.zeros(n, dtype=torch.bool, device=env.device)
        self.tail_xy = torch.zeros_like(self.start_xy)
        self.tail_saved = torch.zeros_like(self.start_saved)
        self.paid = torch.zeros_like(self.bad_contact)
        self.total_payments = torch.zeros((), dtype=torch.long, device=env.device)
        self.total_checks = torch.zeros_like(self.total_payments)

    def __call__(self, env):
        t = env.episode_length_buf * env.step_dt
        robot = env.scene["robot"]
        current_xy = robot.data.root_link_pos_w[:, :2]
        self.start_xy = torch.where(self.start_saved[:, None], self.start_xy, current_xy)
        self.start_saved[:] = True
        self.bad_contact |= _bad_contact(env)
        window = (t >= env.stop_feedback_stop_end - 1.0) & (t < env.stop_feedback_stop_end)
        self.window_steps += window.int()
        self.support_steps += (window & _upright_and_feet(env)).int()
        self.rated_steps += (window & _rated_leg(env)).int()
        tail = (t >= env.stop_feedback_stop_end - 0.5) & ~self.tail_saved
        self.tail_xy = torch.where(tail[:, None], current_xy, self.tail_xy)
        self.tail_saved |= tail
        at_end = (t >= env.stop_feedback_stop_end) & ~self.paid
        travel = (current_xy - self.start_xy).norm(dim=-1)
        expected = env.stop_feedback_speed * env.stop_feedback_walk_end
        tail_travel = (current_xy - self.tail_xy).norm(dim=-1)
        enough_window = self.window_steps >= 48
        denom = self.window_steps.clamp_min(1).float()
        success = (at_end & enough_window &
                   (self.support_steps.float() / denom >= 0.95) &
                   (self.rated_steps.float() / denom <= 0.10) &
                   (travel >= 0.5 * expected) & self.tail_saved &
                   (tail_travel <= 0.03) &
                   _upright_and_feet(env) & ~self.bad_contact)
        self.total_payments += success.long().sum()
        self.total_checks += at_end.long().sum()
        self.paid |= at_end
        return success.float()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.window_steps[ids] = 0
        self.support_steps[ids] = 0
        self.rated_steps[ids] = 0
        self.bad_contact[ids] = False
        self.start_xy[ids] = 0.0
        self.start_saved[ids] = False
        self.tail_xy[ids] = 0.0
        self.tail_saved[ids] = False
        self.paid[ids] = False
