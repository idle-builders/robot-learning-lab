"""Parallel saved-actor warmup with a learned stop/settle leg residual.

The underlying MJLab world runs every gait, stop and posture step. PPO actions
are ignored while walking, then bounded around the saved router's leg targets.
This is an unrun training interface until the native V2-prefix gate passes.
"""

from __future__ import annotations

from typing import Callable

import torch

from batched_controller import POSTURE, STOP, WALK, BatchedPortfolioRouter
from mjlab.rl import RslRlVecEnvWrapper


Policy = Callable[[object], torch.Tensor]


class StopFeedbackVecEnv(RslRlVecEnvWrapper):
    """RSL-RL VecEnv with physical saved-actor warmup in every MJLab world."""

    def __init__(self, env, *, clip_actions: float | None,
                 target_names: list[str], action_scale: torch.Tensor,
                 seed: int, residual_gain: float = 0.15,
                 speed_range: tuple[float, float] = (0.20, 0.35),
                 walk_s_range: tuple[float, float] = (1.5, 2.5),
                 stop_s_range: tuple[float, float] = (1.5, 2.0)):
        super().__init__(env, clip_actions=clip_actions)
        if not 0.0 < residual_gain <= 0.3:
            raise ValueError("stop residual gain must be in (0, 0.3]")
        for lo, hi in (speed_range, walk_s_range, stop_s_range):
            if lo > hi:
                raise ValueError("invalid stop-feedback randomization interval")
        if abs(self.unwrapped.step_dt - 0.02) > 1e-8:
            raise ValueError("saved actors require 50 Hz")
        self.router = BatchedPortfolioRouter(
            target_names, action_scale, self.num_envs, str(self.device))
        self.residual_gain = float(residual_gain)
        self.speed_range = speed_range
        self.walk_s_range = walk_s_range
        self.stop_s_range = stop_s_range
        self.random = torch.Generator(device=self.device).manual_seed(seed)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.walk_end = torch.zeros_like(self.speed)
        self.stop_end = torch.zeros_like(self.speed)
        self.zero_action = torch.zeros((self.num_envs, 21), device=self.device)
        self.zero_flag = torch.zeros(self.num_envs, device=self.device)
        self.reset_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.sample_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.sample_min = torch.full((3,), float("inf"), device=self.device)
        self.sample_max = torch.full((3,), float("-inf"), device=self.device)
        self.gait_policy: Policy | None = None
        self.stop_policy: Policy | None = None
        self.posture_policy: Policy | None = None
        self._resample(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        # Reward terms inspect these only after wrapper construction.
        self.unwrapped.stop_feedback_residual = self.zero_action.clone()
        self.unwrapped.stop_feedback_speed = self.speed
        self.unwrapped.stop_feedback_walk_end = self.walk_end
        self.unwrapped.stop_feedback_stop_end = self.stop_end

    def set_priors(self, gait: Policy, stop: Policy, posture: Policy) -> None:
        self.gait_policy = gait
        self.stop_policy = stop
        self.posture_policy = posture

    @torch.no_grad()
    def _draw(self, bounds: tuple[float, float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.rand(
            self.num_envs, generator=self.random, device=self.device)

    @torch.no_grad()
    def _resample(self, mask: torch.Tensor) -> None:
        self.speed = torch.where(mask, self._draw(self.speed_range), self.speed)
        self.walk_end = torch.where(mask, self._draw(self.walk_s_range), self.walk_end)
        self.stop_end = torch.where(
            mask, self.walk_end + self._draw(self.stop_s_range), self.stop_end)
        values = torch.stack((self.speed, self.walk_end, self.stop_end), dim=1)
        self.sample_min = torch.minimum(
            self.sample_min, torch.where(mask[:, None], values, float("inf")).amin(dim=0))
        self.sample_max = torch.maximum(
            self.sample_max, torch.where(mask[:, None], values, float("-inf")).amax(dim=0))
        self.sample_count += mask.long().sum()
        self.unwrapped.stop_feedback_speed = self.speed
        self.unwrapped.stop_feedback_walk_end = self.walk_end
        self.unwrapped.stop_feedback_stop_end = self.stop_end

    @torch.no_grad()
    def _phase(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.unwrapped.episode_length_buf * self.unwrapped.step_dt
        walking = t < self.walk_end
        stopping = (t >= self.walk_end) & (t < self.stop_end)
        mode = torch.where(walking, WALK, torch.where(stopping, STOP, POSTURE))
        return t, walking, mode

    @torch.no_grad()
    def _set_command(self, walking: torch.Tensor) -> None:
        command = self.unwrapped.command_manager.get_term("twist")
        command.vel_command_b[:, 0] = torch.where(walking, self.speed, 0.0)
        command.vel_command_b[:, 1:] = 0.0

    def get_observations(self):
        _, walking, _ = self._phase()
        self._set_command(walking)
        return super().get_observations()

    def reset(self):
        _, extras = super().reset()
        self.router.reset()
        self._resample(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        return self.get_observations(), extras

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        if self.gait_policy is None or self.stop_policy is None or self.posture_policy is None:
            raise RuntimeError("saved gait, stop and posture policies must be loaded first")
        if actions.shape != (self.num_envs, 21):
            raise ValueError("stop residual must have 21 native action channels")
        t, walking, mode = self._phase()
        self._set_command(walking)
        source_obs = super().get_observations()
        actor_obs = source_obs["actor"]
        posture_action = self.posture_policy(source_obs)
        gait_obs = source_obs.clone()
        gait_obs["actor"] = self.router.gait_observation(
            actor_obs, mode, self.speed, self.zero_flag)
        gait_action = self.gait_policy(gait_obs)
        stop_action = self.stop_policy(gait_obs)
        prior = torch.where(
            walking[:, None], gait_action,
            torch.where((mode == STOP)[:, None], stop_action, self.zero_action))
        baseline = self.router.route(
            mode, posture_action, self.zero_action, prior, self.zero_flag)
        # Zero at the walk boundary and full authority after 0.2 s. This
        # avoids the abrupt residual switch that failed in support V19.
        ramp = ((t - self.walk_end) / 0.2).clamp(0.0, 1.0)
        stop_authority = (mode == STOP).float()
        correction = actions.clamp(-1.0, 1.0) * (
            self.residual_gain * ramp * stop_authority)[:, None]
        correction[:, 10:] = 0.0
        self.unwrapped.stop_feedback_residual = correction
        applied = baseline + correction
        _, reward, dones, extras = super().step(applied)
        reset_mask = dones.bool()
        self.reset_count += reset_mask.long().sum()
        self.router.reset_where(reset_mask)
        self._resample(reset_mask)
        # The underlying environment computes post-reset observations before
        # this wrapper draws the next speed. Recompute them with the new command.
        return self.get_observations(), reward, dones, extras
