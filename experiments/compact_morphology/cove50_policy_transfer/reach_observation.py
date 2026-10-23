"""Timed RGB target observation for the experimental Direct Frame reach actor.

The actor receives image pixels reduced to a red-toy centroid and area, plus
capture health. Object pose, contact truth and the lift command stay outside
this module and may be used by the critic/reward only.
"""

from __future__ import annotations

import torch
from mjlab.managers.manager_base import ManagerTermBase


class TimedRgbTarget(ManagerTermBase):
    """Stateful image observation: u, v, area, fresh, age/max_age, missing.

    The native MJLab camera renders once per environment step, so that step is
    its capture time. An external camera adapter can provide its older capture
    step explicitly. A missing or frozen frame holds the last image-derived
    centroid only while it is young; after max_age_s the coordinate clears.
    No simulator target position is consulted. Reset clears per-world history.
    """

    def __init__(self, cfg, env):
        super().__init__(env)
        self.last_uv_area = torch.zeros((env.num_envs, 3), device=env.device)
        self.last_seen_step = torch.full((env.num_envs,), -1, dtype=torch.long,
                                         device=env.device)
        self.last_processed_step = torch.full_like(self.last_seen_step, -1)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.last_uv_area[env_ids] = 0
        self.last_seen_step[env_ids] = -1
        self.last_processed_step[env_ids] = -1

    def __call__(self, env, *, sensor_name: str, max_age_s: float = 0.2,
                 min_pixels: int = 10, rgb_override=None,
                 capture_step_override=None):
        if max_age_s <= 0 or min_pixels < 1:
            raise ValueError("Expected a positive image age and pixel threshold")
        rgb = (env.scene[sensor_name].data.rgb if rgb_override is None
               else rgb_override)
        step = env.episode_length_buf
        capture_step = (step if capture_step_override is None else
                        torch.as_tensor(capture_step_override, device=env.device,
                                        dtype=step.dtype).expand_as(step))
        if capture_step_override is not None and bool(
                ((capture_step < 0) | (capture_step > step)).any()):
            raise ValueError("RGB capture step must be within the current episode")
        new_step = step != self.last_processed_step
        if rgb is None:
            found = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        else:
            if rgb.ndim != 4 or rgb.shape[0] != env.num_envs or rgb.shape[-1] != 3:
                raise ValueError(f"Expected batched RGB [world,H,W,3], got {rgb.shape}")
            pixels = rgb.to(torch.float32)
            red, green, blue = pixels.unbind(-1)
            # The task's free cube is red (0.8, 0.2, 0.2). This is image
            # classification only; MuJoCo segmentation and cube pose are absent.
            mask = (red > 100) & (red > green * 1.5) & (red > blue * 1.5)
            count = mask.sum(dim=(1, 2))
            found = count >= min_pixels
            h, w = mask.shape[1:]
            yy = torch.arange(h, device=rgb.device, dtype=torch.float32)
            xx = torch.arange(w, device=rgb.device, dtype=torch.float32)
            denom = count.clamp_min(1)
            u = (mask.sum(dim=1).float() * xx).sum(dim=1) / denom / max(w - 1, 1)
            v = (mask.sum(dim=2).float() * yy).sum(dim=1) / denom / max(h - 1, 1)
            area = count.float() / (h * w)
            sample = torch.stack((u, v, area), dim=-1)
            accepted = found & new_step & (capture_step > self.last_seen_step)
            self.last_uv_area = torch.where(accepted[:, None], sample,
                                            self.last_uv_area)
            self.last_seen_step = torch.where(accepted, capture_step,
                                              self.last_seen_step)
        self.last_processed_step = torch.where(new_step, step,
                                               self.last_processed_step)
        age_steps = (step - self.last_seen_step).clamp_min(0)
        age_s = torch.where(self.last_seen_step >= 0,
                            age_steps.float() * env.step_dt,
                            torch.full_like(age_steps.float(), max_age_s))
        fresh = (self.last_seen_step >= 0) & (age_s <= max_age_s)
        uv_area = torch.where(fresh[:, None], self.last_uv_area,
                              torch.zeros_like(self.last_uv_area))
        # A currently missing target is distinct from a retained but still
        # usable image estimate. The age is capped for bounded actor input.
        return torch.cat((uv_area, fresh.float()[:, None],
                          (age_s / max_age_s).clamp(0, 1)[:, None],
                          (~found).float()[:, None]), dim=-1)
