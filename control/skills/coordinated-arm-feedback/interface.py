"""Causal whole22 reference-plus-residual interface; no training/native launcher.

Construct only under a separately released execution. Observation preparation
belongs after reset/command updates and BEFORE returned next observations enter
PPO normalization. Action issue consumes that same cached observation. Recorder
must snapshot inputs before issue; terminal pre-reset rows retain issued inputs.
"""

ACTOR_WIDTH = 120
RESIDUAL_RAD = .03
BLOCKS = (("delayed_q22_rad", 22), ("delayed_qd22_rad_s", 22),
          ("delayed_gyro3_rad_s", 3), ("estimated_gravity3", 3),
          ("beta_and_beta_dot", 2), ("motor_reference22_rad", 22),
          ("pose_goal22_rad", 22), ("own_previous_raw_residual22", 22),
          ("sensor_valid_and_age_s", 2))


def fresh_actor():
    """New 120→64→64→22 mean, no inherited actor/normalizer/Adam tensors.

    Exploration distribution and privileged value inputs are intentionally not
    selected here. A zero mean is baseline identity only for deterministic mean
    action; a sampled nonzero exploration action is a physical intervention.
    """
    import torch
    actor = torch.nn.Sequential(torch.nn.Linear(ACTOR_WIDTH, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 22))
    torch.nn.init.zeros_(actor[-1].weight)
    torch.nn.init.zeros_(actor[-1].bias)
    return actor


class Interface:
    def __init__(self, num_envs, joint_order, reference_module, donor, goals,
                 *, device, dtype=None):
        import torch
        self.torch = torch
        self.names = list(joint_order)
        if len(self.names) != 22 or len(set(self.names)) != 22:
            raise ValueError("Physical named22 order required")
        self.reference, self.donor, self.goals = reference_module, donor, goals
        dtype = torch.float32 if dtype is None else dtype
        self.obs = torch.zeros((num_envs, ACTOR_WIDTH), device=device, dtype=dtype)
        self.previous_raw = self.obs.new_zeros((num_envs, 22))
        self.bootstrap = self.previous_raw.clone()
        self.motor_reference = self.previous_raw.clone()
        self.pose_goal = self.previous_raw.clone()
        self.delta = self.previous_raw.clone()
        self.requested = self.previous_raw.clone()
        self.valid = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.generation = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.prepared_generation = torch.full_like(self.generation, -1)
        self.prepared_tick = self.prepared_generation.clone()
        self.prepared_control = self.prepared_generation.clone()
        self.initialized = self.valid.clone()
        self.issued = self.valid.clone()

    def reset(self, ids, actual_reset_q22):
        """Called after real reset events/clamps, even for first-control 0→0.

        Does NOT reset the physical runtime, clocks, sensor FIFO or motor delay;
        their existing actual-reset owner remains responsible for that operation.
        Command boundaries never call this method. q22 is bootstrap-only and is
        never substituted for a missing actor sensor packet.
        """
        if actual_reset_q22.shape != self.bootstrap[ids].shape:
            raise ValueError("Reset q22 rows/order differ")
        self.bootstrap[ids] = actual_reset_q22
        self.previous_raw[ids] = 0
        self.delta[ids] = 0
        self.requested[ids] = actual_reset_q22
        self.generation[ids] += 1
        self.prepared_generation[ids] = -1
        self.prepared_tick[ids] = -1
        self.prepared_control[ids] = -1
        self.initialized[ids] = True
        self.issued[ids] = False

    def observe(self, runtime, local_control):
        """Only delayed deployed packet plus command-owned information enter120.

        Repeat observations at one generation/tick/control return the same cache,
        including previous raw BEFORE that control's issue. Per-row generation
        changes identify actual partial resets; counter decrease is insufficient.
        """
        t = self.torch
        if runtime.names != self.names or runtime.last_values.shape != (len(self.obs), 50):
            raise ValueError("Delayed packet/order differs from frozen physical22")
        if not bool(self.initialized.all()):
            raise RuntimeError("Actual reset bootstrap has not been captured")
        needed = ((self.prepared_generation != self.generation) |
                  (self.prepared_tick != runtime.tick) |
                  (self.prepared_control != local_control))
        for row in needed.nonzero(as_tuple=False).flatten().tolist():
            beta, beta_dot = self.reference.command(int(local_control[row]))
            # Preserve donor CPU float32 inference and the exact float(beta) path
            # used by the baseline; no vectorized polynomial replacement.
            motor = self.donor.motor_reference(beta).to(self.obs)
            pose = self.obs.new_tensor(self.goals.at(beta)["q_pose_goal22_rad"])
            valid = runtime.valid[row]
            age = t.where(valid, runtime.tick[row] * runtime.physics_dt_s -
                          runtime.capture_time_s[row], 0.).to(self.obs.dtype)
            self.motor_reference[row] = motor
            self.pose_goal[row] = pose
            self.valid[row] = valid
            self.obs[row] = t.cat((runtime.last_values[row].to(self.obs),
                self.obs.new_tensor([beta, beta_dot]), motor, pose,
                self.previous_raw[row], t.stack((valid.to(self.obs.dtype), age))))
        self.prepared_generation[needed] = self.generation[needed]
        self.prepared_tick[needed] = runtime.tick[needed]
        self.prepared_control[needed] = local_control[needed]
        self.issued[needed] = False
        return self.obs.clone()

    def issue(self, raw_residual22, runtime, local_control):
        """One action per prepared control, upstream of unchanged physical runtime.

        Return requested motor radians, NOT delayed applied targets. The ActionTerm
        must pass these through unchanged to existing guards/80 substeps. Keep
        accepted and applied targets separately from this request and raw history.
        """
        t = self.torch
        if raw_residual22.shape != self.previous_raw.shape:
            raise ValueError("Residual action must be raw22")
        stale = ((self.prepared_generation != self.generation) |
                 (self.prepared_tick != runtime.tick) |
                 (self.prepared_control != local_control) | self.issued)
        if bool(stale.any()):
            raise RuntimeError("Action needs one fresh matching causal observation")
        raw = raw_residual22.detach()
        proposed = RESIDUAL_RAD * t.tanh(raw)
        self.delta.copy_(t.where(self.valid[:, None], proposed, 0.))
        # Explicit zero branch preserves the donor's exact float32 bytes (including
        # signed zero), rather than relying on x+0 identity across dtypes/devices.
        corrected = t.where(self.delta == 0, self.motor_reference,
                            self.motor_reference + self.delta)
        self.requested.copy_(t.where(self.valid[:, None], corrected, self.bootstrap))
        self.previous_raw.copy_(raw)  # Even invalid/held ticks remember OWN raw22.
        self.issued[:] = True
        return self.requested.clone()
