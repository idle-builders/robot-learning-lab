"""Hidden physical episode scenarios; construction never consumes randomness."""


class Scenarios:
    def __init__(self, seeds, *, device, fixed=None):
        import torch
        self.torch = torch
        self.generators = [torch.Generator(device='cpu').manual_seed(seed) for seed in seeds]
        self.seeds, self.fixed = list(seeds), fixed
        self.onset = torch.zeros(len(seeds), dtype=torch.long, device=device)
        self.stop = torch.zeros_like(self.onset)
        self.force_Y = torch.zeros(len(seeds), device=device)
        self.generation = torch.zeros_like(self.onset)
        self.applied_dose = torch.zeros(len(seeds), 3, dtype=torch.float64, device=device)
        self.applied_substeps = torch.zeros_like(self.onset)
        self.before, self.after, self.current, self.history = {}, {}, {}, []

    def reset(self, ids, generations):
        torch = self.torch
        for i, generation in zip(ids.tolist(), generations.tolist()):
            if generation != int(self.generation[i]) + 1:
                raise ValueError('Scenario draw requires an actual new reset generation')
            if i in self.current:
                self.finalize(i)
            generator = self.generators[i]
            self.before[i] = generator.get_state().clone()
            if self.fixed is None:
                values = torch.rand(4, generator=generator, dtype=torch.float64).tolist()
                none, sign, magnitude, time = values
                force = 0. if none < .25 else (-1. if sign < .5 else 1.) * (2. if magnitude < .5 else 4.)
                onset, duration = 120 + int(time * 231), 10
                kind = 'episode_randomized'
            else:
                row = self.fixed[i]
                onset, duration, force = row['onset_control'], row['duration_controls'], row['force_world_Y_N']
                values, kind = [], 'fixed_matched_development_case'
            self.after[i] = generator.get_state().clone()
            self.onset[i], self.stop[i], self.force_Y[i] = onset, onset + duration, force
            self.generation[i] = generation
            self.applied_dose[i] = 0
            self.applied_substeps[i] = 0
            row = dict(world_slot=i, reset_generation=generation, scenario_seed=self.seeds[i],
                       kind=kind, uniform_draws=values, onset_control=onset,
                       stop_control=onset + duration, force_world_Y_N=force,
                       intended_dose_world_N_s=[0., force * duration * .02, 0.])
            self.current[i] = row
            self.history.append(row)

    def force(self, local_control, physics_dt_s):
        torch = self.torch
        if bool((self.generation == 0).any()):
            raise RuntimeError('Physical scenario requires real reset before stepping')
        active = (local_control >= self.onset) & (local_control < self.stop)
        force = torch.zeros((len(self.seeds), 1, 3), device=self.force_Y.device)
        force[:, 0, 1] = active * self.force_Y
        return force

    def sample_substep(self, issued_force, physics_dt_s):
        """Only after an actual successful sim.step return, never on request."""
        self.applied_dose += issued_force[:, 0].to(self.torch.float64) * physics_dt_s
        self.applied_substeps += (issued_force[:, 0].abs().sum(dim=1) != 0).long()

    def finalize(self, i):
        self.current[i]['observed_applied_dose_world_N_s'] = self.applied_dose[i].tolist()
        self.current[i]['observed_force_substeps'] = int(self.applied_substeps[i])

    def summary(self):
        for i in self.current:
            self.finalize(i)
        return self.history

    def initialization(self, i):
        torch = self.torch
        return dict(scenario_rng_before=self.before[i], scenario_rng_after=self.after[i],
                    scenario_uniform_draws=torch.tensor(self.current[i]['uniform_draws'], dtype=torch.float64),
                    scenario_seed=torch.tensor(self.seeds[i]),
                    scenario_generation=self.generation[i], scenario_onset_control=self.onset[i],
                    scenario_stop_control=self.stop[i], scenario_force_Y_N=self.force_Y[i],
                    scenario_intended_dose_world_N_s=torch.tensor(self.current[i]['intended_dose_world_N_s'], dtype=torch.float64),
                    scenario_applied_dose_world_N_s=self.applied_dose[i],
                    scenario_applied_substeps=self.applied_substeps[i])
