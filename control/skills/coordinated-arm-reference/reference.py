"""Frozen beta-only whole22 donor and separately owned geometric goals."""
import json


def command(control):
    """Exact source50Hz schedule; time is external, the actor only receives beta."""
    t = control * .02
    beta = min(1., max(0., (t - 2.) / 2.)) if t < 6. else max(0., 1. - (t - 6.) / 2.)
    beta_dot = .5 if 2. <= t < 4. else -.5 if 6. <= t < 8. else 0.
    return beta, beta_dot


class Donor:
    def __init__(self, checkpoint, joint_order):
        import torch
        # Source command_student.model: final Linear has NO output Tanh,
        # action scale, source-home offset, raw clamp, or EMA.
        self.model = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 22))
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['joint_order'] != joint_order:
            raise ValueError('Frozen donor named22 motor-output order differs')
        self.model.load_state_dict(saved['actor_state_dict'], strict=True)
        self.model.eval()
        self.queries = 0

    def motor_reference(self, beta):
        import torch
        with torch.inference_mode():
            result = self.model(torch.tensor([[beta]], dtype=torch.float32))[0]
        self.queries += 1
        return result


class Goals:
    """Exact source-audited desiredq table; never drives the donor network."""
    def __init__(self, path, joint_order):
        import numpy as np
        table = json.loads(path.read_text())
        if table['joint_order'] != joint_order:
            raise ValueError('Geometric goal-table order differs')
        self.nodes = np.asarray(table['beta'])
        self.q = np.asarray(table['q_pose_goal22_rad'])
        self.root = np.asarray(table['geometric_root_xyz_m'])
        self.com = np.asarray(table['geometric_com_xyz_m'])

    def at(self, beta):
        import numpy as np
        def interpolate(values):
            return np.asarray([np.interp(beta, self.nodes, values[:, i])
                               for i in range(values.shape[1])])
        return dict(q_pose_goal22_rad=interpolate(self.q),
            geometric_root_xyz_m=interpolate(self.root),
            geometric_com_xyz_m=interpolate(self.com))
