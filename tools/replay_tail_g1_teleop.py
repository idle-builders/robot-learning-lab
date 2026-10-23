"""Replay the desktop demonstration through Arena's recorder with a longer release tail."""

import os

import h5py
import numpy as np
import torch
from isaaclab.utils.datasets import HDF5DatasetFileHandler


class ReplayTailDevice:
    def __init__(self, _callbacks):
        source = os.environ['REPLAY_TAIL_SOURCE']
        dataset = HDF5DatasetFileHandler()
        dataset.open(source)
        episode_name = os.environ.get('REPLAY_TAIL_EPISODE', 'demo_0')
        self.episode = dataset.load_episode(episode_name, 'cpu')
        with h5py.File(source) as recording:
            self.actions = np.asarray(recording[f'data/{episode_name}/actions'])
        self.extra_steps = int(os.environ.get('REPLAY_TAIL_EXTRA_STEPS', '180'))
        self.index = 0
        self.env = None

    def bind(self, env):
        self.env = env
        env.reset_to(self.episode.get_initial_state(), torch.tensor([0], device=env.device), is_relative=True)
        self.reset()

    def reset(self):
        self.index = 0

    def advance(self):
        if self.index > len(self.actions) + self.extra_steps:
            raise KeyboardInterrupt('Replay tail ended without a successful recording')
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return torch.as_tensor(action, dtype=torch.float32, device=self.env.device)
