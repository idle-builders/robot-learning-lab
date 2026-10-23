"""Export a labeled, unlearned zero-action control for the native evaluator."""
import sys
import torch


class ZeroAction(torch.nn.Module):
    def forward(self, observation):
        return observation.new_zeros((observation.shape[0], 6))


policy = torch.jit.trace(ZeroAction(), torch.zeros(1, 230))
assert policy(torch.zeros(4, 230)).shape == (4, 6)
policy.save(sys.argv[1])
