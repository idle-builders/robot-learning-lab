"""Bounded interventions around NVIDIA's released example action loader."""
import json
import os
from pathlib import Path


def change_actions(actions, mode):
    """Preserve gripper semantics; vary only the authored arm motion."""
    result = actions.copy()
    if mode == 'hold_arm':
        result[:, :6] = 0
    elif mode == 'reverse_translation':
        result[:, :3] *= -1
    elif mode != 'recorded':
        raise ValueError('Unsupported action intervention')
    return result


def load_actions():
    from cosmos_predict2.action_conditioned import load_default_action_fn
    original = load_default_action_fn()

    def load(data, video_path, args):
        result = original(data, video_path, args)
        result['actions'] = change_actions(result['actions'], os.environ['ROBOTLAB_ACTION_MODE'])
        output = Path(os.environ['ROBOTLAB_ACTION_OUTPUT'])
        output.write_text(json.dumps({'mode': os.environ['ROBOTLAB_ACTION_MODE'],
            'actions': result['actions'][:12].tolist(), 'source': 'upstream state-derived example actions',
            'scope': 'Offline replay/intervention input; not a causal Pip policy or a new observed outcome.'}, indent=2))
        return result
    return load
