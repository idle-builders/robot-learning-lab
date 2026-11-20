"""Explicit Direct Frame actor interface on the selected assembled body.

The old 72 observations and all 21 raw outputs survive as a prefix. Nine
observations and three action heads make the added axes observable/trainable.
This mapping is an initial neutral-axis approximation, not IK retargeting.
"""
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'control/skills/forward-walk'))
from contracts import OLD_BODY, CHECKPOINT, MODEL_DIR, MODEL_POINTER, names, mapping, old_home, read, ref
import mujoco


class Interface:
    def __init__(self):
        self.old = mujoco.MjModel.from_xml_path(str(OLD_BODY))
        self.current = mujoco.MjModel.from_xml_path(str(MODEL_DIR / 'cove50-rigid.xml'))
        self.source_names = names(self.old)
        self.current_names = names(self.current)
        self.rows = mapping(self.old, self.current)
        self.new_names = [r['current_name'] for r in self.rows if r['source_index'] is None]
        self.home = torch.tensor(old_home(self.source_names), dtype=torch.float32)
        self.new_rows = {n: 21 + i for i, n in enumerate(self.new_names)}
        saved_env = yaml.load((ROOT / 'artifacts/ape-cove50-recovery-20261008/ankleonly100-params/env.yaml').read_text(), Loader=yaml.BaseLoader)
        clips = saved_env['actions']['joint_pos']['clip']
        self.source_clips = {n.strip('^$'): tuple(map(float, limits)) for n, limits in clips.items()}
        self._device_tensors = {}

    def tensors(self, device):
        if str(device) not in self._device_tensors:
            indices = [0] * 21; signs = [0.] * 21
            for r in self.rows:
                if r['source_index'] is not None:
                    indices[r['source_index']] = r['current_index']; signs[r['source_index']] = r['sign']
            action_ids = [r['source_index'] if r['source_index'] is not None else self.new_rows[r['current_name']] for r in self.rows]
            action_scale = [r['sign'] * .3 if r['source_index'] is not None and r['source_index'] < 10
                else .3 if r['source_index'] is None else 0. for r in self.rows]
            action_offset = [r['sign'] * self.home[r['source_index']].item() if r['source_index'] is not None else 0. for r in self.rows]
            bounds = []
            for r in self.rows:
                lower, upper = self.source_clips.get(r['source_name'], (-float('inf'), float('inf')))
                bounds.append((lower, upper) if r.get('sign', 1.) > 0 else (-upper, -lower))
            self._device_tensors[str(device)] = dict(qids=torch.tensor(indices, device=device),
                signs=torch.tensor(signs, device=device),
                new_ids=torch.tensor([self.current_names.index(n) for n in self.new_names], device=device),
                action_ids=torch.tensor(action_ids, device=device),
                scale=torch.tensor(action_scale, device=device), offset=torch.tensor(action_offset, device=device),
                home=self.home.to(device), lower=torch.tensor([x[0] for x in bounds], device=device),
                upper=torch.tensor([x[1] for x in bounds], device=device))
        return self._device_tensors[str(device)]

    def observe(self, q, qd, gyro, gravity, previous_raw, command):
        """q/qd in exact current actuator order; sensor inputs are delayed."""
        t = self.tensors(q.device)
        old_q = q[:, t['qids']] * t['signs']; old_qd = qd[:, t['qids']] * t['signs']
        return torch.cat((gyro, gravity, old_q - t['home'], old_qd,
                          previous_raw[:, :21], command, q[:, t['new_ids']],
                          qd[:, t['new_ids']], previous_raw[:, 21:]), dim=1)

    def target(self, raw):
        """Output-joint radians BEFORE the exact R3 runtime command clipping."""
        t = self.tensors(raw.device)
        return torch.clamp(raw[:, t['action_ids']] * t['scale'] + t['offset'], t['lower'], t['upper'])

    def contract(self):
        return dict(source_checkpoint=ref(CHECKPOINT), source_body=ref(OLD_BODY),
            selected_pointer=ref(MODEL_POINTER), selected_model=read(MODEL_POINTER),
            source_order=self.source_names, physical_order=self.current_names,
            mapping=self.rows, source_home_rad=self.home.tolist(), new_axes=self.new_names,
            actor_observation81='old72 prefix: gyro3,gravity3,(mapped_q-home)21,mapped_qd21,previous_raw21,command3; then new_q3,new_qd3,previous_raw_new3',
            raw_action24='source raw21 including both absent gripper histories, then new3',
            physical_target22='signed source home+.3*raw for ten old leg axes; signed source home for old upper axes; .3*raw for three new axes. Exact saved source ankle-target clips retained before R3 runtime independently clips accepted targets.',
            retained_source_target_clips_rad=self.source_clips,
            limitations=['Neutral-axis signs do not preserve whole-body poses through changed serial axes.',
                'Removed gripper q/qd are filled zero; their raw history is retained.',
                'The original upper11 action scale stays zero; added neck pitch and ankle rolls have .3rad/raw-unit scale.',
                'Delayed measured gyro/estimated gravity replace historical instantaneous native state.'])


def expand_checkpoint(saved):
    """Keep the learned prefix exactly; discard incompatible optimizer moments."""
    import copy
    result = copy.deepcopy(saved)
    for group, old_width in [('actor_state_dict', 72), ('critic_state_dict', 87)]:
        state = result[group]
        if state['mlp.0.weight'].shape[1] != old_width:
            raise ValueError(f'{group} requires original width{old_width}')
        for key, fill in [('obs_normalizer._mean', 0.), ('obs_normalizer._var', 1.), ('obs_normalizer._std', 1.)]:
            state[key] = torch.cat((state[key], state[key].new_full((1, 9), fill)), dim=1)
        weight=state['mlp.0.weight']
        state['mlp.0.weight'] = torch.cat((weight, weight.new_zeros((weight.shape[0], 9))), dim=1)
        if group.startswith('actor'):
            weight=state['mlp.6.weight'];bias=state['mlp.6.bias'];std=state['distribution.std_param']
            state['mlp.6.weight'] = torch.cat((weight, weight.new_zeros((3, weight.shape[1]))), dim=0)
            state['mlp.6.bias'] = torch.cat((bias, bias.new_zeros(3)))
            state['distribution.std_param'] = torch.cat((std, std.new_full((3,), .05)))
    result['optimizer_state_dict']['state'] = {}
    # RSL optimizer parameter count is unchanged; tensor shapes changed but no
    # moment tensors survive. iter=0 counts NEW PPO updates separately.
    result['iter'] = 0
    result['infos'] = dict(result.get('infos') or {}, transfer_source_iter=saved.get('iter'),
        transfer_source=ref(CHECKPOINT), transfer_initialization='Preserve old21 output exactly; new3 mean heads zero; new input columns zero; critic87 prefix retained.')
    return result
