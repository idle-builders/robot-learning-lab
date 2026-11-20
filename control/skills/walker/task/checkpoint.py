"""Load a cross-body saved actor/value pair or an exact R3 continuation.

Expanded source tensors need fresh Adam moments. R3 continuation tensors keep
their Adam state, effective learning rate, normalizers and curriculum counter.
"""
from copy import deepcopy
from pathlib import Path
import torch
from adapter import CHECKPOINT, expand_checkpoint, ref

LAYOUT = 'r3-forward-v2:actor72+9:critic-native-imu65mm87+9:raw21+3'
CRITIC = ('native old-IMU point body velocity3; native body gyro3; native projected gravity3; '
    'named mapped native q-home21; native qd21; previous raw21; command3; '
    'foot heights2; air times2; contact flags2; signed log1p contact forces6; '
    'new native q3,qd3,previous raw3')


def initialize(runner, env, *, source_checkpoint=None, checkpoint=None):
    path = Path(checkpoint or source_checkpoint or CHECKPOINT)
    saved = torch.load(path, map_location=runner.device, weights_only=False)
    if checkpoint:
        info = saved.get('infos') or {}
        if info.get('r3_contract') != LAYOUT:
            raise ValueError('R3 continuation lacks the exact actor/value/privileged-layout contract. Use --source-checkpoint for original72/87 networks; prior actor-only pilot checkpoints cannot be resumed as learned-value seeds.')
        state = saved
        provenance = deepcopy(info['source_seed'])
        rounds = int(info['new_ppo_rounds_total'])
        optimizer_steps = int(info['new_optimizer_steps_total'])
        kind = 'Compatible R3 actor, value, both normalizers and Adam restored'
    else:
        if saved['actor_state_dict']['mlp.0.weight'].shape[1] != 72 or saved['critic_state_dict']['mlp.0.weight'].shape[1] != 87:
            raise ValueError('Source transfer requires original actor72 and privileged critic87')
        state = expand_checkpoint(saved)
        provenance = dict(checkpoint=ref(path), inherited_iteration=saved.get('iter'),
            source_effective_adam=[{k:v for k,v in g.items() if k!='params'} for g in saved['optimizer_state_dict']['param_groups']],
            source_curriculum_counter=(saved.get('infos') or {}).get('env_state', {}).get('common_step_counter',20400),
            actor_value='Both saved actor and saved value with learned normalizers; zero new feature columns; new actor means zero',
            moment_policy='Fresh moments only for initial cross-body expansion; source effective LR/betas/eps retained')
        rounds = optimizer_steps = 0
        kind = 'Cross-body saved actor/value transfer; fresh compatible Adam moments'
    runner.alg.actor.load_state_dict(state['actor_state_dict'], strict=True)
    runner.alg.critic.load_state_dict(state['critic_state_dict'], strict=True)
    runner.alg.optimizer.load_state_dict(state['optimizer_state_dict'])
    lrs = {g['lr'] for g in runner.alg.optimizer.param_groups}
    if len(lrs) != 1: raise ValueError('PPO adaptive scheduler requires one effective LR')
    runner.alg.learning_rate = next(iter(lrs))
    runner.current_learning_iteration = rounds
    env.common_step_counter = int((state.get('infos') or {}).get('env_state', {}).get('common_step_counter',20400))
    env.curriculum_manager.compute(env_ids=torch.arange(env.num_envs, device=env.device))
    return dict(kind=kind, checkpoint=ref(path), source_seed=provenance,
        r3_contract=LAYOUT, privileged_observation96=CRITIC,
        initial_effective_learning_rate=runner.alg.learning_rate,
        initial_new_ppo_rounds_total=rounds, initial_new_optimizer_steps_total=optimizer_steps,
        source_curriculum_counter=env.common_step_counter)


def save(runner, path, transfer, counts):
    rounds = transfer.get('initial_new_ppo_rounds_total',0) + counts['ppo_rounds']
    optimizer_steps = transfer.get('initial_new_optimizer_steps_total',0) + counts['optimizer_steps']
    state = runner.alg.save()
    # This field is a completed-round count, deliberately distinct from the
    # upstream runner's last zero-based loop index.
    state['iter'] = rounds
    state['infos'] = dict(r3_contract=LAYOUT, source_seed=transfer.get('source_seed',{}),
        new_ppo_rounds_total=rounds, new_optimizer_steps_total=optimizer_steps,
        effective_learning_rate=runner.alg.learning_rate,
        env_state=dict(common_step_counter=runner.env.unwrapped.common_step_counter))
    torch.save(state, path)
    return ref(path)
