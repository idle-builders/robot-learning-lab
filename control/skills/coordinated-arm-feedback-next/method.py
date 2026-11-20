"""Same seven rewards/interfaces; exact full-state warm continuation only."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('feedback_r1_method', ROOT / 'control/skills/coordinated-arm-feedback/method.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
REWARDS = base.REWARDS
reward_terms, value_observation = base.reward_terms, base.value_observation
operational_json, lateral_force = base.operational_json, base.lateral_force


def agent_config():
    cfg = base.agent_config()
    cfg.max_iterations, cfg.save_interval = 12, 12
    return cfg


def restore_training(runner, saved, expected_method):
    if saved['method'] != expected_method or saved['layout'] != dict(actor=120, critic=174, raw=22):
        raise ValueError('Exact original full120/174/raw22 checkpoint required')
    if saved['counts'] != dict(simulator_controls=8000, physics_substeps=640000,
                               PPO_rounds=4, optimizer_updates=80, new_weight_sets=1):
        raise ValueError('Expected actual first-lesson checkpoint counters')
    if saved['common_counter'] != 2000 or saved['normalizer_counts'] != dict(actor=8000, critic=8000):
        raise ValueError('Expected original common2000 and both normalizers8000')
    runner.alg.actor.load_state_dict(saved['actor_state_dict'], strict=True)
    runner.alg.critic.load_state_dict(saved['critic_state_dict'], strict=True)
    runner.alg.optimizer.load_state_dict(saved['optimizer_state_dict'])
    if len(runner.alg.optimizer.state) != 13 or {int(value['step']) for value in runner.alg.optimizer.state.values()} != {80}:
        raise ValueError('All13 original Adam parameter states/80steps required')
    if any(int(model.obs_normalizer.count) != 8000 for model in (runner.alg.actor, runner.alg.critic)):
        raise ValueError('Actual restored normalizers differ')
    runner.current_learning_iteration = 4
    runner.alg.train_mode()
    return dict(PPO_rounds=4, optimizer_updates=80, simulator_controls=8000, common_counter=2000)
