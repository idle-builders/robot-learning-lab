"""Thin full-state continuation of the completed first-feedback native runner."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[3] / 'control/skills/coordinated-arm-feedback/run.py'
source = BASE.read_text()
changes = [
    ('data/coordinated-arm-feedback-20261030', 'data/coordinated-arm-feedback-next-20261030', 1),
    ('from method import agent_config,operational_json,zero_fresh_mean',
     'from method import agent_config,operational_json,restore_training', 1),
    ("cases=method['training_cases'];signs=method['training_impulse_signs']",
     "cases=method['training_cases'];signs=method['training_scenarios']", 1),
    ("signs=[method['comparisons']['impulse_signs'][args.case_index]]",
     "signs=dict(seeds=[cases[0]['seed']],fixed=[method['comparisons']['force_scenarios'][args.case_index]])", 1),
    ('schema=\'cove50.coordinated-arm-feedback.native.v1\'', 'schema=\'cove50.coordinated-arm-feedback-next.native.v1\'', 1),
    ('counts=counts,cases=cases,impulse_signs=signs,', 'counts=counts,cases=cases,hidden_scenario_config=signs,', 1),
    ("One shared four-row Runtime generator and Torch stream seeded242100; per-row condition labels identify fixed entry shapes, not independently reseeded streams.",
     "One shared eight-row Runtime/Torch stream seeded252100. Four entry templates repeated twice; hidden per-world scenario RNGs252201..208 are separate from Runtime noise. Every actual reset draws; shape queries never draw.", 1),
    ('fresh=training,checkpoint=ref(args.checkpoint)if args.checkpoint else None,',
     "fresh=False,warm_continuation=training,checkpoint=method['warm_start']['candidate']if training else ref(args.checkpoint)if args.checkpoint else None,", 1),
    ('action.sampled_substep();return value',
     'action.sampled_substep();action.scenarios.sample_substep(action.issued_force,rt.physics_dt_s);return value', 1),
    ('vec=RslRlVecEnvWrapper(env)',
     "if training:env.common_step_counter=method['warm_start']['common_before']\n        vec=RslRlVecEnvWrapper(env)", 1),
    ('zero_fresh_mean(runner)',
     "saved=torch.load(ROOT/method['warm_start']['candidate']['path'],map_location=args.device,weights_only=False)\n                report['restored_learner_state']=restore_training(runner,saved,method['warm_start']['method'])", 1),
    ('runner.learn(num_learning_iterations=4,init_at_random_ep_len=False)',
     'runner.learn(num_learning_iterations=12,init_at_random_ep_len=False)', 1),
    ("frozen_donor=cfg['checkpoint'],goal_table=cfg['goal_table'],counts=dict(counts),common_counter=env.common_step_counter,",
     "frozen_donor=cfg['checkpoint'],goal_table=cfg['goal_table'],counts=dict(counts),common_counter=env.common_step_counter,\n                warm_start=method['warm_start'],training_lineage_totals=dict(simulator_controls=8000+counts['simulator_controls'],physics_substeps=640000+counts['physics_substeps'],PPO_rounds=4+counts['PPO_rounds'],optimizer_updates=80+counts['optimizer_updates']),\n                last_runner_iteration=runner.current_learning_iteration,", 1),
    ('own_previous_raw22=action.feedback.previous_raw,physical_delta22_rad=action.feedback.delta)',
     "own_previous_raw22=action.feedback.previous_raw,physical_delta22_rad=action.feedback.delta,\n                    issued_force_world_N=action.issued_force,scenario_generation=action.scenarios.generation,\n                    scenario_onset_control=action.scenarios.onset,scenario_stop_control=action.scenarios.stop,\n                    scenario_force_Y_N=action.scenarios.force_Y,scenario_applied_dose_world_N_s=action.scenarios.applied_dose,\n                    scenario_applied_substeps=action.scenarios.applied_substeps,\n                    scenario_rng_states=torch.stack([g.get_state()for g in action.scenarios.generators]))", 1),
    ('report.update(final_common_counter=env.common_step_counter,donor_queries=',
     'report.update(hidden_scenario_history=action.scenarios.summary(),final_common_counter=env.common_step_counter,donor_queries=', 1),
    ("choices=range(4)", "choices=range(8)", 1),
    ("default=242100", "default=252100", 1),
    ("Fresh training cannot load old/candidate state", "Training uses only the method-bound full-state warm checkpoint", 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained runner anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
