"""Same full episode recorder, with hidden schedule/RNG and actual dose evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'control/skills/coordinated-arm-feedback/record.py'
source = BASE.read_text()
changes = [
    ("np.savez_compressed(path,**{k:v.detach().cpu().numpy().copy()for k,v in values.items()})",
     "values.update(action.scenarios.initialization(i))\n                np.savez_compressed(path,**{k:v.detach().cpu().numpy().copy()for k,v in values.items()})", 1),
    ('common_counter=env.common_step_counter,generation=int(action.feedback.generation[i]))',
     'common_counter=env.common_step_counter,generation=int(action.feedback.generation[i]),\n                    hidden_scenario=dict(action.scenarios.current[i]))', 1),
    ('physical_external_force_world_N=action.issued_force,feet_force_N=',
     'physical_external_force_world_N=action.issued_force,\n                scenario_onset_control=action.scenarios.onset,scenario_stop_control=action.scenarios.stop,\n                scenario_force_Y_N=action.scenarios.force_Y,scenario_generation=action.scenarios.generation,\n                scenario_applied_dose_world_N_s=action.scenarios.applied_dose,\n                scenario_applied_substeps=action.scenarios.applied_substeps,feet_force_N=', 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained recorder anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
