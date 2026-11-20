"""Keep reviewed task intact except hidden real-reset scenario/force hooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'control/skills/coordinated-arm-feedback/task.py'
source = BASE.read_text()
changes = [
    ('from interface import Interface', 'from interface import Interface\nfrom scenario import Scenarios', 1),
    ('signs=torch.tensor(impulse_signs,device=env.device,dtype=torch.float32)',
     "scenario=Scenarios(impulse_signs['seeds'],device=env.device,fixed=impulse_signs.get('fixed'))\n            action.scenarios=scenario", 1),
    ('action.physical_delta_rate[ids]=0;action.total_request_rate[ids]=0',
     "action.physical_delta_rate[ids]=0;action.total_request_rate[ids]=0\n                scenario.reset(ids,s.generation[ids])\n                action.issued_force[ids]=0\n                zero=torch.zeros((len(ids),1,3),device=env.device)\n                action._entity.write_external_wrench_to_sim(zero,zero,env_ids=ids,body_ids=[torso])", 1),
    ('force=lateral_force(env.episode_length_buf,signs,torch=torch)',
     'force=scenario.force(env.episode_length_buf,native.runtime_package().get_runtime(env).physics_dt_s)', 1)]
for old, new, count in changes:
    if source.count(old) != count:
        raise ValueError('Exact retained task anchor differs: ' + old)
    source = source.replace(old, new)
exec(compile(source, str(__file__), 'exec'), globals())
