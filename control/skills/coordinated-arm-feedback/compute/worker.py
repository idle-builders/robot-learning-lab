"""Counted composition of the observed finite arm stage/closure worker.

Dispatch source() as the provider function source. The prior worker stays frozen.
"""
from pathlib import Path


def source():
    root=Path(__file__).resolve().parents[4]
    raw=(root/'control/skills/coordinated-arm-reference/compute/worker.py').read_text()
    changes=[('run_coordinated_arm_reference','run_coordinated_arm_feedback',2),
        ('coordinated-arm-reference','coordinated-arm-feedback',7),
        ("reference_source=str(Path(sys.modules['reference'].__file__).resolve())",
         "interface_source=str(Path(sys.modules['interface'].__file__).resolve())",1),
        ("def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\\n')",
         "def write(path,value):Path(path).write_text(json.dumps(common.operational_json(value) if common is not None else value,indent=2,allow_nan=False)+'\\n')",1),
        ("report['distinct_fixed_policies']=1 if any(r.get('native',{}).get('distinct_fixed_policies')==1 for r in native_rows)else 0 if not native_rows else None",
         "report['distinct_fixed_policies']=len({r['native']['controller_identity']for r in native_rows if r.get('native',{}).get('controller_identity')})",1),
        ("report['start_families']=sum(r.get('native',{}).get('independent_start_families',0)for r in native_rows)if all('independent_start_families'in r.get('native',{})for r in native_rows)else None",
         "report['start_families']=len({f for r in native_rows for f in r.get('native',{}).get('start_family_ids',[])});report['start_family_count_semantics']='Distinct frozen condition shapes/seeds, paired evaluations deduplicated. Training uses one shared4row RNG stream; not4independently reseeded streams.'",1),
        ("report['NN_queries']=sum(r.get('native',{}).get('donor_actor_queries',0)for r in native_rows)if all('donor_actor_queries'in r.get('native',{})for r in native_rows)else None",
         "report['donor_queries']=sum(r.get('native',{}).get('donor_queries',0)for r in native_rows);report['NN_queries_note']='Only explicit donor reference queries counted here; PPO/control/value/minibatch network work is not a single shared inference counter.'",1)]
    for old,new,count in changes:
        if raw.count(old)!=count:raise ValueError('Exact retained worker anchor differs: '+old)
        raw=raw.replace(old,new)
    return raw


if __name__=='__main__':exec(compile(source(),str(__file__),'exec'),globals())
