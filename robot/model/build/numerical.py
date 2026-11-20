"""Three supplemental same-start numerical probes, separate from primary cohort."""
from replay import episode
from common import *

def main():
    rows=[];primary=read(OUT/'replay-summary.json')['episodes']
    for variant,dt in [('rigid',2e-6),('compliant',5e-6),('compliant',1e-6)]:
        row,_=episode(variant,dict(name=f'numerical-dt{int(dt*1e6)}us',rail_V=12.,delay_s=.02,physics_timestep_s=dt),0,.8);rows.append(row)
    comparisons=[]
    for row in rows:
        base=next(r for r in primary if r['variant']==row['variant']and r['condition']['name']=='nominal'and r['start_family']==0)
        a=np.load(ROOT/base['trace']['path']);b=np.load(ROOT/row['trace']['path'])
        difference=np.linalg.norm(a['root_position_m']-b['root_position_m'],axis=1)
        comparisons.append(dict(variant=row['variant'],primary_dt_s=base['physics_timestep_s'],probe_dt_s=row['physics_timestep_s'],
            primary_rms_rad=base['rms_joint_tracking_error_rad'],probe_rms_rad=row['rms_joint_tracking_error_rad'],
            relative_rms_change=row['rms_joint_tracking_error_rad']/base['rms_joint_tracking_error_rad']-1,
            root_position_difference_m_percentiles=dict(zip(['p50','p90','p99'],np.quantile(difference,[.5,.9,.99]).tolist()))))
    write(OUT/'numerical-sensitivity.json',dict(episodes=rows,comparisons=comparisons,native_physics_steps=sum(r['physics_steps']for r in rows),
        scope='Three supplemental matched-step numerical probes, same start0 and target stream; separate from six-condition primary cohort.',
        interpretation='Native timestep sensitivity is evidence, not proof of convergence or physical-current fidelity. Interpret rigid and flexible branches separately.'))

if __name__=='__main__':main()
