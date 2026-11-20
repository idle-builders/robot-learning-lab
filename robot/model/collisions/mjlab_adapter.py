"""A new22-axis MJLab entity; intentionally separate from sealed21-axis tasks.

Import under .venv-control. New policies must define a task around this entity;
existing Direct Frame checkpoint/task guards and registry are unchanged.
"""
from functools import partial
from pathlib import Path
import re
from common import *

def robot_spec():
    import mujoco
    spec=mujoco.MjSpec.from_file(str(OUT/'cove50-rigid.xml'))
    spec.delete(spec.geom('floor'))
    # The framework supplies terrain and contacts. Drive its DcMotorActuator
    # from output target radians rather than preserving raw torque XML motors.
    for actuator in list(spec.actuators):spec.delete(actuator)
    return spec

def robot_cfg(*, rail_V=12., command_delay_s=.02):
    from mjlab.actuator import DcMotorActuatorCfg
    from mjlab.entity import EntityArticulationInfoCfg,EntityCfg
    from mjlab.utils.spec_config import CollisionCfg
    acts=read(OUT/'actuators.json')['actuators']
    configs=[]
    for a in acts:
        scale=(rail_V if a['model']=='c047' else min(rail_V,12.))/12.
        configs.append(DcMotorActuatorCfg(target_names_expr=('^'+re.escape(a['joint'])+'$',),
            stiffness=a['simulation_kp_Nm_rad'],damping=a['simulation_kd_Nm_s_rad'],
            effort_limit=(.980665 if a['model']=='c047' else .2)*scale,
            saturation_effort=(2.941995 if a['model']=='c047' else .71)*scale,
            velocity_limit=(np.pi/3/.222 if a['model']=='c047' else 57*2*np.pi/60)*scale,
            delay_min_lag=round(command_delay_s/.001),delay_max_lag=round(command_delay_s/.001)))
    return EntityCfg(spec_fn=robot_spec,sort_actuators=False,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.30201),joint_pos={a['joint']:0. for a in acts}),
        collisions=(CollisionCfg(geom_names_expr=(r'^collision_.*$',),contype=1,conaffinity=1,condim=3,friction=(.8,),disable_other_geoms=True),),
        articulation=EntityArticulationInfoCfg(actuators=tuple(configs),soft_joint_pos_limit_factor=.9))

def describe():
    return dict(joint_order=[a['joint']for a in read(OUT/'actuators.json')['actuators']],
        observation='New75-value runtime contract; task must wire it explicitly. Old72-value observation functions must not be reused silently.',
        motor_difference='MJLab upstream DC envelope is a batched approximation. CPU runtime uses digitized nonlinear XM335 curve; here neck0.20Nm cap and0.71Nm saturation preserve its distinct curve/stall distinction. No backlash/thermal identification.',
        flexible='Full coupled FEM variant is native CPU MJCF. MJWarp tendon support/performance not yet exercised; do not silently substitute it in this rigid entity.',
        source=ref(OUT/'cove50-rigid.xml'))

if __name__=='__main__':
    from mjlab.entity import Entity
    cfg=robot_cfg();entity=Entity(cfg);m=entity.compile();write(OUT/'mjlab-entity-interface.json',dict(**describe(),
        compiled=dict(nq=m.nq,nv=m.nv,nu=m.nu,mass_kg=float(m.body_mass.sum())),actuator_configs=len(cfg.articulation.actuators),
        installed_actuator_order=list(entity.actuator_names),
        qualification='Actual MJLab Entity constructed with22actuator configs and compiled locally; zero trainer transitions or optimizer updates. GPU/MJWarp rollout not exercised; not a learned skill.'))
    print(dict(nq=m.nq,nv=m.nv,actuator_configs=len(cfg.articulation.actuators)))
