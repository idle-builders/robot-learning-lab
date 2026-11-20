"""Regenerate compact CPU/source parity evidence; no GPU or training claim.

Run: .venv-control/bin/python control/common/plant/audit.py
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np
import mujoco
import torch
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0,str(HERE.parent))
from runtime import BatchedRuntime, robot_cfg
from runtime.boundary import MODEL, source_contract


def load_scalar():
    source = ROOT / "robot/model/timestep"
    sys.path.insert(0,str(source))
    spec = importlib.util.spec_from_file_location("scalar_r3_runtime",source/"runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Runtime


class RecordedNoise:
    def __init__(self,generator):
        self.generator=generator
        self.draws=[]

    def normal(self,*args,**kwargs):
        value=self.generator.normal(*args,**kwargs)
        self.draws.append(value)
        return value


def reference(path):
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)


def native_properties(compiled):
    source=mujoco.MjModel.from_xml_path(str(MODEL/"cove50-rigid.xml"))
    differences={}
    for kind,fields in (("body",("mass","inertia","ipos","iquat","pos","quat")),
                        ("joint",("pos","axis","range","limited","stiffness")),
                        ("geom",("pos","quat","size","friction","solref","solimp","contype","conaffinity","condim"))):
        count={"body":source.nbody,"joint":source.njnt,"geom":source.ngeom}[kind]
        for field in fields:
            maximum=0.
            for i in range(count):
                a=getattr(source,kind)(i)
                if kind=="geom" and a.name=="floor":
                    continue
                b=getattr(compiled,kind)(a.name)
                maximum=max(maximum,float(np.max(abs(np.asarray(getattr(a,field),dtype=float)-np.asarray(getattr(b,field),dtype=float)))))
            differences[f"{kind}_{field}"]=maximum
    for field in ("dof_armature","dof_damping","dof_frictionloss"):
        differences[field]=float(np.max(abs(getattr(source,field)-getattr(compiled,field))))
    for field in ("ctrlrange","ctrllimited","forcelimited","gear","user"):
        a=getattr(source,"actuator_"+field)
        b=getattr(compiled,"actuator_"+field)
        differences["actuator_"+field]=float(np.max(abs(a.astype(float)-b.astype(float))))
    return differences


def reset_write_clock_audit():
    from runtime.mjlab_adapter import R3Actuator
    from mjlab.actuator.actuator import ActuatorCmd
    # Exercise the real custom actuator method, without another physics engine.
    actuator=R3Actuator.__new__(R3Actuator)
    actuator.boundary=BatchedRuntime(2,dtype=torch.float64)
    actuator.sensordata=torch.zeros((2,6),dtype=torch.float64)
    actuator.accel_adr=0
    actuator.gyro_adr=3
    q=torch.zeros((2,22),dtype=torch.float64)
    cmd=ActuatorCmd(position_target=q,velocity_target=q,effort_target=q,pos=q,vel=q)
    before=actuator.boundary.generator.get_state().clone()
    actuator.compute(cmd)  # non-integrating initial reset write
    after=actuator.boundary.generator.get_state().clone()
    reset_tick=actuator.boundary.tick.tolist()
    for _ in range(80):
        actuator.boundary.arm_physics_step()
        actuator.compute(cmd)
    control_tick=actuator.boundary.tick.tolist()
    actuator.boundary.reset(torch.tensor([1]))
    actuator.compute(cmd)  # non-integrating partial reset write for every world
    partial_tick=actuator.boundary.tick.tolist()
    for _ in range(80):
        actuator.boundary.arm_physics_step()
        actuator.compute(cmd)
    return dict(method="actual R3Actuator.compute called with explicit actual-step arm and unarmed MJLab reset writes",unarmed_initial_write_ticks=reset_tick,unarmed_initial_write_rng_state_difference_bytes=int(torch.count_nonzero(before!=after)),ticks_after80_armed_calls=control_tick,ticks_after_partial_reset_unarmed_write=partial_tick,ticks_after80_more_armed_calls=actuator.boundary.tick.tolist(),physics_steps=0,scope="Clock/queue mutation check only; separate task worker retains full CPU Warp integration")


def paired(runtime_class, *, native, dtype, rail, delay, feedback, peak):
    n=2
    model=mujoco.MjModel.from_xml_path(str(MODEL/"cove50-rigid.xml"))
    datas=[mujoco.MjData(model) for _ in range(n)]
    def scalar(i):
        rt=runtime_class(model,datas[i],rail_V=rail,command_delay_s=delay,
                         feedback_delay_s=feedback,peak=peak,seed=491+i)
        rt.rng=RecordedNoise(rt.rng)
        return rt
    scalars=[scalar(i) for i in range(n)]
    batch=BatchedRuntime(n,dtype=dtype,rail_V=rail,command_delay_s=delay,
                         feedback_delay_s=feedback,peak=peak,seed=44)
    qids,vids=scalars[0].qids,scalars[0].vids
    maxima={key:0. for key in ("torque_Nm","estimated_current_A","target_rad","previous_target_rad","observation","capture_clock_s","imu_capture_clock_s","gravity","qd_rad_s")}
    capture_steps=[]
    command_receipt=[]
    availability=[]
    requested=np.zeros((n,22))
    control_steps=0
    frames=[]
    for step in range(241 if native else 401):
        if not native and step==161:
            # Reset one world while the other retains filter, FIFO and clock.
            datas[1]=mujoco.MjData(model)
            scalars[1]=scalar(1)
            batch.reset(torch.tensor([1]))
            requested[1]=0
        inputs=[]
        noises=[]
        for i,rt in enumerate(scalars):
            d=datas[i]
            tick=step if i==0 or native or step<161 else step-161
            if not native:
                d.time=tick*.00025
                d.qpos[qids]=.31*np.sin(np.arange(22)*.37+tick*.017+i)
                d.qvel[vids]=8*np.sin(np.arange(22)*.29+tick*.011+i)
                d.sensor("imu_accel").data[:]=[.4*np.sin(tick*.03),.3,9.81]
                d.sensor("imu_gyro").data[:]=[.03,.02*np.sin(tick*.05),-.04]
            elif step==0:
                mujoco.mj_forward(model,d)
            if tick%80==0:
                if tick==0:
                    # Scalar constructor is conceptually after reset pose write,
                    # matching MJLab's lazy initialization at the first substep.
                    rt.target=d.qpos[qids].copy()
                    rt.previous_command=rt.target.copy()
                requested[i]=.13*np.sin(np.arange(22)*.31+tick*.01+i)
                rt.command(rt.names,requested[i])
                control_steps+=1
            batch.accel_bias[i]=batch.tensor(rt.accel_bias)
            batch.gyro_bias[i]=batch.tensor(rt.gyro_bias)
            inputs.append((d.qpos[qids].copy(),d.qvel[vids].copy(),d.sensor("imu_accel").data.copy(),d.sensor("imu_gyro").data.copy()))
            rt.rng.draws=[]
            old_capture=rt.seq
            rt.update()
            if rt.seq!=old_capture and i==0:
                capture_steps.append(dict(tick=step,capture_time_s=float(d.time),imu_capture_time_s=float(rt.sample["capture_time_s"])))
            noises.append(np.concatenate(rt.rng.draws) if rt.rng.draws else np.zeros(6))
        arrays=[batch.tensor(np.array([row[j] for row in inputs])) for j in range(4)]
        batch.step(batch.tensor(requested),*arrays,sensor_noise=batch.tensor(np.array(noises)))
        for i,rt in enumerate(scalars):
            comparisons={"torque_Nm":(batch.torque[i],rt.torque),"estimated_current_A":(batch.current[i],rt.current),"target_rad":(batch.target[i],rt.target),"previous_target_rad":(batch.previous_command[i],rt.previous_command),"gravity":(batch.gravity_estimate[i],rt.gravity_estimate)}
            obs=rt.observation((.3,0.,0.))
            if bool(batch.valid[i]) != (obs is not None):
                availability.append(dict(step=step,world=i,scalar_available=obs is not None,batch_available=bool(batch.valid[i])))
            if obs is not None:
                comparisons["observation"]=(batch.observation((.3,0.,0.))[i],obs["values"])
                comparisons["capture_clock_s"]=(batch.capture_time_s[i],obs["capture_time_s"])
                comparisons["imu_capture_clock_s"]=(batch.imu_capture_time_s[i],obs["imu_capture_time_s"])
                comparisons["qd_rad_s"]=(batch.last_values[i,22:44],rt.last["qd"])
            for key,(actual,expected) in comparisons.items():
                maxima[key]=max(maxima[key],float(np.max(abs(actual.detach().cpu().numpy()-expected))))
        if step in (0,39,40,79,80,81,120,160,200,240):
            command_receipt.append(dict(tick=step,pre_physics_time_s=float(datas[0].time),first_target_rad=float(batch.target[0,0]),first_previous_target_rad=float(batch.previous_command[0,0]),valid=bool(batch.valid[0]),capture_time_s=float(batch.capture_time_s[0]),imu_capture_time_s=float(batch.imu_capture_time_s[0])))
        if native:
            frames.append(dict(tick=step,qpos=[d.qpos.tolist() for d in datas],qvel=[d.qvel.tolist() for d in datas],torque=[rt.torque.tolist() for rt in scalars],imu_native=[np.r_[d.sensor("imu_accel").data,d.sensor("imu_gyro").data].tolist() for d in datas]))
            for d in datas:
                mujoco.mj_step(model,d)
    return dict(mode="native scalar physics inputs; matched batched algebra" if native else "synthetic state inputs with one-world partial reset",dtype=str(dtype),world_slots=n,independent_start_families=1 if native else 0,full_episodes=2 if native else 0,physics_steps=482 if native else 0,batched_boundary_calls=241 if native else 401,control_decisions=control_steps if native else 0,rail_V=rail,command_delay_s=delay,feedback_delay_s=feedback,peak=peak,max_absolute_difference=maxima,availability_differences=availability,packet_captures_world0=capture_steps,selected_clock_receipts=command_receipt),frames


def main():
    start=time.monotonic()
    rt=load_scalar()
    cases=[]
    for rail,delay,feedback,peak in ((12.,.02,.01,False),(10.,.04,.03,True),(12.,0.,0.,True)):
        for dtype in (torch.float64,torch.float32):
            case,_=paired(rt,native=False,dtype=dtype,rail=rail,delay=delay,feedback=feedback,peak=peak)
            cases.append(case)
    native,frames=paired(rt,native=True,dtype=torch.float64,rail=12.,delay=.02,feedback=.01,peak=False)
    from mjlab.entity import Entity
    entity=Entity(robot_cfg())
    model=entity.compile()
    source=source_contract()
    report=dict(schema="cove50.r3-batched-boundary-audit.v1",source_model=source,
        implementation=[reference(p) for p in sorted(HERE.glob("*.py"))],
        frozen_scalar=reference(ROOT/"robot/model/timestep/runtime.py"),
        synthetic_cases=cases,native_case=native,
        reset_write_clock_audit=reset_write_clock_audit(),
        entity_compile=dict(nq=model.nq,nv=model.nv,nu=model.nu,mass_kg=float(model.body_mass.sum()),
            installed_actuator_names=list(entity.actuator_names),boundary_joint_names=entity.actuators[0].target_names,
            maximum_native_property_difference=native_properties(model),
            passive_joint_dynamics="preserved source XML armature/frictionloss/damping"),
        clock="Runtime called before each native step; boundary tick increments after its pre-step update. Observation age at control boundary includes elapsed last physics tick.",
        noise="Matched check injects exact scalar NumPy biases/noise; production Torch uses same source distributions but a different random stream. RNG stream is shared across worlds; reset draw order changes future samples, not marginal distributions.",
        remaining=["No GPU/MJWarp integration/throughput measured; Entity compilation only.","Torch float32 differences measured on source inputs, not independent native GPU trajectories.","Motor/sensor law remains provisional simulation assumptions; no physical calibration.","Native audit is 60.25ms from neutral source state, not walking policy evaluation."],
        evidence_grades=[dict(dimension="deterministic motor algebra",grade_0_to_5=4.5,confidence="high within sampled inputs",basis="Exact source scalar comparison with finite-precision differences reported; no physical motor identification"),dict(dimension="sensor cadence and delayed history",grade_0_to_5=4.,confidence="high within paired fixtures",basis="Matched scalar acquisition/delivery clocks, complementary filter and one-world partial reset; production random streams differ"),dict(dimension="MJLab integration evidence",grade_0_to_5=2.,confidence="limited",basis="Entity compiled with exact actuator order and named native properties; GPU runtime/throughput not exercised here")],
        training=dict(simulator_transitions=0,optimizer_updates=0,distinct_policies=0),elapsed_wall_s=time.monotonic()-start)
    out=ROOT/"data/walk-resume-r1-20261026/runtime"
    out.mkdir(parents=True,exist_ok=True)
    (out/"audit.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    # Short original native observations retained in full, not a disposable render.
    (out/"native-input-trace.json").write_text(json.dumps(dict(schema="cove50.r3-boundary-native-inputs.v1",frames=frames),separators=(",",":"),allow_nan=False)+"\n")
    print(json.dumps(dict(report=str(out/"audit.json"),cases=len(cases),maxima=[c["max_absolute_difference"] for c in cases],native=native["max_absolute_difference"],entity_order=list(entity.actuator_names)==entity.actuators[0].target_names,elapsed_wall_s=report["elapsed_wall_s"])))


if __name__=="__main__":
    main()
