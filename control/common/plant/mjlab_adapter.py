"""MJLab custom actuator: exact R3 motor and pre-physics sensor boundary."""
from dataclasses import dataclass
from functools import partial
import json
import math
import re
import xml.etree.ElementTree as ET
import mujoco
import warp as wp
from mjlab.actuator.actuator import Actuator, ActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec import create_motor_actuator
from mjlab.utils.spec_config import CollisionCfg
from .boundary import BatchedRuntime, MODEL, source_contract


@dataclass(kw_only=True)
class R3ActuatorCfg(ActuatorCfg):
    rail_V: float = 12.
    command_delay_s: float = .02
    feedback_delay_s: float = .01
    peak: bool = False
    seed: int = 0
    c047: dict | None = None  # substitute servo law on the C047 joints (BatchedRuntime)
    duty: dict | None = None  # r4 Feetech firmware duty loop on the C047 joints (dynamics.resolve()["duty"])
    ranges_deg: dict | None = None  # r4 best-estimate ranges for joints without a source range (dynamics.resolve()["ranges_deg"])
    joint_models: dict | None = None  # mixed sizing: per-joint servo law and duty loop (dynamics.resolve()["joints"])
    physics_dt_s: float | None = None  # coarser physics step for r4 training (None keeps R3's .25 ms)

    def build(self, entity, target_ids, target_names):
        names = [a["joint"] for a in json.loads((MODEL/"actuators.json").read_text())["actuators"]]
        lookup = dict(zip(target_names,target_ids))
        return R3Actuator(self,entity,[lookup[n] for n in names],names)


class R3Actuator(Actuator):
    def edit_spec(self,spec,target_names):
        lookup = {n.split("/")[-1]:n for n in target_names}
        source_axes = {a.attrib["joint"]:a for a in ET.parse(MODEL/"cove50-rigid.xml").getroot().find("actuator")}
        models = {a["joint"]:a["model"] for a in json.loads((MODEL/"actuators.json").read_text())["actuators"]}
        for source_name in self.target_names:
            name = lookup[source_name]
            # Preserve the XML's final control clamp as well as the Runtime
            # law. This matters for >12 V peak sensitivity: source Runtime
            # requested torque can exceed the XML's fixed nominal ctrlrange.
            source = source_axes[source_name]
            lower,upper = map(float,source.attrib["ctrlrange"].split())
            if lower != -upper:
                raise ValueError("Expected symmetric selected XML motor control range")
            law = ((self.cfg.joint_models or {}).get(source_name) or {}).get("law") or self.cfg.c047
            if law and models[source_name] == "c047":
                # A substitute servo's own stall torque bounds its control range.
                upper *= law["stall_torque_Nm"]/float(source.attrib["user"].split()[0])
            actuator = create_motor_actuator(spec,name,effort_limit=upper)
            actuator.forcelimited = False  # source has ctrl limit, no force limit
            actuator.userdata = list(map(float,source.attrib["user"].split()))
            self._mjs_actuators.append(actuator)

    def initialize(self,mj_model,model,data,device):
        expected = self.cfg.physics_dt_s or BatchedRuntime.physics_dt_s
        if abs(mj_model.opt.timestep-expected)>1e-12:
            raise ValueError(f"R3 boundary requires the {expected} s physics timestep it was configured with")
        super().initialize(mj_model,model,data,device)
        self.boundary = BatchedRuntime(data.nworld,device=device,rail_V=self.cfg.rail_V,
            command_delay_s=self.cfg.command_delay_s,feedback_delay_s=self.cfg.feedback_delay_s,
            peak=self.cfg.peak,seed=self.cfg.seed,c047=self.cfg.c047,duty=self.cfg.duty,ranges_deg=self.cfg.ranges_deg,joint_models=self.cfg.joint_models,physics_dt_s=self.cfg.physics_dt_s)
        if list(self.target_names) != self.boundary.names:
            raise ValueError("MJLab target order differs from selected actuator order")
        # MJLab DataBridge already exposes a TorchArray tensor proxy, while
        # direct MJWarp use may still supply a Warp array.
        self.sensordata = wp.to_torch(data.sensordata) if isinstance(data.sensordata,wp.array) else data.sensordata
        self.accel_adr = self._sensor_address(mj_model,"imu_accel")
        self.gyro_adr = self._sensor_address(mj_model,"imu_gyro")

    def _sensor_address(self,model,suffix):
        ids = [i for i in range(model.nsensor) if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_SENSOR,i).split("/")[-1] == suffix]
        if len(ids)!=1 or model.sensor_dim[ids[0]]!=3:
            raise ValueError(f"Expected one native 3-axis {suffix} sensor")
        return int(model.sensor_adr[ids[0]])

    def compute(self,cmd):
        # ManagerBasedRlEnv also calls scene.write_data_to_sim at reset and
        # partial reset without integrating physics. The action term arms only
        # actual decimation-loop calls; extra writes must not advance ANY world.
        if not self.boundary.physics_armed:
            return self.boundary.torque
        self.boundary.physics_armed = False
        return self.boundary.step(cmd.position_target,cmd.pos,cmd.vel,
            self.sensordata[:,self.accel_adr:self.accel_adr+3],
            self.sensordata[:,self.gyro_adr:self.gyro_adr+3])

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,"boundary"):
            self.boundary.reset(env_ids)


def robot_spec(dynamics=None):
    """The selected R3 spec; with dynamics (dynamics.resolve()), r4 rotor inertia and Coulomb friction per servo model."""
    source_contract()
    spec = mujoco.MjSpec.from_file(str(MODEL / "cove50-rigid.xml"))
    spec.delete(spec.geom("floor"))
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    if dynamics:
        per = dynamics.get("joints") or {}
        for axis in json.loads((MODEL/"actuators.json").read_text())["actuators"]:
            joint = spec.joint(axis["joint"])
            own = per.get(axis["joint"]) or {}
            joint.armature = own.get("armature", dynamics["armature"][axis["model"]])
            joint.frictionloss = own.get("frictionloss", dynamics["frictionloss"][axis["model"]])
        for name, (lo, hi) in dynamics.get("ranges_deg", {}).items():
            joint = spec.joint(name)
            joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            joint.range = [lo, hi] if spec.compiler.degree else [math.radians(lo), math.radians(hi)]
    return spec


def robot_cfg(*,rail_V=12.,command_delay_s=.02,feedback_delay_s=.01,peak=False,seed=0,c047=None,dynamics=None,physics_dt_s=None):
    """dynamics: dynamics.resolve() output for r4 best-estimate actuator dynamics (None keeps R3 exactly)."""
    # Resolve declared names without constructing a random sensor runtime.
    names = [a["joint"] for a in json.loads((MODEL/"actuators.json").read_text())["actuators"]]
    if dynamics:
        c047 = dict(dynamics["law"], **(c047 or {}))
        rail_V = dynamics.get("rail_V", rail_V)  # the best-estimate supply rail (catalog curves scale with it)
    return EntityCfg(spec_fn=partial(robot_spec,dynamics),sort_actuators=False,
        init_state=EntityCfg.InitialStateCfg(pos=(0.,0.,.30201),joint_pos={n:0. for n in names}),
        collisions=(CollisionCfg(geom_names_expr=(r"^collision_.*$",),contype=1,conaffinity=1,
            condim=3,friction=(.8,),disable_other_geoms=True),),
        articulation=EntityArticulationInfoCfg(actuators=(R3ActuatorCfg(
            target_names_expr=tuple("^"+re.escape(n)+"$" for n in names),
            rail_V=rail_V,command_delay_s=command_delay_s,feedback_delay_s=feedback_delay_s,
            peak=peak,seed=seed,c047=c047,duty=dynamics and dynamics["duty"],ranges_deg=dynamics and dynamics.get("ranges_deg"),
            joint_models=dynamics and dynamics.get("joints") and {j: dict(law=v["law"], duty=v["duty"]) for j, v in dynamics["joints"].items()},
            physics_dt_s=physics_dt_s),),
        soft_joint_pos_limit_factor=.9))
